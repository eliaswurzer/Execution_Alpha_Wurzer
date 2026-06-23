"""Consolidate per-period processed TAQ roots into one loader root.

The streaming preprocessing produced three sibling roots
(``sp500_preprocess_2018h1_streaming``, ``..._2018h2_streaming``,
``..._2019_streaming``).  The analysis loader resolves one root per year, so
this script merges all date directories into a single target root with the
standard ``<YYYYMMDD>/{trades,nbbo,qc}/`` layout.

Default mode is ``move`` (same-volume ``os.replace``: instant and
space-neutral).  Duplicate (date, side, symbol) files across sources are
resolved by a documented precedence rule and the losing copies are left in
place and counted, never overwritten; cleaning up the leftover source trees
is an explicit manual step after verification.

The script also verifies the result (file counts and sampled sizes per date
and side) and checks point-in-time S&P 500 membership coverage per date so
that missing active constituents are visible before any downstream run.

Usage::

    python -m preprocessing.consolidate_processed_roots --dry-run
    python -m preprocessing.consolidate_processed_roots
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

_CODING_ROOT = Path(__file__).resolve().parents[1]
if str(_CODING_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODING_ROOT))

from analysis.utils.symbols import canonical_symbol  # noqa: E402

DATA_ROOT = Path(os.environ.get("THESIS_DATA_ROOT", "data"))
DEFAULT_SOURCES = (
    DATA_ROOT / "sp500_preprocess_2018h1_streaming",
    DATA_ROOT / "sp500_preprocess_2018h2_streaming",
    DATA_ROOT / "sp500_preprocess_2019_streaming",
)
DEFAULT_TARGET = DATA_ROOT / "sp500_preprocess_2018_2019"
DEFAULT_MEMBERSHIP = (
    Path(__file__).resolve().parents[2]
    / "reference" / "index_membership" / "sp500_membership_intervals.csv"
)
SIDES = ("trades", "nbbo")


def _is_date_dir(name: str) -> bool:
    return len(name) == 8 and name.isdigit()


def _ordered_sources(date_str: str, sources: list[Path]) -> list[Path]:
    """Precedence per date.

    With the three default streaming roots, dates from 2018-07-01 onward
    prefer the h2 root over the h1 root (the h2 root holds the complete
    second-half build; the h1 root only carries partial late-2018 spillover).
    Otherwise the CLI order is the precedence order.
    """
    names = [s.name for s in sources]
    if (
        date_str >= "20180701"
        and "sp500_preprocess_2018h1_streaming" in names
        and "sp500_preprocess_2018h2_streaming" in names
    ):
        ranked = sorted(
            sources,
            key=lambda s: 0 if s.name == "sp500_preprocess_2018h2_streaming" else 1
            if s.name == "sp500_preprocess_2018h1_streaming" else 2,
        )
        return ranked
    return list(sources)


def _plan_date(
    date_str: str,
    ordered: list[Path],
    target: Path,
) -> tuple[list[tuple[Path, Path, int]], dict]:
    """Return (moves, stats) for one date directory.

    ``moves`` are (source_file, target_file, size) tuples.  Side files are
    deduplicated by file name across sources (winner first); every other
    entry under the date directory (qc/, manifests) is taken from the first
    source that provides it.
    """
    moves: list[tuple[Path, Path, int]] = []
    stats = {
        "files_planned": 0,
        "bytes_planned": 0,
        "duplicates_skipped": 0,
        "already_present": 0,
        "conflict_skipped": 0,
        "per_source": defaultdict(int),
        "side_counts": {side: 0 for side in SIDES},
    }
    seen: dict[tuple[str, str], Path] = {}

    def _plan_file(rel: Path, src_file: Path, source: Path) -> None:
        key = (str(rel.parent), rel.name)
        if key in seen:
            stats["duplicates_skipped"] += 1
            return
        seen[key] = src_file
        dst = target / date_str / rel
        size = src_file.stat().st_size
        if dst.exists():
            if dst.stat().st_size == size:
                stats["already_present"] += 1
            else:
                # On a rerun the winning copy already lives in the target and
                # only the losing duplicate remains in its source tree; the
                # source file is never allowed to overwrite the target. The
                # count is surfaced in the manifest for review.
                stats["conflict_skipped"] += 1
            return
        moves.append((src_file, dst, size))
        stats["files_planned"] += 1
        stats["bytes_planned"] += size
        stats["per_source"][source.name] += 1
        if rel.parts and rel.parts[0] in SIDES:
            stats["side_counts"][rel.parts[0]] += 1

    for source in ordered:
        date_dir = source / date_str
        if not date_dir.is_dir():
            continue
        for path in sorted(date_dir.rglob("*")):
            if path.is_file():
                _plan_file(path.relative_to(date_dir), path, source)
    return moves, stats


def _execute(moves: list[tuple[Path, Path, int]], mode: str) -> None:
    made_dirs: set[Path] = set()
    for src, dst, _size in moves:
        parent = dst.parent
        if parent not in made_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            made_dirs.add(parent)
        if mode == "move":
            src.replace(dst)
        else:
            shutil.copy2(src, dst)


def _verify_date(
    date_str: str,
    target: Path,
    expected_stats: dict,
    planned_sizes: dict[str, int],
    sample_n: int,
) -> list[str]:
    problems: list[str] = []
    date_dir = target / date_str
    for side in SIDES:
        expected = expected_stats["side_counts"][side] + expected_stats.get(
            f"pre_existing_{side}", 0,
        )
        side_dir = date_dir / side
        actual = (
            sum(1 for p in side_dir.iterdir() if p.suffix == ".parquet")
            if side_dir.is_dir() else 0
        )
        if actual < expected:
            problems.append(
                f"{date_str}/{side}: {actual} files in target, expected >= {expected}"
            )
    for rel, size in list(planned_sizes.items())[:sample_n]:
        dst = date_dir / rel
        if not dst.exists():
            problems.append(f"{date_str}: planned file missing after move: {rel}")
        elif dst.stat().st_size != size:
            problems.append(
                f"{date_str}: size mismatch {rel}: {dst.stat().st_size} != {size}"
            )
    return problems


def _load_membership(path: Path) -> list[tuple[str, _dt.date, _dt.date | None]]:
    import csv

    rows: list[tuple[str, _dt.date, _dt.date | None]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            symbol = canonical_symbol(row.get("symbol", ""))
            if not symbol:
                continue
            start = _dt.date.fromisoformat(str(row["effective_from"])[:10])
            raw_to = str(row.get("effective_to", "") or "").strip()
            end = _dt.date.fromisoformat(raw_to[:10]) if raw_to else None
            rows.append((symbol, start, end))
    return rows


def _membership_report(
    target: Path,
    dates: list[str],
    membership: list[tuple[str, _dt.date, _dt.date | None]],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    total_missing = 0
    dates_with_missing = 0
    for date_str in dates:
        day = _dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        active = {
            sym for sym, start, end in membership
            if start <= day and (end is None or day <= end)
        }
        trades_dir = target / date_str / "trades"
        present = set()
        if trades_dir.is_dir():
            present = {
                canonical_symbol(p.stem)
                for p in trades_dir.iterdir() if p.suffix == ".parquet"
            }
        missing = sorted(active - present)
        extra = sorted(present - active)
        rows.append({
            "date": date_str,
            "active_members": len(active),
            "present_trades": len(present),
            "active_missing_trades": len(missing),
            "present_non_members": len(extra),
            "missing_symbols": ";".join(missing[:25]),
        })
        if missing:
            dates_with_missing += 1
            total_missing += len(missing)
    summary = {
        "dates_checked": len(dates),
        "dates_with_missing_active_trades": dates_with_missing,
        "total_missing_active_symbol_days": total_missing,
    }
    return rows, summary


def _safe_symbol(symbol: str) -> str:
    """File-safe symbol form used for parquet stems (matches the audit)."""
    return str(symbol).replace(" ", "_").replace("/", "_")


def _read_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    """Manifest rows keyed by the file-safe symbol form.

    Manifests store the raw TAQ symbol (``DISC A``) while parquet stems use
    the file-safe form (``DISC_A``); keying by the safe form keeps the merge
    comparable with the parquet sets, and the original symbol string is
    preserved inside the row for writing back.
    """
    import csv

    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            symbol = row.get("symbol")
            if symbol:
                rows[_safe_symbol(symbol)] = row
    return rows


def _side_listed(row: dict[str, str] | None, col: str) -> bool:
    if not row:
        return False
    try:
        return int(float(row.get(col, 0) or 0)) > 0
    except ValueError:
        return False


def repair_aux(sources: list[Path], target: Path) -> dict:
    """Post-consolidation repair of root-level auxiliaries.

    The file move only covered date directories. The audit additionally
    requires (a) ``coverage/<date>_summary.json`` at the target root and
    (b) per-date ``manifest.csv`` files whose per-side symbol sets match the
    consolidated parquet sets. Overlap dates merged from two sources carry
    only the winner's manifest, so losing-source rows are merged in per side.
    """
    import csv

    stats = {"coverage_moved": 0, "coverage_skipped": 0,
             "manifests_merged": 0, "manifest_problems": []}
    coverage_dir = target / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    all_dates = sorted(
        e.name for e in target.iterdir() if e.is_dir() and _is_date_dir(e.name)
    )
    for date_str in all_dates:
        ordered = _ordered_sources(date_str, sources)
        dst_cov = coverage_dir / f"{date_str}_summary.json"
        if not dst_cov.exists():
            for source in ordered:
                src_cov = source / "coverage" / f"{date_str}_summary.json"
                if src_cov.exists():
                    src_cov.replace(dst_cov)
                    stats["coverage_moved"] += 1
                    break
        else:
            stats["coverage_skipped"] += 1

        date_dir = target / date_str
        trade_set = {p.stem for p in (date_dir / "trades").glob("*.parquet")}
        nbbo_set = {p.stem for p in (date_dir / "nbbo").glob("*.parquet")}
        manifest_path = date_dir / "manifest.csv"
        rows = _read_manifest_rows(manifest_path)
        manifest_trade = {s for s, r in rows.items() if _side_listed(r, "trade_rows")}
        manifest_nbbo = {s for s, r in rows.items() if _side_listed(r, "nbbo_rows")}
        if manifest_trade == trade_set and manifest_nbbo == nbbo_set:
            continue
        # Merge in rows from the losing sources, per side and per symbol.
        source_rows = [
            _read_manifest_rows(source / date_str / "manifest.csv")
            for source in ordered
        ]
        merged: dict[str, dict[str, str]] = {}
        problems_before = len(stats["manifest_problems"])
        for symbol in sorted(trade_set | nbbo_set):
            candidates = [rows.get(symbol)] + [sr.get(symbol) for sr in source_rows]
            trade_rows = next(
                (c["trade_rows"] for c in candidates
                 if symbol in trade_set and _side_listed(c, "trade_rows")), "0",
            )
            nbbo_rows = next(
                (c["nbbo_rows"] for c in candidates
                 if symbol in nbbo_set and _side_listed(c, "nbbo_rows")), "0",
            )
            if symbol in trade_set and trade_rows == "0":
                stats["manifest_problems"].append(
                    f"{date_str}: no source manifest row for trade {symbol}"
                )
            if symbol in nbbo_set and nbbo_rows == "0":
                stats["manifest_problems"].append(
                    f"{date_str}: no source manifest row for nbbo {symbol}"
                )
            raw_symbol = next(
                (c["symbol"] for c in candidates if c and c.get("symbol")), symbol,
            )
            merged[symbol] = {
                "symbol": raw_symbol, "trade_rows": trade_rows, "nbbo_rows": nbbo_rows,
            }
        if len(stats["manifest_problems"]) > problems_before:
            continue
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["symbol", "trade_rows", "nbbo_rows"],
            )
            writer.writeheader()
            writer.writerows(merged.values())
        stats["manifests_merged"] += 1
    log.info("Aux repair: %s", {k: (len(v) if isinstance(v, list) else v)
                                for k, v in stats.items()})
    return stats


def consolidate(
    sources: list[Path],
    target: Path,
    *,
    mode: str = "move",
    dry_run: bool = False,
    membership_file: Path | None = None,
    hash_samples: int = 3,
) -> dict:
    started = time.perf_counter()
    sources = [Path(s) for s in sources if Path(s).is_dir()]
    if not sources:
        raise SystemExit("No existing source roots given")
    target.mkdir(parents=True, exist_ok=True)

    all_dates = sorted({
        entry.name
        for source in sources
        for entry in source.iterdir()
        if entry.is_dir() and _is_date_dir(entry.name)
    })
    log.info("Consolidating %d dates from %d sources into %s (mode=%s%s)",
             len(all_dates), len(sources), target, mode,
             ", DRY-RUN" if dry_run else "")

    totals = defaultdict(int)
    per_source_totals = defaultdict(int)
    problems: list[str] = []
    date_records: list[dict] = []

    for i, date_str in enumerate(all_dates, 1):
        ordered = _ordered_sources(date_str, sources)
        moves, stats = _plan_date(date_str, ordered, target)
        planned_sizes = {
            str(dst.relative_to(target / date_str)): size
            for _src, dst, size in moves
        }
        if not dry_run:
            _execute(moves, mode)
            problems.extend(_verify_date(
                date_str, target, stats, planned_sizes, hash_samples,
            ))
        totals["files"] += stats["files_planned"]
        totals["bytes"] += stats["bytes_planned"]
        totals["duplicates_skipped"] += stats["duplicates_skipped"]
        totals["already_present"] += stats["already_present"]
        totals["conflict_skipped"] += stats["conflict_skipped"]
        for name, count in stats["per_source"].items():
            per_source_totals[name] += count
        date_records.append({
            "date": date_str,
            "winner_source": ordered[0].name if ordered else "",
            **{k: v for k, v in stats.items() if k != "per_source"},
            "side_counts": dict(stats["side_counts"]),
        })
        if i % 50 == 0 or i == len(all_dates):
            log.info("  %d/%d dates  files=%d  dup_skipped=%d",
                     i, len(all_dates), totals["files"], totals["duplicates_skipped"])

    membership_summary: dict = {}
    if membership_file and membership_file.exists() and not dry_run:
        membership = _load_membership(membership_file)
        rows, membership_summary = _membership_report(target, all_dates, membership)
        import csv as _csv
        report_path = target / "membership_check.csv"
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Membership check: %s", membership_summary)

    manifest = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dry_run": dry_run,
        "sources": [str(s) for s in sources],
        "target": str(target),
        "dates": len(all_dates),
        "totals": dict(totals),
        "per_source_files": dict(per_source_totals),
        "verify_problems": problems,
        "status": "dry_run" if dry_run else ("failed_verify" if problems else "complete"),
        "membership_summary": membership_summary,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    manifest_path = target / "consolidation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    detail_path = target / "consolidation_dates.json"
    detail_path.write_text(json.dumps(date_records, indent=2), encoding="utf-8")

    if problems:
        for p in problems[:20]:
            log.error("VERIFY: %s", p)
        raise SystemExit(
            f"Consolidation verification failed with {len(problems)} problem(s); "
            f"see {manifest_path}"
        )
    log.info("Consolidation %s: %d files, %.1f GB, %d duplicates skipped, %.1fs",
             manifest["status"], totals["files"], totals["bytes"] / 1e9,
             totals["duplicates_skipped"], manifest["runtime_seconds"])
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", type=Path, default=None,
                   help="Source root (repeatable; default: the three streaming roots)")
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--mode", choices=["move", "copy"], default="move")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--membership-file", type=Path, default=DEFAULT_MEMBERSHIP)
    p.add_argument("--hash-samples", type=int, default=3,
                   help="Sampled per-date size verifications")
    p.add_argument("--aux-only", action="store_true",
                   help="Only repair root-level coverage files and merge "
                        "per-date manifests in an already consolidated target")
    args = p.parse_args()
    sources = args.source or list(DEFAULT_SOURCES)
    if args.aux_only:
        stats = repair_aux([Path(s) for s in sources if Path(s).is_dir()], args.target)
        if stats["manifest_problems"]:
            for problem in stats["manifest_problems"][:20]:
                log.error("AUX: %s", problem)
            raise SystemExit(
                f"Aux repair found {len(stats['manifest_problems'])} problem(s)"
            )
        return
    consolidate(
        sources,
        args.target,
        mode=args.mode,
        dry_run=args.dry_run,
        membership_file=args.membership_file,
        hash_samples=args.hash_samples,
    )


if __name__ == "__main__":
    main()
