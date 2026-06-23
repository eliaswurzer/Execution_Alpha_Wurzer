#!/usr/bin/env python3
"""Restartable full-run wrapper for streaming index preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from . import preprocessing_status as ps
except ImportError:  # direct script execution or import-by-file tests
    _preprocessing_dir = Path(__file__).resolve().parent
    if str(_preprocessing_dir) not in sys.path:
        sys.path.insert(0, str(_preprocessing_dir))
    import preprocessing_status as ps


PYTHON = sys.executable
STREAMING_SCRIPT = Path(__file__).resolve().with_name("preprocess_taq_streaming.py")


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback))



# Compatibility adapters around shared preprocessing status logic.  Existing
# tests and scripts still import these private names, but the implementation now
# has a single source of truth in preprocessing_status.py.
def _safe_symbol(symbol: str) -> str:
    return ps.safe_symbol(symbol)


def _load_symbols(path: Path) -> list[str]:
    return ps.load_symbols(path)


def _discover_date_sides(raw_root: Path) -> dict[str, dict[str, bool]]:
    discovered = ps.discover_raw_date_sides([raw_root])
    return {
        date: {"trade": bool(row["trade"]), "nbbo": bool(row["nbbo"])}
        for date, row in discovered.items()
    }


def _discover_dates(raw_root: Path) -> list[str]:
    sides = _discover_date_sides(raw_root)
    return sorted(d for d, available in sides.items() if available["trade"] and available["nbbo"])


def _manifest_symbols(path: Path) -> set[str]:
    return ps.manifest_symbols(path)


def _manifest_side_symbols(path: Path, side: str) -> set[str]:
    return ps.manifest_side_symbols(path, side)


def _date_side_status(out_root: Path, date: str) -> dict:
    return ps.date_side_status(out_root, date).to_dict()


def _done_status(out_root: Path, date: str, symbols: list[str]) -> tuple[bool, list[str]]:
    return ps.done_status(out_root, date, symbols)


def _select_preprocessing_todo(
    *,
    raw_root: Path,
    out_root: Path,
    symbols: list[str],
    mode: str = "paired",
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> tuple[list[str], list[dict], int]:
    """Return selected dates, work items, and skipped count for resume logic."""
    sides_by_date = _discover_date_sides(raw_root)
    if mode == "paired":
        dates = sorted(
            d for d, available in sides_by_date.items()
            if available["trade"] and available["nbbo"]
        )
    elif mode == "available":
        dates = sorted(sides_by_date)
    else:
        raise ValueError("mode must be 'paired' or 'available'")
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    if limit:
        dates = dates[:limit]

    todo: list[dict] = []
    skipped = 0
    for date in dates:
        raw_sides = sides_by_date[date]
        status_obj = ps.date_side_status(
            out_root,
            date,
            symbols,
            raw_trade=bool(raw_sides["trade"]),
            raw_nbbo=bool(raw_sides["nbbo"]),
        )
        status = status_obj.to_dict()
        done, missing = _done_status(out_root, date, symbols)
        if done and not overwrite:
            skipped += 1
            continue

        missing_trade = list(status_obj.missing_trade_symbols)
        missing_nbbo = list(status_obj.missing_nbbo_symbols)
        trade_manifest_mismatch = bool(
            status_obj.trade_symbols or status_obj.trade_manifest_symbols
        ) and status_obj.trade_symbols != status_obj.trade_manifest_symbols
        nbbo_manifest_mismatch = bool(
            status_obj.nbbo_symbols or status_obj.nbbo_manifest_symbols
        ) and status_obj.nbbo_symbols != status_obj.nbbo_manifest_symbols
        if status_obj.status == ps.STATUS_MANIFEST_INCONSISTENT:
            supplement_trade = False
            supplement_nbbo = False
            need_trade = bool(raw_sides["trade"]) and (overwrite or trade_manifest_mismatch or not status_obj.trade_done)
            need_nbbo = bool(raw_sides["nbbo"]) and (overwrite or nbbo_manifest_mismatch or not status_obj.nbbo_done)
        else:
            supplement_trade = bool(status_obj.trade_done and missing_trade and not overwrite)
            supplement_nbbo = bool(status_obj.nbbo_done and missing_nbbo and not overwrite)
            need_trade = bool(raw_sides["trade"]) and (
                overwrite or not status_obj.trade_done or supplement_trade
            )
            need_nbbo = bool(raw_sides["nbbo"]) and (
                overwrite or not status_obj.nbbo_done or supplement_nbbo
            )
        if mode == "paired" and not (raw_sides["trade"] and raw_sides["nbbo"]):
            continue
        if not need_trade and not need_nbbo:
            skipped += 1
            continue

        processed_side = (
            "trade+nbbo" if need_trade and need_nbbo
            else "trade" if need_trade
            else "nbbo"
        )
        todo.append({
            "date": date,
            "skip_trades": not need_trade,
            "skip_nbbo": not need_nbbo,
            "processed_side": processed_side,
            "raw_trade": bool(raw_sides["trade"]),
            "raw_nbbo": bool(raw_sides["nbbo"]),
            "status_before": status.get("status", ""),
            "trade_done_before": bool(status.get("trade_done", False)),
            "nbbo_done_before": bool(status.get("nbbo_done", False)),
            "missing_complete_symbols": missing,
            "missing_trade_symbols": missing_trade,
            "missing_nbbo_symbols": missing_nbbo,
            "trade_symbols_override": missing_trade if supplement_trade else None,
            "nbbo_symbols_override": missing_nbbo if supplement_nbbo else None,
            "supplement_missing": bool(supplement_trade or supplement_nbbo),
        })
    return dates, todo, skipped


def _write_coverage(out_root: Path, date: str, symbols: list[str]) -> dict:
    import pandas as pd

    date_dir = out_root / date
    expected = {_safe_symbol(s) for s in symbols}
    trade = {p.stem for p in (date_dir / "trades").glob("*.parquet")}
    nbbo = {p.stem for p in (date_dir / "nbbo").glob("*.parquet")}
    rows = [
        {
            "symbol": sym,
            "trade_parquet": sym in trade,
            "nbbo_parquet": sym in nbbo,
            "complete": sym in trade and sym in nbbo,
        }
        for sym in sorted(expected)
    ]
    cov_dir = out_root / "coverage"
    cov_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cov_dir / f"{date}_symbol_coverage.csv", index=False)
    summary = {
        "date": date,
        "expected_symbols": len(expected),
        "trade_symbols": len(trade),
        "nbbo_symbols": len(nbbo),
        "complete_symbols": len(trade & nbbo),
        "trade_only_symbols": len(trade - nbbo),
        "nbbo_only_symbols": len(nbbo - trade),
        "status": (
            "complete" if trade and nbbo and trade == nbbo
            else "trade_only" if trade and not nbbo
            else "nbbo_only" if nbbo and not trade
            else "partial"
        ),
        "missing_trade_symbols": sorted(expected - trade),
        "missing_nbbo_symbols": sorted(expected - nbbo),
    }
    (cov_dir / f"{date}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    return summary


def _write_symbol_override_file(args: argparse.Namespace, date: str, side: str, symbols: list[str] | None) -> Path | None:
    if not symbols:
        return None
    path = args.out_root / "logs" / "supplement_symbols" / f"{date}_{side}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(symbols)) + "\n", encoding="utf-8")
    return path


def _run_date(
    args: argparse.Namespace,
    date: str,
    *,
    skip_trades: bool = False,
    skip_nbbo: bool = False,
    trade_symbols_override: list[str] | None = None,
    nbbo_symbols_override: list[str] | None = None,
    supplement_missing: bool = False,
) -> tuple[bool, float, dict | None, int, str]:
    cmd = [
        PYTHON, str(STREAMING_SCRIPT),
        "--date", date,
        "--raw-root", str(args.raw_root),
        "--output-dir", str(args.out_root),
        "--symbols-file", str(args.symbols_file),
        "--chunksize", str(args.chunksize),
        "--schema", args.schema,
        "--trade-filter-policy", args.trade_filter_policy,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if supplement_missing:
        cmd.append("--supplement-missing")
    trade_override_path = _write_symbol_override_file(args, date, "trade", trade_symbols_override)
    nbbo_override_path = _write_symbol_override_file(args, date, "nbbo", nbbo_symbols_override)
    if trade_override_path is not None:
        cmd.extend(["--trade-symbols-file", str(trade_override_path)])
    if nbbo_override_path is not None:
        cmd.extend(["--nbbo-symbols-file", str(nbbo_override_path)])
    if skip_trades:
        cmd.append("--skip-trades")
    if skip_nbbo:
        cmd.append("--skip-nbbo")
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{date}.stdout.log").write_text(result.stdout, encoding="utf-8", errors="replace")
    (log_dir / f"{date}.stderr.log").write_text(result.stderr, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        return False, elapsed, None, result.returncode, detail
    return True, elapsed, _write_coverage(args.out_root, date, _load_symbols(args.symbols_file)), 0, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=_env_path("THESIS_RAW_TAQ_ROOT", "data/raw"),
        help="Root containing licensed raw DTAQ files. Defaults to THESIS_RAW_TAQ_ROOT or data/raw.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_env_path("THESIS_TAQ_OUTPUT_ROOT", "data/processed/sp500_preprocess_2018h1_streaming"),
        help="Output root for processed Parquet files. Defaults to THESIS_TAQ_OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("reference/index_membership/public_sp500_2018h1_union_taq_symbols.txt"),
    )
    parser.add_argument("--start", default=None, help="YYYYMMDD lower bound")
    parser.add_argument("--end", default=None, help="YYYYMMDD upper bound")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["paired", "available"],
        default="paired",
        help=(
            "paired processes only dates with Trade and NBBO raw files; "
            "available also processes trade-only/nbbo-only partial dates."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--schema", choices=["slim", "rich"], default="slim")
    parser.add_argument(
        "--trade-filter-policy",
        choices=["preprocessing", "evaluation"],
        default="preprocessing",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = _load_symbols(args.symbols_file)
    dates, todo, skipped = _select_preprocessing_todo(
        raw_root=args.raw_root,
        out_root=args.out_root,
        symbols=symbols,
        mode=args.mode,
        start=args.start,
        end=args.end,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    for item in todo:
        date = item["date"]
        missing = item.get("missing_complete_symbols", [])
        mode_note = {"trade+nbbo": "trade+nbbo", "trade": "trade-only", "nbbo": "nbbo-only"}[item["processed_side"]]
        if missing:
            print(
                f"[TODO] {date}: {mode_note}; missing complete symbols "
                f"{len(missing)}: {', '.join(missing[:12])}"
            )
        else:
            print(f"[TODO] {date}: {mode_note}")

    print(f"Symbols expected : {len(symbols)}")
    print(f"Dates selected   : {len(dates)}")
    print(f"Already complete : {skipped}")
    print(f"To process       : {len(todo)}")
    print(f"Workers          : {args.workers}")
    print(f"Mode             : {args.mode}")
    print(f"Raw root         : {args.raw_root}")
    print(f"Output root      : {args.out_root}")
    if args.dry_run or not todo:
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    def _row_from_result(
        date: str,
        ok: bool,
        elapsed: float,
        summary: dict | None,
        before: dict | None = None,
        exit_code: int = 0,
        error_detail: str = "",
    ) -> dict:
        after = _date_side_status(args.out_root, date)
        row = {
            "date": date,
            "ok": ok,
            "elapsed_seconds": elapsed,
            "status_before": (before or {}).get("status_before", ""),
            "trade_done_before": bool((before or {}).get("trade_done_before", False)),
            "nbbo_done_before": bool((before or {}).get("nbbo_done_before", False)),
            "status_after": after.get("status", ""),
            "trade_done_after": bool(after.get("trade_done", False)),
            "nbbo_done_after": bool(after.get("nbbo_done", False)),
            "complete_done_after": bool(after.get("complete_done", False)),
            "qc_status": after.get("trade_qc_status", ""),
            "qc_status_after": after.get("trade_qc_status", ""),
            "exit_code": int(exit_code),
            "error_detail": error_detail or "",
            "supplement_missing": bool((before or {}).get("supplement_missing", False)),
            "supplement_trade_symbols": " ".join((before or {}).get("trade_symbols_override") or []),
            "supplement_nbbo_symbols": " ".join((before or {}).get("nbbo_symbols_override") or []),
        }
        if summary:
            row.update({
                "status": summary.get("status"),
                "expected_symbols": summary["expected_symbols"],
                "complete_symbols": summary["complete_symbols"],
                "trade_symbols": summary["trade_symbols"],
                "nbbo_symbols": summary["nbbo_symbols"],
                "trade_only_symbols": summary.get("trade_only_symbols", 0),
                "nbbo_only_symbols": summary.get("nbbo_only_symbols", 0),
                "missing_trade_count": len(summary["missing_trade_symbols"]),
                "missing_nbbo_count": len(summary["missing_nbbo_symbols"]),
            })
        return row

    run_rows = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, item in enumerate(todo, 1):
            date = item["date"]
            print(f"[{i}/{len(todo)}] {date}")
            ok, elapsed, summary, exit_code, error_detail = _run_date(
                args,
                date,
                skip_trades=item["skip_trades"],
                skip_nbbo=item["skip_nbbo"],
                trade_symbols_override=item.get("trade_symbols_override"),
                nbbo_symbols_override=item.get("nbbo_symbols_override"),
                supplement_missing=bool(item.get("supplement_missing", False)),
            )
            row = _row_from_result(date, ok, elapsed, summary, item, exit_code, error_detail)
            row["processed_side"] = item["processed_side"]
            run_rows.append(row)
            print(f"  ok={ok} elapsed={elapsed:.0f}s")
            if not ok:
                break
    else:
        print(f"Running up to {workers} dates in parallel.")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_date,
                    args,
                    item["date"],
                    skip_trades=item["skip_trades"],
                    skip_nbbo=item["skip_nbbo"],
                    trade_symbols_override=item.get("trade_symbols_override"),
                    nbbo_symbols_override=item.get("nbbo_symbols_override"),
                    supplement_missing=bool(item.get("supplement_missing", False)),
                ): item
                for item in todo
            }
            completed = 0
            for fut in as_completed(futures):
                item = futures[fut]
                date = item["date"]
                completed += 1
                try:
                    ok, elapsed, summary, exit_code, error_detail = fut.result()
                except Exception as exc:
                    ok, elapsed, summary, exit_code, error_detail = False, 0.0, None, -1, f"{type(exc).__name__}: {exc}"
                    print(f"[{completed}/{len(todo)}] {date} failed: {exc}")
                row = _row_from_result(date, ok, elapsed, summary, item, exit_code, error_detail)
                row["processed_side"] = (
                    "trade+nbbo" if not item["skip_trades"] and not item["skip_nbbo"]
                    else "trade" if not item["skip_trades"]
                    else "nbbo"
                )
                run_rows.append(row)
                print(f"[{completed}/{len(todo)}] {date} ok={ok} elapsed={elapsed:.0f}s")

    if run_rows:
        import pandas as pd

        new_summary = pd.DataFrame(run_rows)
        summary_path = args.out_root / "run_summary.csv"
        if summary_path.exists():
            old_summary = pd.read_csv(summary_path)
            old_summary["date"] = old_summary["date"].astype(str)
            new_summary["date"] = new_summary["date"].astype(str)
            old_summary = old_summary[
                ~old_summary["date"].isin(new_summary["date"])
            ]
            new_summary = pd.concat([old_summary, new_summary], ignore_index=True)
        new_summary["date"] = new_summary["date"].astype(str)
        new_summary.sort_values("date").to_csv(summary_path, index=False)
    if any(not r["ok"] for r in run_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

