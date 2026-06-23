#!/usr/bin/env python3
"""Build S&P 500 membership intervals from Refinitiv constituent snapshots.

The Refinitiv file is the membership source of record. RIC roots are mapped to
historical TAQ symbols through an explicit date-bounded crosswalk before any
intervals are written.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_REFINITIV_FILE = Path(
    os.environ.get("THESIS_REFINITIV_CONSTITUENTS", "data/sp500_constituents/constituents_sp500.csv")
)
DEFAULT_REFERENCE_DIR = Path("reference/index_membership")
DEFAULT_CROSSWALK = DEFAULT_REFERENCE_DIR / "sp500_refinitiv_ric_to_taq_crosswalk.csv"
DEFAULT_INTERVAL_OVERRIDES = DEFAULT_REFERENCE_DIR / "sp500_refinitiv_interval_overrides.csv"
DEFAULT_PREVIOUS = DEFAULT_REFERENCE_DIR / "sp500_membership_intervals_public_approx_2018_2019.csv"
DEFAULT_OUT = DEFAULT_REFERENCE_DIR / "sp500_membership_intervals.csv"
DEFAULT_AUDIT = DEFAULT_REFERENCE_DIR / "sp500_membership_intervals_refinitiv_audit.json"
DEFAULT_DIFF = DEFAULT_REFERENCE_DIR / "sp500_refinitiv_vs_previous_membership_diff.csv"

START_DATE = dt.date(2018, 1, 2)
END_DATE = dt.date(2019, 12, 31)

FIELDNAMES = [
    "index_id",
    "symbol",
    "effective_from",
    "effective_to",
    "company_name",
    "sector",
    "listing_exchange",
    "source",
    "source_note",
]


def parse_date(value: Any) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def ric_root_to_symbol(ric: str) -> str:
    """Return the normalized RIC root before explicit TAQ crosswalk mapping."""
    raw = str(ric or "").strip()
    if not raw:
        return ""
    raw = raw.split("^", 1)[0]
    root = raw.split(".", 1)[0]
    match = re.match(r"^([A-Z]+)([a-z])$", root)
    if match:
        root = f"{match.group(1)} {match.group(2).upper()}"
    root = root.upper().replace("/", " ").replace("_", " ").replace(".", " ")
    root = " ".join(root.split())
    if root == "GOOGL":
        root = "GOOG L"
    return root


def plausible_taq_symbol(symbol: str) -> bool:
    compact = str(symbol).replace(" ", "")
    return bool(compact) and compact.isalnum() and 1 <= len(compact) <= 8


def load_crosswalk(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"refinitiv_symbol", "taq_symbol", "effective_from", "effective_to"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            ref = ric_root_to_symbol(row.get("refinitiv_symbol", ""))
            taq = ric_root_to_symbol(row.get("taq_symbol", ""))
            if not ref or not taq:
                continue
            rows.append({
                "refinitiv_symbol": ref,
                "taq_symbol": taq,
                "effective_from": parse_date(row["effective_from"]),
                "effective_to": parse_date(row["effective_to"]),
                "reason": row.get("reason", ""),
                "source_note": row.get("source_note", ""),
            })
    return rows


def map_refinitiv_symbol(symbol: str, as_of: dt.date, crosswalk: list[dict[str, Any]]) -> tuple[str, str]:
    base = ric_root_to_symbol(symbol)
    for row in crosswalk:
        if row["refinitiv_symbol"] == base and row["effective_from"] <= as_of <= row["effective_to"]:
            return row["taq_symbol"], row.get("reason", "crosswalk")
    return base, "direct"


def load_refinitiv_snapshots(path: Path) -> dict[dt.date, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Refinitiv constituents file: {path}")
    snapshots: dict[dt.date, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if {"Date", "RIC"} - set(reader.fieldnames or []):
            raise ValueError(f"{path} must contain Date and RIC columns")
        for row in reader:
            snapshots[parse_date(row["Date"])].append(str(row["RIC"]).strip())
    if not snapshots:
        raise ValueError(f"{path} did not contain any Refinitiv rows")
    return dict(sorted(snapshots.items()))


def _snapshot_dates_for_window(snapshots: dict[dt.date, list[str]], start: dt.date, end: dt.date) -> list[dt.date]:
    dates = sorted(snapshots)
    seed = [d for d in dates if d <= start]
    in_window = [d for d in dates if start < d <= end]
    if not seed:
        raise ValueError(f"No Refinitiv seed snapshot on or before {start.isoformat()}")
    return [seed[-1], *in_window]


def _mapped_snapshot(
    snapshot_date: dt.date,
    rics: list[str],
    crosswalk: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], list[dict[str, str]], list[str]]:
    mapped: dict[str, set[str]] = defaultdict(set)
    mapping_rows: list[dict[str, str]] = []
    invalid: list[str] = []
    for ric in sorted(set(rics)):
        ref = ric_root_to_symbol(ric)
        taq, reason = map_refinitiv_symbol(ref, snapshot_date, crosswalk)
        if not plausible_taq_symbol(taq):
            invalid.append(ric)
            continue
        mapped[taq].add(ric)
        mapping_rows.append({
            "date": snapshot_date.isoformat(),
            "ric": ric,
            "refinitiv_symbol": ref,
            "taq_symbol": taq,
            "mapping_reason": reason,
        })
    return dict(mapped), mapping_rows, invalid


def _close_interval(
    intervals: list[dict[str, str]],
    symbol: str,
    start: dt.date,
    end: dt.date,
    rics: set[str],
) -> None:
    if end < start:
        return
    intervals.append({
        "index_id": "sp500",
        "symbol": symbol,
        "effective_from": start.isoformat(),
        "effective_to": end.isoformat(),
        "company_name": "",
        "sector": "",
        "listing_exchange": "",
        "source": "refinitiv_constituents_sp500_csv",
        "source_note": "Primary Refinitiv constituent snapshot mapped to historical TAQ symbol; RIC roots: "
        + " ".join(sorted(rics)),
    })


def _active_from_intervals(intervals: list[dict[str, str]], d: dt.date) -> set[str]:
    out: set[str] = set()
    for row in intervals:
        if parse_date(row["effective_from"]) <= d <= parse_date(row["effective_to"]):
            out.add(row["symbol"])
    return out


def _load_previous_intervals(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_interval_overrides(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "effective_from", "effective_to", "action"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows: list[dict[str, str]] = []
        for row in reader:
            symbol = ric_root_to_symbol(row.get("symbol", ""))
            action = str(row.get("action", "")).strip().lower()
            if not symbol or action not in {"replace", "drop"}:
                continue
            if action == "replace":
                parse_date(row["effective_from"])
                parse_date(row["effective_to"])
            rows.append({
                "symbol": symbol,
                "effective_from": row.get("effective_from", ""),
                "effective_to": row.get("effective_to", ""),
                "action": action,
                "reason": row.get("reason", ""),
                "source_note": row.get("source_note", ""),
            })
        return rows


def apply_interval_overrides(
    intervals: list[dict[str, str]],
    overrides: list[dict[str, str]],
    *,
    start: dt.date,
    end: dt.date,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not overrides:
        return intervals, []
    override_symbols = {row["symbol"] for row in overrides}
    kept = [row for row in intervals if row["symbol"] not in override_symbols]
    audit_rows: list[dict[str, Any]] = []
    original_counts = defaultdict(int)
    for row in intervals:
        if row["symbol"] in override_symbols:
            original_counts[row["symbol"]] += 1
    for row in overrides:
        symbol = row["symbol"]
        if row["action"] == "drop":
            audit_rows.append({
                "symbol": symbol,
                "action": "drop",
                "replacement_interval": "",
                "original_interval_count": original_counts[symbol],
                "reason": row.get("reason", ""),
            })
            continue
        from_date = parse_date(row["effective_from"])
        to_date = parse_date(row["effective_to"])
        if to_date < start or from_date > end or to_date < from_date:
            continue
        clipped_from = max(from_date, start)
        clipped_to = min(to_date, end)
        kept.append({
            "index_id": "sp500",
            "symbol": symbol,
            "effective_from": clipped_from.isoformat(),
            "effective_to": clipped_to.isoformat(),
            "company_name": "",
            "sector": "",
            "listing_exchange": "",
            "source": "refinitiv_constituents_sp500_csv_with_taq_boundary_overrides",
            "source_note": (
                "Primary Refinitiv membership with audited TAQ tradability/ticker-boundary override; "
                f"reason={row.get('reason', '')}; note={row.get('source_note', '')}"
            ),
        })
        audit_rows.append({
            "symbol": symbol,
            "action": "replace",
            "replacement_interval": f"{clipped_from.isoformat()}..{clipped_to.isoformat()}",
            "original_interval_count": original_counts[symbol],
            "reason": row.get("reason", ""),
        })
    return sorted(kept, key=lambda item: (item["effective_from"], item["symbol"], item["effective_to"])), audit_rows


def build_refinitiv_intervals(
    refinitiv_file: Path,
    crosswalk_file: Path,
    interval_overrides_file: Path,
    out_file: Path,
    audit_file: Path,
    diff_file: Path,
    *,
    previous_file: Path = DEFAULT_PREVIOUS,
    start: dt.date = START_DATE,
    end: dt.date = END_DATE,
) -> dict[str, Any]:
    snapshots = load_refinitiv_snapshots(refinitiv_file)
    crosswalk = load_crosswalk(crosswalk_file)
    interval_overrides = load_interval_overrides(interval_overrides_file)
    selected_dates = _snapshot_dates_for_window(snapshots, start, end)

    active: dict[str, tuple[dt.date, set[str]]] = {}
    intervals: list[dict[str, str]] = []
    all_mapping_rows: list[dict[str, str]] = []
    invalid_rics: list[str] = []
    duplicate_rows: list[dict[str, str]] = []
    snapshot_counts: list[dict[str, Any]] = []

    for ix, snapshot_date in enumerate(selected_dates):
        effective = start if ix == 0 and snapshot_date <= start else snapshot_date
        mapped, mapping_rows, invalid = _mapped_snapshot(snapshot_date, snapshots[snapshot_date], crosswalk)
        invalid_rics.extend(invalid)
        all_mapping_rows.extend(mapping_rows)
        for symbol, rics in mapped.items():
            if len(rics) > 1:
                duplicate_rows.append({
                    "date": snapshot_date.isoformat(),
                    "taq_symbol": symbol,
                    "rics": " ".join(sorted(rics)),
                })

        new_symbols = set(mapped)
        old_symbols = set(active)
        for removed in sorted(old_symbols - new_symbols):
            start_date, rics = active.pop(removed)
            _close_interval(intervals, removed, start_date, effective - dt.timedelta(days=1), rics)
        for added in sorted(new_symbols - old_symbols):
            active[added] = (effective, set(mapped[added]))
        for kept in sorted(new_symbols & old_symbols):
            start_date, rics = active[kept]
            active[kept] = (start_date, rics | set(mapped[kept]))

        snapshot_counts.append({
            "snapshot_date": snapshot_date.isoformat(),
            "effective_date": effective.isoformat(),
            "ric_count": len(set(snapshots[snapshot_date])),
            "taq_symbol_count": len(new_symbols),
        })

    for symbol, (start_date, rics) in sorted(active.items()):
        _close_interval(intervals, symbol, start_date, end, rics)

    if invalid_rics:
        sample = ", ".join(sorted(set(invalid_rics))[:20])
        raise ValueError(
            "Refinitiv active RICs could not be mapped to plausible TAQ symbols: "
            f"{sample}"
        )

    intervals = sorted(intervals, key=lambda row: (row["effective_from"], row["symbol"], row["effective_to"]))
    intervals, interval_override_audit = apply_interval_overrides(
        intervals,
        interval_overrides,
        start=start,
        end=end,
    )
    _write_csv(out_file, intervals, FIELDNAMES)

    previous = _load_previous_intervals(previous_file)
    diff_rows: list[dict[str, Any]] = []
    for item in snapshot_counts:
        d = parse_date(item["effective_date"])
        ref_active = _active_from_intervals(intervals, d)
        prev_active = _active_from_intervals(previous, d)
        only_ref = sorted(ref_active - prev_active)
        only_prev = sorted(prev_active - ref_active)
        diff_rows.append({
            "date": d.isoformat(),
            "refinitiv_count": len(ref_active),
            "previous_count": len(prev_active),
            "only_refinitiv_count": len(only_ref),
            "only_previous_count": len(only_prev),
            "only_refinitiv": " ".join(only_ref),
            "only_previous": " ".join(only_prev),
            "mapped_equivalent_note": "Known ticker-history mappings are applied before this comparison.",
        })
    _write_csv(diff_file, diff_rows, [
        "date",
        "refinitiv_count",
        "previous_count",
        "only_refinitiv_count",
        "only_previous_count",
        "only_refinitiv",
        "only_previous",
        "mapped_equivalent_note",
    ])

    audit = {
        "index_id": "sp500",
        "source": "refinitiv_constituents_sp500_csv",
        "refinitiv_file": str(refinitiv_file),
        "crosswalk_file": str(crosswalk_file),
        "interval_overrides_file": str(interval_overrides_file),
        "previous_file": str(previous_file),
        "out_file": str(out_file),
        "diff_file": str(diff_file),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "selected_snapshot_count": len(selected_dates),
        "interval_rows": len(intervals),
        "unique_symbols": len({row["symbol"] for row in intervals}),
        "active_on_end": len(_active_from_intervals(intervals, end)),
        "snapshot_counts": snapshot_counts,
        "duplicate_mapped_ric_rows": duplicate_rows,
        "invalid_rics": sorted(set(invalid_rics)),
        "crosswalk_rows": len(crosswalk),
        "interval_override_rows": len(interval_overrides),
        "interval_override_audit": interval_override_audit,
        "diff_summary": {
            "max_only_refinitiv_count": max((int(row["only_refinitiv_count"]) for row in diff_rows), default=0),
            "max_only_previous_count": max((int(row["only_previous_count"]) for row in diff_rows), default=0),
        },
        "note": (
            "Refinitiv is the primary point-in-time membership source. RIC roots are mapped "
            "to historical TAQ symbols through the explicit crosswalk before evaluation."
        ),
    }
    audit_file.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinitiv-file", type=Path, default=DEFAULT_REFINITIV_FILE)
    parser.add_argument("--crosswalk-file", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--interval-overrides-file", type=Path, default=DEFAULT_INTERVAL_OVERRIDES)
    parser.add_argument("--previous-file", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--diff", type=Path, default=DEFAULT_DIFF)
    parser.add_argument("--start", default=START_DATE.isoformat())
    parser.add_argument("--end", default=END_DATE.isoformat())
    args = parser.parse_args()
    audit = build_refinitiv_intervals(
        args.refinitiv_file,
        args.crosswalk_file,
        args.interval_overrides_file,
        args.out,
        args.audit,
        args.diff,
        previous_file=args.previous_file,
        start=parse_date(args.start),
        end=parse_date(args.end),
    )
    print(json.dumps({
        "out_file": audit["out_file"],
        "interval_rows": audit["interval_rows"],
        "unique_symbols": audit["unique_symbols"],
        "active_on_end": audit["active_on_end"],
        "selected_snapshot_count": audit["selected_snapshot_count"],
        "max_only_refinitiv_count": audit["diff_summary"]["max_only_refinitiv_count"],
        "max_only_previous_count": audit["diff_summary"]["max_only_previous_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
