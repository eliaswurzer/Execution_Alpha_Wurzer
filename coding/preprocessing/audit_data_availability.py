#!/usr/bin/env python3
"""Unified raw/processed TAQ availability audit for final-run preflight."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from . import preprocessing_status as ps
except ImportError:  # direct script execution or import-by-file tests
    _preprocessing_dir = Path(__file__).resolve().parent
    if str(_preprocessing_dir) not in sys.path:
        sys.path.insert(0, str(_preprocessing_dir))
    import preprocessing_status as ps


CSV_LIST_SEPARATOR = " "


def _normal_date(value: str | None) -> str | None:
    if value is None:
        return None
    out = str(value).strip().replace("-", "")
    if len(out) != 8 or not out.isdigit():
        raise ValueError(f"Date must be YYYYMMDD or YYYY-MM-DD: {value}")
    return out


def _json_list(items: Iterable[str]) -> str:
    return json.dumps(list(items), ensure_ascii=True)


def _join_paths(items: Iterable[str]) -> str:
    return CSV_LIST_SEPARATOR.join(str(x) for x in items)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _date_filter(dates: Iterable[str], start: str | None, end: str | None) -> list[str]:
    out = sorted(set(str(d) for d in dates))
    if start:
        out = [d for d in out if d >= start]
    if end:
        out = [d for d in out if d <= end]
    return out


def _load_membership_by_date(path: Path | None, dates: list[str]) -> dict[str, set[str]]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    required = {"symbol", "effective_from", "effective_to"}
    if not required.issubset(frame.columns):
        raise ValueError(f"membership file must contain {sorted(required)}: {path}")
    frame = frame.copy()
    frame["effective_from"] = pd.to_datetime(frame["effective_from"]).dt.strftime("%Y%m%d")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"].fillna("2099-12-31")).dt.strftime("%Y%m%d")
    out: dict[str, set[str]] = {}
    for date in dates:
        active = frame[(frame["effective_from"] <= date) & (frame["effective_to"] >= date)]
        out[date] = {ps.safe_symbol(symbol) for symbol in active["symbol"].astype(str)}
    return out


def build_audit(
    *,
    raw_roots: list[Path],
    processed_roots: list[Path],
    symbols: list[str],
    start: str | None,
    end: str | None,
    membership_file: Path | None = None,
    exclude_dates: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    raw_by_date = ps.discover_raw_date_sides(raw_roots)
    processed_dates: set[str] = set()
    for root in processed_roots:
        processed_dates |= ps.discover_processed_dates(root)
    dates = _date_filter(set(raw_by_date) | processed_dates, start, end)
    exclude_dates = exclude_dates or set()
    excluded_present = sorted(set(dates) & exclude_dates)
    dates = [d for d in dates if d not in exclude_dates]
    membership_by_date = _load_membership_by_date(membership_file, dates)

    audit_rows: list[dict] = []
    todo_rows: list[dict] = []
    missing_rows: list[dict] = []

    for date in dates:
        raw = raw_by_date.get(
            date,
            {"trade": False, "nbbo": False, "trade_paths": [], "nbbo_paths": [], "raw_roots": []},
        )
        statuses = [
            ps.date_side_status(
                root,
                date,
                symbols,
                raw_trade=bool(raw["trade"]),
                raw_nbbo=bool(raw["nbbo"]),
            )
            for root in processed_roots
        ]
        best = ps.choose_best_status(statuses)
        if best is None:
            continue
        status_dict = best.to_dict()
        raw_trade = bool(raw["trade"])
        raw_nbbo = bool(raw["nbbo"])
        todo_trade = raw_trade and not best.trade_done
        todo_nbbo = raw_nbbo and not best.nbbo_done
        partial = bool(best.trade_done) ^ bool(best.nbbo_done)
        active_symbols = membership_by_date.get(date, set())
        active_complete = active_symbols & best.complete_symbols
        active_missing_trade = sorted(active_symbols - best.trade_symbols)
        active_missing_nbbo = sorted(active_symbols - best.nbbo_symbols)
        extra_complete = sorted(best.complete_symbols - active_symbols) if active_symbols else []
        row = {
            "date": date,
            "raw_trade_gz_available": raw_trade,
            "raw_nbbo_gz_available": raw_nbbo,
            "raw_trade_paths": _join_paths(raw.get("trade_paths", [])),
            "raw_nbbo_paths": _join_paths(raw.get("nbbo_paths", [])),
            "raw_roots": _join_paths(raw.get("raw_roots", [])),
            "processed_root": best.processed_root,
            "trade_parquet_available": bool(best.trade_symbols),
            "nbbo_parquet_available": bool(best.nbbo_symbols),
            "trade_side_done": best.trade_done,
            "nbbo_side_done": best.nbbo_done,
            "complete_date": best.complete_done,
            "partial_date": partial,
            "status": best.status,
            "expected_symbols": best.expected_symbols,
            "trade_symbols": len(best.trade_symbols),
            "nbbo_symbols": len(best.nbbo_symbols),
            "complete_symbols": len(best.complete_symbols),
            "missing_trade_count": len(best.missing_trade_symbols),
            "missing_nbbo_count": len(best.missing_nbbo_symbols),
            "missing_trade_symbols": _json_list(best.missing_trade_symbols),
            "missing_nbbo_symbols": _json_list(best.missing_nbbo_symbols),
            "active_membership_symbols": len(active_symbols),
            "active_complete_symbols": len(active_complete),
            "active_missing_trade_count": len(active_missing_trade),
            "active_missing_nbbo_count": len(active_missing_nbbo),
            "active_missing_trade_symbols": _json_list(active_missing_trade),
            "active_missing_nbbo_symbols": _json_list(active_missing_nbbo),
            "extra_preprocessed_complete_symbols": len(extra_complete),
            "extra_preprocessed_symbols": _json_list(extra_complete),
            "manifest_exists": best.manifest_exists,
            "manifest_consistent": best.manifest_consistent,
            "coverage_exists": best.coverage_exists,
            "trade_qc_status": best.trade_qc_status,
            "trade_qc_detail": best.trade_qc_detail,
            "todo_trade": todo_trade,
            "todo_nbbo": todo_nbbo,
            "suggested_processed_side": (
                "trade+nbbo" if todo_trade and todo_nbbo else
                "trade" if todo_trade else
                "nbbo" if todo_nbbo else
                "none"
            ),
        }
        audit_rows.append(row)

        for side, is_todo in (("trade", todo_trade), ("nbbo", todo_nbbo)):
            if is_todo:
                todo_rows.append({
                    "date": date,
                    "side": side,
                    "status": best.status,
                    "processed_root": best.processed_root,
                    "raw_paths": _join_paths(raw.get(f"{side}_paths", [])),
                    "reason": f"raw_{side}_available_but_{side}_side_not_done",
                })

        for side, missing in (
            ("trade", best.missing_trade_symbols),
            ("nbbo", best.missing_nbbo_symbols),
        ):
            for symbol in missing:
                missing_rows.append({
                    "date": date,
                    "side": side,
                    "symbol": symbol,
                    "status": best.status,
                    "processed_root": best.processed_root,
                })

    status_counts = Counter(row["status"] for row in audit_rows)
    union_symbol_days = sum(int(row["expected_symbols"]) for row in audit_rows)
    union_complete_symbol_days = sum(int(row["complete_symbols"]) for row in audit_rows)
    active_symbol_days = sum(int(row["active_membership_symbols"]) for row in audit_rows)
    active_complete_symbol_days = sum(int(row["active_complete_symbols"]) for row in audit_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_roots": [str(p) for p in raw_roots],
        "processed_roots": [str(p) for p in processed_roots],
        "excluded_dates": excluded_present,
        "start": start,
        "end": end,
        "expected_symbols": len(symbols),
        "membership_file": str(membership_file) if membership_file else None,
        "dates": len(audit_rows),
        "raw_trade_dates": sum(1 for row in audit_rows if row["raw_trade_gz_available"]),
        "raw_nbbo_dates": sum(1 for row in audit_rows if row["raw_nbbo_gz_available"]),
        "complete_dates": sum(1 for row in audit_rows if row["complete_date"]),
        "partial_dates": sum(1 for row in audit_rows if row["partial_date"]),
        "trade_only_dates": status_counts.get(ps.STATUS_TRADE_ONLY, 0),
        "nbbo_only_dates": status_counts.get(ps.STATUS_NBBO_ONLY, 0),
        "unprocessed_dates": status_counts.get(ps.STATUS_UNPROCESSED, 0),
        "complete_with_missing_symbols_dates": status_counts.get(ps.STATUS_COMPLETE_WITH_MISSING, 0),
        "manifest_inconsistent_dates": status_counts.get(ps.STATUS_MANIFEST_INCONSISTENT, 0),
        "qc_problem_dates": status_counts.get(ps.STATUS_QC_PROBLEM, 0),
        "raw_trade_unprocessed_dates": sum(1 for row in audit_rows if row["todo_trade"]),
        "raw_nbbo_unprocessed_dates": sum(1 for row in audit_rows if row["todo_nbbo"]),
        "todo_rows": len(todo_rows),
        "missing_symbol_rows": len(missing_rows),
        "union_symbol_days": union_symbol_days,
        "union_complete_symbol_days": union_complete_symbol_days,
        "union_coverage": union_complete_symbol_days / max(union_symbol_days, 1),
        "active_membership_symbol_days": active_symbol_days,
        "active_membership_complete_symbol_days": active_complete_symbol_days,
        "active_membership_coverage": active_complete_symbol_days / max(active_symbol_days, 1),
        "active_membership_missing_trade_symbol_days": sum(int(row["active_missing_trade_count"]) for row in audit_rows),
        "active_membership_missing_nbbo_symbol_days": sum(int(row["active_missing_nbbo_count"]) for row in audit_rows),
        "status_counts": dict(sorted(status_counts.items())),
    }
    return audit_rows, todo_rows, missing_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", action="append", type=Path, default=[])
    parser.add_argument("--processed-root", action="append", type=Path, required=True)
    parser.add_argument("--symbols-file", type=Path, required=True)
    parser.add_argument("--membership-file", type=Path)
    parser.add_argument("--start", type=_normal_date)
    parser.add_argument("--end", type=_normal_date)
    parser.add_argument(
        "--exclude-date", action="append", type=_normal_date, default=[],
        help="Documented date exclusions (e.g. 20190513: raw trade file "
             "unobtainable); recorded in the summary as excluded_dates",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = ps.load_symbols(args.symbols_file)
    audit_rows, todo_rows, missing_rows, summary = build_audit(
        raw_roots=args.raw_root,
        processed_roots=args.processed_root,
        symbols=symbols,
        start=args.start,
        end=args.end,
        membership_file=args.membership_file,
        exclude_dates=set(args.exclude_date),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out_dir / "date_side_audit.csv",
        audit_rows,
        [
            "date", "raw_trade_gz_available", "raw_nbbo_gz_available",
            "raw_trade_paths", "raw_nbbo_paths", "raw_roots", "processed_root",
            "trade_parquet_available", "nbbo_parquet_available",
            "trade_side_done", "nbbo_side_done", "complete_date",
            "partial_date", "status", "expected_symbols", "trade_symbols",
            "nbbo_symbols", "complete_symbols", "missing_trade_count",
            "missing_nbbo_count", "missing_trade_symbols", "missing_nbbo_symbols",
            "active_membership_symbols", "active_complete_symbols",
            "active_missing_trade_count", "active_missing_nbbo_count",
            "active_missing_trade_symbols", "active_missing_nbbo_symbols",
            "extra_preprocessed_complete_symbols", "extra_preprocessed_symbols",
            "manifest_exists", "manifest_consistent", "coverage_exists",
            "trade_qc_status", "trade_qc_detail", "todo_trade", "todo_nbbo",
            "suggested_processed_side",
        ],
    )
    _write_csv(
        args.out_dir / "preprocessing_todo.csv",
        todo_rows,
        ["date", "side", "status", "processed_root", "raw_paths", "reason"],
    )
    _write_csv(
        args.out_dir / "missing_symbols_by_date.csv",
        missing_rows,
        ["date", "side", "symbol", "status", "processed_root"],
    )
    (args.out_dir / "date_side_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


