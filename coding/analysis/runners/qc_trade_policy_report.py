"""Aggregate trade preprocessing QC manifests into a CSV report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


QC_FILE = "trade_qc_summary.json"


def _row_from_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    counts = summary.get("qc_counts", {})
    row = {
        "date": summary.get("date"),
        "schema": summary.get("schema"),
        "trade_filter_policy": summary.get("trade_filter_policy"),
        "trade_condition_policy_version": summary.get("trade_condition_policy_version"),
        "symbols_filter_count": summary.get("symbols_filter_count"),
        "symbols_written": summary.get("symbols_written"),
        "total_read_rows": summary.get("total_read_rows"),
        "total_symbol_filter_rows": summary.get("total_symbol_filter_rows"),
        "total_kept_rows": summary.get("total_kept_rows"),
        "manifest_path": str(path),
    }
    row.update(counts)
    input_rows = float(row.get("policy_input_rows") or 0)
    kept_rows = float(row.get("total_kept_rows") or 0)
    row["preprocess_drop_rate"] = (
        1.0 - kept_rows / input_rows if input_rows > 0 else pd.NA
    )
    row["eval_bad_if_applied_rate"] = (
        float(row.get("dropped_eval_bad_condition_if_applied") or 0) / input_rows
        if input_rows > 0 else pd.NA
    )
    return row


def build_report(root: Path) -> pd.DataFrame:
    manifests = sorted(root.glob(f"**/qc/{QC_FILE}"))
    rows = [_row_from_manifest(path) for path in manifests]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate trade preprocessing QC manifests."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Preprocessed TAQ root, e.g. data/pilot_preprocess/2018",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    report = build_report(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    print(f"QC manifests: {len(report)}")
    print(f"CSV written: {args.out}")
    if not report.empty:
        cols = [
            "date",
            "trade_filter_policy",
            "policy_input_rows",
            "total_kept_rows",
            "dropped_bad_correction",
            "dropped_preprocess_bad_condition",
            "dropped_eval_bad_condition_if_applied",
            "kept_opening_auction_condition",
            "kept_closing_auction_condition",
        ]
        print(report[[c for c in cols if c in report.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
