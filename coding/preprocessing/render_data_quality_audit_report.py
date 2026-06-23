#!/usr/bin/env python3
"""Render a human-readable Markdown report from data-quality audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORT = "DATA_QUALITY_AUDIT_REPORT.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required audit summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pct(value: Any) -> str:
    return f"{_float(value) * 100:.4f}%"


def _month(date_value: str) -> str:
    value = str(date_value or "")
    if len(value) >= 6 and value[:6].isdigit():
        return f"{value[:4]}-{value[4:6]}"
    return "unknown"


def _count_by(rows: list[dict[str, str]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "") or "missing") for row in rows)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return out


def _top_rows(rows: list[dict[str, str]], *, key: str, limit: int = 20) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (-_int(row.get(key)), str(row.get("date", ""))))[:limit]


def _active_gap_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in audit_rows
        if _int(row.get("active_missing_trade_count")) > 0
        or _int(row.get("active_missing_nbbo_count")) > 0
    ]


def _union_gap_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in audit_rows
        if _int(row.get("missing_trade_count")) > 0
        or _int(row.get("missing_nbbo_count")) > 0
    ]


def _raw_present_supplement_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in audit_rows:
        if _int(row.get("active_missing_trade_count")) > 0 and str(row.get("raw_trade_gz_available", "")).lower() == "true":
            rows.append({**row, "supplement_side": "trade", "supplement_count": row.get("active_missing_trade_count", "0")})
        if _int(row.get("active_missing_nbbo_count")) > 0 and str(row.get("raw_nbbo_gz_available", "")).lower() == "true":
            rows.append({**row, "supplement_side": "nbbo", "supplement_count": row.get("active_missing_nbbo_count", "0")})
    return rows


def _raw_missing_active_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in audit_rows:
        if _int(row.get("active_missing_trade_count")) > 0 and str(row.get("raw_trade_gz_available", "")).lower() != "true":
            rows.append({**row, "missing_side": "trade", "missing_count": row.get("active_missing_trade_count", "0")})
        if _int(row.get("active_missing_nbbo_count")) > 0 and str(row.get("raw_nbbo_gz_available", "")).lower() != "true":
            rows.append({**row, "missing_side": "nbbo", "missing_count": row.get("active_missing_nbbo_count", "0")})
    return rows


def build_report(audit_dir: Path) -> str:
    audit_dir = Path(audit_dir)
    summary = _read_json(audit_dir / "date_side_audit_summary.json")
    audit_rows = _read_csv(audit_dir / "date_side_audit.csv")
    todo_rows = _read_csv(audit_dir / "preprocessing_todo.csv")
    missing_rows = _read_csv(audit_dir / "missing_symbols_by_date.csv")
    supplement_rows = _raw_present_supplement_rows(audit_rows)
    raw_missing_active_rows = _raw_missing_active_rows(audit_rows)

    blockers = {
        "manifest_inconsistent_dates": _int(summary.get("manifest_inconsistent_dates")),
        "qc_problem_dates": _int(summary.get("qc_problem_dates")),
        "active_membership_missing_trade_symbol_days": _int(summary.get("active_membership_missing_trade_symbol_days")),
        "active_membership_missing_nbbo_symbol_days": _int(summary.get("active_membership_missing_nbbo_symbol_days")),
    }
    hard_blockers = {key: value for key, value in blockers.items() if value != 0}
    gate_status = "GO" if not hard_blockers else "NO-GO"
    build_status = "READY" if gate_status == "GO" and _int(summary.get("todo_rows")) == 0 and not supplement_rows else "WAIT"

    lines: list[str] = []
    lines.append("# Data-Quality Audit Report")
    lines.append("")
    lines.append("This report is generated from audit artifacts. It does not run preprocessing and does not modify data.")
    lines.append("")
    lines.append("## Executive Status")
    lines.append("")
    lines.extend(_markdown_table(
        ["Gate", "Status", "Interpretation"],
        [
            ["Active membership quality", gate_status, "Manifest, QC, and active S&P symbol coverage gates."],
            ["Final data build", build_status, "Volume DB and final calibration should wait if todo rows or hard blockers remain."],
        ],
    ))
    lines.append("")

    if hard_blockers:
        lines.append("Hard blockers:")
        for key, value in hard_blockers.items():
            lines.append(f"- `{key}` = `{value}`")
    else:
        lines.append("No hard blockers were found in the active-membership quality gates.")
    lines.append("")

    lines.append("## Key Metrics")
    lines.append("")
    lines.extend(_markdown_table(
        ["Metric", "Value"],
        [
            ["Created at", summary.get("created_at", "")],
            ["Date range", f"{summary.get('start', '')} to {summary.get('end', '')}"],
            ["Audit dates", summary.get("dates", 0)],
            ["Complete dates", summary.get("complete_dates", 0)],
            ["Partial dates", summary.get("partial_dates", 0)],
            ["Todo rows", summary.get("todo_rows", 0)],
            ["Raw-present active supplement rows", len(supplement_rows)],
            ["Raw-missing active blocker rows", len(raw_missing_active_rows)],
            ["Union coverage", _pct(summary.get("union_coverage"))],
            ["Active membership coverage", _pct(summary.get("active_membership_coverage"))],
            ["Manifest inconsistent dates", summary.get("manifest_inconsistent_dates", 0)],
            ["QC problem dates", summary.get("qc_problem_dates", 0)],
            ["Active missing Trade symbol-days", summary.get("active_membership_missing_trade_symbol_days", 0)],
            ["Active missing NBBO symbol-days", summary.get("active_membership_missing_nbbo_symbol_days", 0)],
        ],
    ))
    lines.append("")

    lines.append("## Preprocessing Todo")
    lines.append("")
    if todo_rows:
        side_counts = _count_by(todo_rows, "side")
        status_counts = _count_by(todo_rows, "status")
        month_side: dict[tuple[str, str], int] = defaultdict(int)
        for row in todo_rows:
            month_side[(_month(row.get("date", "")), str(row.get("side", "missing")))] += 1
        lines.append("Todo by side:")
        lines.extend(_markdown_table(["Side", "Rows"], [[side, count] for side, count in sorted(side_counts.items())]))
        lines.append("")
        lines.append("Todo by current status:")
        lines.extend(_markdown_table(["Status", "Rows"], [[status, count] for status, count in sorted(status_counts.items())]))
        lines.append("")
        lines.append("Todo by month and side:")
        lines.extend(_markdown_table(
            ["Month", "Side", "Rows"],
            [[month, side, count] for (month, side), count in sorted(month_side.items())],
        ))
    else:
        lines.append("No preprocessing todo rows were found.")
    lines.append("")

    lines.append("## Raw-Present Active-Member Supplement Queue")
    lines.append("")
    if supplement_rows:
        month_side: dict[tuple[str, str], int] = defaultdict(int)
        symbol_month_side: dict[tuple[str, str], int] = defaultdict(int)
        for row in supplement_rows:
            key = (_month(row.get("date", "")), str(row.get("supplement_side", "missing")))
            month_side[key] += 1
            symbol_month_side[key] += _int(row.get("supplement_count"))
        lines.append(
            "These rows have raw gz files available and should be supplemented before the final Volume DB build."
        )
        lines.extend(_markdown_table(
            ["Month", "Side", "Rows", "Active missing symbol-days"],
            [
                [month, side, count, symbol_month_side[(month, side)]]
                for (month, side), count in sorted(month_side.items())
            ],
        ))
        lines.append("")
        lines.append("Largest supplement rows:")
        lines.extend(_markdown_table(
            ["Date", "Side", "Active missing symbols", "Status"],
            [
                [
                    row.get("date", ""),
                    row.get("supplement_side", ""),
                    row.get("supplement_count", "0"),
                    row.get("status", ""),
                ]
                for row in sorted(supplement_rows, key=lambda item: (-_int(item.get("supplement_count")), str(item.get("date", ""))))[:20]
            ],
        ))
    else:
        lines.append("No raw-present active membership supplement rows were found.")
    lines.append("")

    lines.append("## Active Membership Gaps")
    lines.append("")
    active_gaps = _active_gap_rows(audit_rows)
    if active_gaps:
        lines.append("These rows are final-run blockers unless the underlying raw data is unavailable and documented.")
        lines.extend(_markdown_table(
            ["Date", "Status", "Missing Trade", "Missing NBBO"],
            [
                [
                    row.get("date", ""),
                    row.get("status", ""),
                    row.get("active_missing_trade_count", "0"),
                    row.get("active_missing_nbbo_count", "0"),
                ]
                for row in _top_rows(active_gaps, key="active_missing_trade_count")
            ],
        ))
    else:
        lines.append("No active S&P membership Trade/NBBO symbol gaps were found.")
    lines.append("")

    lines.append("## Raw-Missing Active-Member Blockers")
    lines.append("")
    if raw_missing_active_rows:
        lines.append("These rows cannot be fixed by preprocessing until the corresponding raw gz files are provided.")
        lines.extend(_markdown_table(
            ["Date", "Side", "Active missing symbols", "Expected raw prefix"],
            [
                [
                    row.get("date", ""),
                    row.get("missing_side", ""),
                    row.get("missing_count", "0"),
                    "EQY_US_ALL_TRADE" if row.get("missing_side") == "trade" else "EQY_US_ALL_NBBO",
                ]
                for row in sorted(raw_missing_active_rows, key=lambda item: (str(item.get("date", "")), str(item.get("missing_side", ""))))[:40]
            ],
        ))
    else:
        lines.append("No raw-missing active membership blockers were found.")
    lines.append("")

    lines.append("## Conservative-Union Gaps")
    lines.append("")
    union_gaps = _union_gap_rows(audit_rows)
    if union_gaps:
        lines.append(
            "These gaps are not automatically fatal because the preprocessing union deliberately over-includes tickers."
        )
        lines.extend(_markdown_table(
            ["Date", "Status", "Missing Trade", "Missing NBBO"],
            [
                [
                    row.get("date", ""),
                    row.get("status", ""),
                    row.get("missing_trade_count", "0"),
                    row.get("missing_nbbo_count", "0"),
                ]
                for row in _top_rows(union_gaps, key="missing_trade_count")
            ],
        ))
    else:
        lines.append("No conservative-union symbol gaps were found.")
    lines.append("")

    lines.append("## Raw Availability")
    lines.append("")
    raw_trade_missing = [row for row in audit_rows if str(row.get("raw_trade_gz_available", "")).lower() != "true"]
    raw_nbbo_missing = [row for row in audit_rows if str(row.get("raw_nbbo_gz_available", "")).lower() != "true"]
    lines.extend(_markdown_table(
        ["Metric", "Value"],
        [
            ["Raw Trade dates available", summary.get("raw_trade_dates", 0)],
            ["Raw NBBO dates available", summary.get("raw_nbbo_dates", 0)],
            ["Audit rows without raw Trade", len(raw_trade_missing)],
            ["Audit rows without raw NBBO", len(raw_nbbo_missing)],
        ],
    ))
    lines.append("")

    lines.append("## Missing Symbol Diagnostics")
    lines.append("")
    if missing_rows:
        side_counts = _count_by(missing_rows, "side")
        status_counts = _count_by(missing_rows, "status")
        lines.append("Missing-symbol rows by side:")
        lines.extend(_markdown_table(["Side", "Rows"], [[side, count] for side, count in sorted(side_counts.items())]))
        lines.append("")
        lines.append("Missing-symbol rows by date status:")
        lines.extend(_markdown_table(["Status", "Rows"], [[status, count] for status, count in sorted(status_counts.items())]))
    else:
        lines.append("No missing-symbol rows were found.")
    lines.append("")

    lines.append("## Next Commands")
    lines.append("")
    if supplement_rows:
        lines.append("Raw-present active membership gaps remain. Supplement them first:")
        lines.append("")
        lines.append("```powershell")
        lines.append('cd "C:\\Users\\elias\\Documents\\master thesis\\repo"')
        lines.append("$env:PYTHONPATH='coding'")
        lines.append("python -B coding\\preprocessing\\preprocess_active_membership_gaps.py `")
        lines.append(f'  --audit-dir "{audit_dir}" `')
        lines.append("  --workers 6")
        lines.append("```")
    elif todo_rows:
        lines.append("Preprocessing todo rows remain. Resume missing sides before the final Volume DB build:")
        lines.append("")
        lines.append("```powershell")
        lines.append('cd "C:\\Users\\elias\\Documents\\master thesis\\repo"')
        lines.append("powershell -NoProfile -ExecutionPolicy Bypass `")
        lines.append("  -File coding\\preprocessing\\launch_available_preprocessing.ps1 `")
        lines.append("  -Workers 6")
        lines.append("```")
    else:
        lines.append("No preprocessing todo rows remain.")
    lines.append("")
    if hard_blockers:
        lines.append("Do not build the final Volume DuckDB until the hard blockers above are resolved or explicitly documented.")
    elif supplement_rows:
        lines.append("Do not build the final Volume DuckDB until raw-present active supplement rows have been processed.")
    elif todo_rows:
        lines.append("Do not build the final Volume DuckDB until preprocessing todo rows are resolved or explicitly waived.")
    else:
        lines.append("The audit is clean for the checked gates. The next pipeline step is the final 2018-2019 Volume DuckDB build.")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    lines.extend(_markdown_table(
        ["Artifact", "Path"],
        [
            ["Audit directory", str(audit_dir)],
            ["Summary", str(audit_dir / "date_side_audit_summary.json")],
            ["Date-side audit", str(audit_dir / "date_side_audit.csv")],
            ["Preprocessing todo", str(audit_dir / "preprocessing_todo.csv")],
            ["Missing symbols", str(audit_dir / "missing_symbols_by_date.csv")],
        ],
    ))
    lines.append("")
    return "\n".join(lines)


def render_report(audit_dir: Path, out: Path | None = None) -> Path:
    audit_dir = Path(audit_dir)
    out_path = Path(out) if out is not None else audit_dir / DEFAULT_REPORT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(audit_dir), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = render_report(args.audit_dir, args.out)
    print(str(out))


if __name__ == "__main__":
    main()
