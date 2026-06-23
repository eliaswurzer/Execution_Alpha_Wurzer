#!/usr/bin/env python3
"""Audit point-in-time index-membership transitions against the TAQ tape.

Validates the three structural properties the hypothesis runs rely on:

1. Every membership DROP ends on a day where the symbol still has a regular
   closing auction (acquisition-completion halts must carry a documented
   ``taq_tradability_boundary`` override that trims ``effective_to`` to the
   last day with a regular close, as for AET/COL/TWX/CA/RHT).
2. Every membership ADD has trade data and a closing print from its first
   active day.
3. The daily constituent count stays inside the expected band (505 share
   classes plus or minus replacement-gap days, e.g. 2018-11-29/30 at 503).

Writes ``drop_audit.csv``, ``add_audit.csv``, ``daily_counts.csv`` and a
``summary.json`` with problem counts; exits non-zero when problems remain.

Usage::

    python -m preprocessing.audit_membership_transitions \
        --out ../artifacts/audits/membership_transitions_20260611
(run from ``coding/``; PYTHONPATH must include ``coding``.)
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from analysis.runners._common import _eval_dates, _load_vc_one
from analysis.data.taq_loader import trades_parquet_path
from analysis.data.index_universe import membership_path

log = logging.getLogger(__name__)

COUNT_BAND = (504, 506)


def _load_membership(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["effective_from"] = pd.to_datetime(frame["effective_from"]).dt.date
    frame["effective_to"] = pd.to_datetime(frame["effective_to"]).dt.date
    return frame


def audit(
    membership_file: Path,
    start: dt.date,
    end: dt.date,
    out_dir: Path,
    count_band: tuple[int, int] = COUNT_BAND,
) -> dict:
    membership = _load_membership(membership_file)
    trading_days = _eval_dates(start, end)
    if not trading_days:
        raise RuntimeError("No preprocessed TAQ trading days available")

    drop_rows: list[dict] = []
    for row in membership[membership["effective_to"] < end].itertuples(index=False):
        i = bisect.bisect_right(trading_days, row.effective_to) - 1
        if i < 0:
            continue
        last_day = trading_days[i]
        has_parquet = trades_parquet_path(last_day, row.symbol).exists()
        has_close = (
            _load_vc_one(last_day, row.symbol) is not None if has_parquet else False
        )
        drop_rows.append({
            "symbol": row.symbol,
            "effective_to": row.effective_to,
            "last_trading_day": last_day,
            "has_parquet": has_parquet,
            "has_regular_close": has_close,
            "problem": not (has_parquet and has_close),
        })

    add_rows: list[dict] = []
    for row in membership[membership["effective_from"] > start].itertuples(index=False):
        i = bisect.bisect_left(trading_days, row.effective_from)
        if i >= len(trading_days):
            continue
        first_day = trading_days[i]
        has_parquet = trades_parquet_path(first_day, row.symbol).exists()
        has_close = (
            _load_vc_one(first_day, row.symbol) is not None if has_parquet else False
        )
        add_rows.append({
            "symbol": row.symbol,
            "effective_from": row.effective_from,
            "first_trading_day": first_day,
            "has_parquet": has_parquet,
            "has_regular_close": has_close,
            "problem": not (has_parquet and has_close),
        })

    count_rows = []
    for day in trading_days:
        n = int((
            (membership["effective_from"] <= day)
            & (membership["effective_to"] >= day)
        ).sum())
        count_rows.append({
            "date": day,
            "constituents": n,
            "outside_band": not (count_band[0] <= n <= count_band[1]),
        })

    drop_df = pd.DataFrame(drop_rows)
    add_df = pd.DataFrame(add_rows)
    count_df = pd.DataFrame(count_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_df.to_csv(out_dir / "drop_audit.csv", index=False)
    add_df.to_csv(out_dir / "add_audit.csv", index=False)
    count_df.to_csv(out_dir / "daily_counts.csv", index=False)

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "membership_file": str(membership_file),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "trading_days": len(trading_days),
        "drops_checked": len(drop_df),
        "drop_problems": int(drop_df["problem"].sum()) if not drop_df.empty else 0,
        "drop_problem_symbols": (
            sorted(drop_df.loc[drop_df["problem"], "symbol"].tolist())
            if not drop_df.empty else []
        ),
        "adds_checked": len(add_df),
        "add_problems": int(add_df["problem"].sum()) if not add_df.empty else 0,
        "add_problem_symbols": (
            sorted(add_df.loc[add_df["problem"], "symbol"].tolist())
            if not add_df.empty else []
        ),
        "count_band": list(count_band),
        "days_outside_count_band": [
            row["date"].isoformat()
            for row in count_rows if row["outside_band"]
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--membership-file", type=Path, default=None)
    parser.add_argument("--start", type=dt.date.fromisoformat,
                        default=dt.date(2018, 1, 2))
    parser.add_argument("--end", type=dt.date.fromisoformat,
                        default=dt.date(2019, 12, 31))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    membership_file = args.membership_file or membership_path(args.universe)
    summary = audit(membership_file, args.start, args.end, args.out)
    print(json.dumps(summary, indent=2))
    problems = summary["drop_problems"] + summary["add_problems"]
    if problems:
        log.error("Membership transition audit found %d problems", problems)
        sys.exit(1)


if __name__ == "__main__":
    main()
