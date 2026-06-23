"""Intraday dollar-volume bucket shares for the thesis data chapter.

Computes volume-weighted and equal-weighted shares of the seven intraday
dollar-volume buckets from the DuckDB volume panel, restricted to the
evaluation calendar (documented exclusions applied, so early-close sessions
without a 15:30-16:00 segment cannot distort the shares) and to point-in-time
S&P 500 members. The summary backs the Window-B liquidity argument and the
closing-auction share statistics in the thesis; numbers quoted there must
come from this artifact.

Usage::

    python -m analysis.runners.volume_bucket_shares \
        --db "<...>/dollar_volume_sp500_2018_2019.duckdb" \
        --out ../artifacts/audits/volume_bucket_shares_<date>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from .. import config as cfg
from ..data.index_universe import load_index_membership
from ..utils.symbols import canonical_symbol
from ._common import _eval_dates

log = logging.getLogger(__name__)

BUCKET_COLUMNS = {
    "open_auction": "Open_Auction_Val",
    "morning_30m": "Morning_30m_Val",
    "mid_day": "Mid_Day_Val",
    "afternoon_30m": "Afternoon_30m_Val",
    "close_auction": "Close_Auction_Val",
    "pre_market": "Pre_Market_Val",
    "post_market": "Post_Market_Val",
}
MID_DAY_HOURS = 5.5  # 10:00-15:30


def _load_panel(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = ", ".join(BUCKET_COLUMNS.values())
        frame = con.execute(
            f"SELECT Ticker, Date, Is_Witching_Day, {cols}, Total_Daily_Val "
            "FROM daily_volume WHERE Total_Daily_Val > 0"
        ).df()
    finally:
        con.close()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
    frame["symbol"] = frame["Ticker"].map(canonical_symbol)
    return frame


def _filter_panel(
    panel: pd.DataFrame,
    start: _dt.date,
    end: _dt.date,
    universe: str,
    membership_root: Path | None,
) -> pd.DataFrame:
    calendar = set(_eval_dates(start, end))
    out = panel[panel["Date"].isin(calendar)].copy()
    membership = load_index_membership(universe, membership_root)
    merged = out.merge(
        membership[["symbol", "effective_from", "effective_to"]],
        on="symbol", how="inner",
    )
    active = merged[
        (merged["effective_from"] <= merged["Date"])
        & (merged["effective_to"] >= merged["Date"])
    ]
    return active.drop(columns=["effective_from", "effective_to"]).drop_duplicates(
        ["symbol", "Date"],
    )


def _share_rows(panel: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    total = float(panel["Total_Daily_Val"].sum())
    for name, col in BUCKET_COLUMNS.items():
        vw = float(panel[col].sum()) / total if total > 0 else float("nan")
        ew = float((panel[col] / panel["Total_Daily_Val"]).mean())
        rows.append({
            "scope": scope,
            "bucket": name,
            "share_volume_weighted": vw,
            "share_equal_weighted": ew,
        })
    return rows


def build_bucket_shares(
    db_path: Path,
    out_dir: Path,
    *,
    start: _dt.date,
    end: _dt.date,
    universe: str = "sp500",
    membership_root: Path | None = None,
) -> dict:
    panel = _load_panel(db_path)
    panel = _filter_panel(panel, start, end, universe, membership_root)
    if panel.empty:
        raise RuntimeError("No symbol-days left after calendar/membership filters")

    rows: list[dict] = []
    rows.extend(_share_rows(panel, "all_days"))
    rows.extend(_share_rows(panel[~panel["Is_Witching_Day"]], "non_witching"))
    witching = panel[panel["Is_Witching_Day"]]
    if not witching.empty:
        rows.extend(_share_rows(witching, "witching"))
    shares = pd.DataFrame(rows)

    def _vw(scope: str, bucket: str) -> float:
        sel = shares[(shares["scope"] == scope) & (shares["bucket"] == bucket)]
        return float(sel["share_volume_weighted"].iloc[0])

    aft = _vw("all_days", "afternoon_30m")
    mid = _vw("all_days", "mid_day")
    continuous = aft + mid + _vw("all_days", "morning_30m")
    summary = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "db_path": str(db_path),
        "universe": universe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "trading_days": int(panel["Date"].nunique()),
        "symbol_days": int(len(panel)),
        "witching_symbol_days": int(panel["Is_Witching_Day"].sum()),
        "afternoon_30m_share_vw": aft,
        "mid_day_share_vw": mid,
        "mid_day_share_per_30min_vw": mid / (MID_DAY_HOURS * 2),
        "afternoon_to_midday_intensity_ratio": (
            aft / (mid / (MID_DAY_HOURS * 2)) if mid > 0 else float("nan")
        ),
        "afternoon_30m_share_of_continuous_vw": (
            aft / continuous if continuous > 0 else float("nan")
        ),
        "close_auction_share_vw": _vw("all_days", "close_auction"),
        "close_auction_share_ew": float(shares[
            (shares["scope"] == "all_days") & (shares["bucket"] == "close_auction")
        ]["share_equal_weighted"].iloc[0]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    shares.to_csv(out_dir / "bucket_shares.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--start", type=_dt.date.fromisoformat,
                        default=_dt.date(2018, 1, 2))
    parser.add_argument("--end", type=_dt.date.fromisoformat,
                        default=_dt.date(2019, 12, 31))
    args = parser.parse_args()
    summary = build_bucket_shares(
        args.db, args.out, start=args.start, end=args.end, universe=args.universe,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
