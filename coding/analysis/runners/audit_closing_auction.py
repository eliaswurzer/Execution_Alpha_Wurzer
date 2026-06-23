"""Audit closing-auction extraction sources across a TAQ parquet panel."""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..data.index_universe import build_index_universe_panel
from ..data.taq_loader import (
    extract_closing_auction_details,
    filter_valid_trades,
    list_symbols,
    load_trades,
)
from ._common import _eval_dates

log = logging.getLogger(__name__)


def _date_arg(value: str) -> _dt.date:
    return _dt.date.fromisoformat(value)


def _symbol_panel(
    dates: list[_dt.date],
    *,
    symbols: list[str] | None,
    universe: str | None,
) -> list[tuple[_dt.date, str]]:
    if symbols:
        return [(d, s) for d in dates for s in symbols]
    if universe:
        panel = build_index_universe_panel(universe, dates)
        return [
            (pd.Timestamp(row.date).date(), str(row.symbol))
            for row in panel.itertuples(index=False)
        ]
    return [(d, s) for d in dates for s in list_symbols(d)]


def audit_closing_auction(
    start: _dt.date,
    end: _dt.date,
    *,
    symbols: list[str] | None = None,
    universe: str | None = None,
    max_symbol_days: int | None = None,
) -> pd.DataFrame:
    dates = _eval_dates(start, end)
    pairs = _symbol_panel(dates, symbols=symbols, universe=universe)
    if max_symbol_days is not None:
        pairs = pairs[:max_symbol_days]

    rows: list[dict] = []
    for d, sym in pairs:
        try:
            trades = load_trades(d, sym, rth_only=False)
        except FileNotFoundError as exc:
            rows.append({
                "date": d,
                "symbol": sym,
                "price": float("nan"),
                "vc_shares": 0.0,
                "price_source": "missing_parquet",
                "vc_source": "missing_parquet",
                "close_trade_volume": 0.0,
                "close_trade_rows": 0,
                "official_close_marker_volume": 0.0,
                "official_close_marker_rows": 0,
                "official_close_marker_fallback_volume": 0.0,
                "error": str(exc),
            })
            continue
        trades = filter_valid_trades(trades)
        auction = extract_closing_auction_details(trades)
        rows.append({
            "date": d,
            "symbol": sym,
            "price": auction.price,
            "vc_shares": auction.volume,
            "price_source": auction.price_source,
            "vc_source": auction.volume_source,
            "close_trade_volume": auction.close_trade_volume,
            "close_trade_rows": auction.close_trade_rows,
            "official_close_marker_volume": auction.official_marker_volume,
            "official_close_marker_rows": auction.official_marker_rows,
            "official_close_marker_fallback_volume": auction.official_marker_fallback_volume,
            "error": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", type=_date_arg, required=True)
    ap.add_argument("--end-date", type=_date_arg, required=True)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--max-symbol-days", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=cfg.ARTIFACTS_DIR / "closing_auction_audit",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = audit_closing_auction(
        args.start_date,
        args.end_date,
        symbols=args.symbols,
        universe=args.universe,
        max_symbol_days=args.max_symbol_days,
    )
    detail_path = out_dir / "closing_auction_detail.csv"
    summary_path = out_dir / "closing_auction_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary = (
        detail.groupby(["vc_source", "price_source"], dropna=False)
        .agg(
            symbol_days=("symbol", "size"),
            vc_shares=("vc_shares", "sum"),
            close_trade_rows=("close_trade_rows", "sum"),
            official_marker_rows=("official_close_marker_rows", "sum"),
        )
        .reset_index()
        .sort_values(["symbol_days", "vc_source"], ascending=[False, True])
    )
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", detail_path)
    log.info("Wrote %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
