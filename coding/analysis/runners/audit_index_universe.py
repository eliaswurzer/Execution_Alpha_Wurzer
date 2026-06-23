"""Audit point-in-time index membership against available TAQ parquet files."""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..data.index_universe import build_index_universe_panel
from ..data.taq_loader import nbbo_parquet_path, trades_parquet_path
from ._common import _eval_dates


def audit_index_universe(
    universe: str,
    start: _dt.date,
    end: _dt.date,
    out_dir: Path,
    max_dates: int | None = None,
) -> dict[str, pd.DataFrame]:
    dates = _eval_dates(start, end)
    if max_dates is not None:
        dates = dates[:max_dates]
    panel = build_index_universe_panel(universe, dates)
    rows = []
    for row in panel[["date", "symbol", "index_id"]].itertuples(index=False):
        d = pd.Timestamp(row.date).date()
        sym = str(row.symbol)
        trade_ok = trades_parquet_path(d, sym).exists()
        nbbo_ok = nbbo_parquet_path(d, sym).exists()
        rows.append({
            "date": d,
            "symbol": sym,
            "index_id": row.index_id,
            "trade_parquet": trade_ok,
            "nbbo_parquet": nbbo_ok,
            "complete": trade_ok and nbbo_ok,
        })
    coverage = pd.DataFrame(rows)
    if coverage.empty:
        summary = pd.DataFrame([{
            "universe": universe,
            "start": start,
            "end": end,
            "dates": len(dates),
            "symbol_days": 0,
            "unique_symbols": 0,
            "complete_symbol_days": 0,
            "coverage_rate": 0.0,
        }])
    else:
        summary = pd.DataFrame([{
            "universe": universe,
            "start": start,
            "end": end,
            "dates": len(dates),
            "symbol_days": len(coverage),
            "unique_symbols": coverage["symbol"].nunique(),
            "complete_symbol_days": int(coverage["complete"].sum()),
            "coverage_rate": float(coverage["complete"].mean()),
        }])

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{universe}_coverage_summary.csv", index=False)
    coverage.to_csv(out_dir / f"{universe}_symbol_day_coverage.csv", index=False)
    missing = coverage[~coverage["complete"]].copy()
    missing.to_csv(out_dir / f"{universe}_missing_symbol_days.csv", index=False)
    return {"summary": summary, "coverage": coverage, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit point-in-time index membership against TAQ parquet availability."
    )
    parser.add_argument("--universe", choices=["sp500", "nasdaq100"], required=True)
    parser.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    parser.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.EVAL_END)
    parser.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "universe_audit")
    parser.add_argument("--max-dates", type=int, default=None)
    args = parser.parse_args()

    result = audit_index_universe(
        args.universe, args.start, args.end, args.out, args.max_dates,
    )
    print(result["summary"].to_string(index=False))
    if not result["missing"].empty:
        print(f"Missing symbol-days written: {args.out / f'{args.universe}_missing_symbol_days.csv'}")


if __name__ == "__main__":
    main()
