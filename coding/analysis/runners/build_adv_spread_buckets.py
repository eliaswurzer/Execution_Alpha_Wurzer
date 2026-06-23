"""Build fixed ADV x spread reporting buckets from a calibration window."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..data.adv_spread_buckets import (
    BUCKET_POLICY_VERSION,
    assign_adv_spread_buckets,
    summarize_adv_spread_buckets,
    write_bucket_manifest,
)
from ..data.features import compute_daily_features
from ..data.index_universe import build_index_universe_panel
from ..data.taq_loader import (
    TradePolicyMismatchError,
    filter_trades_near_quotes,
    filter_valid_quotes,
    filter_valid_trades,
    load_symbol_day,
)
from . import _common

log = logging.getLogger(__name__)


def _symbol_day_features(date: dt.date, symbol: str) -> tuple[dict | None, dict]:
    started = time.perf_counter()
    status = {
        "date": date,
        "symbol": symbol,
        "status": "ok",
        "reason": "",
        "runtime_seconds": 0.0,
    }
    try:
        trades, nbbo = load_symbol_day(date, symbol)
        trades = filter_valid_trades(trades)
        nbbo = filter_valid_quotes(nbbo)
        if nbbo.empty:
            status.update(status="skipped", reason="insufficient_quotes")
            return None, status
        trades = filter_trades_near_quotes(trades, nbbo)
        if trades.empty:
            status.update(status="skipped", reason="empty_after_filter")
            return None, status
        row = compute_daily_features(trades, nbbo, symbol, date).to_dict()
        return row, status
    except FileNotFoundError:
        status.update(status="skipped", reason="missing_parquet")
        return None, status
    except TradePolicyMismatchError as exc:
        status.update(status="failed", reason="policy_mismatch", detail=str(exc))
        return None, status
    except Exception as exc:  # diagnostics only, one bad symbol-day must not kill the map
        status.update(status="failed", reason=type(exc).__name__, detail=str(exc))
        return None, status
    finally:
        status["runtime_seconds"] = time.perf_counter() - started


def build_bucket_artifacts(
    *,
    start: dt.date,
    end: dt.date,
    universe: str,
    out_dir: Path,
    workers: int = 4,
    min_days: int = 1,
    limit: int | None = None,
) -> dict:
    dates = _common._eval_dates(start, end)
    dates = [d for d in dates if start <= d <= end]
    membership = build_index_universe_panel(universe, dates, expand_aliases=False)
    pairs = (
        membership[["date", "symbol"]]
        .drop_duplicates()
        .sort_values(["date", "symbol"])
        .itertuples(index=False)
    )
    pair_list = [(pd.Timestamp(row.date).date(), str(row.symbol)) for row in pairs]
    if limit is not None:
        pair_list = pair_list[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    statuses: list[dict] = []
    workers = max(1, int(workers))
    if workers <= 1:
        for date, symbol in pair_list:
            row, status = _symbol_day_features(date, symbol)
            if row is not None:
                rows.append(row)
            statuses.append(status)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
            futures = {pool.submit(_symbol_day_features, d, s): (d, s) for d, s in pair_list}
            for i, future in enumerate(as_completed(futures), start=1):
                row, status = future.result()
                if row is not None:
                    rows.append(row)
                statuses.append(status)
                if i % 500 == 0:
                    log.info("bucket feature progress: %d/%d", i, len(pair_list))

    daily_features = pd.DataFrame(rows)
    status_df = pd.DataFrame(statuses)
    bucket_map = assign_adv_spread_buckets(daily_features, min_days=min_days)
    summary = summarize_adv_spread_buckets(bucket_map)

    map_path = out_dir / "symbol_adv_spread_bucket_map.csv"
    summary_path = out_dir / "adv_spread_bucket_summary.csv"
    manifest_path = out_dir / "adv_spread_bucket_manifest.json"
    status_path = out_dir / "adv_spread_bucket_status.csv"
    bucket_map.to_csv(map_path, index=False)
    summary.to_csv(summary_path, index=False)
    status_df.to_csv(status_path, index=False)

    ok_pairs = int((status_df["status"] == "ok").sum()) if not status_df.empty else 0
    manifest = {
        "status": "complete" if not bucket_map.empty else "empty",
        "bucket_policy": BUCKET_POLICY_VERSION,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe": universe,
        "n_dates": len(dates),
        "n_symbol_days_expected": len(pair_list),
        "n_symbol_days_ok": ok_pairs,
        "coverage": ok_pairs / max(len(pair_list), 1),
        "n_symbols_bucketed": int((bucket_map["adv_spread_bucket"] != "unassigned").sum()) if not bucket_map.empty else 0,
        "n_symbols_unassigned": int((bucket_map["adv_spread_bucket"] == "unassigned").sum()) if not bucket_map.empty else 0,
        "min_days": int(min_days),
        "adv_metric": "mean_adv_dollar",
        "spread_metric": "avg_quoted_spread_bps",
        "map_path": str(map_path),
        "summary_path": str(summary_path),
        "status_path": str(status_path),
    }
    write_bucket_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2018, 1, 2))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2018, 6, 29))
    parser.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-days", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = build_bucket_artifacts(
        start=args.start,
        end=args.end,
        universe=args.universe,
        out_dir=args.out,
        workers=args.workers,
        min_days=args.min_days,
        limit=args.limit,
    )
    print(manifest)


if __name__ == "__main__":
    main()
