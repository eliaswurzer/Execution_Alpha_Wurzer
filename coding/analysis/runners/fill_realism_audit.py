"""
fill_realism_audit.py -- Compare tape-replay fill rules on real symbol-days.

Runs the same parent orders through every tape-replay fill specification and
reports per-spec fill rates, time-to-first-fill, adverse selection, and alpha
versus MOC. The table quantifies the bounds bracket used in the thesis:

    tape_replay_strict  <=  tape_replay_queue  <=  tape_replay (at-or-through)

Usage::

    python -m analysis.runners.fill_realism_audit --symbols AAPL MSFT JPM \
        --start 2018-02-01 --n-days 5

Output: ``<artifacts>/fill_realism_audit/fill_realism_audit.csv`` plus a
Markdown summary, and the summary table on stdout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.taq_loader import (
    extract_closing_auction_details,
    filter_regular_hours,
    filter_valid_trades,
    list_dates,
    load_trades,
    trades_parquet_path,
)
from ..metrics.alpha import attach_alpha_columns, attach_moc_differential_columns
from ..simulation.engine import simulate_symbol_day
from ..simulation.parent_orders import build_parent_orders

log = logging.getLogger(__name__)

DEFAULT_SPECS = (
    "tape_replay",
    "tape_replay_queue",
    "tape_replay_strict",
    "tape_replay_haircut",
    "tape_replay_volume",
)
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "AMZN", "FB", "JPM", "XOM", "JNJ", "PG")
DEFAULT_STRATEGIES = ("S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE")


def _expected_vc(date: _dt.date, symbol: str) -> float:
    """Same-day V_C as the order-sizing anchor.

    The audit compares fill mechanics across specs on identical orders, so the
    pilot-style same-day fallback is acceptable here (documented deviation in
    ``parent_orders.same_day_vc_fallback``); it is NOT used for headline runs.
    """
    try:
        trades = load_trades(date, symbol, rth_only=False)
    except FileNotFoundError:
        return 0.0
    auction = extract_closing_auction_details(filter_valid_trades(trades))
    return float(auction.volume)


def run_audit(
    symbols: list[str],
    dates: list[_dt.date],
    specs: list[str],
    strategies: list[str],
    out_dir: Path,
    *,
    window: str = cfg.PRIMARY_WINDOW,
    size_frac: float = cfg.PARENT_ORDER_PRIMARY_FRACTION,
    delta_max_bps: float | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_by_tier = (
        {t: float(delta_max_bps) for t in cfg.DELTA_MAX_BPS}
        if delta_max_bps is not None else cfg.DELTA_MAX_BPS
    )
    rows: list[pd.DataFrame] = []

    for date in dates:
        for symbol in symbols:
            if not trades_parquet_path(date, symbol).exists():
                continue
            expected_vc = _expected_vc(date, symbol)
            if expected_vc <= 0:
                continue
            parents = build_parent_orders(symbol, date, expected_vc)
            parents = parents[
                (parents["window"] == window)
                & np.isclose(parents["size_frac"].astype(float), size_frac)
            ]
            if parents.empty:
                continue
            # All specs must see the identical symbol-day sample; a silent
            # per-spec drop (e.g. a transient parquet read error swallowed as
            # FileNotFoundError) would bias the cross-spec comparison.
            per_spec: dict[str, pd.DataFrame] = {}
            for spec in specs:
                res = pd.DataFrame()
                for _attempt in range(2):
                    res = simulate_symbol_day(
                        symbol, date, parents, strategies,
                        fill_model=None,
                        delta_max_bps_by_tier=delta_by_tier,
                        tier=1,
                        fill_specification=spec,
                    )
                    if not res.empty:
                        break
                per_spec[spec] = res
            n_nonempty = sum(1 for r in per_spec.values() if not r.empty)
            if n_nonempty == 0:
                continue
            if n_nonempty != len(specs):
                missing = [s for s, r in per_spec.items() if r.empty]
                raise RuntimeError(
                    f"Inconsistent symbol-day sample for {symbol} {date}: "
                    f"specs {missing} returned no rows while others did"
                )
            for spec, res in per_spec.items():
                res = res.copy()
                res["fill_spec"] = spec
                rows.append(res)

    if not rows:
        raise RuntimeError("Fill realism audit produced no rows — check data paths")

    panel = pd.concat(rows, ignore_index=True)
    panel = attach_alpha_columns(panel)
    panel = pd.concat(
        [
            attach_moc_differential_columns(grp)
            for _, grp in panel.groupby("fill_spec", sort=False)
        ],
        ignore_index=True,
    )

    first_fill = pd.to_datetime(panel["first_fill_time"], errors="coerce")
    arrival = pd.to_datetime(panel["arrival_time"], errors="coerce")
    panel["seconds_to_first_fill"] = (first_fill - arrival).dt.total_seconds()

    passive = panel[panel["strategy"] != "S0_MOC"]
    summary = (
        passive.groupby(["fill_spec", "strategy"])
        .agg(
            n_orders=("order_id", "count"),
            mean_fill_rate=("fill_rate", "mean"),
            median_fill_rate=("fill_rate", "median"),
            share_fully_passive=("fill_rate", lambda s: float((s >= 0.999).mean())),
            median_secs_to_first_fill=("seconds_to_first_fill", "median"),
            mean_as_bps=("adverse_selection_bps", "mean"),
            mean_net_alpha_bps=("net_alpha_bps", "mean"),
            mean_net_alpha_vs_moc_bps=("net_alpha_vs_moc_bps", "mean"),
        )
        .reset_index()
        .sort_values(["strategy", "fill_spec"])
    )

    panel.to_parquet(out_dir / "fill_realism_panel.parquet", index=False)
    summary.to_csv(out_dir / "fill_realism_audit.csv", index=False)
    md_lines = [
        "# Fill realism audit",
        "",
        f"- Symbols: {', '.join(symbols)}",
        f"- Dates: {dates[0].isoformat()} .. {dates[-1].isoformat()} ({len(dates)} days)",
        f"- Window {window}, size_frac {size_frac:g}, specs: {', '.join(specs)}",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (out_dir / "fill_realism_audit.md").write_text(
        "\n".join(md_lines), encoding="utf-8",
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--n-days", type=int, default=5)
    p.add_argument("--specs", nargs="*", default=list(DEFAULT_SPECS))
    p.add_argument("--strategies", nargs="*", default=list(DEFAULT_STRATEGIES))
    p.add_argument("--window", default=cfg.PRIMARY_WINDOW)
    p.add_argument("--size-frac", type=float, default=cfg.PARENT_ORDER_PRIMARY_FRACTION)
    p.add_argument("--delta-max-bps", type=float, default=None,
                   help="Override delta_max for all tiers (0 = at-touch posting, "
                        "where the queue rule binds against displayed depth)")
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "fill_realism_audit")
    args = p.parse_args()

    dates = [d for d in list_dates(args.start.year) if d >= args.start][: args.n_days]
    if not dates:
        raise SystemExit(f"No preprocessed dates found from {args.start}")

    summary = run_audit(
        args.symbols, dates, args.specs, args.strategies, args.out,
        window=args.window, size_frac=args.size_frac,
        delta_max_bps=args.delta_max_bps,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
