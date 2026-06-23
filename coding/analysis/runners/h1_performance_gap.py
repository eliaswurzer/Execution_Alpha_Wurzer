"""
h1_performance_gap.py -- Hypothesis 1 (thesis Section 3.2, H1 performance gap).

Runs S0 through S3 on the evaluation panel and reports:

* mean net alpha by strategy with two-way clustered standard errors,
* the primary t-test for S3_FULL in Window B,
* subgroup tests by tier, year, size, listing exchange, and dissemination state.

Usage::

    python -m analysis.runners.h1_performance_gap --max-dates 30 --symbols AAPL MSFT GOOG
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..data.listing_exchange import build_listing_map, merge_listing_into_panel
from ..inference.tests import primary_ttest, subgroup_ttests
from ..metrics import tracking_error_variance
from ..metrics.alpha import break_even_impact_coef
from ._common import require_headline_panel, run_panel, validate_run


log = logging.getLogger(__name__)


def _attach_dissemination_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the exchange-specific post-dissemination indicator."""
    if "arrival_time" not in panel.columns or "listing_exchange" not in panel.columns:
        return panel
    out = panel.copy()
    arrival_time = pd.to_datetime(out["arrival_time"]).dt.time
    cutoffs = out["listing_exchange"].map(cfg.DISSEMINATION_START_BY_LISTING)
    cutoffs = cutoffs.fillna(cfg.DISSEMINATION_START_BY_LISTING["NASDAQ"])
    out["post_dissemination"] = [
        bool(t >= c) if (t is not None and c is not None) else False
        for t, c in zip(arrival_time, cutoffs)
    ]
    return out


def _primary_surface(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the Window-B, one-percent parent-order surface used by H1."""
    out = panel.copy()
    if "window" in out.columns:
        out = out[out["window"] == cfg.PRIMARY_WINDOW]
    if "size_frac" in out.columns:
        size = pd.to_numeric(out["size_frac"], errors="coerce")
        out = out[(size - cfg.PARENT_ORDER_PRIMARY_FRACTION).abs() <= 1e-12]
    return out.copy()


def analyze_panel(
    panel: pd.DataFrame,
    out_dir: Path,
    *,
    panel_path: Path | None = None,
) -> None:
    if panel.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_path or (out_dir / "h1_panel.parquet")
    require_headline_panel(panel, "H1", require_moc=True)

    # Enrich listing exchange and dissemination timing. Panel rows carry
    # listing_exchange from the membership reference map; the tape heuristic
    # only backfills symbols whose rows arrived blank.
    if "symbol" in panel.columns:
        if "listing_exchange" in panel.columns:
            vals = panel["listing_exchange"].astype("string")
            blank = vals.isna() | (vals.str.strip() == "")
        else:
            blank = pd.Series(True, index=panel.index)
        missing_syms = sorted(panel.loc[blank, "symbol"].unique().tolist())
        if missing_syms:
            ref_date = panel["date"].min()
            if isinstance(ref_date, str):
                ref_date = _dt.date.fromisoformat(ref_date)
            listing_map = build_listing_map(
                missing_syms, pd.Timestamp(ref_date).date(),
            )
            if "listing_exchange" not in panel.columns:
                panel = merge_listing_into_panel(panel, listing_map)
            else:
                panel = panel.copy()
                panel.loc[blank, "listing_exchange"] = (
                    panel.loc[blank, "symbol"].map(listing_map).fillna("NASDAQ")
                )
    panel = _attach_dissemination_flag(panel)
    if "post_dissemination" not in panel.columns:
        log.warning("H1 post-dissemination subgroup unavailable; metadata missing.")
    panel.to_parquet(panel_path, index=False)

    tev = tracking_error_variance(panel, alpha_col="net_alpha_bps")
    tev.to_csv(out_dir / "h1_tev.csv", index=False)
    log.info("\n%s", tev.to_string(index=False))

    primary = primary_ttest(panel)
    log.info("Primary: %s", primary)
    pd.DataFrame([vars(primary)]).to_csv(out_dir / "h1_primary_ttest.csv", index=False)

    subgroup_panel = _primary_surface(panel)
    for by in ("tier", "adv_bucket", "spread_bucket", "adv_spread_bucket", "year", "size_frac", "listing_exchange", "post_dissemination"):
        if by in subgroup_panel.columns:
            sub = subgroup_ttests(
                subgroup_panel, by=by, alpha_col="net_alpha_vs_moc_bps",
            )
            if not sub.empty:
                sub.to_csv(out_dir / f"h1_subgroup_{by}.csv", index=False)

    # Break-even impact coefficient for large-parent extension
    if "size_frac" in panel.columns and "impact_bps" in panel.columns:
        non_headline = panel[panel["size_frac"] > cfg.IMPACT_ACTIVATION_THRESHOLD].copy()
        if not non_headline.empty:
            non_headline["net_alpha_no_impact"] = (
                non_headline["net_alpha_bps"] + non_headline["impact_bps"]
            )
            non_headline["break_even_coef_bps"] = [
                break_even_impact_coef(a, s)
                for a, s in zip(
                    non_headline["net_alpha_no_impact"], non_headline["size_frac"]
                )
            ]
            be_summary = (
                non_headline.groupby(["strategy", "size_frac"])["break_even_coef_bps"]
                .agg(["median", "mean"])
                .reset_index()
            )
            be_summary.to_csv(out_dir / "h1_breakeven_impact.csv", index=False)
            log.info("\n[Break-even impact coefficients (large-parent extension)]\n%s",
                     be_summary.to_string(index=False))


def run(symbols: list[str] | None, start, end, artifacts_dir: Path, out_dir: Path,
        max_dates: int | None = None, workers: int = 1,
        fill_specification: str = "tape_replay_queue",
        universe: str | None = None) -> None:
    panel_path = out_dir / "h1_panel.parquet"
    panel = run_panel(
        strategies=["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
                    "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD"],
        start=start, end=end,
        artifacts_dir=artifacts_dir, out_path=panel_path,
        symbols=symbols, universe=universe, max_dates=max_dates, workers=workers,
        fill_specification=fill_specification,
        size_fractions=(cfg.PARENT_ORDER_PRIMARY_FRACTION,),
    )
    analyze_panel(panel, out_dir, panel_path=panel_path)


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500",
                   help="Point-in-time index universe (default: sp500 = full "
                        "point-in-time S&P 500 membership); ignored when --symbols is set")
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.EVAL_END)
    p.add_argument("--artifacts", type=Path, default=cfg.ARTIFACTS_DIR / "fill_model")
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "h1")
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for simulation (default: 1; try 4)")
    p.add_argument("--fill-spec", default="tape_replay_queue",
                   choices=["tape_replay", "tape_replay_haircut",
                            "tape_replay_volume", "tape_replay_volume_haircut",
                            "tape_replay_strict", "tape_replay_queue",
                            "cox", "km", "infinite_depth",
                            "infinite_depth_haircut", "xgb"],
                   help="Fill mechanism (default: tape_replay_queue = "
                        "volume-ahead queue model)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config and artifacts without running simulation")
    args = p.parse_args()
    if args.dry_run:
        strategies = ["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
                      "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD"]
        validate_run(strategies, args.start, args.end, args.artifacts,
                     args.symbols, args.universe, args.fill_spec)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.symbols, args.start, args.end, args.artifacts, args.out,
        args.max_dates, args.workers, args.fill_spec, args.universe)


if __name__ == "__main__":
    main()

