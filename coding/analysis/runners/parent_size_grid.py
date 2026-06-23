"""
parent_size_grid.py -- resumable parent-size robustness run over the grid
``{0.005, 0.01, 0.02, 0.05, 0.10} * E[V_C]`` at the primary arrival window.

Runs the same resumable, gated master-panel pipeline as the headline run
(point-in-time universe, daily shards, fingerprints, coverage gates, tier
policy) with strategies ``S0_MOC`` and ``S3_FULL`` and window ``B`` only,
which is exactly what the thesis size-robustness table reports. The
hypothesis analyzers are intentionally NOT invoked: they guard against
mixed-size panels.

Outputs under ``<run-root>/``::

    panel_shards/...                      resumable daily shards
    robustness_panel.parquet              all sizes, long format
    size_buckets/size_<frac>/panel.parquet
    size_buckets/size_<frac>/summary_by_strategy.csv
    robustness_summary_clustered.csv      (size, strategy) means + two-way SEs
    impact_sweep_summary.csv              net alpha under IMPACT_COEF_BPS_GRID

Usage::

    python -m analysis.runners.parent_size_grid --run-id size_grid_<date> \\
        --universe sp500 --workers 6 --artifacts <fill_model_v2>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..inference.clustering import mean_with_twoway_se
from .master_panel import (
    TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    TIER_POLICY_CHOICES,
    materialize_panel,
    run_master_panel,
)

log = logging.getLogger(__name__)

DEFAULT_SIZE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10)
GRID_STRATEGIES = ["S0_MOC", "S3_FULL"]
GRID_WINDOWS = ("B",)
HEADLINE_SIZE = cfg.PARENT_ORDER_PRIMARY_FRACTION


def _filter_panel_to_size(panel: pd.DataFrame, size_frac: float, tol: float = 1e-9) -> pd.DataFrame:
    if "size_frac" not in panel.columns:
        return panel.copy()
    return panel[(panel["size_frac"] - size_frac).abs() < tol].copy()


def _clustered_rows(panel: pd.DataFrame, value_col: str) -> list[dict]:
    rows = []
    for (size, strategy), grp in panel.groupby(["size_frac", "strategy"]):
        vals = grp[value_col].dropna()
        sub = grp.loc[vals.index]
        if sub.empty:
            continue
        m, se = mean_with_twoway_se(sub[value_col], sub["symbol"], sub["date"])
        rows.append({
            "size_bucket": float(size),
            "strategy": strategy,
            "metric": value_col,
            "mean": m,
            "se_twoway": se,
            "t": m / se if se and np.isfinite(se) and se > 0 else np.nan,
            "n": int(len(sub)),
        })
    return rows


def summarize_grid_panel(panel: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Clustered (size, strategy) summaries for the thesis size table."""
    rows: list[dict] = []
    rows.extend(_clustered_rows(panel, "net_alpha_bps"))
    if "net_alpha_vs_moc_bps" in panel.columns:
        rows.extend(_clustered_rows(
            panel[panel["strategy"] != "S0_MOC"], "net_alpha_vs_moc_bps",
        ))
    summary = pd.DataFrame(rows).sort_values(
        ["metric", "size_bucket", "strategy"],
    ).reset_index(drop=True)
    summary.to_csv(out_root / "robustness_summary_clustered.csv", index=False)

    # Tidy per-size table input for the thesis renderer (Panel A of
    # tab:parent-window-robustness): S3_FULL clustered net alpha plus the
    # diagnostic columns of the table.
    from ..metrics import tracking_error_variance

    table_rows: list[dict] = []
    focus = panel[panel["strategy"] == "S3_FULL"]
    for size, grp in focus.groupby("size_frac"):
        vals = grp["net_alpha_bps"].dropna()
        sub = grp.loc[vals.index]
        if sub.empty:
            continue
        m, se = mean_with_twoway_se(
            sub["net_alpha_bps"], sub["symbol"], sub["date"],
        )
        tev_frame = tracking_error_variance(sub, alpha_col="net_alpha_bps")
        tev = (
            float(tev_frame.loc[tev_frame["strategy"] == "S3_FULL", "tev"].iloc[0])
            if not tev_frame.empty else np.nan
        )
        # Primary AS diagnostic: negated mean SIGNED markout conditional on a
        # passive fill (positive = adverse); avoids the Jensen markup of the
        # one-sided cost and matches the markout literature.
        filled = sub[sub["fill_rate"] > 0]
        as_markout = (
            float(-filled["adverse_selection_bps"].mean())
            if "adverse_selection_bps" in sub.columns and not filled.empty
            else np.nan
        )
        table_rows.append({
            "size_bucket": float(size),
            "strategy": "S3_FULL",
            "mean_net_alpha_bps": m,
            "se_twoway": se,
            "t": m / se if se and np.isfinite(se) and se > 0 else np.nan,
            "mean_fill_rate": float(sub["fill_rate"].mean()),
            "mean_as_markout_bps": as_markout,
            "tev": tev,
            "n": int(len(sub)),
        })
    pd.DataFrame(table_rows).sort_values("size_bucket").to_csv(
        out_root / "size_table_summary.csv", index=False,
    )
    return summary


def impact_sweep(panel: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Post-hoc net-alpha reconstruction under the impact-coefficient grid.

    Impact enters net alpha additively as ``impact_bps = kappa * sqrt(size)``
    for active rows (size above the activation threshold, passive strategies
    only), so for an alternative coefficient ``kappa'`` the panel value is
    ``net' = net + impact_bps * (1 - kappa'/kappa_headline)`` row by row.
    No re-simulation is required and inactive rows are untouched because
    their stored ``impact_bps`` is zero.
    """
    if "impact_bps" not in panel.columns:
        log.warning("impact sweep skipped: panel lacks impact_bps")
        return pd.DataFrame()
    rows: list[dict] = []
    base = panel[panel["strategy"] != "S0_MOC"]
    for kappa in cfg.IMPACT_COEF_BPS_GRID:
        scale = 1.0 - kappa / cfg.IMPACT_COEF_BPS
        adjusted = base.copy()
        adjusted["net_alpha_kappa_bps"] = (
            adjusted["net_alpha_bps"] + adjusted["impact_bps"] * scale
        )
        for row in _clustered_rows(adjusted, "net_alpha_kappa_bps"):
            row["impact_coef_bps"] = float(kappa)
            rows.append(row)
    sweep = pd.DataFrame(rows).sort_values(
        ["impact_coef_bps", "size_bucket", "strategy"],
    ).reset_index(drop=True)
    sweep.to_csv(out_root / "impact_sweep_summary.csv", index=False)
    return sweep


def run_size_grid(
    *,
    start: _dt.date,
    end: _dt.date,
    artifacts_dir: Path,
    run_root: Path,
    universe: str | None = "sp500",
    symbols: list[str] | None = None,
    sizes: tuple[float, ...] = DEFAULT_SIZE_GRID,
    windows: tuple[str, ...] = GRID_WINDOWS,
    strategies: list[str] | None = None,
    workers: int = 1,
    fill_specification: str = "tape_replay_queue",
    tier_policy: str = TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    max_dates: int | None = None,
    resume: bool = True,
) -> dict[float, pd.DataFrame]:
    strategies = list(strategies or GRID_STRATEGIES)
    sizes = tuple(sorted(set(float(s) for s in sizes)))
    run_root.mkdir(parents=True, exist_ok=True)

    summary = run_master_panel(
        strategies=strategies,
        start=start, end=end,
        artifacts_dir=artifacts_dir,
        run_root=run_root,
        symbols=symbols, universe=universe,
        max_dates=max_dates, workers=workers,
        fill_specification=fill_specification,
        size_fractions=sizes,
        windows=windows,
        resume=resume,
        tier_policy=tier_policy,
    )
    log.info("Size-grid master panel complete: %s", {
        k: summary.get(k) for k in (
            "status", "eligible_coverage", "critical_failures",
        )
    })

    master_path = run_root / "robustness_panel.parquet"
    materialize_panel(run_root / "panel_shards", strategies, master_path)
    panel = pd.read_parquet(master_path)
    if panel.empty:
        log.warning("Size-grid panel empty.")
        return {}

    panels: dict[float, pd.DataFrame] = {}
    bucket_root = run_root / "size_buckets"
    for s in sizes:
        sub = _filter_panel_to_size(panel, s)
        if sub.empty:
            log.warning("size %.4f: no rows -- skipping", s)
            continue
        sub_dir = bucket_root / f"size_{s:.4f}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(sub_dir / "panel.parquet", index=False)
        per_strategy = (
            sub.groupby("strategy")
               .agg(mean_alpha_bps=("alpha_bps", "mean"),
                    mean_net_alpha_bps=("net_alpha_bps", "mean"),
                    mean_fill_rate=("fill_rate", "mean"),
                    n=("order_id", "count"))
               .round(4)
        )
        per_strategy.to_csv(sub_dir / "summary_by_strategy.csv")
        panels[s] = sub

    clustered = summarize_grid_panel(panel, run_root)
    sweep = impact_sweep(panel, run_root)
    log.info(
        "Size-grid reporting written: %d clustered rows, %d sweep rows",
        len(clustered), len(sweep),
    )
    return panels


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", type=Path, default=cfg.ARTIFACTS_DIR / "runs")
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.EVAL_END)
    p.add_argument("--sizes", nargs="*", type=float, default=list(DEFAULT_SIZE_GRID))
    p.add_argument("--windows", nargs="*", default=list(GRID_WINDOWS))
    p.add_argument("--strategies", nargs="*", default=list(GRID_STRATEGIES))
    p.add_argument("--artifacts", type=Path, default=cfg.ARTIFACTS_DIR / "fill_model_v2")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fill-spec", default="tape_replay_queue")
    p.add_argument("--tier-policy", choices=sorted(TIER_POLICY_CHOICES),
                   default=TIER_POLICY_CALIBRATED_PLUS_FALLBACK)
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    run_size_grid(
        start=args.start, end=args.end,
        artifacts_dir=args.artifacts,
        run_root=args.run_root / args.run_id,
        universe=None if args.symbols else args.universe,
        symbols=args.symbols,
        sizes=tuple(args.sizes),
        windows=tuple(args.windows),
        strategies=args.strategies,
        workers=args.workers,
        fill_specification=args.fill_spec,
        tier_policy=args.tier_policy,
        max_dates=args.max_dates,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
