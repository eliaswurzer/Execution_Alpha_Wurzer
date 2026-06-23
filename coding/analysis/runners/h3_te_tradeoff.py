"""
h3_te_tradeoff.py -- Hypothese 3 (Thesis Â§3.2, H3 "Tracking Error Trade-off").

Berechnet fuer jede Strategie TEV, RAEAR(eta), Break-Even-eta* und die
Portfolio-TE-Boundaries (independence + perfect correlation).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..metrics import (
    portfolio_tracking_error, raear_panel, tracking_error_variance,
)
from ._common import require_headline_panel, run_panel, validate_run

log = logging.getLogger(__name__)

H3_ALPHA_COL = "net_alpha_vs_moc_bps"
H3_METRIC_POLICY_VERSION = "h3_moc_relative_primary_cell_v2"


def primary_surface(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the headline H3 cell aligned with the primary H1 test."""
    out = panel.copy()
    if "window" in out.columns:
        out = out[out["window"] == cfg.PRIMARY_WINDOW]
    if "size_frac" in out.columns:
        size = pd.to_numeric(out["size_frac"], errors="coerce")
        out = out[(size - cfg.PARENT_ORDER_PRIMARY_FRACTION).abs() <= 1e-12]
    return out.copy()


def _pin_moc_benchmark(tev: pd.DataFrame) -> pd.DataFrame:
    out = tev.copy()
    if {"strategy", "mean_alpha", "tev"}.issubset(out.columns):
        mask = out["strategy"] == "S0_MOC"
        out.loc[mask, ["mean_alpha", "tev"]] = 0.0
    return out


def _write_grouped_raear(panel: pd.DataFrame, out_dir: Path, group_col: str, etas: list[float]) -> None:
    if panel.empty or group_col not in panel.columns:
        return
    tev_parts = []
    raear_parts = []
    for group_value, grp in panel.groupby(group_col, dropna=False):
        if grp.empty:
            continue
        label = group_value if pd.notna(group_value) else "unassigned"
        tev = _pin_moc_benchmark(tracking_error_variance(grp, alpha_col=H3_ALPHA_COL))
        tev[group_col] = label
        tev_parts.append(tev)
        ra = raear_panel(tev.drop(columns=[group_col]), etas)
        ra[group_col] = label
        raear_parts.append(ra)
    if tev_parts:
        pd.concat(tev_parts, ignore_index=True).to_csv(out_dir / f"h3_tev_by_{group_col}.csv", index=False)
    if raear_parts:
        pd.concat(raear_parts, ignore_index=True).to_csv(out_dir / f"h3_raear_by_{group_col}.csv", index=False)


def analyze_panel(
    panel,
    out_dir: Path,
    *,
    etas: list[float] | None = None,
) -> None:
    if etas is None:
        etas = [0.01, 0.05, 0.10, 0.25, 0.5]
    if panel.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_panel = primary_surface(panel)
    require_headline_panel(analysis_panel, "H3", require_moc=True)
    if H3_ALPHA_COL not in analysis_panel.columns:
        raise ValueError(
            f"H3: required benchmark-relative alpha column missing: {H3_ALPHA_COL}"
        )

    tev = _pin_moc_benchmark(
        tracking_error_variance(analysis_panel, alpha_col=H3_ALPHA_COL)
    )
    n_positions = max(1, analysis_panel["symbol"].nunique())
    tev_port = portfolio_tracking_error(tev, n_positions)
    tev_port.to_csv(out_dir / "h3_tev.csv", index=False)

    ra = raear_panel(tev, etas)
    ra.to_csv(out_dir / "h3_raear.csv", index=False)
    (out_dir / "h3_metric_manifest.json").write_text(
        json.dumps({
            "metric_policy_version": H3_METRIC_POLICY_VERSION,
            "alpha_col": H3_ALPHA_COL,
            "benchmark_strategy": "S0_MOC",
            "description": (
                "H3 TEV, TES, IR, and RAEAR are computed from net execution "
                "alpha differentials versus the Market-on-Close benchmark on "
                "the primary H1 Window-B parent-order cell."
            ),
            "etas": etas,
            "n_rows": int(len(analysis_panel)),
            "n_symbols": (
                int(analysis_panel["symbol"].nunique())
                if "symbol" in analysis_panel.columns else None
            ),
            "strategies": sorted(
                str(s) for s in analysis_panel["strategy"].dropna().unique()
            ) if "strategy" in analysis_panel.columns else [],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for group_col in ("adv_bucket", "spread_bucket", "adv_spread_bucket"):
        _write_grouped_raear(analysis_panel, out_dir, group_col, etas)
    log.info("\n%s", ra.to_string(index=False))


def run(symbols, start, end, artifacts_dir: Path, out_dir: Path,
        etas: list[float] | None = None, max_dates: int | None = None,
        workers: int = 1, fill_specification: str = "tape_replay_queue",
        universe: str | None = None) -> None:
    panel_path = out_dir / "h3_panel.parquet"
    panel = run_panel(
        strategies=["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE", "S3_FULL", "S4_TOD"],
        start=start, end=end,
        artifacts_dir=artifacts_dir, out_path=panel_path,
        symbols=symbols, universe=universe, max_dates=max_dates, workers=workers,
        fill_specification=fill_specification,
        size_fractions=(cfg.PARENT_ORDER_PRIMARY_FRACTION,),
    )
    analyze_panel(panel, out_dir, etas=etas)


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
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "h3")
    p.add_argument("--etas", type=float, nargs="*", default=None)
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
        validate_run(["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE", "S3_FULL", "S4_TOD"],
                     args.start, args.end, args.artifacts, args.symbols,
                     args.universe, args.fill_spec)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.symbols, args.start, args.end, args.artifacts, args.out,
        args.etas, args.max_dates, args.workers, args.fill_spec, args.universe)


if __name__ == "__main__":
    main()

