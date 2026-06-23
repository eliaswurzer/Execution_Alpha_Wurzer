"""Bootstrap inference for the H3 risk-adjusted ranking.

The standard H3 analysis reports tracking-error variance, the tracking-error
standard deviation, the information ratio, the RAEAR grid, and the strategy
ranking as point estimates only. This module attaches a clustered block
bootstrap (resampling whole trading dates) so the risk-ranking claims can be
stated with uncertainty: percentile confidence intervals for TEV, TES, and the
information ratio, the bootstrap probability that the information-ratio ranking
is preserved, pairwise ordering probabilities, and the probability that the
RAEAR ranking flips across the risk-aversion grid (a direct inferential version
of the H3 rank-stability null). The point-estimate methodology is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics.raear import information_ratio
from analysis.metrics.tracking_error import tracking_error_variance
from analysis.runners.h3_te_tradeoff import (
    H3_ALPHA_COL,
    _pin_moc_benchmark,
    primary_surface,
)

from . import config as st_cfg

BENCHMARK = "S0_MOC"
DEFAULT_ETAS = [0.01, 0.05, 0.10, 0.25, 0.5]


def _strategy_moments(panel: pd.DataFrame) -> pd.DataFrame:
    """MOC-pinned per-strategy mean alpha and TEV (the H3 point estimates)."""
    tev = _pin_moc_benchmark(tracking_error_variance(panel, alpha_col=H3_ALPHA_COL))
    return tev[["strategy", "mean_alpha", "tev"]].copy()


def _moments_stat(panel: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    moments = _strategy_moments(panel)
    for _, r in moments.iterrows():
        s = str(r["strategy"])
        out[f"mean_alpha::{s}"] = float(r["mean_alpha"])
        out[f"tev::{s}"] = float(r["tev"])
    return out


def _date_strategy_stat_mats(panel: pd.DataFrame):
    """Sufficient statistics for exact date-block H3 resampling.

    The original block bootstrap resampled all rows for a drawn date and then
    recomputed per-strategy means and sample variances. For H3, those moments
    are fully determined by per-date, per-strategy counts, sums, and squared
    sums, so the same resampling design can be evaluated without materializing
    a multi-million-row bootstrap panel in every replication.
    """
    cols = ["date", "strategy", H3_ALPHA_COL]
    frame = panel[cols].copy()
    frame[H3_ALPHA_COL] = pd.to_numeric(frame[H3_ALPHA_COL], errors="coerce")
    frame = frame.dropna(subset=["date", "strategy", H3_ALPHA_COL])
    if frame.empty:
        return [], [], np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
    frame["_sq"] = frame[H3_ALPHA_COL] * frame[H3_ALPHA_COL]
    grouped = (
        frame.groupby(["date", "strategy"], sort=True, dropna=False)[H3_ALPHA_COL]
        .agg(n="count", total="sum")
        .reset_index()
    )
    sq = (
        frame.groupby(["date", "strategy"], sort=True, dropna=False)["_sq"]
        .sum()
        .reset_index(name="sumsq")
    )
    grouped = grouped.merge(sq, on=["date", "strategy"], how="left")
    date_labels = list(pd.Index(grouped["date"]).drop_duplicates())
    strategy_labels = list(pd.Index(grouped["strategy"].astype(str)).drop_duplicates())
    date_codes = pd.Categorical(grouped["date"], categories=date_labels).codes
    strategy_codes = pd.Categorical(
        grouped["strategy"].astype(str), categories=strategy_labels,
    ).codes
    shape = (len(date_labels), len(strategy_labels))
    n_mat = np.zeros(shape, dtype=float)
    sum_mat = np.zeros(shape, dtype=float)
    sumsq_mat = np.zeros(shape, dtype=float)
    np.add.at(n_mat, (date_codes, strategy_codes), grouped["n"].to_numpy(dtype=float))
    np.add.at(sum_mat, (date_codes, strategy_codes), grouped["total"].to_numpy(dtype=float))
    np.add.at(sumsq_mat, (date_codes, strategy_codes), grouped["sumsq"].to_numpy(dtype=float))
    return date_labels, strategy_labels, n_mat, sum_mat, sumsq_mat


def _moments_from_totals(
    strategy_labels: list[str],
    total_n: np.ndarray,
    total_sum: np.ndarray,
    total_sumsq: np.ndarray,
) -> dict:
    out: dict[str, float] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(total_n > 0, total_sum / total_n, np.nan)
        numer = total_sumsq - (total_sum * total_sum / total_n)
        tev = np.where(total_n > 1, numer / (total_n - 1.0), np.nan)
    tev = np.where(np.isfinite(tev) & (tev >= 0), tev, np.nan)
    for i, strategy in enumerate(strategy_labels):
        if strategy == BENCHMARK:
            out[f"mean_alpha::{strategy}"] = 0.0
            out[f"tev::{strategy}"] = 0.0
        else:
            out[f"mean_alpha::{strategy}"] = float(means[i])
            out[f"tev::{strategy}"] = float(tev[i])
    return out


def _bootstrap_moments_from_date_stats(
    panel: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    date_labels, strategy_labels, n_mat, sum_mat, sumsq_mat = _date_strategy_stat_mats(panel)
    n_dates = len(date_labels)
    if n_dates < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_dates, size=n_dates)
        counts = np.bincount(drawn, minlength=n_dates).astype(float)
        rows.append(_moments_from_totals(
            strategy_labels,
            counts @ n_mat,
            counts @ sum_mat,
            counts @ sumsq_mat,
        ))
    return pd.DataFrame(rows)


def _ir(mean_alpha: float, tev: float) -> float:
    return information_ratio(mean_alpha, tev)


def _ranking(ir_by_strategy: dict[str, float]) -> tuple[str, ...]:
    """Strategies ordered by descending information ratio (ties broken by name)."""
    items = [(s, v) for s, v in ir_by_strategy.items() if np.isfinite(v)]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return tuple(s for s, _ in items)


def h3_bootstrap(
    panel: pd.DataFrame,
    *,
    etas: list[float] | None = None,
    n_boot: int = st_cfg.BOOTSTRAP_B_H3,
    ci_alpha: float = st_cfg.CI_ALPHA,
    seed: int = st_cfg.BOOTSTRAP_SEED,
) -> dict[str, pd.DataFrame]:
    """Run the H3 block bootstrap and return CI, rank-stability, and pairwise tables."""
    if panel.empty or "date" not in panel.columns:
        return {"strategy_ci": pd.DataFrame(), "rank_stability": pd.DataFrame(),
                "pairwise": pd.DataFrame()}
    panel = primary_surface(panel)
    if panel.empty:
        return {"strategy_ci": pd.DataFrame(), "rank_stability": pd.DataFrame(),
                "pairwise": pd.DataFrame()}
    etas = etas if etas is not None else DEFAULT_ETAS

    obs_moments = _strategy_moments(panel)
    strategies = [s for s in obs_moments["strategy"].astype(str) if s != BENCHMARK]

    obs_ir = {}
    for _, r in obs_moments.iterrows():
        s = str(r["strategy"])
        if s == BENCHMARK:
            continue
        obs_ir[s] = _ir(float(r["mean_alpha"]), float(r["tev"]))
    observed_order = _ranking(obs_ir)

    reps = _bootstrap_moments_from_date_stats(panel, n_boot=n_boot, seed=seed)
    if reps.empty:
        return {"strategy_ci": pd.DataFrame(), "rank_stability": pd.DataFrame(),
                "pairwise": pd.DataFrame()}

    # Derive per-replication IR, TES, and RAEAR-grid rankings.
    ir_reps: dict[str, np.ndarray] = {}
    tev_reps: dict[str, np.ndarray] = {}
    tes_reps: dict[str, np.ndarray] = {}
    mean_reps: dict[str, np.ndarray] = {}
    for s in strategies:
        ma = reps.get(f"mean_alpha::{s}")
        tv = reps.get(f"tev::{s}")
        if ma is None or tv is None:
            continue
        ma = ma.to_numpy(dtype=float)
        tv = tv.to_numpy(dtype=float)
        mean_reps[s] = ma
        tev_reps[s] = tv
        tes_reps[s] = np.sqrt(np.clip(tv, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            ir_reps[s] = np.where(tv > 1e-12, ma / np.sqrt(np.clip(tv, 0, None)), np.nan)

    avail = list(ir_reps.keys())
    n_rep = len(reps)

    # Confidence intervals per strategy.
    ci_rows: list[dict] = []
    lo, hi = ci_alpha / 2.0, 1.0 - ci_alpha / 2.0
    for s in avail:
        ci_rows.append({
            "strategy": s,
            "ir": obs_ir.get(s, float("nan")),
            "ir_lo": float(np.nanquantile(ir_reps[s], lo)) if np.isfinite(ir_reps[s]).any() else float("nan"),
            "ir_hi": float(np.nanquantile(ir_reps[s], hi)) if np.isfinite(ir_reps[s]).any() else float("nan"),
            "tev": float(obs_moments.loc[obs_moments["strategy"] == s, "tev"].iloc[0]),
            "tev_lo": float(np.nanquantile(tev_reps[s], lo)),
            "tev_hi": float(np.nanquantile(tev_reps[s], hi)),
            "tes": float(np.sqrt(max(obs_moments.loc[obs_moments["strategy"] == s, "tev"].iloc[0], 0.0))),
            "tes_lo": float(np.nanquantile(tes_reps[s], lo)),
            "tes_hi": float(np.nanquantile(tes_reps[s], hi)),
            "n_boot": n_rep,
        })
    strategy_ci = pd.DataFrame(ci_rows)

    # IR ranking preservation and RAEAR rank-flip probability.
    rank_match = 0
    flip = 0
    valid_rank = 0
    for i in range(n_rep):
        ir_i = {s: ir_reps[s][i] for s in avail}
        order_i = _ranking(ir_i)
        if order_i:
            valid_rank += 1
            if order_i == observed_order:
                rank_match += 1
        # RAEAR rank flip across the eta grid (raear = mean_alpha - eta * tev).
        order_at_eta = []
        ok = True
        for eta in (min(etas), max(etas)):
            ra = {s: mean_reps[s][i] - eta * tev_reps[s][i] for s in avail}
            ra = {s: v for s, v in ra.items() if np.isfinite(v)}
            if not ra:
                ok = False
                break
            order_at_eta.append(tuple(sorted(ra, key=lambda k: (-ra[k], k))))
        if ok and len(order_at_eta) == 2 and order_at_eta[0] != order_at_eta[1]:
            flip += 1
    rank_stability = pd.DataFrame([{
        "observed_ir_ranking": " > ".join(observed_order),
        "p_ir_ranking_preserved": rank_match / valid_rank if valid_rank else float("nan"),
        "p_raear_rank_flip_across_eta": flip / n_rep if n_rep else float("nan"),
        "eta_min": min(etas),
        "eta_max": max(etas),
        "n_boot": n_rep,
    }])

    # Pairwise IR ordering probabilities.
    pair_rows: list[dict] = []
    for a_i in range(len(avail)):
        for b_i in range(a_i + 1, len(avail)):
            s_a, s_b = avail[a_i], avail[b_i]
            diff = ir_reps[s_a] - ir_reps[s_b]
            valid = np.isfinite(diff)
            if not valid.any():
                continue
            pair_rows.append({
                "strategy_a": s_a,
                "strategy_b": s_b,
                "ir_a": obs_ir.get(s_a, float("nan")),
                "ir_b": obs_ir.get(s_b, float("nan")),
                "p_a_beats_b": float(np.mean(diff[valid] > 0)),
                "n_boot": int(valid.sum()),
            })
    pairwise = pd.DataFrame(pair_rows)

    return {
        "strategy_ci": strategy_ci,
        "rank_stability": rank_stability,
        "pairwise": pairwise,
    }


def read_h3_panel(run_root: Path) -> pd.DataFrame:
    path = Path(run_root) / "hypotheses" / "h3" / "h3_panel.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
