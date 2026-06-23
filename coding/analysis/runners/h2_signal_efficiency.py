"""
h2_signal_efficiency.py -- Hypothesis 2 (thesis Section 3.2, H2 extension).
The runner asks which signals contribute to S3 alpha.

Four panels:
* S2, the baseline without signal conditioning.
* S3_OFI, using only OFI.
* S3_IMB, using only the auction-imbalance proxy.
* S3_FULL, using both signals in the thesis specification.

Three differentials per cell:
* OFI marginal: S3_OFI minus S2.
* IMB marginal: S3_IMB minus S2.
* Interaction: S3_FULL minus S3_OFI minus S3_IMB plus S2.

Matched-fill binning uses the S2 reference metric available in the panel. The
tape-replay headline uses realized passive S2 fill rate; a calibrated fill
probability is used only when a model-based run persists one.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from math import erf, sqrt

from .. import config as cfg
from ..inference.bootstrap import max_t_union_test
from ..inference.clustering import mean_with_twoway_se
from ..inference.tests import holm_step_down
from ._common import require_headline_panel, run_panel, validate_run


log = logging.getLogger(__name__)


def _two_sided_p(t: float) -> float:
    if t is None or not np.isfinite(t):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(float(t)) / sqrt(2.0)))))


# ---------------------------------------------------------------------------
# Differentials by matched-fill bin and in the pooled surface
# ---------------------------------------------------------------------------

def _attach_s2_match_metric(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the available S2 matched-fill metric to each order."""
    if "fill_probability" in panel.columns:
        source_col = "fill_probability"
        source_label = "calibrated_fill_probability"
    else:
        source_col = "fill_rate"
        source_label = "realized_passive_fill_rate"
    base = (
        panel[panel["strategy"] == "S2_TIME_ADAPTIVE"][["order_id", source_col]]
        .rename(columns={source_col: "s2_match_metric"})
    )
    out = panel.merge(base, on="order_id", how="left")
    out["s2_match_metric_source"] = source_label
    return out


def _per_bin_means(
    panel: pd.DataFrame, n_bins: int = 10, alpha_col: str = "net_alpha_bps",
) -> pd.DataFrame:
    if panel.empty or "s2_match_metric" not in panel.columns:
        return pd.DataFrame()
    p = panel.dropna(subset=["s2_match_metric"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["bin"] = pd.qcut(
        p["s2_match_metric"], q=n_bins, labels=False, duplicates="drop",
    )
    n_bins_effective = int(p["bin"].nunique(dropna=True))
    rows = []
    for (strat, b), grp in p.groupby(["strategy", "bin"], dropna=False):
        m, se = mean_with_twoway_se(grp[alpha_col], grp["symbol"], grp["date"])
        rows.append({
            "strategy": strat, "bin": int(b) if pd.notna(b) else -1,
            "mean": m, "se_twoway": se, "n": len(grp),
            "matching_metric": grp["s2_match_metric_source"].iloc[0],
            "n_bins_requested": int(n_bins),
            "n_bins_effective": n_bins_effective,
        })
    return pd.DataFrame(rows)


def _per_bin_differentials(
    panel: pd.DataFrame, n_bins: int = 10, alpha_col: str = "net_alpha_bps",
) -> pd.DataFrame:
    """Matched-fill H2 differentials computed within frozen S2-fill bins."""
    if panel.empty or "s2_match_metric" not in panel.columns:
        return pd.DataFrame()
    p = panel.dropna(subset=["s2_match_metric"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["bin"] = pd.qcut(
        p["s2_match_metric"], q=n_bins, labels=False, duplicates="drop",
    )
    n_bins_effective = int(p["bin"].nunique(dropna=True))
    rows = []
    for b, grp in p.groupby("bin", dropna=False):
        for label, a, base in (
            ("OFI_marginal", "S3_OFI", "S2_TIME_ADAPTIVE"),
            ("IMB_marginal", "S3_IMB", "S2_TIME_ADAPTIVE"),
            ("FULL_vs_S2", "S3_FULL", "S2_TIME_ADAPTIVE"),
        ):
            m, se, n = _diff_with_se(grp, a, base, alpha_col=alpha_col)
            t = m / se if se and se > 0 else float("nan")
            rows.append({
                "label": label,
                "bin": int(b) if pd.notna(b) else -1,
                "mean": m,
                "se_twoway": se,
                "t": t,
                "p_value": _two_sided_p(t),
                "n": n,
                "matching_metric": grp["s2_match_metric_source"].iloc[0],
                "n_bins_requested": int(n_bins),
                "n_bins_effective": n_bins_effective,
            })
        inter_m, inter_se, inter_n = _interaction_with_se(grp, alpha_col=alpha_col)
        inter_t = inter_m / inter_se if inter_se and inter_se > 0 else float("nan")
        rows.append({
            "label": "interaction",
            "bin": int(b) if pd.notna(b) else -1,
            "mean": inter_m,
            "se_twoway": inter_se,
            "t": inter_t,
            "p_value": _two_sided_p(inter_t),
            "n": inter_n,
            "matching_metric": grp["s2_match_metric_source"].iloc[0],
            "n_bins_requested": int(n_bins),
            "n_bins_effective": n_bins_effective,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm-correct across bins within each label family: the per-bin t-stats are
    # an exploratory family, so the raw bin-level p-values must not be read as
    # if each bin were a separate confirmatory test.
    out["p_holm_within_label"] = np.nan
    for label, grp in out.groupby("label", sort=False):
        out.loc[grp.index, "p_holm_within_label"] = holm_step_down(
            grp["p_value"].to_numpy()
        )
    return out


def per_bin_union_test(
    panel: pd.DataFrame,
    *,
    n_bins: int = 10,
    alpha_col: str = "net_alpha_bps",
    n_boot: int = 2000,
    seed: int = cfg.DEFAULT_SEED,
) -> pd.DataFrame:
    """Bootstrap max-t union test of the H2 "at least one bin" alternative.

    H2a and H2b are stated as ``m(q) > 0`` for at least one fill-rate bin ``q``.
    Reading any single significant bin off the per-bin table would inflate the
    type-I error across the ten bins; this test reports one multiplicity-aware
    p-value per signal family by bootstrapping the joint null distribution of
    the maximum studentized bin differential.
    """
    panel = _attach_s2_match_metric(panel)
    if panel.empty or "s2_match_metric" not in panel.columns:
        return pd.DataFrame()
    p = panel.dropna(subset=["s2_match_metric"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["bin"] = pd.qcut(p["s2_match_metric"], q=n_bins, labels=False, duplicates="drop")
    n_bins_effective = int(p["bin"].nunique(dropna=True))

    rows: list[dict] = []
    specs = (
        ("OFI_marginal", "S3_OFI", "S2_TIME_ADAPTIVE"),
        ("IMB_marginal", "S3_IMB", "S2_TIME_ADAPTIVE"),
        ("FULL_vs_S2", "S3_FULL", "S2_TIME_ADAPTIVE"),
    )
    for label, a, base in specs:
        frame = _label_diff_frame(p, a, base, alpha_col)
        if frame.empty:
            continue
        res = max_t_union_test(
            frame, value_col="diff", group_col="bin", alternative="greater",
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            "label": label,
            "n_bins_requested": int(n_bins),
            "n_bins_effective": n_bins_effective,
            **res,
        })
    inter = _label_interaction_frame(p, alpha_col)
    if not inter.empty:
        # The interaction term has no registered direction; keep it two-sided.
        res = max_t_union_test(
            inter, value_col="diff", group_col="bin", alternative="two-sided",
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            "label": "interaction",
            "n_bins_requested": int(n_bins),
            "n_bins_effective": n_bins_effective,
            **res,
        })
    return pd.DataFrame(rows)


def _label_diff_frame(
    panel: pd.DataFrame, a: str, b: str, alpha_col: str,
) -> pd.DataFrame:
    a_df = panel[panel["strategy"] == a][["order_id", "symbol", "date", "bin", alpha_col]]
    b_df = panel[panel["strategy"] == b][["order_id", alpha_col]].rename(
        columns={alpha_col: f"{alpha_col}_b"}
    )
    j = a_df.merge(b_df, on="order_id", how="inner")
    if j.empty:
        return pd.DataFrame()
    j["diff"] = j[alpha_col] - j[f"{alpha_col}_b"]
    return j[["symbol", "date", "bin", "diff"]]


def _label_interaction_frame(panel: pd.DataFrame, alpha_col: str) -> pd.DataFrame:
    pieces = {}
    for s in ("S3_FULL", "S3_OFI", "S3_IMB", "S2_TIME_ADAPTIVE"):
        pieces[s] = panel[panel["strategy"] == s][
            ["order_id", "symbol", "date", "bin", alpha_col]
        ].rename(columns={alpha_col: s})
    j = pieces["S3_FULL"]
    for s in ("S3_OFI", "S3_IMB", "S2_TIME_ADAPTIVE"):
        j = j.merge(pieces[s].drop(columns=["symbol", "date", "bin"]), on="order_id", how="inner")
    if j.empty:
        return pd.DataFrame()
    j["diff"] = j["S3_FULL"] - j["S3_OFI"] - j["S3_IMB"] + j["S2_TIME_ADAPTIVE"]
    return j[["symbol", "date", "bin", "diff"]]


def _diff_with_se(
    panel: pd.DataFrame, a: str, b: str, alpha_col: str = "net_alpha_bps",
) -> tuple[float, float, int]:
    """Mean and two-way clustered SE of ``a - b`` on shared order IDs."""
    a_df = panel[panel["strategy"] == a][["order_id", "symbol", "date", alpha_col]]
    b_df = panel[panel["strategy"] == b][["order_id", alpha_col]].rename(
        columns={alpha_col: f"{alpha_col}_b"}
    )
    j = a_df.merge(b_df, on="order_id", how="inner")
    if j.empty:
        return float("nan"), float("nan"), 0
    diff = j[alpha_col] - j[f"{alpha_col}_b"]
    m, se = mean_with_twoway_se(diff, j["symbol"], j["date"])
    return m, se, len(j)


def _interaction_with_se(
    panel: pd.DataFrame, alpha_col: str = "net_alpha_bps",
) -> tuple[float, float, int]:
    """Interaction equals S3_FULL minus S3_OFI minus S3_IMB plus S2."""
    pieces = {}
    for s in ("S3_FULL", "S3_OFI", "S3_IMB", "S2_TIME_ADAPTIVE"):
        pieces[s] = panel[panel["strategy"] == s][
            ["order_id", "symbol", "date", alpha_col]
        ].rename(columns={alpha_col: s})
    j = pieces["S3_FULL"]
    for s in ("S3_OFI", "S3_IMB", "S2_TIME_ADAPTIVE"):
        j = j.merge(pieces[s].drop(columns=["symbol", "date"]), on="order_id", how="inner")
    if j.empty:
        return float("nan"), float("nan"), 0
    inter = (
        j["S3_FULL"] - j["S3_OFI"] - j["S3_IMB"] + j["S2_TIME_ADAPTIVE"]
    )
    m, se = mean_with_twoway_se(inter, j["symbol"], j["date"])
    return m, se, len(j)


def compute_decomposition(panel: pd.DataFrame, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """Return per-bin and pooled OFI, IMB, and interaction differentials."""
    panel = _attach_s2_match_metric(panel)
    per_bin = _per_bin_means(panel, n_bins=n_bins)
    per_bin_diffs = _per_bin_differentials(panel, n_bins=n_bins)

    rows = []
    for label, a, b in (
        ("OFI_marginal", "S3_OFI", "S2_TIME_ADAPTIVE"),
        ("IMB_marginal", "S3_IMB", "S2_TIME_ADAPTIVE"),
        ("FULL_vs_S2", "S3_FULL", "S2_TIME_ADAPTIVE"),
    ):
        m, se, n = _diff_with_se(panel, a, b)
        t = m / se if se and se > 0 else float("nan")
        rows.append({"label": label, "mean": m, "se_twoway": se, "t": t, "n": n})
    inter_m, inter_se, inter_n = _interaction_with_se(panel)
    inter_t = inter_m / inter_se if inter_se and inter_se > 0 else float("nan")
    rows.append({
        "label": "interaction", "mean": inter_m, "se_twoway": inter_se,
        "t": inter_t, "n": inter_n,
    })
    pooled = pd.DataFrame(rows)
    source = panel["s2_match_metric_source"].dropna()
    pooled["matching_metric"] = source.iloc[0] if not source.empty else "unavailable"
    return {
        "per_bin": per_bin,
        "per_bin_differentials": per_bin_diffs,
        "pooled_differentials": pooled,
    }


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------

def compute_grouped_decomposition(
    panel: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute pooled H2 differentials inside a reporting group."""
    if panel.empty or group_col not in panel.columns:
        return pd.DataFrame()
    rows = []
    for group_value, grp in panel.groupby(group_col, dropna=False):
        out = compute_decomposition(grp)
        pooled = out["pooled_differentials"].copy()
        if pooled.empty:
            continue
        pooled[group_col] = group_value if pd.notna(group_value) else "unassigned"
        rows.append(pooled)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def analyze_panel(
    panel: pd.DataFrame,
    out_dir: Path,
    *,
    n_bins: int = 10,
) -> None:
    if panel.empty:
        log.warning("Empty panel; skipping decomposition.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    require_headline_panel(panel, "H2")

    out = compute_decomposition(panel, n_bins=n_bins)
    out["per_bin"].to_csv(out_dir / "h2_per_bin.csv", index=False)
    out["per_bin_differentials"].to_csv(
        out_dir / "h2_per_bin_differentials.csv", index=False,
    )
    out["pooled_differentials"].to_csv(out_dir / "h2_pooled.csv", index=False)
    for group_col in ("adv_bucket", "spread_bucket", "adv_spread_bucket"):
        grouped = compute_grouped_decomposition(panel, group_col)
        if not grouped.empty:
            grouped.to_csv(out_dir / f"h2_pooled_by_{group_col}.csv", index=False)
    log.info("\n[H2 pooled differentials]\n%s",
             out["pooled_differentials"].to_string(index=False))


def run(symbols, start, end, artifacts_dir: Path, out_dir: Path,
        max_dates: int | None = None, n_bins: int = 10, workers: int = 1,
        fill_specification: str = "tape_replay_queue",
        universe: str | None = None) -> None:
    panel_path = out_dir / "h2_panel.parquet"
    panel = run_panel(
        strategies=["S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL"],
        start=start, end=end,
        artifacts_dir=artifacts_dir, out_path=panel_path,
        symbols=symbols, universe=universe, max_dates=max_dates, workers=workers,
        fill_specification=fill_specification,
        size_fractions=(cfg.PARENT_ORDER_PRIMARY_FRACTION,),
    )
    analyze_panel(panel, out_dir, n_bins=n_bins)


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
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "h2")
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--n-bins", type=int, default=10)
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
        validate_run(["S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL"],
                     args.start, args.end, args.artifacts, args.symbols,
                     args.universe, args.fill_spec)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.symbols, args.start, args.end, args.artifacts, args.out,
        args.max_dates, args.n_bins, args.workers, args.fill_spec, args.universe)


if __name__ == "__main__":
    main()

