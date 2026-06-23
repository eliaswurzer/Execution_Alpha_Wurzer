"""Economic significance tests across fill specifications."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.inference.bootstrap import wild_cluster_bootstrap_mean
from analysis.inference.clustering import mean_with_twoway_se
from analysis.inference.power import minimum_detectable_effect

from . import config as st_cfg
from .multiple_testing import (
    attach_fdr, attach_holm, one_sided_p_from_t, two_sided_p_from_t,
)


H1_COLUMNS = [
    "strategy",
    "window",
    "size_frac",
    "order_id",
    "symbol",
    "date",
    "net_alpha_vs_moc_bps",
    "net_alpha_bps",
    "fill_rate",
    "adverse_selection_bps",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_h1_panel(run_root: Path) -> pd.DataFrame:
    path = Path(run_root) / "hypotheses" / "h1" / "h1_panel.parquet"
    try:
        return pd.read_parquet(path, columns=H1_COLUMNS)
    except (KeyError, ValueError):
        return pd.read_parquet(path)


def primary_s3_window(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "strategy" in out.columns:
        out = out[out["strategy"] == "S3_FULL"]
    if "window" in out.columns:
        out = out[out["window"] == st_cfg.PRIMARY_WINDOW]
    if "size_frac" in out.columns:
        out = out[np.isclose(out["size_frac"].astype(float), st_cfg.PRIMARY_SIZE_FRAC)]
    return out.copy()


def _assert_unique_pairing_keys(frame: pd.DataFrame, key: list[str], label: str) -> None:
    dupes = frame.duplicated(key, keep=False)
    if dupes.any():
        example = frame.loc[dupes, key].head(5).to_dict("records")
        raise ValueError(
            f"{label}: primary S3 comparison key is not unique on {key}; "
            f"examples={example}"
        )


def _as_markout(frame: pd.DataFrame) -> float:
    if "adverse_selection_bps" not in frame.columns or "fill_rate" not in frame.columns:
        return float("nan")
    filled = frame[pd.to_numeric(frame["fill_rate"], errors="coerce") > 0]
    if filled.empty:
        return 0.0
    return float(-pd.to_numeric(filled["adverse_selection_bps"], errors="coerce").mean())


def _mean_test(values: pd.Series, symbols: pd.Series, dates: pd.Series) -> tuple[float, float, float, float, int]:
    mean, se = mean_with_twoway_se(values, symbols, dates)
    t_val = mean / se if se and se > 0 else float("nan")
    return mean, se, float(t_val), two_sided_p_from_t(float(t_val)), int(values.dropna().shape[0])


def fill_spec_summary(
    runs: dict[str, Path] | None = None,
    *,
    headline_spec: str = "tape_replay_queue",
    run_bootstrap: bool = True,
    n_boot: int = st_cfg.BOOTSTRAP_B,
    alternative: str = st_cfg.PRIMARY_ALTERNATIVE,
) -> pd.DataFrame:
    """S3-full vs MOC summary by fill specification.

    Reports the asymptotic two-way clustered test plus a wild cluster bootstrap
    p-value, the registered one-sided p-value, a percentile-t confidence
    interval, and the design-based minimum detectable effect, so the headline
    differential can be read robustly against the normal-reference and
    distributional assumptions.
    """
    runs = runs or st_cfg.FILL_SPEC_RUNS
    rows: list[dict] = []
    for spec in st_cfg.FILL_SPEC_ORDER:
        root = runs.get(spec)
        if root is None:
            continue
        panel_path = Path(root) / "hypotheses" / "h1" / "h1_panel.parquet"
        if not panel_path.exists():
            continue
        sub = primary_s3_window(read_h1_panel(root))
        if sub.empty:
            continue
        alpha_col = "net_alpha_vs_moc_bps" if "net_alpha_vs_moc_bps" in sub.columns else "net_alpha_bps"
        mean, se, t_val, p_val, n = _mean_test(sub[alpha_col], sub["symbol"], sub["date"])
        greater = alternative != "less"
        mde = minimum_detectable_effect(
            se, alpha=st_cfg.MDE_ALPHA, power=st_cfg.MDE_POWER,
            one_sided=alternative != "two-sided",
        )
        row = {
            "family": "fill_spec_robustness",
            "spec": spec,
            "label": st_cfg.FILL_SPEC_LABELS.get(spec, spec),
            "is_headline": spec == headline_spec,
            "mean_net_alpha_vs_moc_bps": mean,
            "se_twoway": se,
            "t": t_val,
            "p_value": p_val,
            "p_one_sided": one_sided_p_from_t(t_val, greater=greater),
            "alternative": alternative,
            "p_bootstrap_two_sided": float("nan"),
            "p_bootstrap_one_sided": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "mde_bps": mde,
            "mean_fill_rate": float(pd.to_numeric(sub["fill_rate"], errors="coerce").mean()),
            "as_markout_bps": _as_markout(sub),
            "residual_moc": float(1.0 - pd.to_numeric(sub["fill_rate"], errors="coerce").mean()),
            "n": n,
            "run_root": str(root),
            "panel_sha256": sha256_file(panel_path),
        }
        if run_bootstrap:
            boot = wild_cluster_bootstrap_mean(
                sub[alpha_col], sub["symbol"], sub["date"],
                alternative=alternative, n_boot=n_boot,
                weights=st_cfg.BOOTSTRAP_WEIGHTS, two_way=st_cfg.BOOTSTRAP_TWO_WAY,
                ci_alpha=st_cfg.CI_ALPHA, seed=st_cfg.BOOTSTRAP_SEED,
            )
            row["p_bootstrap_two_sided"] = boot.p_bootstrap_two_sided
            row["p_bootstrap_one_sided"] = boot.p_bootstrap_one_sided
            row["ci_lo"] = boot.ci_lo
            row["ci_hi"] = boot.ci_hi
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return attach_holm(out, mask=~out["is_headline"].astype(bool))


def _metric_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "as_markout_bps":
        if "adverse_selection_bps" not in frame.columns:
            return pd.Series(index=frame.index, dtype=float)
        out = -pd.to_numeric(frame["adverse_selection_bps"], errors="coerce")
        if "fill_rate" in frame.columns:
            out = out.where(pd.to_numeric(frame["fill_rate"], errors="coerce") > 0)
        return out
    return pd.to_numeric(frame[metric], errors="coerce")


def paired_vs_queue_tests(
    runs: dict[str, Path] | None = None,
    *,
    headline_spec: str = "tape_replay_queue",
    metrics: tuple[str, ...] = ("net_alpha_vs_moc_bps", "fill_rate", "as_markout_bps"),
) -> pd.DataFrame:
    """Paired alternative-minus-queue tests on shared parent orders.

    Orders are matched on the stable ``(symbol, date)`` key within the primary
    S3 window rather than on the raw ``order_id`` string, because the order_id
    size token differs across simulation batches (for example ``01`` versus
    ``0100bp``) even when the parent orders are identical. Matching on the
    economic key keeps the paired comparison valid across batches. A merge that
    yields no rows is recorded as an explicit zero-overlap skip rather than
    dropped silently.
    """
    runs = runs or st_cfg.FILL_SPEC_RUNS
    base_panel_path = Path(runs[headline_spec]) / "hypotheses" / "h1" / "h1_panel.parquet"
    if not base_panel_path.exists():
        return pd.DataFrame()
    base = primary_s3_window(read_h1_panel(runs[headline_spec]))
    key = ["symbol", "date"]
    _assert_unique_pairing_keys(base, key, headline_spec)
    rows: list[dict] = []
    skips: list[dict] = []
    for spec, root in runs.items():
        if spec == headline_spec or spec not in st_cfg.FILL_SPEC_ORDER:
            continue
        # Skip fill-spec run roots that were not produced (e.g. model-based runs
        # not present on this machine); a missing root is a structured skip, not
        # a crash.
        if not (Path(root) / "hypotheses" / "h1" / "h1_panel.parquet").exists():
            skips.append({"spec": spec, "reason": "missing_run_root", "n": 0})
            continue
        alt = primary_s3_window(read_h1_panel(root))
        _assert_unique_pairing_keys(alt, key, spec)
        for metric in metrics:
            b = base[key].copy()
            a = alt[key].copy()
            b["metric_base"] = _metric_series(base, metric).to_numpy()
            a["metric_alt"] = _metric_series(alt, metric).to_numpy()
            paired = a.merge(b, on=key, how="inner")
            if paired.empty:
                skips.append({"spec": spec, "metric": metric,
                              "reason": "zero_symbol_date_overlap", "n": 0})
                continue
            diff = paired["metric_alt"] - paired["metric_base"]
            valid = diff.notna()
            if not valid.any():
                skips.append({"spec": spec, "metric": metric,
                              "reason": "all_diffs_nan", "n": 0})
                continue
            mean, se, t_val, p_val, n = _mean_test(
                diff[valid], paired.loc[valid, "symbol"], paired.loc[valid, "date"]
            )
            rows.append({
                "family": f"alt_vs_queue_{metric}",
                "spec": spec,
                "label": st_cfg.FILL_SPEC_LABELS.get(spec, spec),
                "metric": metric,
                "mean_diff": mean,
                "se_twoway": se,
                "t": t_val,
                "p_value": p_val,
                "mde_bps": minimum_detectable_effect(
                    se, alpha=st_cfg.MDE_ALPHA, power=st_cfg.MDE_POWER,
                    one_sided=False,
                ),
                "n": n,
                "run_root": str(root),
            })
    if skips:
        import logging
        logging.getLogger(__name__).warning(
            "paired_vs_queue_tests skipped %d spec/metric cells: %s",
            len(skips), skips,
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    parts = []
    for _, grp in out.groupby("metric", sort=False):
        parts.append(attach_holm(grp))
    return pd.concat(parts, ignore_index=True)


def adjusted_h2_pooled(headline_run: Path | None = None) -> pd.DataFrame:
    """Attach raw, one-sided, Holm, and FDR p-values to the pooled H2 family."""
    root = Path(headline_run or st_cfg.HEADLINE_RUN)
    path = root / "hypotheses" / "h2" / "h2_pooled.csv"
    h2 = pd.read_csv(path)
    h2["p_value"] = [two_sided_p_from_t(float(t)) for t in h2["t"]]
    # H2a/H2b are registered one-sided (incremental alpha predicted positive).
    h2["p_one_sided"] = [one_sided_p_from_t(float(t), greater=True) for t in h2["t"]]
    h2["alternative"] = st_cfg.PRIMARY_ALTERNATIVE
    if "se_twoway" in h2.columns:
        h2["mde_bps"] = [
            minimum_detectable_effect(
                float(se), alpha=st_cfg.MDE_ALPHA, power=st_cfg.MDE_POWER, one_sided=True,
            )
            for se in h2["se_twoway"]
        ]
    # Two-sided Holm is retained for continuity; the registered one-sided family
    # is corrected separately and drives the confirmatory decision.
    h2 = attach_holm(h2)
    h2 = attach_holm(h2, p_col="p_one_sided", out_col="p_holm_one_sided")
    return attach_fdr(h2)
