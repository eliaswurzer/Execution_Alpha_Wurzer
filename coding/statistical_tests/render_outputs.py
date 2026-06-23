"""Render supplementary statistical-test CSVs into thesis-ready snippets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .multiple_testing import star_suffix


def fmt_num(value, nd: int = 2, na: str = "n.a.") -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(val):
        return na
    text = f"{val:.{nd}f}"
    return text.replace("-", "$-$")


def fmt_int(value, na: str = "n.a.") -> str:
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        return na
    return f"{val:,}"


def with_stars(value, p_value, nd: int = 2) -> str:
    return f"{fmt_num(value, nd=nd)}{star_suffix(float(p_value) if pd.notna(p_value) else np.nan)}"


def _first_finite(*values):
    """Return the first value that is present and finite, else NaN."""
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return f
    return np.nan


def _table(caption: str, label: str, legend: str, colspec: str, header: str, body: list[str], notes: str) -> str:
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\joflegend{{{legend}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\jofnotes{{{notes}}}",
        r"\end{table}",
        "",
    ])


def render_fill_model_calibration(calibration: pd.DataFrame) -> str:
    body: list[str] = []
    if calibration.empty:
        body.append(r"No OOS calibration rows available & n.a. & n.a. & n.a. & n.a. & n.a. & n.a. & n.a. \\")
    else:
        labels = {"cox": "Cox", "km": "Kaplan-Meier", "xgb": "XGBoost"}
        for _, row in calibration.sort_values(["model", "tier"]).iterrows():
            body.append(
                f"{labels.get(str(row['model']), row['model'])} & Tier {int(row['tier'])} & "
                f"{fmt_num(row['observed_fill_rate'], nd=3)} & "
                f"{fmt_num(row['mean_predicted_probability'], nd=3)} & "
                f"{fmt_num(row['absolute_calibration_error'], nd=3)} & "
                f"{fmt_num(row['brier'], nd=3)} & {fmt_num(row['auc'], nd=3)} & "
                f"{fmt_int(row['n'])} \\\\"
            )
    return _table(
        caption="Out-of-Sample Calibration of Model-Based Fill Specifications",
        label="tab:fill-model-oos-calibration",
        legend=(
            "This table scores the model-based fill specifications on a deterministic "
            "stratified sample of evaluation-period symbol-days. Observed is the "
            "realized event rate within the 30-second horizon, predicted is the "
            "mean model-implied fill probability, and absolute error is the absolute "
            "difference between the two."
        ),
        colspec="llrrrrrr",
        header="Model & Tier & Observed & Predicted & Abs. error & Brier & AUC & $N$",
        body=body,
        notes=(
            "The models are loaded from the pre-sample calibration artifacts and are "
            "not refitted on the evaluation-period OOS panel. Lower Brier and "
            "absolute calibration error indicate better probability calibration; "
            "higher AUC indicates better ordinal discrimination."
        ),
    )


def render_fill_robustness(economic: pd.DataFrame) -> str:
    body: list[str] = []
    ordered = economic.sort_values(["is_headline", "spec"], ascending=[False, True])
    spec_order = {
        "tape_replay_queue": 0,
        "tape_replay_strict": 1,
        "tape_replay": 2,
        "cox": 3,
        "km": 4,
        "xgb": 5,
    }
    ordered = ordered.assign(_order=ordered["spec"].map(spec_order).fillna(99)).sort_values("_order")
    for _, row in ordered.iterrows():
        is_head = bool(row.get("is_headline", False))
        # Headline stars use the registered one-sided (or bootstrap one-sided)
        # p-value; robustness rows use the Holm-adjusted family p-value.
        if is_head:
            p_for_stars = _first_finite(
                row.get("p_bootstrap_one_sided"), row.get("p_one_sided"), row.get("p_value"),
            )
        else:
            p_for_stars = _first_finite(row.get("p_holm"), row.get("p_value"))
        body.append(
            f"{row['label']} & {with_stars(row['mean_net_alpha_vs_moc_bps'], p_for_stars)} & "
            f"{fmt_num(row['t'])} & {fmt_num(row.get('p_one_sided'), nd=3)} & "
            f"{fmt_num(row.get('p_holm'), nd=3)} & {fmt_num(row.get('p_bootstrap_one_sided'), nd=3)} & "
            f"{fmt_num(row.get('mean_fill_rate'), nd=3)} & {fmt_num(row.get('as_markout_bps'))} & "
            f"{fmt_num(row.get('residual_moc'), nd=3)} & {fmt_num(row.get('mde_bps'))} & "
            f"{fmt_int(row['n'])} \\\\"
        )
    return _table(
        caption="Headline Results across Fill Specifications with Bootstrap and Multiple-Testing Adjustment",
        label="tab:fill-robustness-adjusted",
        legend=(
            "This table reports the Window-B S3-full net-alpha differential against "
            "MOC across deterministic and model-based fill specifications. The "
            "one-sided p-value is the registered directional test based on two-way "
            "clustered standard errors by symbol and date; the bootstrap p-value is "
            "a wild cluster bootstrap that does not rely on the normal reference. "
            "Holm p-values control the exploratory fill-specification family and are "
            "applied to non-headline specifications only. MDE is the design-based "
            "minimum detectable effect in basis points at five-percent size and "
            "eighty-percent power."
        ),
        colspec="lrrrrrrrrrr",
        header=(
            "Fill specification & Net alpha & $t$ & $p$ (1s) & $p$ (Holm) & "
            "$p$ (boot) & Fill rate & AS markout & Residual MOC & MDE & $N$"
        ),
        body=body,
        notes=(
            "Stars on the headline queue-aware row are based on the wild cluster "
            "bootstrap one-sided p-value; robustness rows use Holm-adjusted "
            "p-values. Model-based rows are interpreted as robustness bounds rather "
            "than as replacements for queue-aware tape replay."
        ),
    )


def render_h2_adjusted(h2: pd.DataFrame) -> str:
    labels = {
        "OFI_marginal": "S3 OFI $-$ S2",
        "IMB_marginal": "S3 IMB $-$ S2",
        "FULL_vs_S2": "S3 Full $-$ S2",
        "interaction": "Interaction",
    }
    body: list[str] = []
    for _, row in h2.iterrows():
        # The registered H2a/H2b family is one-sided; stars and the Holm column
        # use the one-sided Holm-adjusted p-value, with a two-sided fallback.
        p_holm_one = _first_finite(row.get("p_holm_one_sided"), row.get("p_holm"))
        body.append(
            f"{labels.get(str(row['label']), row['label'])} & "
            f"{with_stars(row['mean'], p_holm_one)} & {fmt_num(row['t'])} & "
            f"{fmt_num(row.get('p_one_sided'), nd=3)} & {fmt_num(p_holm_one, nd=3)} & "
            f"{fmt_num(row.get('p_fdr_bh'), nd=3)} & {fmt_int(row['n'])} \\\\"
        )
    matching = h2["matching_metric"].iloc[0] if not h2.empty and "matching_metric" in h2.columns else "n.a."
    return _table(
        caption="Pooled H2 Signal Contributions with Holm and FDR Adjustment",
        label="tab:h2-pooled-adjusted",
        legend=(
            "This table reports pooled matched-fill-rate differentials of the S3 "
            "signal variants against S2. The one-sided p-value is the registered "
            "directional test based on two-way clustered standard errors; the Holm "
            "column controls the family-wise error rate of the four-test signal "
            "family on the same one-sided p-values, and the Benjamini-Hochberg "
            "column reports the corresponding false-discovery-rate adjustment."
        ),
        colspec="lrrrrrr",
        header=r"Signal differential & Mean (bps) & $t$ & $p$ (1s) & $p$ (Holm, 1s) & $p$ (FDR) & $N$",
        body=body,
        notes=f"Matching metric: {matching}. Stars are based on one-sided Holm-adjusted p-values.",
    )


def render_h3_risk_inference(strategy_ci: pd.DataFrame, rank_stability: pd.DataFrame) -> str:
    body: list[str] = []
    if strategy_ci is None or strategy_ci.empty:
        body.append(r"No H3 bootstrap rows available & n.a. & n.a. & n.a. & n.a. \\")
    else:
        for _, row in strategy_ci.iterrows():
            body.append(
                f"{row['strategy']} & "
                f"{fmt_num(row['ir'], nd=3)} & "
                f"[{fmt_num(row['ir_lo'], nd=3)}, {fmt_num(row['ir_hi'], nd=3)}] & "
                f"{fmt_num(row['tev'], nd=3)} & "
                f"[{fmt_num(row['tev_lo'], nd=3)}, {fmt_num(row['tev_hi'], nd=3)}] \\\\"
            )
    if rank_stability is not None and not rank_stability.empty:
        r = rank_stability.iloc[0]
        rank_note = (
            f"Bootstrap probability that the information-ratio ranking is preserved: "
            f"{fmt_num(r.get('p_ir_ranking_preserved'), nd=3)}. Bootstrap probability "
            f"that the RAEAR ranking flips across the risk-aversion grid: "
            f"{fmt_num(r.get('p_raear_rank_flip_across_eta'), nd=3)}."
        )
    else:
        rank_note = "Rank-stability probabilities unavailable."
    return _table(
        caption="Bootstrap Inference for the H3 Risk-Adjusted Ranking",
        label="tab:h3-risk-inference",
        legend=(
            "This table reports block-bootstrap percentile confidence intervals for "
            "the information ratio and the tracking-error variance of each strategy, "
            "resampling whole trading dates. The intervals quantify the sampling "
            "uncertainty of the H3 risk-adjusted statistics that are otherwise "
            "reported only as point estimates."
        ),
        colspec="lrrrr",
        header=r"Strategy & IR & IR 95\% CI & TEV & TEV 95\% CI",
        body=body,
        notes=rank_note,
    )


def render_h2_union(union: pd.DataFrame) -> str:
    labels = {
        "OFI_marginal": "S3 OFI $-$ S2",
        "IMB_marginal": "S3 IMB $-$ S2",
        "FULL_vs_S2": "S3 Full $-$ S2",
        "interaction": "Interaction",
    }
    body: list[str] = []
    if union is None or union.empty:
        body.append(r"No per-bin union rows available & n.a. & n.a. & n.a. \\")
    else:
        for _, row in union.iterrows():
            body.append(
                f"{labels.get(str(row['label']), row['label'])} & "
                f"{fmt_num(row.get('max_abs_t'), nd=2)} & "
                f"{fmt_int(row.get('n_groups'))} & "
                f"{fmt_num(row.get('p_bootstrap'), nd=3)} \\\\"
            )
    return _table(
        caption="Exploratory Per-Bin H2 Union Test across Matched-Fill Bins",
        label="tab:h2-union-bins",
        legend=(
            "This diagnostic table reports the multiplicity-aware per-bin union "
            "surface. The confirmatory H2 decision remains the pooled matched "
            "differential; this table asks whether a signal differential appears "
            "in at least one matched-fill bin after accounting for the bin search."
        ),
        colspec="lrrr",
        header=r"Signal differential & Max. bin statistic & Bins & $p$ (boot)",
        body=body,
        notes=(
            "A single cluster-weight draw is shared across bins in each replication "
            "so the null distribution of the maximum respects cross-bin dependence. "
            "Rows are exploratory and do not replace the registered pooled H2 test."
        ),
    )


def write_outputs(
    out_dir: Path,
    *,
    calibration: pd.DataFrame,
    economic: pd.DataFrame,
    paired: pd.DataFrame,
    h2: pd.DataFrame,
    manifest: dict,
    h3_strategy_ci: pd.DataFrame | None = None,
    h3_rank_stability: pd.DataFrame | None = None,
    h3_pairwise: pd.DataFrame | None = None,
    h2_union: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(out_dir / "fill_model_oos_calibration.csv", index=False)
    economic.to_csv(out_dir / "fill_model_economic_tests.csv", index=False)
    paired.to_csv(out_dir / "fill_model_vs_queue_tests.csv", index=False)
    h2.to_csv(out_dir / "h2_pooled_adjusted.csv", index=False)

    files = {
        "tab_fill_model_calibration.tex": render_fill_model_calibration(calibration),
        "tab_fill_robustness_adjusted.tex": render_fill_robustness(economic),
        "tab_h2_pooled_adjusted.tex": render_h2_adjusted(h2),
    }

    extra_csvs: dict[str, pd.DataFrame] = {}
    if h3_strategy_ci is not None:
        extra_csvs["h3_risk_inference_ci.csv"] = h3_strategy_ci
        files["tab_h3_risk_inference.tex"] = render_h3_risk_inference(
            h3_strategy_ci, h3_rank_stability,
        )
    if h3_rank_stability is not None:
        extra_csvs["h3_rank_stability.csv"] = h3_rank_stability
    if h3_pairwise is not None:
        extra_csvs["h3_pairwise_ir.csv"] = h3_pairwise
    if h2_union is not None:
        extra_csvs["h2_per_bin_union.csv"] = h2_union
        files["tab_h2_union_bins.tex"] = render_h2_union(h2_union)
    if registry is not None:
        extra_csvs["test_registry.csv"] = registry

    for name, frame in extra_csvs.items():
        frame.to_csv(out_dir / name, index=False)
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")

    readme = [
        "# Statistical tests output",
        "",
        "This folder contains supplementary statistical validation artifacts for the thesis.",
        "",
        "Core files:",
        "",
        "- `fill_model_oos_calibration.csv`: stratified evaluation-period fill calibration.",
        "- `fill_model_economic_tests.csv`: S3-full vs MOC by fill specification, with",
        "  wild cluster bootstrap p-values, one-sided p-values, and MDE.",
        "- `fill_model_vs_queue_tests.csv`: paired alternative-minus-queue tests.",
        "- `h2_pooled_adjusted.csv`: H2 pooled signal family with Holm and FDR p-values.",
        "- `h2_per_bin_union.csv`: exploratory multiplicity-aware per-bin H2 union test.",
        "- `h3_risk_inference_ci.csv`, `h3_rank_stability.csv`, `h3_pairwise_ir.csv`:",
        "  descriptive block-bootstrap diagnostics for the H3 risk-adjusted ranking.",
        "- `test_registry.csv`: consolidated enumeration of every emitted test with",
        "  its role and correction family.",
        "- `tab_*.tex`: copy-ready thesis snippets.",
        "",
        "Model-based fill rows are robustness bounds, not replacements for the queue-aware headline specification.",
        "",
    ]
    (out_dir / "README_statistical_tests.md").write_text("\n".join(readme), encoding="utf-8")
    manifest["outputs"] = sorted([
        "fill_model_oos_calibration.csv",
        "fill_model_economic_tests.csv",
        "fill_model_vs_queue_tests.csv",
        "h2_pooled_adjusted.csv",
        "README_statistical_tests.md",
        *extra_csvs.keys(),
        *files.keys(),
    ])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {name: str(out_dir / name) for name in manifest["outputs"]}
