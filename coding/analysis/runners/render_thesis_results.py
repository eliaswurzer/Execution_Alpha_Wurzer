"""
render_thesis_results.py -- Run bundle to copy-ready thesis LaTeX snippets.

Reads the validated hypothesis artifacts of a run bundle produced by
``run_all_hypotheses`` and renders the Chapter-7 tables and figures of the
thesis in the Journal-of-Finance style already defined in the thesis
preamble. Each ``.tex`` snippet is a complete table or figure environment
carrying the same label as the placeholder template it replaces, so the
workflow is: run completes, snippets appear under ``<run-root>/thesis_exports/``,
copy each snippet over the matching template in ``thesis0506.tex``.

Usage::

    python -m analysis.runners.render_thesis_results --run-root <dir>
    python -m analysis.runners.render_thesis_results --run-root <dir> \
        --compare-run tape_replay_strict=<dir> --compare-run tape_replay=<dir>

A run that fails the acceptance gate (run status, hypothesis statuses,
calibration feature policy) is refused unless ``--allow-incomplete`` is set,
in which case every output is visibly marked DRAFT and file names carry a
``_draft`` suffix. Draft output must never be pasted into the thesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..inference.tests import _two_sided_p
from ._common import rolling_window_panel
from ..reporting.jof_latex import (
    escape_latex,
    fmt_int,
    fmt_num,
    jof_figure,
    jof_table,
    paren_t,
    with_stars,
)
from ..reporting.thesis_figures.style import (
    FILL_ALPHA,
    THEME,
    THEME_DIVERGING,
    apply_thesis_style,
    color_for,
    emphasis_color,
)

log = logging.getLogger(__name__)

STRATEGY_NAMES = {
    "S0_MOC": "S0 MOC",
    "S1_STATIC": "S1 Static",
    "S2_TIME_ADAPTIVE": "S2 Time-Adaptive",
    "S3_OFI": "S3 OFI",
    "S3_IMB": "S3 IMB",
    "S3_FULL": "S3 Full",
    "S4_TOD": "S4 TOD",
    "S5_VALUE_AWARE_XGB": "S5 Value-Aware",
}
STRATEGY_ORDER = list(STRATEGY_NAMES)

H2_LABELS = {
    "OFI_marginal": "S3 OFI $-$ S2",
    "IMB_marginal": "S3 IMB $-$ S2",
    "FULL_vs_S2": "S3 Full $-$ S2",
    "interaction": "Interaction",
}

FILL_SPEC_NAMES = {
    "tape_replay_queue": "Queue-aware replay (headline)",
    "tape_replay_strict": "Strictly-through replay (lower bound)",
    "tape_replay": "At-or-through replay (upper bound)",
    "tape_replay_volume": "Volume-capped replay",
    "tape_replay_haircut": "Queue haircut",
    "tape_replay_volume_haircut": "Volume-capped queue haircut",
    "cox": "Cox proportional hazards",
    "xgb": "XGBoost survival",
    "km": "Kaplan-Meier",
    "infinite_depth": "Touch benchmark",
    "infinite_depth_haircut": "Touch benchmark (haircut)",
}
BRACKET_SPECS = ("tape_replay_queue", "tape_replay_strict", "tape_replay")

# Model-based survival specifications. Their fills are probabilistic and assigned
# at scheduled refresh times, so they are not anchored to the realized adverse
# trade prints; their net alpha is an optimistic bound (overstated) and their
# adverse selection is understated. Flagged with a marker in the fill-spec table
# and figure so readers do not read them as tape-comparable execution evidence.
MODEL_SPECS = frozenset({"cox", "xgb", "km"})


class RunNotValidatedError(RuntimeError):
    """Raised when the run bundle fails the thesis acceptance gate."""


# ---------------------------------------------------------------------------
# Bundle access and acceptance gate
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_simulation_source_sha256() -> str:
    from .master_panel import _simulation_source_signature

    return _simulation_source_signature()


class RunBundle:
    """Thin accessor for a run_all_hypotheses bundle."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inputs: dict[str, str] = {}

    def path(self, rel: str) -> Path:
        return self.root / rel

    def read_csv(self, rel: str) -> pd.DataFrame:
        path = self.path(rel)
        frame = pd.read_csv(path)
        self.inputs[rel] = _sha256(path)
        return frame

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self.path(rel)
        frame = pd.read_parquet(path, columns=columns)
        self.inputs[rel] = _sha256(path)
        return frame

    def validate(self) -> tuple[list[str], str, str]:
        """Return (problems, simulation_fingerprint, feature_policy)."""
        problems: list[str] = []
        fingerprint = "unknown"
        feature_policy = "unknown"

        status_path = self.path("run_status.json")
        if not status_path.exists():
            problems.append("run_status.json missing")
        else:
            self.inputs["run_status.json"] = _sha256(status_path)
            status = _read_json(status_path)
            if status.get("status") != "complete":
                problems.append(
                    f"run_status.json status is {status.get('status')!r}, expected 'complete'"
                )
            fingerprint = str(
                (status.get("simulation") or {}).get("fingerprint", "unknown")
            )

        for name in ("h1", "h2", "h3"):
            hp = self.path(f"hypotheses/{name}/status.json")
            if not hp.exists():
                problems.append(f"hypotheses/{name}/status.json missing")
                continue
            hstatus = _read_json(hp)
            if hstatus.get("status") != "complete":
                problems.append(
                    f"{name} status is {hstatus.get('status')!r}, expected 'complete'"
                )

        sim_config_path = self.path("metadata/simulation_config.json")
        if not sim_config_path.exists():
            problems.append("metadata/simulation_config.json missing")
        else:
            self.inputs["metadata/simulation_config.json"] = _sha256(sim_config_path)
            sim_config = _read_json(sim_config_path)
            stored_source = str(sim_config.get("simulation_source_sha256", "unknown"))
            current_source = _current_simulation_source_sha256()
            if stored_source != current_source:
                problems.append(
                    "simulation source hash mismatch: "
                    f"run has {stored_source!r}, current source is {current_source!r}"
                )

        config_path = self.path("metadata/run_config.json")
        if config_path.exists():
            self.inputs["metadata/run_config.json"] = _sha256(config_path)
            run_config = _read_json(config_path)
            artifacts = run_config.get("artifacts")
            manifest_path = Path(artifacts) / "calibration_manifest.json" if artifacts else None
            if manifest_path is None or not manifest_path.exists():
                problems.append("calibration_manifest.json not found via run_config artifacts path")
            else:
                self.inputs["calibration_manifest.json"] = _sha256(manifest_path)
                manifest = _read_json(manifest_path)
                feature_policy = str(manifest.get("feature_policy", "unknown"))
                if manifest.get("status") != "complete":
                    problems.append(
                        f"calibration manifest status is {manifest.get('status')!r}"
                    )
                if feature_policy != cfg.FEATURE_POLICY_VERSION:
                    problems.append(
                        f"feature policy {feature_policy!r} does not match "
                        f"current {cfg.FEATURE_POLICY_VERSION!r}"
                    )
        else:
            problems.append("metadata/run_config.json missing")

        return problems, fingerprint, feature_policy


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _strategy_label(strategy: str) -> str:
    return STRATEGY_NAMES.get(strategy, escape_latex(strategy))


def _as_markout(sub: pd.DataFrame) -> float:
    """Primary AS diagnostic: minus the mean SIGNED post-fill markout over
    rows with passive fills, so positive values denote adverse drift. The
    signed convention follows the markout literature and avoids the Jensen
    markup of the one-sided cost (which stays in the panel as a supplementary
    downside measure)."""
    if "adverse_selection_bps" not in sub.columns:
        return float("nan")
    filled = sub[pd.to_numeric(sub["fill_rate"], errors="coerce") > 0]
    if filled.empty:
        return 0.0
    return float(-pd.to_numeric(
        filled["adverse_selection_bps"], errors="coerce",
    ).mean())


def _panel_strategy_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy means used in Panel B of the H1 table."""
    out = panel.copy()
    agg = out.groupby("strategy").agg(
        gross_alpha=("alpha_bps", "mean"),
        net_alpha=("net_alpha_bps", "mean"),
        fill_rate=("fill_rate", "mean"),
        vs_moc=("net_alpha_vs_moc_bps", "mean"),
    )
    agg["as_markout"] = [
        _as_markout(out[out["strategy"] == s]) for s in agg.index
    ]
    # Exact gross-identity component: mean of fill_rate * signed markout.
    if "adverse_selection_bps" in out.columns:
        comp = (
            pd.to_numeric(out["fill_rate"], errors="coerce")
            * pd.to_numeric(out["adverse_selection_bps"], errors="coerce")
        )
        agg["as_component"] = comp.groupby(out["strategy"]).mean().reindex(agg.index)
    else:
        agg["as_component"] = np.nan
    return agg.reindex([s for s in STRATEGY_ORDER if s in agg.index])


def _primary_h1_surface(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the primary H1 cell used by the paired headline test."""
    out = panel.copy()
    if "window" in out.columns:
        out = out[out["window"] == cfg.PRIMARY_WINDOW]
    if "size_frac" in out.columns:
        size = pd.to_numeric(out["size_frac"], errors="coerce")
        out = out[np.isclose(size, cfg.PARENT_ORDER_PRIMARY_FRACTION)]
    return out.copy()


def _sample_span(panel: pd.DataFrame) -> str:
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    if dates.empty:
        return "the evaluation sample"
    return f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"


def build_h1_table(bundle: RunBundle, provenance: str, draft_note: str) -> str:
    primary = bundle.read_csv("hypotheses/h1/h1_primary_ttest.csv").iloc[0]
    panel = bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
    primary_panel = _primary_h1_surface(panel)
    diag = _panel_strategy_diagnostics(primary_panel)
    span = _sample_span(panel)

    body: list[str] = []
    body.append(r"\jofpanel{6}{Panel A: Primary paired differential}")
    body.append(r" & Mean diff.\ (bps) & & $t$ & $p$ & $N$ \\")
    body.append(r"\midrule")
    p_value = primary.get("p_value", primary.get("p", np.nan))
    body.append(
        "S3 full $-$ S0 MOC & "
        f"{with_stars(primary['mean'], p_value)} & & "
        f"{fmt_num(primary['t'])} & {fmt_num(p_value, nd=3)} & "
        f"{fmt_int(primary['n'])} \\\\"
    )
    body.append(f" & {paren_t(primary['t'])} & & & & \\\\")
    body.append(r"\addlinespace")
    body.append(r"\jofpanel{6}{Panel B: Strategy-level diagnostics}")
    body.append(r" & Gross alpha & Net alpha & Fill rate & AS markout & vs.\ MOC \\")
    body.append(r"\midrule")
    for strategy, row in diag.iterrows():
        body.append(
            f"{_strategy_label(strategy)} & {fmt_num(row['gross_alpha'])} & "
            f"{fmt_num(row['net_alpha'])} & {fmt_num(row['fill_rate'], nd=3)} & "
            f"{fmt_num(row['as_markout'])} & {fmt_num(row['vs_moc'])} \\\\"
        )

    legend = (
        "This table reports the primary H1 test of the paired net-execution-alpha "
        "differential between the signal-conditioned passive strategy (S3 full) and "
        "the Market-on-Close benchmark (S0). Panel A reports the panel-mean "
        "differential in basis points for the primary configuration, with Window~B "
        "arrival at 15:30~ET, a parent size of one percent of expected "
        "closing-auction volume, and queue-aware tape-replay fills. Panel B reports "
        "strategy-level diagnostics for the same Window~B, one-percent parent-size "
        "cell. $t$-statistics "
        "based on standard errors clustered two-way by symbol and date appear in "
        f"parentheses. The sample covers {span}.{draft_note}"
    )
    notes = (
        "Panel A reports the validated paired primary differential, and Panel B "
        "reports the corresponding strategy-level diagnostics from the same "
        "primary H1 cell. Alphas are expressed in basis points relative to the "
        "official closing price. The AS markout is the negated mean signed "
        "post-fill mid-quote drift over the configured horizon, conditional on "
        "a passive fill, so positive values denote adverse drift."
    )
    return jof_table(
        caption="Net Execution Alpha of the Signal-Conditioned Strategy versus Market-on-Close",
        label="tab:h1-primary-template",
        legend=legend,
        column_format="lrrrrr",
        header="",
        body_lines=body,
        notes=notes,
        provenance=provenance,
    )


def build_h2_table(bundle: RunBundle, provenance: str, draft_note: str) -> str:
    pooled = bundle.read_csv("hypotheses/h2/h2_pooled.csv")
    per_bin = bundle.read_csv("hypotheses/h2/h2_per_bin_differentials.csv")

    bins = sorted(int(b) for b in per_bin["bin"].dropna().unique()) if not per_bin.empty else []
    n_cols = max(len(bins), 4) + 1

    body: list[str] = []
    body.append(rf"\jofpanel{{{n_cols}}}{{Panel A: Pooled matched differentials}}")
    pad = " &" * (n_cols - 5)
    body.append(rf" & Mean (bps) & $t$ & $N$ &{pad} \\".replace("&  \\\\", "& \\\\"))
    body.append(r"\midrule")
    for _, row in pooled.iterrows():
        label = H2_LABELS.get(str(row["label"]), escape_latex(str(row["label"])))
        p_value = _two_sided_p(float(row["t"])) if np.isfinite(row.get("t", np.nan)) else np.nan
        cells = [
            label,
            with_stars(row["mean"], p_value),
            fmt_num(row["t"]),
            fmt_int(row["n"]),
        ]
        cells += [""] * (n_cols - len(cells))
        body.append(" & ".join(cells) + r" \\")
        t_cells = ["", paren_t(row["t"])] + [""] * (n_cols - 2)
        body.append(" & ".join(t_cells) + r" \\")
    body.append(r"\addlinespace")

    if bins:
        body.append(
            rf"\jofpanel{{{n_cols}}}{{Panel B: Differentials by realized S2 fill-rate bin}}"
        )
        header_cells = [""] + [f"B{b + 1}" for b in bins]
        header_cells += [""] * (n_cols - len(header_cells))
        body.append(" & ".join(header_cells) + r" \\")
        body.append(r"\midrule")
        for raw_label in ("OFI_marginal", "IMB_marginal", "FULL_vs_S2"):
            sub = per_bin[per_bin["label"] == raw_label].set_index("bin")
            cells = [H2_LABELS[raw_label]]
            for b in bins:
                if b in sub.index:
                    row = sub.loc[b]
                    p_value = _two_sided_p(float(row["t"])) if np.isfinite(row["t"]) else np.nan
                    cells.append(with_stars(row["mean"], p_value))
                else:
                    cells.append("--")
            cells += [""] * (n_cols - len(cells))
            body.append(" & ".join(cells) + r" \\")

    matching = escape_latex(str(pooled["matching_metric"].iloc[0])) if not pooled.empty else "unavailable"
    legend = (
        "This table reports the H2 signal-decomposition tests. Panel A reports "
        "pooled net-alpha differentials of each S3 variant against the "
        "time-adaptive baseline S2, matched on the realized S2 passive fill rate. "
        "The interaction is computed as S3 full minus S3 OFI minus S3 IMB plus S2. "
        "Panel B reports the differentials within realized fill-rate bins. "
        "$t$-statistics based on standard errors clustered two-way by symbol and "
        f"date appear in parentheses. Matching metric: {matching}.{draft_note}"
    )
    notes = (
        r"Means are net-alpha differentials in basis points under the headline "
        r"queue-aware tape-replay mechanism. Panel A is the pooled matched "
        r"signal-family comparison. Panel B reports the same signal "
        r"differentials within realized S2 fill-rate bins."
    )
    return jof_table(
        caption="Marginal Contributions of the OFI Signal and the Closing-Pressure Proxy",
        label="tab:h2-pooled-template",
        legend=legend,
        column_format="l" + "r" * (n_cols - 1),
        header="",
        body_lines=body,
        notes=notes,
        provenance=provenance,
        resize=len(bins) > 6,
    )


def build_h3_table(bundle: RunBundle, provenance: str, draft_note: str) -> str:
    raear = bundle.read_csv("hypotheses/h3/h3_raear.csv")
    eta_cols = [c for c in raear.columns if c.startswith("raear_eta_")]
    etas = [c.replace("raear_eta_", "") for c in eta_cols]

    header_cells = ["Strategy", "Mean alpha vs. MOC", "TEV", "TES", "IR"]
    header_cells += [rf"RAEAR$_{{{eta}}}$" for eta in etas]
    header_cells.append(r"$\eta^*$")

    body: list[str] = []
    raear = raear.set_index("strategy")
    for strategy in [s for s in STRATEGY_ORDER if s in raear.index]:
        row = raear.loc[strategy]
        cells = [
            _strategy_label(strategy),
            fmt_num(row["mean_alpha"]),
            fmt_num(row["tev"]),
            fmt_num(row["tes"]),
            fmt_num(row["ir"], nd=3),
        ]
        cells += [fmt_num(row[c]) for c in eta_cols]
        cells.append(fmt_num(row["eta_star"], nd=4))
        body.append(" & ".join(cells) + r" \\")

    bounds_note = ""
    tev_path = bundle.path("hypotheses/h3/h3_tev.csv")
    if tev_path.exists():
        tev = bundle.read_csv("hypotheses/h3/h3_tev.csv")
        if {"te_port_indep", "te_port_perf_corr", "strategy"}.issubset(tev.columns):
            ref = tev[tev["strategy"] == "S3_FULL"]
            if not ref.empty:
                bounds_note = (
                    " Portfolio tracking-error bounds for S3 full are "
                    f"{fmt_num(ref['te_port_indep'].iloc[0])} (independence) and "
                    f"{fmt_num(ref['te_port_perf_corr'].iloc[0])} (perfect correlation) "
                    "basis points."
                )

    legend = (
        "This table reports the H3 risk-adjusted comparison across strategies on "
        "the validated evaluation panel. Mean net alpha is expressed as a "
        "basis-point differential versus S0 MOC, TEV is the sample variance of "
        "that differential in squared basis points, and TES is its square root "
        "in basis points. IR is the dimensionless execution-alpha information "
        "ratio. RAEAR columns evaluate the MOC-relative penalized statistic "
        f"at the indicated risk-aversion coefficients, and $\\eta^{{*}}$ is "
        f"the break-even risk aversion.{draft_note}"
    )
    notes = (
        "The table reports the MOC-relative net-alpha object used in H3 and "
        "treats strategy-level rankings as descriptive risk-return diagnostics "
        f"rather than clustered pairwise tests.{bounds_note}"
    )
    return jof_table(
        caption="Risk-Adjusted Execution Alpha and Tracking-Error Statistics by Strategy",
        label="tab:h3-raear-template",
        legend=legend,
        column_format="l" + "r" * (len(header_cells) - 1),
        header=" & ".join(header_cells),
        body_lines=body,
        notes=notes,
        provenance=provenance,
        resize=True,
        include_stars_note=False,
    )


def build_tier_subgroup_table(bundle: RunBundle, provenance: str, draft_note: str) -> str | None:
    rel = "hypotheses/h1/h1_subgroup_tier.csv"
    if not bundle.path(rel).exists():
        return None
    sub = bundle.read_csv(rel)
    body: list[str] = []
    for _, row in sub.iterrows():
        p_holm = row.get("p_holm", np.nan)
        body.append(
            f"Tier {escape_latex(str(row['level']))} & "
            f"{with_stars(row['mean'], p_holm)} & {fmt_num(row['t'])} & "
            f"{fmt_num(row.get('p_value', np.nan), nd=3)} & "
            f"{fmt_num(p_holm, nd=3)} & {fmt_int(row['n'])} \\\\"
        )
        body.append(f" & {paren_t(row['t'])} & & & & \\\\")
    legend = (
        "This table reports the exploratory H1 net-alpha differential of S3 full "
        "against MOC by liquidity tier. Tiers are assigned from pre-sample median "
        "relative half-spreads, with Tier 1 the tightest-spread group. "
        "$t$-statistics based on two-way clustered standard errors appear in "
        "parentheses, and stars refer to the Holm-adjusted $p$-values, which "
        f"control the family-wise error rate within this subgroup family.{draft_note}"
    )
    notes = (
        "Subgroup tests are exploratory, evaluated on the headline Window-B "
        "parent-order surface, and Holm-corrected within the liquidity-tier "
        "family."
    )
    return jof_table(
        caption="Heterogeneity of the Performance Gap by Liquidity Tier",
        label="tab:h1-tier-subgroup",
        legend=legend,
        column_format="lrrrrr",
        header=r"Tier & Mean diff.\ (bps) & $t$ & $p$ & $p$ (Holm) & $N$",
        body_lines=body,
        notes=notes,
        provenance=provenance,
    )


def build_side_split_robustness_table(
    bundle: RunBundle, provenance: str, draft_note: str,
) -> str | None:
    """Buy/sell diagnostic for the primary S3-full Window-B comparison."""
    from ..inference.clustering import mean_with_twoway_se, two_way_cluster_ols

    panel = bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
    required = {"strategy", "window", "side", "symbol", "date"}
    if panel.empty or not required.issubset(panel.columns):
        return None
    alpha_col = (
        "net_alpha_vs_moc_bps"
        if "net_alpha_vs_moc_bps" in panel.columns else "net_alpha_bps"
    )
    if alpha_col not in panel.columns:
        return None

    sub = panel[
        (panel["strategy"] == "S3_FULL")
        & (panel["window"] == cfg.PRIMARY_WINDOW)
    ].copy()
    if "size_frac" in sub.columns:
        sub = sub[np.isclose(
            pd.to_numeric(sub["size_frac"], errors="coerce"),
            cfg.PARENT_ORDER_PRIMARY_FRACTION,
        )]
    sub = sub.dropna(subset=[alpha_col, "side", "symbol", "date"])
    if sub.empty or sub["side"].nunique() < 2:
        return None

    body: list[str] = []
    side_stats: dict[str, dict[str, float]] = {}
    for side in ("BUY", "SELL"):
        grp = sub[sub["side"].astype(str).str.upper() == side]
        if grp.empty:
            side_stats[side] = {
                "n": 0, "mean": float("nan"), "se": float("nan"),
                "t": float("nan"), "p": float("nan"),
            }
            continue
        mean, se = mean_with_twoway_se(grp[alpha_col], grp["symbol"], grp["date"])
        t = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
        side_stats[side] = {
            "n": int(len(grp)), "mean": mean, "se": se, "t": t,
            "p": _two_sided_p(t),
        }

    for side in ("BUY", "SELL"):
        row = side_stats[side]
        body.append(
            f"{side.title()} parent orders & {fmt_int(row['n'])} & "
            f"{fmt_num(row['mean'])} & {fmt_num(row['se'])} & "
            f"{fmt_num(row['t'])} & {fmt_num(row['p'], nd=3)} \\\\"
        )

    x_buy = (sub["side"].astype(str).str.upper() == "BUY").astype(float).to_numpy()
    y = pd.to_numeric(sub[alpha_col], errors="coerce").to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), x_buy])
    res = two_way_cluster_ols(
        y,
        X,
        sub["symbol"].to_numpy(),
        pd.to_datetime(sub["date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
        names=["sell_mean", "buy_minus_sell"],
    )
    contrast = float(res.coef[1])
    contrast_se = float(res.se_cluster_twoway[1])
    contrast_t = float(res.tstat("twoway")[1])
    body.append(r"\addlinespace")
    body.append(
        f"Buy $-$ Sell contrast & {fmt_int(len(sub))} & "
        f"{fmt_num(contrast)} & {fmt_num(contrast_se)} & "
        f"{fmt_num(contrast_t)} & {fmt_num(_two_sided_p(contrast_t), nd=3)} \\\\"
    )

    span = _sample_span(sub)
    legend = (
        "This table reports a buy/sell side diagnostic for the primary "
        "S3-full comparison against MOC. Parent-order sides are assigned by "
        "the deterministic balanced rotation described in Section~\\ref{sec:parent_orders}, "
        "and all alpha measures are side-signed, so buy and sell means are "
        "reported on the same economic scale. The first two rows test each "
        "side mean against zero; the final row reports the buy-minus-sell "
        "contrast from an intercept-plus-buy-dummy regression. Standard errors "
        f"are clustered two-way by symbol and date. The sample covers {span}.{draft_note}"
    )
    notes = (
        "Means are S3-full net-alpha differentials versus the MOC benchmark "
        "in basis points for Window~B and the headline one-percent parent "
        "size. A positive buy-minus-sell contrast indicates stronger "
        "MOC-relative performance on buy parent orders. This diagnostic is "
        "reported as robustness evidence on the side-balanced design and is "
        "not part of the primary hypothesis family."
    )
    return jof_table(
        caption="Buy-Sell Side-Split Robustness",
        label="tab:side-split-robustness",
        legend=legend,
        column_format="lrrrrr",
        header=r"Side / contrast & $N$ & Mean diff.\ (bps) & SE & $t$ & $p$",
        body_lines=body,
        notes=notes,
        provenance=provenance,
        include_stars_note=False,
    )


def _spec_diagnostics(panel: pd.DataFrame) -> dict[str, float]:
    sub = panel[panel["strategy"] == "S3_FULL"]
    # Report the Window-B headline cell so the bracket matches the H1 primary
    # test exactly. Fall back to the full panel only if no window column or no
    # Window-B rows are present.
    if "window" in sub.columns:
        win_b = sub[sub["window"] == "B"]
        if not win_b.empty:
            sub = win_b
    if sub.empty:
        return {}
    # Net alpha is reported as the differential against MOC (the H1 object), so
    # the common commission paid by the MOC benchmark cancels; fall back to the
    # raw net alpha only when the differential column is absent.
    alpha_col = (
        "net_alpha_vs_moc_bps" if "net_alpha_vs_moc_bps" in sub.columns
        else "net_alpha_bps"
    )
    return {
        "net_alpha": float(sub[alpha_col].mean()),
        "fill_rate": float(sub["fill_rate"].mean()),
        "as_markout": _as_markout(sub),
        "residual_moc": float(1.0 - sub["fill_rate"].mean()),
        "n": int(len(sub)),
    }


def _primary_pvalue(bundle: RunBundle) -> float | None:
    """Two-way clustered p-value of the Window-B S3-full vs MOC differential,
    read from the bundle's stored primary t-test so the fill-robustness stars
    match the H1 primary inference exactly."""
    try:
        tt = bundle.read_csv("hypotheses/h1/h1_primary_ttest.csv")
    except Exception:
        return None
    if tt.empty or "p_value" not in tt.columns:
        return None
    row = tt
    if "label" in tt.columns:
        win_b = tt[tt["label"].astype(str).str.contains(":B:")]
        if not win_b.empty:
            row = win_b
    try:
        return float(row.iloc[0]["p_value"])
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def build_fill_robustness_table(
    bundle: RunBundle,
    compare_runs: dict[str, Path],
    headline_spec: str,
    provenance: str,
    draft_note: str,
) -> str:
    rows: dict[str, dict[str, float]] = {}
    rows[headline_spec] = _spec_diagnostics(
        bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
    )
    if rows[headline_spec]:
        rows[headline_spec]["p"] = _primary_pvalue(bundle)
    for spec, root in compare_runs.items():
        try:
            cmp_bundle = RunBundle(root)
            rows[spec] = _spec_diagnostics(
                cmp_bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
            )
            if rows[spec]:
                rows[spec]["p"] = _primary_pvalue(cmp_bundle)
            bundle.inputs.update({
                f"compare:{spec}:{k}": v for k, v in cmp_bundle.inputs.items()
            })
        except Exception as exc:
            log.warning("compare run %s unreadable: %s", spec, exc)
            rows[spec] = {}

    def _row(spec: str) -> str:
        name = FILL_SPEC_NAMES.get(spec, escape_latex(spec))
        if spec in MODEL_SPECS:
            name += r"$^{\dagger}$"  # optimistic model-based bound (see notes)
        d = rows.get(spec) or {}
        # Stars sit on the net-alpha differential only; the other columns are
        # descriptive diagnostics.
        return (
            f"{name} & {with_stars(d.get('net_alpha'), d.get('p'))} & "
            f"{fmt_num(d.get('fill_rate'), nd=3)} & {fmt_num(d.get('as_markout'))} & "
            f"{fmt_num(d.get('residual_moc'), nd=3)} & {fmt_int(d.get('n'))} \\\\"
        )

    body: list[str] = []
    body.append(r"\jofpanel{6}{Panel A: Tape-replay bracket}")
    for spec in BRACKET_SPECS:
        body.append(_row(spec))
    body.append(r"\addlinespace")
    body.append(r"\jofpanel{6}{Panel B: Stylized and model-based alternatives}")
    for spec in FILL_SPEC_NAMES:
        if spec in BRACKET_SPECS:
            continue
        if spec in rows or spec in (
            "tape_replay_volume", "tape_replay_haircut", "cox", "xgb", "km",
            "infinite_depth",
        ):
            body.append(_row(spec))

    missing = [
        FILL_SPEC_NAMES.get(s, s)
        for s in list(BRACKET_SPECS) + ["tape_replay_volume", "tape_replay_haircut",
                                        "cox", "xgb", "km", "infinite_depth"]
        if not rows.get(s)
    ]
    missing_note = (
        " Rows shown as -- await the corresponding robustness run "
        f"({escape_latex(', '.join(missing))})." if missing else ""
    )
    legend = (
        "This table reports the S3-full net-execution-alpha differential against "
        "the Market-on-Close benchmark and its diagnostics under each implemented "
        "fill specification, evaluated at the Window-B headline cell. Panel A "
        "contains the deterministic tape-replay bracket, in which the queue-aware "
        "headline rule is bounded from below by strictly-through replay and from "
        "above by at-or-through replay. Panel B contains the stylized and "
        "model-based alternatives. The differential and the adverse-selection "
        "markout are in basis points, fill rate and residual MOC share in "
        f"fractions of the parent quantity.{draft_note}"
    )
    notes = (
        "Each row reruns the identical strategy panel under the indicated fill "
        "specification. Net alpha is the Window-B differential of S3-full against "
        "the MOC benchmark, so the common commission cancels and a positive value "
        "denotes execution superior to MOC. The AS markout is the negated mean "
        "signed post-fill mid-quote drift conditional on a passive fill; positive "
        "values denote adverse drift. Significance stars refer to the net-alpha "
        "differential against MOC and use the same two-way clustered $t$-statistics "
        "as the H1 primary test; for the model-based rows they reflect the "
        "statistical precision of the estimate rather than its economic validity. "
        "Rows marked $\\dagger$ are model-based survival specifications whose "
        "probabilistic fills are not anchored to the realized adverse trade prints, "
        "so their net alpha is an optimistic upper bound and their adverse-selection "
        "markout is understated; the realistic estimates are the Panel~A "
        "tape-replay bracket."
        f"{missing_note}"
    )
    return jof_table(
        caption="Headline Results across Fill Specifications",
        label="tab:fill-robustness-template",
        legend=legend,
        column_format="lrrrrr",
        header=r"Fill specification & Net alpha vs.\ MOC & Fill rate & AS markout & Residual MOC & $N$",
        body_lines=body,
        notes=notes,
        provenance=provenance,
        include_stars_note=True,
    )


def build_size_window_robustness_table(
    bundle: RunBundle,
    compare_runs: dict[str, Path],
    provenance: str,
    draft_note: str,
) -> str:
    """Panel A: parent-size grid (from the size-grid run, key ``size_grid``);
    Panel B: arrival windows A/B/C at the headline size from the own panel."""
    from ..inference.clustering import mean_with_twoway_se
    from ..metrics import tracking_error_variance

    grid = pd.DataFrame()
    grid_root = compare_runs.get("size_grid")
    if grid_root is not None:
        csv_path = Path(grid_root) / "size_table_summary.csv"
        if csv_path.exists():
            grid = pd.read_csv(csv_path)
            bundle.inputs[f"compare:size_grid:{csv_path.name}"] = _sha256(csv_path)
        else:
            log.warning("size grid summary missing: %s", csv_path)
        # Merge the MOC-relative differential (and its t-statistic) so Panel A
        # reports the same object as H1 rather than the raw net alpha, which
        # embeds the commission that the MOC benchmark also pays.
        clustered_path = Path(grid_root) / "robustness_summary_clustered.csv"
        if not grid.empty and clustered_path.exists():
            clustered = pd.read_csv(clustered_path)
            vs = clustered[
                (clustered["strategy"] == "S3_FULL")
                & (clustered["metric"] == "net_alpha_vs_moc_bps")
            ][["size_bucket", "mean", "t"]].rename(
                columns={"mean": "vs_moc_net", "t": "vs_moc_t"},
            )
            if not vs.empty:
                grid = grid.merge(vs, on="size_bucket", how="left")
                bundle.inputs[f"compare:size_grid:{clustered_path.name}"] = _sha256(clustered_path)

    def _size_row(size: float, label: str) -> str:
        row = grid[np.isclose(grid["size_bucket"], size)] if not grid.empty else pd.DataFrame()
        if row.empty:
            return f"{label} & -- & -- & -- & -- & -- \\\\"
        r = row.iloc[0]
        # Prefer the MOC-relative differential where available (matches H1).
        if "vs_moc_net" in row.columns and pd.notna(r.get("vs_moc_net")):
            net = with_stars(r["vs_moc_net"], _two_sided_p(float(r["vs_moc_t"])))
        else:
            net = with_stars(r["mean_net_alpha_bps"], _two_sided_p(float(r["t"])))
        as_key = (
            "mean_as_markout_bps" if "mean_as_markout_bps" in row.columns
            else "mean_as_cost_bps"
        )
        return (
            f"{label} & {net} & {fmt_num(r['mean_fill_rate'], nd=3)} & "
            f"{fmt_num(r[as_key])} & {fmt_num(r['tev'], nd=1)} & "
            f"{fmt_int(r['n'])} \\\\"
        )

    panel = bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
    window_rows: list[str] = []
    window_labels = {
        "A": "Window A (15:00)",
        "B": "Window B (15:30, headline)",
        "C": "Window C (15:45)",
    }
    sub_all = panel[panel["strategy"] == "S3_FULL"] if not panel.empty else pd.DataFrame()
    win_alpha_col = (
        "net_alpha_vs_moc_bps"
        if not sub_all.empty and "net_alpha_vs_moc_bps" in sub_all.columns
        else "net_alpha_bps"
    )
    for window, label in window_labels.items():
        sub = (
            sub_all[sub_all["window"] == window]
            if not sub_all.empty and "window" in sub_all.columns else pd.DataFrame()
        )
        sub = sub.dropna(subset=[win_alpha_col]) if not sub.empty else sub
        if sub.empty:
            window_rows.append(f"{label} & -- & -- & -- & -- & -- \\\\")
            continue
        m, se = mean_with_twoway_se(
            sub[win_alpha_col], sub["symbol"], sub["date"],
        )
        t = m / se if se and np.isfinite(se) and se > 0 else float("nan")
        tev_frame = tracking_error_variance(sub, alpha_col=win_alpha_col)
        tev = (
            float(tev_frame["tev"].iloc[0]) if not tev_frame.empty else float("nan")
        )
        window_rows.append(
            f"{label} & {with_stars(m, _two_sided_p(t))} & "
            f"{fmt_num(sub['fill_rate'].mean(), nd=3)} & "
            f"{fmt_num(_as_markout(sub))} & {fmt_num(tev, nd=1)} & "
            f"{fmt_int(len(sub))} \\\\"
        )

    body: list[str] = []
    body.append(r"\jofpanel{6}{Panel A: Parent-size grid, Window B}")
    size_labels = [
        (0.005, r"0.5\%"),
        (0.01, r"1.0\% (headline)"),
        (0.02, r"2.0\%"),
        (0.05, r"5.0\%"),
        (0.10, r"10.0\%"),
    ]
    for size, label in size_labels:
        body.append(_size_row(size, label))
    body.append(r"\addlinespace")
    body.append(r"\jofpanel{6}{Panel B: Arrival windows, 1\% parent}")
    body.extend(window_rows)

    grid_note = (
        "" if not grid.empty
        else " Panel A rows shown as -- await the parent-size-grid run."
    )
    legend = (
        "This table reports the S3-full net-execution-alpha differential against "
        "the Market-on-Close benchmark across the parent-size grid at the primary "
        "arrival window and across the alternative arrival windows at the headline "
        "parent size. Parent sizes are fractions of expected closing-auction "
        "volume. The self-impact component is active for parent sizes above one "
        "percent. The differential and the adverse-selection markout are in basis "
        f"points, TEV in squared basis points.{draft_note}"
    )
    notes = (
        "Panel A reports the parent-size-grid robustness pass and Panel B "
        "reports the arrival-window cells of the headline panel. Net alpha is the "
        "differential of S3-full against the MOC benchmark, so the common "
        "commission cancels and a positive value denotes execution superior to "
        "MOC. The AS markout is the negated mean signed post-fill mid-quote drift "
        "conditional on a passive fill. Significance stars use two-way clustered "
        f"standard errors by symbol and date.{grid_note} \\jofstars"
    )
    return jof_table(
        caption="Parent-Size and Arrival-Window Robustness",
        label="tab:parent-window-robustness-template",
        legend=legend,
        column_format="lrrrrr",
        header=r"Parent size / window & Net alpha vs.\ MOC & Fill rate & AS markout & TEV & $N$",
        body_lines=body,
        notes=notes,
        provenance=provenance,
        include_stars_note=False,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Path, stem: str) -> str:
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return pdf.name


def build_figures(
    bundle: RunBundle,
    out_dir: Path,
    provenance: str,
    draft_note: str,
    *,
    compare_runs: dict[str, Path] | None = None,
    headline_spec: str = "tape_replay_queue",
) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_thesis_style()

    snippets: dict[str, str] = {}
    panel = bundle.read_parquet("hypotheses/h1/h1_panel.parquet")
    diag = _panel_strategy_diagnostics(panel)
    span = _sample_span(panel)
    labels = [_strategy_label(s).replace("$-$", "-") for s in diag.index]

    # Alpha decomposition (stacked components per strategy). Net alpha is
    # gross plus cash adjustments (implementation shortfall); the realized
    # adverse-selection drift is already inside the close-relative gross and
    # is overlaid as a within-gross diagnostic, not stacked as a deduction.
    components = pd.DataFrame({
        "Gross alpha": diag["gross_alpha"],
        "Maker rebate": diag["fill_rate"] * cfg.MAKER_REBATE_BPS,
        "Commission": -cfg.COMMISSION_BPS,
    }, index=diag.index)
    if "impact_bps" in panel.columns:
        components["Self-impact"] = -panel.groupby("strategy")["impact_bps"].mean().reindex(diag.index)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom_pos = np.zeros(len(components))
    bottom_neg = np.zeros(len(components))
    x = np.arange(len(components))
    for i, col in enumerate(components.columns):
        vals = components[col].fillna(0.0).to_numpy()
        base = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals, bottom=base, label=col, width=0.65, color=color_for("component", col, i))
        bottom_pos = np.where(vals >= 0, bottom_pos + vals, bottom_pos)
        bottom_neg = np.where(vals < 0, bottom_neg + vals, bottom_neg)
    ax.plot(x, diag["net_alpha"].to_numpy(), marker="o", color=emphasis_color(),
            linestyle="none", label="Net alpha")
    ax.plot(
        x, diag["as_component"].to_numpy(),
        marker="v", color=THEME["rose"], linestyle="none", markerfacecolor="none",
        label="Signed AS markout within gross",
    )
    ax.axhline(0.0, color=THEME["gray"], linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Basis points")
    ax.legend(fontsize=8, ncols=2)
    fname = _save_fig(fig, out_dir, "fig_alpha_decomposition")
    snippets["fig_alpha_decomposition"] = jof_figure(
        graphics_file=fname,
        caption="Decomposition of Net Execution Alpha by Strategy",
        label="fig:alpha-decomposition-template",
        legend=(
            "This figure shows the panel-mean net execution alpha of each "
            "strategy as the close-relative gross price alpha plus the cash "
            "adjustments (maker rebate on the filled fraction, commission, and "
            "self-impact where active). Black markers show the resulting net "
            "alpha. Open triangles report the fill-weighted mean of the signed "
            "post-fill adverse-selection markout, which is the exact "
            "adverse-selection component embedded in the gross term; it is not "
            "an additional deduction. All values are in basis points relative "
            f"to the official close. The sample covers {span}.{draft_note}"
        ),
        provenance=provenance,
    )

    # Alpha-fill frontier
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (lab, (_, row)) in enumerate(zip(labels, diag.iterrows())):
        ax.scatter(row["fill_rate"], row["net_alpha"], color=color_for("strategy", lab, i), zorder=3)
        ax.annotate(lab, (row["fill_rate"], row["net_alpha"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Realized passive fill rate")
    ax.set_ylabel("Net execution alpha (bps)")
    ax.axhline(0.0, color=THEME["gray"], linewidth=0.8)
    fname = _save_fig(fig, out_dir, "fig_alpha_fill_frontier")
    snippets["fig_alpha_fill_frontier"] = jof_figure(
        graphics_file=fname,
        caption="Net Execution Alpha against Realized Passive Fill Rate",
        label="fig:alpha-fill-frontier-template",
        legend=(
            "This figure plots strategy-level mean net execution alpha against "
            "the mean realized passive fill rate. The frontier separates "
            "strategies that improve per-fill economics from strategies that "
            f"mainly shift fill-rate exposure. The sample covers {span}.{draft_note}"
        ),
        provenance=provenance,
    )

    # Fill-specification frontier: the Window-B S3-full net-alpha-vs-MOC
    # differential against the realized passive fill rate, one point per fill
    # specification. The deterministic tape-replay bracket (queue, strict,
    # at-or-through) clusters near zero; the model-based survival specifications
    # (KM, XGB, Cox) sit at high fill and high alpha because their probabilistic
    # fills are assigned at scheduled refresh times rather than at realized
    # adverse trade prints, so they do not inherit the adverse-selection
    # asymmetry of tape fills and form an optimistic bound.
    spec_points: dict[str, dict[str, float]] = {}
    hd = _spec_diagnostics(panel)
    if hd:
        spec_points[headline_spec] = hd
    for spec, root in (compare_runs or {}).items():
        if spec not in FILL_SPEC_NAMES:
            continue  # the size_grid compare run is not a fill specification
        try:
            cmp_panel = RunBundle(root).read_parquet("hypotheses/h1/h1_panel.parquet")
            d = _spec_diagnostics(cmp_panel)
            if d:
                spec_points[spec] = d
        except Exception as exc:
            log.warning("fill-spec frontier: compare run %s unreadable: %s", spec, exc)
    if spec_points:
        model_specs = MODEL_SPECS
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # Right-side points anchor their label to the left so it stays inside
        # the axes; the two model specifications sit close together, so stagger
        # them vertically to keep both labels legible.
        y_nudge = {"km": -10, "xgb": 6}
        for i, (spec, d) in enumerate(spec_points.items()):
            is_model = spec in model_specs
            group = "Model-based" if is_model else "Tape replay"
            x, y = d["fill_rate"], d["net_alpha"]
            ax.scatter(
                x, y, zorder=3,
                color=color_for("spec_group", group, i),
                marker="s" if is_model else "o",
            )
            right_side = x > 0.85
            label = FILL_SPEC_NAMES.get(spec, spec).split(" (")[0]
            if is_model:
                label += "*"  # marks the optimistic model-based bound (see legend)
            ax.annotate(
                label, (x, y),
                textcoords="offset points",
                xytext=(-7 if right_side else 6, y_nudge.get(spec, 4)),
                ha="right" if right_side else "left", fontsize=8,
            )
        ax.axhline(0.0, color=THEME["gray"], linewidth=0.8)
        ax.set_xlabel("Realized passive fill rate")
        ax.set_ylabel("Net execution alpha vs. MOC (bps)")
        fname = _save_fig(fig, out_dir, "fig_fill_spec_frontier")
        snippets["fig_fill_spec_frontier"] = jof_figure(
            graphics_file=fname,
            caption="Net Execution Alpha against Fill Rate across Fill Specifications",
            label="fig:fill-spec-frontier-template",
            legend=(
                "This figure plots the Window-B S3-full net-execution-alpha "
                "differential against MOC against the realized passive fill rate, "
                "one point per implemented fill specification. Round markers are "
                "the deterministic tape-replay rules (the queue-aware headline "
                "with its strictly-through and at-or-through bounds); square "
                "markers are the model-based survival specifications. The "
                "deterministic bracket clusters near the zero line, whereas the "
                "survival specifications fill a widely varying share of the parent "
                "at near-zero realized adverse selection and therefore sit well "
                "above it. That vertical gap reflects the fill assignment rather than "
                "superior economics: probabilistic survival fills are scheduled at "
                "refresh times and are not tied to the adverse trade prints that "
                "generate realized adverse selection, so they form an optimistic "
                "bound and are not used to qualify the headline result. The "
                "model-based survival specifications are marked with an asterisk: "
                "their net alpha is overstated and their adverse selection "
                f"understated for this reason.{draft_note}"
            ),
            provenance=provenance,
        )

    # RAEAR curve
    raear = bundle.read_csv("hypotheses/h3/h3_raear.csv").set_index("strategy")
    eta_cols = [c for c in raear.columns if c.startswith("raear_eta_")]
    eta_max = max((float(c.replace("raear_eta_", "")) for c in eta_cols), default=0.5)
    grid = np.linspace(0.0, eta_max, 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, strategy in enumerate([s for s in STRATEGY_ORDER if s in raear.index]):
        row = raear.loc[strategy]
        curve = float(row["mean_alpha"]) - grid * float(row["tev"])
        style = (
            {"linewidth": 2.2, "color": emphasis_color()}
            if strategy == "S0_MOC"
            else {"linewidth": 1.4, "color": color_for("strategy", _strategy_label(strategy).replace("$-$", "-"), i)}
        )
        ax.plot(grid, curve, label=_strategy_label(strategy).replace("$-$", "-"), **style)
    ax.set_xlabel(r"Risk aversion $\eta$ (bps$^{-1}$)")
    ax.set_ylabel("RAEAR (bps)")
    ax.legend(fontsize=8)
    fname = _save_fig(fig, out_dir, "fig_raear_curve")
    snippets["fig_raear_curve"] = jof_figure(
        graphics_file=fname,
        caption="Risk-Adjusted Execution Alpha across the Risk-Aversion Grid",
        label="fig:raear-curve-template",
        legend=(
            "This figure plots the RAEAR of each strategy as a function of the "
            "risk-aversion coefficient, computed from the panel mean net-alpha "
            "differential versus MOC and its tracking-error variance. The MOC "
            "benchmark is the bold reference line. Crossings identify the "
            "break-even aversion levels at which the mean-alpha ranking "
            f"reverses.{draft_note}"
        ),
        provenance=provenance,
    )

    # H2 heatmap
    per_bin = bundle.read_csv("hypotheses/h2/h2_per_bin_differentials.csv")
    if not per_bin.empty:
        pivot = per_bin.pivot_table(index="label", columns="bin", values="mean")
        pivot = pivot.reindex([l for l in H2_LABELS if l in pivot.index])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=THEME_DIVERGING)
        ax.set_yticks(range(len(pivot.index)),
                      [H2_LABELS[l].replace("$-$", "-") for l in pivot.index])
        ax.set_xticks(range(len(pivot.columns)),
                      [f"B{int(b) + 1}" for b in pivot.columns])
        ax.set_xlabel("Realized S2 fill-rate bin")
        fig.colorbar(im, ax=ax, label="Net-alpha differential (bps)")
        fname = _save_fig(fig, out_dir, "fig_h2_heatmap")
        snippets["fig_h2_heatmap"] = jof_figure(
            graphics_file=fname,
            caption="Signal Contributions by Fill-Rate Bin",
            label="fig:h2-signal-heatmap-template",
            legend=(
                "This figure maps the matched net-alpha differentials of the S3 "
                "variants against S2 across realized fill-rate bins. Cell values "
                "are in basis points and correspond to the within-bin estimates "
                f"underlying the H2 table.{draft_note}"
            ),
            provenance=provenance,
        )

    # Rolling stability (only when the panel spans enough calendar time).
    # Uses the shared overlapping six-month windows with two-way clustered
    # standard errors (rolling_window_panel); only complete windows enter.
    dates = pd.to_datetime(panel["date"], errors="coerce")
    span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
    if span_days >= 360:
        sub = panel[panel["strategy"] == "S3_FULL"].copy()
        rolling = rolling_window_panel(sub, alpha_col="net_alpha_vs_moc_bps")
        if not rolling.empty:
            rolling.to_csv(out_dir / "rolling_stability.csv", index=False)
            x = pd.to_datetime(rolling["window_end"])
            mean = rolling["mean_alpha"].to_numpy(dtype=float)
            half_ci = 1.96 * rolling["clustered_se"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, mean, color=THEME["purple"])
            ax.fill_between(
                x, mean - half_ci, mean + half_ci,
                color=THEME["purple"], alpha=FILL_ALPHA, linewidth=0,
            )
            ax.axhline(0.0, color=THEME["gray"], linewidth=0.8)
            ax.set_ylabel("Rolling six-month net alpha vs.\nMOC (bps)")
            fname = _save_fig(fig, out_dir, "fig_rolling_stability")
            snippets["fig_rolling_stability"] = jof_figure(
                graphics_file=fname,
                caption="Within-Sample Stability of Net Execution Alpha",
                label="fig:rolling-stability-template",
                legend=(
                    "This figure plots overlapping six-month panel means of the "
                    "S3-full net-alpha differential against MOC, stepped by one "
                    "month and dated at the window end. The shaded band is the "
                    "95 percent confidence interval from two-way clustered "
                    f"standard errors. The sample covers {span}.{draft_note}"
                ),
                provenance=provenance,
            )
    else:
        log.info("rolling stability figure skipped: span %d days < 360", span_days)

    return snippets


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render(
    run_root: Path,
    out_dir: Path | None = None,
    *,
    compare_runs: dict[str, Path] | None = None,
    allow_incomplete: bool = False,
) -> dict:
    """Render all thesis exports for one run bundle. Returns the manifest."""
    bundle = RunBundle(run_root)
    problems, fingerprint, feature_policy = bundle.validate()
    draft = bool(problems)
    if draft and not allow_incomplete:
        raise RunNotValidatedError(
            "Run bundle failed the thesis acceptance gate:\n  - "
            + "\n  - ".join(problems)
            + "\nUse --allow-incomplete to render DRAFT output."
        )
    if draft:
        log.warning("Rendering DRAFT output despite gate failures: %s", problems)

    out_dir = Path(out_dir) if out_dir else bundle.root / "thesis_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_draft" if draft else ""
    draft_note = " DRAFT: this run is not validated and must not enter the thesis." if draft else ""
    # Snippets stay free of generator comments so the thesis source remains
    # clean; full provenance (run root, fingerprint, feature policy, input
    # hashes) lives in manifest.json and the exports README instead.
    provenance = ""

    headline_spec = "tape_replay_queue"
    config_path = bundle.path("metadata/run_config.json")
    if config_path.exists():
        headline_spec = _read_json(config_path).get("fill_specification", headline_spec)

    outputs: dict[str, str] = {}
    skipped: dict[str, str] = {}

    builders = {
        "tab_h1_primary": lambda: build_h1_table(bundle, provenance, draft_note),
        "tab_h2_pooled": lambda: build_h2_table(bundle, provenance, draft_note),
        "tab_h3_raear": lambda: build_h3_table(bundle, provenance, draft_note),
        "tab_h1_tier_subgroup": lambda: build_tier_subgroup_table(bundle, provenance, draft_note),
        "tab_fill_robustness": lambda: build_fill_robustness_table(
            bundle, compare_runs or {}, headline_spec, provenance, draft_note,
        ),
        "tab_side_split_robustness": lambda: build_side_split_robustness_table(
            bundle, provenance, draft_note,
        ),
        "tab_parent_window_robustness": lambda: build_size_window_robustness_table(
            bundle, compare_runs or {}, provenance, draft_note,
        ),
    }
    for name, builder in builders.items():
        try:
            tex = builder()
        except Exception as exc:
            skipped[name] = str(exc)
            log.warning("%s skipped: %s", name, exc)
            continue
        if tex is None:
            skipped[name] = "source artifact not present"
            continue
        path = out_dir / f"{name}{suffix}.tex"
        path.write_text(tex, encoding="utf-8")
        outputs[name] = path.name

    try:
        figure_snippets = build_figures(
            bundle, out_dir, provenance, draft_note,
            compare_runs=compare_runs or {}, headline_spec=headline_spec,
        )
        for name, tex in figure_snippets.items():
            path = out_dir / f"{name}{suffix}.tex"
            path.write_text(tex, encoding="utf-8")
            outputs[name] = path.name
    except Exception as exc:
        skipped["figures"] = str(exc)
        log.warning("figure rendering failed: %s", exc)

    label_map = {
        "tab_h1_primary": "tab:h1-primary-template",
        "tab_h2_pooled": "tab:h2-pooled-template",
        "tab_h3_raear": "tab:h3-raear-template",
        "tab_h1_tier_subgroup": "tab:h1-tier-subgroup (new subsection table)",
        "tab_fill_robustness": "tab:fill-robustness-template",
        "tab_side_split_robustness": "tab:side-split-robustness (new robustness table)",
        "tab_parent_window_robustness": "tab:parent-window-robustness-template",
        "fig_alpha_decomposition": "fig:alpha-decomposition-template",
        "fig_alpha_fill_frontier": "fig:alpha-fill-frontier-template",
        "fig_fill_spec_frontier": "fig:fill-spec-frontier-template",
        "fig_raear_curve": "fig:raear-curve-template",
        "fig_h2_heatmap": "fig:h2-signal-heatmap-template",
        "fig_rolling_stability": "fig:rolling-stability-template",
    }
    readme_lines = [
        "# Thesis exports",
        "",
        "Copy each snippet over the thesis placeholder carrying the same label.",
        "Figure snippets expect the PDF next to the thesis file or on the",
        "graphics path. DRAFT-suffixed files must never enter the thesis.",
        "",
        "Provenance (kept out of the snippets so the thesis source stays clean):",
        "",
        f"- run root: `{bundle.root}`",
        f"- simulation fingerprint: `{fingerprint}`",
        f"- feature policy: `{feature_policy}`",
        f"- draft: `{draft}`",
        "",
        "| Snippet | Replaces thesis label |",
        "| --- | --- |",
    ]
    for name, filename in outputs.items():
        readme_lines.append(f"| `{filename}` | `{label_map.get(name, name)}` |")
    (out_dir / "README_thesis_exports.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8",
    )

    manifest = {
        "run_root": str(bundle.root),
        "simulation_fingerprint": fingerprint,
        "feature_policy": feature_policy,
        "draft": draft,
        "gate_problems": problems,
        "inputs_sha256": bundle.inputs,
        "outputs": outputs,
        "skipped": skipped,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    log.info("Thesis exports written to %s (%d snippets, %d skipped)",
             out_dir, len(outputs), len(skipped))
    return manifest


def _parse_compare(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--compare-run expects <spec>=<run-root>, got {item!r}")
        spec, root = item.split("=", 1)
        out[spec.strip()] = Path(root.strip())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--compare-run", action="append", default=None,
                   help="Additional fill-spec runs as <spec>=<run-root>; repeatable")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Render DRAFT output even when the acceptance gate fails")
    args = p.parse_args()
    try:
        render(
            args.run_root, args.out,
            compare_runs=_parse_compare(args.compare_run),
            allow_incomplete=args.allow_incomplete,
        )
    except RunNotValidatedError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
