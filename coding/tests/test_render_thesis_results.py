"""Tests for the thesis-results renderer (run bundle -> JoF LaTeX snippets)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.reporting import jof_latex
from analysis.runners import render_thesis_results as rtr


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fmt_num_and_stars() -> None:
    assert jof_latex.fmt_num(-1.234) == "$-$1.23"
    assert jof_latex.fmt_num(float("nan")) == "--"
    assert jof_latex.fmt_int(12345.4) == "12,345"
    assert jof_latex.stars(0.004) == r"$^{***}$"
    assert jof_latex.stars(0.04) == r"$^{**}$"
    assert jof_latex.stars(0.09) == r"$^{*}$"
    assert jof_latex.stars(0.2) == ""
    assert jof_latex.with_stars(2.5, 0.001) == r"2.50$^{***}$"
    assert jof_latex.paren_t(-2.1) == "($-$2.10)"


@pytest.mark.unit
def test_escape_latex() -> None:
    assert jof_latex.escape_latex("a_b & 5%") == r"a\_b \& 5\%"


# ---------------------------------------------------------------------------
# Synthetic run bundle
# ---------------------------------------------------------------------------

def _make_bundle(root: Path, *, complete: bool = True,
                 feature_policy: str | None = None,
                 simulation_source_sha256: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "fill_model_artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "calibration_manifest.json").write_text(json.dumps({
        "status": "complete",
        "feature_policy": feature_policy or cfg.FEATURE_POLICY_VERSION,
    }), encoding="utf-8")

    meta = root / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "simulation_config.json").write_text(json.dumps({
        "fingerprint": "fp123",
        "simulation_source_sha256": (
            simulation_source_sha256 or rtr._current_simulation_source_sha256()
        ),
    }), encoding="utf-8")
    (meta / "run_config.json").write_text(json.dumps({
        "fill_specification": "tape_replay_queue",
        "artifacts": str(artifacts),
    }), encoding="utf-8")

    (root / "run_status.json").write_text(json.dumps({
        "status": "complete" if complete else "running",
        "simulation": {"fingerprint": "fp123"},
    }), encoding="utf-8")

    rng = np.random.default_rng(0)
    strategies = ["S0_MOC", "S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL"]
    rows = []
    for d in range(6):
        date = dt.date(2018, 7, 2) + dt.timedelta(days=d)
        for sym in ("AAPL", "MSFT"):
            for side in ("BUY", "SELL"):
                for strat in strategies:
                    rows.append({
                        "order_id": f"{date}_{sym}_B_01_{side}",
                        "symbol": sym,
                        "date": date,
                        "side": side,
                        "strategy": strat,
                        "alpha_bps": float(rng.normal(1.0, 0.5)),
                        "net_alpha_bps": float(rng.normal(0.5, 0.5)),
                        "net_alpha_vs_moc_bps": float(rng.normal(0.4, 0.5)),
                        "fill_rate": float(rng.uniform(0.6, 1.0)),
                        "adverse_selection_bps": float(rng.normal(-1.0, 0.5)),
                        "adverse_selection_cost_bps": float(rng.uniform(0, 2)),
                        "impact_bps": 0.0,
                        "window": "B",
                        "size_frac": 0.01,
                    })
    panel = pd.DataFrame(rows)

    for name in ("h1", "h2", "h3"):
        hdir = root / "hypotheses" / name
        hdir.mkdir(parents=True, exist_ok=True)
        (hdir / "status.json").write_text(json.dumps({
            "status": "complete" if complete else "running",
            "simulation_fingerprint": "fp123",
        }), encoding="utf-8")

    panel.to_parquet(root / "hypotheses" / "h1" / "h1_panel.parquet", index=False)
    pd.DataFrame([{
        "mean": 1.234, "se": 0.4, "t": 3.085, "p_value": 0.002, "n": 120,
        "label": "primary:S3_FULL-S0_MOC:B:0.01",
    }]).to_csv(root / "hypotheses" / "h1" / "h1_primary_ttest.csv", index=False)
    pd.DataFrame([
        {"group": "tier", "level": 1, "mean": 0.8, "se": 0.5, "t": 1.6,
         "p_value": 0.11, "n": 40, "p_holm": 0.22},
        {"group": "tier", "level": 2, "mean": 1.4, "se": 0.5, "t": 2.8,
         "p_value": 0.005, "n": 40, "p_holm": 0.015},
    ]).to_csv(root / "hypotheses" / "h1" / "h1_subgroup_tier.csv", index=False)

    pooled = pd.DataFrame([
        {"label": "OFI_marginal", "mean": 0.30, "se_twoway": 0.10, "t": 3.0, "n": 100},
        {"label": "IMB_marginal", "mean": -0.05, "se_twoway": 0.10, "t": -0.5, "n": 100},
        {"label": "FULL_vs_S2", "mean": 0.28, "se_twoway": 0.12, "t": 2.33, "n": 100},
        {"label": "interaction", "mean": 0.03, "se_twoway": 0.08, "t": 0.38, "n": 100},
    ])
    pooled["matching_metric"] = "realized_passive_fill_rate"
    pooled.to_csv(root / "hypotheses" / "h2" / "h2_pooled.csv", index=False)
    per_bin = []
    for b in range(5):
        for label in ("OFI_marginal", "IMB_marginal", "FULL_vs_S2", "interaction"):
            per_bin.append({
                "label": label, "bin": b, "mean": 0.1 * (b + 1),
                "se_twoway": 0.1, "t": b + 0.5, "n": 20,
                "matching_metric": "realized_passive_fill_rate",
            })
    pd.DataFrame(per_bin).to_csv(
        root / "hypotheses" / "h2" / "h2_per_bin_differentials.csv", index=False,
    )

    raear = pd.DataFrame([
        {"strategy": s, "mean_alpha": 0.5 + i * 0.1, "tev": 4.0 + i,
         "tes": float(np.sqrt(4.0 + i)), "ir": 0.2,
         "raear_eta_0.01": 0.4, "raear_eta_0.05": 0.2, "eta_star": 0.12}
        for i, s in enumerate(["S0_MOC", "S2_TIME_ADAPTIVE", "S3_FULL"])
    ])
    raear.to_csv(root / "hypotheses" / "h3" / "h3_raear.csv", index=False)
    tev = raear[["strategy", "mean_alpha", "tev"]].copy()
    tev["n"] = 120
    tev["te_port_indep"] = 0.2
    tev["te_port_perf_corr"] = 2.0
    tev.to_csv(root / "hypotheses" / "h3" / "h3_tev.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gate_rejects_incomplete_bundle(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle", complete=False)
    with pytest.raises(rtr.RunNotValidatedError):
        rtr.render(bundle)


@pytest.mark.unit
def test_gate_rejects_feature_policy_mismatch(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle", feature_policy="causal_features_v0")
    with pytest.raises(rtr.RunNotValidatedError, match="feature policy"):
        rtr.render(bundle)


@pytest.mark.unit
def test_gate_rejects_stale_simulation_source_hash(artifact_dir) -> None:
    bundle = _make_bundle(
        artifact_dir / "bundle",
        simulation_source_sha256="stale-source-hash",
    )
    with pytest.raises(rtr.RunNotValidatedError, match="simulation source hash mismatch"):
        rtr.render(bundle)

    manifest = rtr.render(bundle, allow_incomplete=True)
    assert manifest["draft"] is True
    assert any("simulation source hash mismatch" in p for p in manifest["gate_problems"])


@pytest.mark.unit
def test_allow_incomplete_renders_draft(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle", complete=False)
    manifest = rtr.render(bundle, allow_incomplete=True)
    assert manifest["draft"] is True
    out = bundle / "thesis_exports"
    draft_files = list(out.glob("*_draft.tex"))
    assert draft_files, "expected draft-suffixed snippets"
    text = (out / "tab_h1_primary_draft.tex").read_text(encoding="utf-8")
    assert "DRAFT" in text


# ---------------------------------------------------------------------------
# Rendering content
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_render_complete_bundle_outputs(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle")
    manifest = rtr.render(bundle)
    out = bundle / "thesis_exports"

    assert manifest["draft"] is False
    assert manifest["simulation_fingerprint"] == "fp123"
    for name in ("tab_h1_primary", "tab_h2_pooled", "tab_h3_raear",
                 "tab_h1_tier_subgroup", "tab_fill_robustness",
                 "tab_side_split_robustness",
                 "fig_alpha_decomposition", "fig_raear_curve"):
        assert name in manifest["outputs"], f"missing output {name}"
        assert (out / manifest["outputs"][name]).exists()

    h1 = (out / "tab_h1_primary.tex").read_text(encoding="utf-8")
    assert r"\label{tab:h1-primary-template}" in h1
    assert r"1.23$^{***}$" in h1            # mean with stars from p=0.002
    assert "(3.08)" in h1                   # t-stat line (3.085 floors in binary)
    assert r"\jofpanel{6}{Panel B: Strategy-level diagnostics}" in h1
    assert "S3 Full" in h1
    # Snippets stay free of generator comments; provenance lives in the
    # manifest and the exports README instead.
    assert "%" not in h1.split(r"\begin{table}")[0]
    assert "fp123" not in h1
    readme = (out / "README_thesis_exports.md").read_text(encoding="utf-8")
    assert "fp123" in readme

    h2 = (out / "tab_h2_pooled.tex").read_text(encoding="utf-8")
    assert r"\label{tab:h2-pooled-template}" in h2
    assert "S3 OFI $-$ S2" in h2
    assert "B1" in h2 and "B5" in h2        # five bins rendered
    assert r"realized\_passive\_fill\_rate" in h2  # escaped metric name

    h3 = (out / "tab_h3_raear.tex").read_text(encoding="utf-8")
    assert r"RAEAR$_{0.01}$" in h3 and r"RAEAR$_{0.05}$" in h3
    assert "S0 MOC" in h3

    rob = (out / "tab_fill_robustness.tex").read_text(encoding="utf-8")
    assert "Queue-aware replay (headline)" in rob
    assert "Strictly-through replay (lower bound)" in rob
    # No compare runs supplied: bounds rows are dashes and the notes say so.
    assert "await the corresponding robustness run" in rob

    side = (out / "tab_side_split_robustness.tex").read_text(encoding="utf-8")
    assert r"\label{tab:side-split-robustness}" in side
    assert "Buy parent orders" in side
    assert "Sell parent orders" in side
    assert "Buy $-$ Sell contrast" in side

    thesis_text = "\n".join([h1, h2, h3, rob, side])
    for internal_token in ("hypotheses/", r"\texttt{", ".csv", ".parquet", "runner"):
        assert internal_token not in thesis_text

    assert (out / "fig_alpha_decomposition.pdf").exists()
    assert (out / "fig_raear_curve.pdf").exists()
    assert (out / "README_thesis_exports.md").exists()
    assert (out / "manifest.json").exists()


@pytest.mark.unit
def test_h1_primary_panel_b_uses_primary_cell(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle")
    h1_dir = bundle / "hypotheses" / "h1"
    panel = pd.DataFrame([
        {
            "order_id": "2018-07-02_AAPL_A_01_BUY",
            "symbol": "AAPL",
            "date": dt.date(2018, 7, 2),
            "strategy": "S0_MOC",
            "alpha_bps": 0.0,
            "net_alpha_bps": -0.10,
            "moc_net_alpha_bps": -0.10,
            "net_alpha_vs_moc_bps": 0.0,
            "fill_rate": 0.0,
            "adverse_selection_bps": 0.0,
            "window": "A",
            "size_frac": 0.01,
        },
        {
            "order_id": "2018-07-02_AAPL_A_01_BUY",
            "symbol": "AAPL",
            "date": dt.date(2018, 7, 2),
            "strategy": "S3_FULL",
            "alpha_bps": 10.0,
            "net_alpha_bps": 10.00,
            "moc_net_alpha_bps": -0.10,
            "net_alpha_vs_moc_bps": 10.10,
            "fill_rate": 1.0,
            "adverse_selection_bps": -2.0,
            "window": "A",
            "size_frac": 0.01,
        },
        {
            "order_id": "2018-07-02_AAPL_B_01_BUY",
            "symbol": "AAPL",
            "date": dt.date(2018, 7, 2),
            "strategy": "S0_MOC",
            "alpha_bps": 0.0,
            "net_alpha_bps": -0.10,
            "moc_net_alpha_bps": -0.10,
            "net_alpha_vs_moc_bps": 0.0,
            "fill_rate": 0.0,
            "adverse_selection_bps": 0.0,
            "window": cfg.PRIMARY_WINDOW,
            "size_frac": cfg.PARENT_ORDER_PRIMARY_FRACTION,
        },
        {
            "order_id": "2018-07-02_AAPL_B_01_BUY",
            "symbol": "AAPL",
            "date": dt.date(2018, 7, 2),
            "strategy": "S3_FULL",
            "alpha_bps": -0.23,
            "net_alpha_bps": -0.22,
            "moc_net_alpha_bps": -0.10,
            "net_alpha_vs_moc_bps": -0.12,
            "fill_rate": 0.6,
            "adverse_selection_bps": -1.2,
            "window": cfg.PRIMARY_WINDOW,
            "size_frac": cfg.PARENT_ORDER_PRIMARY_FRACTION,
        },
    ])
    panel.to_parquet(h1_dir / "h1_panel.parquet", index=False)
    pd.DataFrame([{
        "mean": -0.12,
        "se": 0.08,
        "t": -1.50,
        "p_value": 0.135,
        "n": 1,
        "label": "primary:S3_FULL-S0_MOC:B:0.01",
    }]).to_csv(h1_dir / "h1_primary_ttest.csv", index=False)

    h1 = rtr.build_h1_table(rtr.RunBundle(bundle), "", "")
    s3_line = next(line for line in h1.splitlines() if line.startswith("S3 Full"))

    assert "$-$0.22" in s3_line
    assert "$-$0.12" in s3_line
    assert "10.00" not in s3_line
    assert "same Window~B, one-percent parent-size cell" in h1


@pytest.mark.unit
def test_compare_run_fills_bracket_row(artifact_dir) -> None:
    headline = _make_bundle(artifact_dir / "headline")
    strict = _make_bundle(artifact_dir / "strict")
    manifest = rtr.render(
        headline, compare_runs={"tape_replay_strict": strict},
    )
    rob = (headline / "thesis_exports" / "tab_fill_robustness.tex").read_text(
        encoding="utf-8",
    )
    strict_line = next(
        line for line in rob.splitlines()
        if line.startswith("Strictly-through replay")
    )
    assert "--" not in strict_line.split("&")[1]
    assert any(k.startswith("compare:tape_replay_strict:") for k in manifest["inputs_sha256"])
    # Significance stars sit on the net-alpha differential (synthetic primary
    # p_value = 0.002 -> ***) and the stars note is emitted for this table.
    assert r"$^{***}$" in strict_line.split("&")[1]
    headline_line = next(
        line for line in rob.splitlines()
        if line.startswith("Queue-aware replay")
    )
    assert r"$^{***}$" in headline_line.split("&")[1]
    assert r"\jofstars" in rob


@pytest.mark.unit
def test_fill_spec_frontier_figure_includes_model_specs(artifact_dir) -> None:
    headline = _make_bundle(artifact_dir / "headline")
    km = _make_bundle(artifact_dir / "km")
    xgb = _make_bundle(artifact_dir / "xgb")
    manifest = rtr.render(headline, compare_runs={"km": km, "xgb": xgb})
    out = headline / "thesis_exports"

    assert "fig_fill_spec_frontier" in manifest["outputs"]
    assert (out / "fig_fill_spec_frontier.pdf").exists()
    frontier = (out / "fig_fill_spec_frontier.tex").read_text(encoding="utf-8")
    assert r"\label{fig:fill-spec-frontier-template}" in frontier
    # The legend must carry the honest evaluation framing.
    assert "optimistic bound" in frontier
    # Model-based specs are flagged with an explained asterisk.
    assert "asterisk" in frontier


@pytest.mark.unit
def test_rolling_stability_skipped_for_short_span(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle")
    manifest = rtr.render(bundle)
    # six trading days of data -> rolling figure must not be produced
    assert "fig_rolling_stability" not in manifest["outputs"]


@pytest.mark.unit
def test_size_window_table_renders_with_and_without_grid_run(artifact_dir) -> None:
    bundle = _make_bundle(artifact_dir / "bundle")
    manifest = rtr.render(bundle)
    tex_path = bundle / "thesis_exports" / "tab_parent_window_robustness.tex"
    tex = tex_path.read_text(encoding="utf-8")
    assert r"\label{tab:parent-window-robustness-template}" in tex
    # No grid run supplied: Panel A rows are dashes, Window B from own panel.
    assert "await the parent-size-grid run" in tex
    headline_row = next(
        line for line in tex.splitlines() if line.startswith("Window B")
    )
    assert "--" not in headline_row.split("&")[1]
    assert "tab_parent_window_robustness" in manifest["outputs"]

    # With a size-grid compare run, Panel A rows are populated.
    grid_root = artifact_dir / "size_grid"
    grid_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"size_bucket": s, "strategy": "S3_FULL", "mean_net_alpha_bps": -1.0,
         "se_twoway": 0.1, "t": -10.0, "mean_fill_rate": 0.8,
         "mean_as_cost_bps": 1.5, "tev": 300.0, "n": 1000}
        for s in (0.005, 0.01, 0.02, 0.05, 0.10)
    ]).to_csv(grid_root / "size_table_summary.csv", index=False)
    manifest = rtr.render(bundle, compare_runs={"size_grid": grid_root})
    tex = tex_path.read_text(encoding="utf-8")
    assert "await the parent-size-grid run" not in tex
    five_pct = next(line for line in tex.splitlines() if line.startswith(r"5.0\%"))
    assert "$-$1.00" in five_pct
    assert any(k.startswith("compare:size_grid:") for k in manifest["inputs_sha256"])


@pytest.mark.unit
def test_size_panel_prefers_moc_relative_differential(artifact_dir) -> None:
    """When the size grid ships the clustered MOC-relative summary, Panel A must
    report that differential rather than the raw net alpha (which embeds the
    commission the MOC benchmark also pays)."""
    bundle = _make_bundle(artifact_dir / "bundle")
    grid_root = artifact_dir / "size_grid"
    grid_root.mkdir(parents=True, exist_ok=True)
    # Raw net alpha is -1.00 (would round to -1.00); the MOC-relative
    # differential is a clearly distinct -0.40 with an insignificant t.
    pd.DataFrame([
        {"size_bucket": s, "strategy": "S3_FULL", "mean_net_alpha_bps": -1.0,
         "se_twoway": 0.1, "t": -10.0, "mean_fill_rate": 0.8,
         "mean_as_markout_bps": 1.5, "tev": 300.0, "n": 1000}
        for s in (0.005, 0.01, 0.02, 0.05, 0.10)
    ]).to_csv(grid_root / "size_table_summary.csv", index=False)
    pd.DataFrame([
        {"size_bucket": s, "strategy": "S3_FULL",
         "metric": "net_alpha_vs_moc_bps", "mean": -0.40,
         "se_twoway": 0.5, "t": -0.80, "n": 1000}
        for s in (0.005, 0.01, 0.02, 0.05, 0.10)
    ]).to_csv(grid_root / "robustness_summary_clustered.csv", index=False)

    rtr.render(bundle, compare_runs={"size_grid": grid_root})
    tex = (
        bundle / "thesis_exports" / "tab_parent_window_robustness.tex"
    ).read_text(encoding="utf-8")
    five_pct = next(line for line in tex.splitlines() if line.startswith(r"5.0\%"))
    # The MOC-relative differential is reported, not the raw -1.00, and the
    # insignificant t carries no stars.
    assert "$-$0.40" in five_pct
    assert "$-$1.00" not in five_pct
    assert "$^{*" not in five_pct


@pytest.mark.unit
def test_as_markout_diagnostics_signed_convention() -> None:
    panel = pd.DataFrame({
        "strategy": ["S1_STATIC"] * 3,
        "alpha_bps": [1.0, 2.0, 0.0],
        "net_alpha_bps": [1.0, 2.0, -0.1],
        "net_alpha_vs_moc_bps": [1.0, 2.0, 0.0],
        "fill_rate": [0.5, 1.0, 0.0],
        # Mixed favorable/adverse signed markouts; the no-fill row must be
        # excluded from the conditional mean.
        "adverse_selection_bps": [-2.0, 1.0, 5.0],
    })
    diag = rtr._panel_strategy_diagnostics(panel)
    row = diag.loc["S1_STATIC"]
    # Conditional signed mean over fills: (-2 + 1) / 2 = -0.5 -> markout +0.5.
    assert row["as_markout"] == pytest.approx(0.5)
    # Exact gross-identity component: mean(fill_rate * signed) over ALL rows.
    assert row["as_component"] == pytest.approx((0.5 * -2.0 + 1.0 * 1.0 + 0.0) / 3)
