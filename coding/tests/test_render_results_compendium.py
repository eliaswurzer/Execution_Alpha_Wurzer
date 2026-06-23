"""Tests for the standalone results compendium renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.runners import render_results_compendium as rrc


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_table(label: str) -> str:
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Synthetic {label}}}",
        rf"\label{{{label}}}",
        r"\joflegend{Synthetic table legend.}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        r"A & 1 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\jofnotes{Synthetic notes.}",
        r"\end{table}",
        "",
    ])


def _minimal_figure(name: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.5\textwidth]{{{name}.pdf}}",
        rf"\caption{{Synthetic {name}}}",
        rf"\label{{fig:{name}}}",
        r"\jofnotes{Synthetic figure notes.}",
        r"\end{figure}",
        "",
    ])


def _make_run(root: Path, *, complete: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_status.json").write_text(json.dumps({
        "status": "complete" if complete else "running",
        "updated_at": "2026-06-11T21:28:00",
        "simulation": {
            "fingerprint": "fp123",
            "dates_expected": 2,
            "dates_with_valid_shards": 2,
            "eligible_coverage": 0.999,
            "critical_failures": 0,
            "failure_reason_counts": {
                "missing_expected_vc": 1,
                "empty_after_filter": 1,
            },
            "tier_fallback_symbols": 3,
            "adv_spread_bucket_map_path": None,
        },
    }), encoding="utf-8")

    _write_text(root / "metadata" / "run_config.json", json.dumps({
        "run_id": "synthetic",
        "start": "2018-07-02",
        "end": "2018-07-03",
        "fill_specification": "tape_replay_queue",
    }))
    _write_csv(root / "metadata" / "simulation_manifest.csv", [
        {"date": "2018-07-02", "status": "complete"},
        {"date": "2018-07-03", "status": "partial"},
    ])
    _write_csv(root / "metadata" / "simulation_failures.csv", [
        {"date": "2018-07-03", "symbol": "XYZ", "reason": "missing_expected_vc"},
        {"date": "2018-07-03", "symbol": "ABC", "reason": "empty_after_filter"},
    ])

    _write_csv(root / "hypotheses" / "h1" / "h1_primary_ttest.csv", [{
        "mean": -1.23, "se": 0.1, "t": -12.3, "p_value": 0.0,
        "n": 100, "label": "primary:S3_FULL-S0_MOC:B:0.01",
    }])
    _write_csv(root / "hypotheses" / "h1" / "h1_tev.csv", [{
        "strategy": "S3_FULL", "mean_alpha": -1.2, "tev": 9.0, "n": 100,
    }])
    _write_csv(root / "hypotheses" / "h2" / "h2_pooled.csv", [
        {"label": "OFI_marginal", "mean": -0.02, "se_twoway": 0.01, "t": -2.0, "n": 100},
        {"label": "IMB_marginal", "mean": -0.01, "se_twoway": 0.01, "t": -1.0, "n": 100},
        {"label": "FULL_vs_S2", "mean": -0.03, "se_twoway": 0.01, "t": -3.0, "n": 100},
        {"label": "interaction", "mean": 0.01, "se_twoway": 0.01, "t": 1.0, "n": 100},
    ])
    _write_csv(root / "hypotheses" / "h2" / "h2_per_bin_differentials.csv", [{
        "label": "OFI_marginal", "bin": 0, "mean": 0.1,
        "se_twoway": 0.01, "t": 10.0, "n": 10,
    }])
    _write_csv(root / "hypotheses" / "h3" / "h3_raear.csv", [
        {"strategy": "S0_MOC", "mean_alpha": -0.1, "tev": 0.0, "tes": 0.0, "ir": 0.0,
         "raear_eta_0.01": -0.1, "raear_eta_0.05": -0.1, "eta_star": 0.0},
        {"strategy": "S3_FULL", "mean_alpha": -1.1, "tev": 10.0, "tes": 3.16, "ir": -0.3,
         "raear_eta_0.01": -1.2, "raear_eta_0.05": -1.6, "eta_star": -0.1},
    ])
    _write_csv(root / "hypotheses" / "h3" / "h3_tev.csv", [{
        "strategy": "S3_FULL", "mean_alpha": -1.1, "tev": 10.0, "n": 100,
        "te_port_indep": 0.5, "te_port_perf_corr": 3.16,
    }])

    export = root / "thesis_exports"
    _write_text(export / "manifest.json", json.dumps({
        "draft": False,
        "feature_policy": "causal_features_v2",
        "simulation_fingerprint": "fp123",
    }))
    snippets = {
        "tab_h1_primary.tex": _minimal_table("tab:h1-primary-template"),
        "tab_h2_pooled.tex": _minimal_table("tab:h2-pooled-template"),
        "tab_h3_raear.tex": _minimal_table("tab:h3-raear-template"),
        "tab_h1_tier_subgroup.tex": _minimal_table("tab:h1-tier-subgroup"),
        "tab_fill_robustness.tex": _minimal_table("tab:fill-robustness-template"),
        "fig_alpha_decomposition.tex": _minimal_figure("fig_alpha_decomposition"),
        "fig_alpha_fill_frontier.tex": _minimal_figure("fig_alpha_fill_frontier"),
        "fig_h2_heatmap.tex": _minimal_figure("fig_h2_heatmap"),
        "fig_raear_curve.tex": _minimal_figure("fig_raear_curve"),
        "fig_rolling_stability.tex": _minimal_figure("fig_rolling_stability"),
    }
    for name, text in snippets.items():
        _write_text(export / name, text)
    for name in (*rrc.REQUIRED_FIGURES, *rrc.OPTIONAL_FIGURES):
        (export / name).write_bytes(b"%PDF-1.4\n% synthetic\n")
    return root


@pytest.mark.unit
def test_results_compendium_writes_tex_manifest_and_readme(artifact_dir: Path) -> None:
    run = _make_run(artifact_dir / "run")
    manifest = rrc.render(run)
    out = run / "results_compendium"

    assert manifest["draft"] is False
    assert (out / "results_compendium.tex").exists()
    assert (out / "results_compendium_manifest.json").exists()
    assert (out / "README_results_compendium.md").exists()
    assert (out / "fig_alpha_decomposition.pdf").exists()
    assert (out / "fig_alpha_decomposition.tex").exists()

    tex = (out / "results_compendium.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\joflegend}" in tex
    assert r"\newcommand{\jofnotes}" in tex
    assert r"\newcommand{\jofstars}" in tex
    assert r"\newcommand{\jofpanel}" in tex
    assert r"\input{tab_h1_primary.tex}" in tex
    assert "The primary paired test compares S3 Full" in tex
    assert r"missing\_expected\_vc" in tex


@pytest.mark.unit
def test_results_compendium_rejects_incomplete_run(artifact_dir: Path) -> None:
    run = _make_run(artifact_dir / "run", complete=False)
    with pytest.raises(rrc.ResultsCompendiumError, match="status"):
        rrc.render(run)


@pytest.mark.unit
def test_results_compendium_reports_missing_required_file(artifact_dir: Path) -> None:
    run = _make_run(artifact_dir / "run")
    (run / "hypotheses" / "h2" / "h2_pooled.csv").unlink()
    with pytest.raises(rrc.ResultsCompendiumError, match="h2_pooled"):
        rrc.render(run)
