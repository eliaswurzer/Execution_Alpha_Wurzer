from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.fill_model.rolling import build_monthly_training_schedule
from analysis.reporting.preliminary_templates import write_preliminary_templates
from analysis.runners import run_value_model_smoke as smoke


@pytest.mark.unit
def test_synthetic_candidate_panel_has_required_value_columns() -> None:
    panel = smoke.build_synthetic_candidate_panel(n_dates=70, symbols=("AAPL", "MSFT"))

    assert set(smoke.REQUIRED_CANDIDATE_COLUMNS).issubset(panel.columns)
    assert panel["side"].nunique() == 2
    assert panel["tier"].nunique() >= 2
    assert panel["sector"].nunique() >= 1
    assert panel["limit_offset_bps"].nunique() >= 4
    assert np.isfinite(panel["target_net_alpha_vs_moc_bps"]).all()
    assert panel["target_net_alpha_vs_moc_bps"].std() > 0


@pytest.mark.unit
def test_rolling_schedule_excludes_evaluation_dates() -> None:
    panel = smoke.build_synthetic_candidate_panel(n_dates=90, symbols=("AAPL", "MSFT"))
    dates = sorted(pd.to_datetime(panel["date"]).dt.date.unique())
    schedule = build_monthly_training_schedule(dates)
    trainable = schedule[schedule["status"] == "trainable"]

    assert not trainable.empty
    assert (pd.to_datetime(trainable["train_end"]).dt.date < pd.to_datetime(trainable["anchor_date"]).dt.date).all()


@pytest.mark.unit
def test_s5_dry_run_branches_are_explicit() -> None:
    pytest.importorskip("xgboost")
    panel = smoke.build_synthetic_candidate_panel(n_dates=40, symbols=("AAPL", "MSFT"))
    from analysis.fill_model.value_model import SideTieredXGBValueModel

    model = SideTieredXGBValueModel().fit_panel(
        panel,
        min_rows_global=20,
        min_rows_side=10,
        min_rows_side_tier=5,
        n_estimators=5,
    )
    decisions = smoke.s5_dry_run_decisions(model, offset_grid_bps=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0))

    positive = decisions.loc[decisions["case"] == "positive_stub"].iloc[0]
    nonpositive = decisions.loc[decisions["case"] == "nonpositive_stub"].iloc[0]
    assert positive["posted_passively"] == 1
    assert nonpositive["posted_passively"] == 0
    assert nonpositive["slice_qty"] == 0


@pytest.mark.integration
def test_synthetic_value_model_smoke_runner_writes_artifacts(artifact_dir: Path) -> None:
    pytest.importorskip("xgboost")

    manifest = smoke.run_smoke(
        mode="synthetic",
        out_dir=artifact_dir,
        symbols=["AAPL", "MSFT"],
        n_estimators=5,
        min_rows_global=20,
        min_rows_side=10,
        min_rows_side_tier=5,
    )

    assert manifest["status"] == "complete"
    for name in [
        "candidate_panel.parquet",
        "candidate_panel_summary.csv",
        "rolling_schedule.csv",
        "rolling_anchor_map.csv",
        "value_model_manifest.json",
        "s5_dry_run_orders.csv",
        "posting_curve_summary.csv",
        "posting_curve.png",
        "smoke_manifest.json",
    ]:
        path = artifact_dir / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    posting_curve = pd.read_csv(artifact_dir / "posting_curve_summary.csv")
    assert posting_curve["fill_probability"].between(0.0, 1.0).all()
    assert np.isfinite(posting_curve["mean_value_bps"]).all()
    dry_run = pd.read_csv(artifact_dir / "s5_dry_run_orders.csv")
    assert {"trained_model", "positive_stub", "nonpositive_stub"}.issubset(set(dry_run["case"]))
    report = artifact_dir / "reporting_templates" / "README_preliminary_reporting_template.md"
    tex = artifact_dir / "reporting_templates" / "latex" / "preliminary_results_tables.tex"
    assert report.exists() and report.read_text(encoding="utf-8").strip()
    assert tex.exists() and tex.read_text(encoding="utf-8").strip()
    assert "not-yet-validated artifacts" in report.read_text(encoding="utf-8")
    saved_manifest = json.loads((artifact_dir / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["value_model_keys"]


@pytest.mark.unit
def test_reporting_template_placeholders_do_not_fabricate_results(artifact_dir: Path) -> None:
    posting = pd.DataFrame({
        "side": ["BUY"],
        "tier": [1],
        "limit_offset_bps": [0.0],
        "n": [10],
        "fill_probability": [0.5],
        "mean_value_bps": [0.1],
        "median_value_bps": [0.1],
    })
    outputs = write_preliminary_templates(artifact_dir, posting_summary=posting)

    h1 = pd.read_csv(artifact_dir / "tables" / "h1_strategy_comparison.csv")
    curve = pd.read_csv(artifact_dir / "tables" / "static_posting_curve.csv")
    assert h1.empty
    assert not curve.empty
    assert "validated final-run artifacts" in Path(outputs["markdown"]).read_text(encoding="utf-8")


@pytest.mark.realdata
def test_realdata_value_model_smoke_is_opt_in(artifact_dir: Path) -> None:
    if os.environ.get("THESIS_ENABLE_REALDATA_TESTS") != "1":
        pytest.skip("real-data value-model smoke is opt-in")
    pytest.importorskip("xgboost")
    manifest = smoke.run_smoke(
        mode="realdata",
        out_dir=artifact_dir,
        symbols=["AAPL", "MSFT"],
        start=dt.date(2018, 2, 1),
        end=dt.date(2018, 2, 2),
        n_estimators=5,
        min_rows_global=10,
        min_rows_side=5,
        min_rows_side_tier=3,
    )
    assert manifest["status"] == "complete"
