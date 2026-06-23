from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import config as analysis_cfg
from analysis.runners import h1_performance_gap
from statistical_tests.economic_tests import fill_spec_summary, paired_vs_queue_tests
from statistical_tests.multiple_testing import holm_step_down
from statistical_tests import config as st_cfg
from statistical_tests import oos_calibration, run_all
from statistical_tests.oos_calibration import select_stratified_oos_dates
from statistical_tests.render_outputs import (
    render_fill_model_calibration,
    render_fill_robustness,
    render_h2_adjusted,
    write_outputs,
)


def test_holm_step_down_is_monotone_and_keeps_nan() -> None:
    adjusted = holm_step_down([0.04, np.nan, 0.01, 0.20, 0.80])

    assert math.isnan(adjusted[1])
    assert adjusted[2] == 0.04
    assert adjusted[0] == 0.12
    assert adjusted[3] == 0.40
    assert adjusted[4] == 0.80
    assert max(x for x in adjusted if not math.isnan(x)) <= 1.0


def test_oos_date_sampler_is_deterministic_by_quarter() -> None:
    dates = pd.date_range("2018-07-02", "2018-12-31", freq="B")

    first = select_stratified_oos_dates(dates, days_per_quarter=3)
    second = select_stratified_oos_dates(list(reversed(dates)), days_per_quarter=3)

    assert first == second
    assert len(first) == 6
    quarters = pd.PeriodIndex(pd.to_datetime(first), freq="Q")
    assert quarters.value_counts().sort_index().tolist() == [3, 3]


def test_oos_model_loader_respects_model_specs(monkeypatch, artifact_dir) -> None:
    for name in ("cox_tier_1.pkl", "km_tier_1.pkl", "xgb_tier_1.ubj"):
        (artifact_dir / name).write_bytes(b"present")

    loaded: list[str] = []

    class DummyLoader:
        @classmethod
        def load(cls, _path):
            loaded.append(cls.__name__)
            return cls()

    monkeypatch.setattr(oos_calibration, "TieredFillModel", type("CoxLoader", (DummyLoader,), {}))
    monkeypatch.setattr(oos_calibration, "TieredKMFillModel", type("KMLoader", (DummyLoader,), {}))
    monkeypatch.setattr(oos_calibration, "TieredXGBFillModel", type("XGBLoader", (DummyLoader,), {}))

    models = oos_calibration.load_models(artifact_dir, model_specs=["cox", "km"])

    assert set(models) == {"cox", "km"}
    assert "XGBLoader" not in loaded


def test_xgb_oos_scoring_uses_public_fill_probability_api() -> None:
    class PublicOnlyXGB:
        def fill_probability(self, horizon_seconds, frame):
            assert horizon_seconds == 30
            assert isinstance(frame, pd.DataFrame)
            return np.full(len(frame), 0.42)

    frame = pd.DataFrame({"limit_offset_bps": [0.0, 1.0, 2.0]})
    pred = oos_calibration._predict_for_tier("xgb", PublicOnlyXGB(), 30, frame)

    assert np.allclose(pred, 0.42)


def test_manifest_records_v4_model_family_and_input_hashes(monkeypatch, artifact_dir) -> None:
    queue = artifact_dir / "queue"
    cox = artifact_dir / "cox"
    fill = artifact_dir / "fill_model_v4_cox_km_xgb_logicfix_20260616"
    for root in (queue, cox):
        (root / "hypotheses" / "h1").mkdir(parents=True)
        (root / "metadata").mkdir(parents=True)
        (root / "run_status.json").write_text('{"status":"complete"}', encoding="utf-8")
        (root / "metadata" / "simulation_config.json").write_text('{"fingerprint":"fp"}', encoding="utf-8")
        (root / "hypotheses" / "h1" / "h1_panel.parquet").write_bytes(b"panel")
        (root / "hypotheses" / "h1" / "h1_primary_ttest.csv").write_text("x\n1\n", encoding="utf-8")
    (queue / "hypotheses" / "h2").mkdir(parents=True)
    (queue / "hypotheses" / "h2" / "h2_pooled.csv").write_text("label,t\nx,1\n", encoding="utf-8")
    fill.mkdir()
    (fill / "calibration_manifest.json").write_text(
        '{"status":"complete","feature_policy":"causal"}',
        encoding="utf-8",
    )
    (fill / "validation.csv").write_text("tier,n\n1,10\n", encoding="utf-8")
    (fill / "km_validation.csv").write_text("tier,n\n1,10\n", encoding="utf-8")

    monkeypatch.setattr(st_cfg, "FILL_SPEC_RUNS", {"tape_replay_queue": queue, "cox": cox})
    monkeypatch.setattr(st_cfg, "FILL_SPEC_ORDER", ["tape_replay_queue", "cox"])
    monkeypatch.setattr(st_cfg, "HEADLINE_RUN", queue)
    monkeypatch.setattr(st_cfg, "FILL_MODEL_DIR", fill)
    monkeypatch.setattr(st_cfg, "MODEL_SPECS", ["cox", "km"])

    manifest = run_all._manifest_base(
        SimpleNamespace(
            skip_oos=False,
            diagnostic_skip_oos=False,
            force_oos=True,
            days_per_quarter=10,
            event_sample_per_symbol_day=48,
            workers=8,
        ),
        selected_dates=[pd.Timestamp("2019-01-02").date()],
    )

    assert "xgb" not in manifest["fill_spec_runs"]
    assert "final_20260616_xgb_v4" not in json.dumps(manifest)
    assert manifest["model_specs"] == ["cox", "km"]
    assert manifest["output_policy"] == "final"
    assert manifest["settings"]["skip_oos"] is False
    assert manifest["settings"]["force_oos"] is True
    assert manifest["fill_model_dir"].endswith("fill_model_v4_cox_km_xgb_logicfix_20260616")
    assert manifest["h2_confirmatory_surface"] == "pooled_matched_differentials"
    assert manifest["h3_inference_role"] == "descriptive_risk_tradeoff"
    assert "tape_replay_queue:h1:h1_panel.parquet" in manifest["input_sha256"]
    assert "fill_model:km_validation.csv" in manifest["input_sha256"]


def test_skip_oos_requires_explicit_diagnostic_flag(artifact_dir) -> None:
    with pytest.raises(SystemExit):
        run_all.main(["--skip-oos", "--out-dir", str(artifact_dir)])


def _write_complete_gate_run(root: Path, calibration_root: Path) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "run_status.json").write_text(
        json.dumps({
            "status": "complete",
            "simulation": {
                "dates_expected": 2,
                "dates_with_valid_shards": 2,
                "critical_failures": 0,
            },
        }),
        encoding="utf-8",
    )
    (root / "metadata" / "simulation_config.json").write_text(
        json.dumps({
            "feature_policy": analysis_cfg.FEATURE_POLICY_VERSION,
            "trade_policy": analysis_cfg.TRADE_CONDITION_POLICY_VERSION,
        }),
        encoding="utf-8",
    )
    (root / "metadata" / "run_config.json").write_text(
        json.dumps({"artifacts": str(calibration_root)}),
        encoding="utf-8",
    )
    for hypothesis, files in run_all.HYPOTHESIS_REQUIRED.items():
        hdir = root / "hypotheses" / hypothesis
        hdir.mkdir(parents=True, exist_ok=True)
        (hdir / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
        for name in files:
            (hdir / name).write_bytes(b"nonempty")


def test_canonical_input_gate_requires_complete_current_runs(monkeypatch, artifact_dir) -> None:
    calibration_root = artifact_dir / "calibration"
    calibration_root.mkdir()
    (calibration_root / "calibration_manifest.json").write_text(
        json.dumps({
            "status": "complete",
            "feature_policy": analysis_cfg.FEATURE_POLICY_VERSION,
        }),
        encoding="utf-8",
    )
    queue = artifact_dir / "queue"
    strict = artifact_dir / "strict"
    for root in (queue, strict):
        _write_complete_gate_run(root, calibration_root)

    monkeypatch.setattr(st_cfg, "EXPECTED_EVAL_DATES", 2)
    monkeypatch.setattr(st_cfg, "FILL_SPEC_ORDER", ["tape_replay_queue", "tape_replay_strict"])
    monkeypatch.setattr(st_cfg, "FILL_SPEC_RUNS", {
        "tape_replay_queue": queue,
        "tape_replay_strict": strict,
    })
    monkeypatch.setattr(st_cfg, "FILL_MODEL_DIR", calibration_root)

    run_all.validate_canonical_inputs()

    (strict / "run_status.json").write_text(
        json.dumps({
            "status": "running",
            "simulation": {
                "dates_expected": 2,
                "dates_with_valid_shards": 1,
                "critical_failures": 0,
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="expected 2 valid shards"):
        run_all.validate_canonical_inputs()


def test_oos_cache_fingerprint_changes_with_sampling_policy() -> None:
    base = oos_calibration._cache_fingerprint(
        headline_run=Path("run_a"),
        h1_panel_sha256="abc",
        days_per_quarter=10,
        event_sample_per_symbol_day=48,
    )
    changed_sample = oos_calibration._cache_fingerprint(
        headline_run=Path("run_a"),
        h1_panel_sha256="abc",
        days_per_quarter=10,
        event_sample_per_symbol_day=24,
    )
    changed_panel = oos_calibration._cache_fingerprint(
        headline_run=Path("run_a"),
        h1_panel_sha256="def",
        days_per_quarter=10,
        event_sample_per_symbol_day=48,
    )

    assert base != changed_sample
    assert base != changed_panel


def test_oos_existing_shard_row_allows_resume(artifact_dir) -> None:
    shard_root = artifact_dir / "oos_event_shards"
    event = shard_root / "events" / "20190102" / "BRK_B.parquet"
    daily = shard_root / "daily" / "20190102" / "BRK_B.parquet"
    event.parent.mkdir(parents=True)
    daily.parent.mkdir(parents=True)
    event.write_bytes(b"event")
    daily.write_bytes(b"daily")

    row = oos_calibration._existing_oos_status_row(
        shard_root, pd.Timestamp("2019-01-02").date(), "BRK B",
    )

    assert row is not None
    assert row["status"] == "ok"
    assert row["reason"] == "existing_shard"
    assert row["event_path"].endswith("BRK_B.parquet")


def _write_h1_panel(root: Path, rows: list[dict]) -> None:
    path = root / "hypotheses" / "h1"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path / "h1_panel.parquet", index=False)


def test_economic_tests_match_alternative_runs_by_order_id() -> None:
    root = Path("repo/coding/artifacts/tests/statistical_tests_pairing")
    queue = root / "queue"
    strict = root / "strict"
    common = {
        "strategy": "S3_FULL",
        "window": "B",
        "size_frac": 0.01,
    }
    queue_rows = [
        {**common, "order_id": "a", "symbol": "AAA", "date": "2019-01-02", "net_alpha_vs_moc_bps": 0.0, "net_alpha_bps": -0.1, "fill_rate": 0.5, "adverse_selection_bps": -1.0},
        {**common, "order_id": "b", "symbol": "BBB", "date": "2019-01-03", "net_alpha_vs_moc_bps": 1.0, "net_alpha_bps": 0.9, "fill_rate": 0.6, "adverse_selection_bps": -2.0},
        {**common, "order_id": "c", "symbol": "CCC", "date": "2019-01-04", "net_alpha_vs_moc_bps": 0.0, "net_alpha_bps": -0.1, "fill_rate": 0.7, "adverse_selection_bps": -1.5},
    ]
    strict_rows = [
        {**common, "order_id": "c", "symbol": "CCC", "date": "2019-01-04", "net_alpha_vs_moc_bps": 2.0, "net_alpha_bps": 1.9, "fill_rate": 0.4, "adverse_selection_bps": -0.5},
        {**common, "order_id": "a", "symbol": "AAA", "date": "2019-01-02", "net_alpha_vs_moc_bps": 1.0, "net_alpha_bps": 0.9, "fill_rate": 0.3, "adverse_selection_bps": -0.2},
        {**common, "order_id": "b", "symbol": "BBB", "date": "2019-01-03", "net_alpha_vs_moc_bps": 1.0, "net_alpha_bps": 0.9, "fill_rate": 0.5, "adverse_selection_bps": -0.4},
    ]
    _write_h1_panel(queue, queue_rows)
    _write_h1_panel(strict, strict_rows)

    runs = {"tape_replay_queue": queue, "tape_replay_strict": strict}
    summary = fill_spec_summary(runs=runs)
    paired = paired_vs_queue_tests(runs=runs)

    assert set(summary["spec"]) == {"tape_replay_queue", "tape_replay_strict"}
    assert summary.loc[summary["spec"] == "tape_replay_queue", "is_headline"].item() is True

    alpha = paired[paired["metric"] == "net_alpha_vs_moc_bps"].iloc[0]
    fill = paired[paired["metric"] == "fill_rate"].iloc[0]
    markout = paired[paired["metric"] == "as_markout_bps"].iloc[0]
    assert alpha["n"] == 3
    assert np.isclose(alpha["mean_diff"], 1.0)
    assert np.isclose(fill["mean_diff"], -0.2)
    assert np.isclose(markout["mean_diff"], -1.1333333333333333)


def test_pairing_rejects_duplicate_symbol_date_keys(artifact_dir) -> None:
    queue = artifact_dir / "queue"
    strict = artifact_dir / "strict"
    common = {
        "strategy": "S3_FULL",
        "window": "B",
        "size_frac": 0.01,
        "net_alpha_vs_moc_bps": 0.0,
        "net_alpha_bps": 0.0,
        "fill_rate": 0.5,
        "adverse_selection_bps": 0.0,
    }
    duplicate_rows = [
        {**common, "order_id": "a", "symbol": "AAA", "date": "2019-01-02"},
        {**common, "order_id": "b", "symbol": "AAA", "date": "2019-01-02"},
    ]
    _write_h1_panel(queue, duplicate_rows)
    _write_h1_panel(strict, [
        {**common, "order_id": "a", "symbol": "AAA", "date": "2019-01-02"},
    ])

    with pytest.raises(ValueError, match="not unique"):
        paired_vs_queue_tests(runs={"tape_replay_queue": queue, "tape_replay_strict": strict})


def test_h1_subgroup_surface_filters_to_headline_window_and_size() -> None:
    rows = []
    for window in ("A", "B", "C"):
        for size_frac in (0.005, 0.01):
            rows.append({
                "strategy": "S3_FULL",
                "window": window,
                "size_frac": size_frac,
                "symbol": f"{window}{size_frac}",
                "date": "2019-01-02",
                "tier": 1,
                "net_alpha_vs_moc_bps": 0.0,
            })
    panel = pd.DataFrame(rows)

    primary = h1_performance_gap._primary_surface(panel)

    assert len(primary) == 1
    assert primary["window"].iloc[0] == "B"
    assert np.isclose(primary["size_frac"].iloc[0], 0.01)


def test_renderers_do_not_emit_draft_or_placeholder_text() -> None:
    calibration = pd.DataFrame([{
        "model": "cox",
        "tier": 1,
        "n": 100,
        "observed_fill_rate": 0.4,
        "mean_predicted_probability": 0.35,
        "absolute_calibration_error": 0.05,
        "brier": 0.2,
        "auc": 0.6,
    }])
    economic = pd.DataFrame([{
        "spec": "tape_replay_queue",
        "label": "Queue-aware replay (headline)",
        "is_headline": True,
        "mean_net_alpha_vs_moc_bps": -0.12,
        "t": -1.5,
        "p_value": 0.13,
        "p_holm": np.nan,
        "mean_fill_rate": 0.65,
        "as_markout_bps": 1.28,
        "residual_moc": 0.35,
        "n": 187309,
    }])
    h2 = pd.DataFrame([{
        "label": "OFI_marginal",
        "mean": -0.01,
        "t": -1.3,
        "p_value": 0.19,
        "p_holm": 0.76,
        "n": 1000,
        "matching_metric": "realized_passive_fill_rate",
    }])

    combined = "\n".join([
        render_fill_model_calibration(calibration),
        render_fill_robustness(economic),
        render_h2_adjusted(h2),
    ])
    lowered = combined.lower()
    assert "placeholder" not in lowered
    assert "await" not in lowered
    assert "draft" not in lowered
    assert " -- " not in combined


def test_write_outputs_manifest_lists_outputs() -> None:
    out_dir = Path("repo/coding/artifacts/tests/statistical_tests_manifest")
    calibration = pd.DataFrame(columns=["model", "tier", "n"])
    economic = pd.DataFrame([{
        "spec": "tape_replay_queue",
        "label": "Queue-aware replay (headline)",
        "is_headline": True,
        "mean_net_alpha_vs_moc_bps": 0.0,
        "t": 0.0,
        "p_value": 1.0,
        "p_holm": np.nan,
        "mean_fill_rate": 0.0,
        "as_markout_bps": 0.0,
        "residual_moc": 1.0,
        "n": 1,
    }])
    paired = pd.DataFrame([{"metric": "net_alpha_vs_moc_bps", "p_holm": 1.0}])
    h2 = pd.DataFrame([{
        "label": "interaction",
        "mean": 0.0,
        "t": 0.0,
        "p_value": 1.0,
        "p_holm": 1.0,
        "n": 1,
        "matching_metric": "realized_passive_fill_rate",
    }])

    write_outputs(
        out_dir,
        calibration=calibration,
        economic=economic,
        paired=paired,
        h2=h2,
        manifest={
            "run_roots": {"tape_replay_queue": "queue"},
            "input_sha256": {"queue:h1_panel": "abc"},
            "oos_dates": ["2019-01-02"],
            "feature_policy": "causal",
        },
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "run_roots" in manifest
    assert "input_sha256" in manifest
    assert "oos_dates" in manifest
    assert "feature_policy" in manifest
    assert "tab_fill_robustness_adjusted.tex" in manifest["outputs"]
    for output in manifest["outputs"]:
        assert (out_dir / output).exists()
