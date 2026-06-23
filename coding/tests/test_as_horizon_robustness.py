from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from analysis.runners import as_horizon_robustness as asr


def _write_fake_run(
    root,
    horizon: int,
    *,
    baseline: bool = False,
    as_bps: float = -1.0,
    net_alpha: float = -0.12,
    fill_rates: tuple[float, ...] = (0.6, 0.7),
    order_start: int = 0,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "hypotheses" / "h1").mkdir(parents=True, exist_ok=True)
    (root / "run_status.json").write_text(
        json.dumps({
            "status": "complete",
            "simulation": {
                "status": "complete",
                "dates_with_valid_shards": 371,
                "critical_failures": 0,
                "eligible_coverage": 0.9999,
                "fingerprint": f"fp-{horizon}",
                "workers": 8,
            },
        }),
        encoding="utf-8",
    )
    sim_config = {"fingerprint": f"fp-{horizon}"}
    if not baseline:
        sim_config["as_horizon_seconds"] = horizon
    (root / "metadata" / "simulation_config.json").write_text(
        json.dumps(sim_config), encoding="utf-8",
    )
    (root / "metadata" / "run_config.json").write_text(
        json.dumps({"as_horizon_seconds": horizon}), encoding="utf-8",
    )
    rows = []
    for idx, fill_rate in enumerate(fill_rates):
        order_idx = order_start + idx
        rows.append({
            "order_id": f"order-{order_idx}",
            "strategy": "S3_FULL",
            "window": "B",
            "size_frac": 0.01,
            "net_alpha_vs_moc_bps": net_alpha,
            "fill_rate": fill_rate,
            "adverse_selection_bps": as_bps / (idx + 1),
            "adverse_selection_cost_bps": max(0.0, -(as_bps / (idx + 1))),
        })
    pd.DataFrame(rows).to_parquet(root / "hypotheses" / "h1" / "h1_panel.parquet", index=False)
    pd.DataFrame([{
        "mean": net_alpha,
        "se": 0.08,
        "t": -1.5,
        "p_value": 0.13,
        "p_one_sided": 0.93,
        "n": len(fill_rates),
    }]).to_csv(root / "hypotheses" / "h1" / "h1_primary_ttest.csv", index=False)
    pd.DataFrame([{"strategy": "S3_FULL", "mean_alpha": -0.1, "tev": 1.0, "n": len(fill_rates)}]).to_csv(
        root / "hypotheses" / "h1" / "h1_tev.csv", index=False,
    )
    (root / "hypotheses" / "h1" / "status.json").write_text(
        json.dumps({
            "status": "complete",
            "panel_rows": len(fill_rates),
            "simulation_fingerprint": f"fp-{horizon}",
            "as_horizon_seconds": horizon,
        }),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_aggregate_horizons_writes_summary_and_manifest(artifact_dir) -> None:
    run_root = artifact_dir / "runs"
    prefix = "as_h"
    baseline = artifact_dir / "baseline_30"
    out_dir = artifact_dir / "summary"

    _write_fake_run(baseline, 30, baseline=True, as_bps=-3.0)
    for horizon, as_bps in ((5, -0.5), (15, -1.5), (60, -6.0), (300, -30.0)):
        _write_fake_run(run_root / asr._horizon_run_id(prefix, horizon), horizon, as_bps=as_bps)

    summary = asr.aggregate_horizons(SimpleNamespace(
        run_root=run_root,
        run_id_prefix=prefix,
        baseline_run=baseline,
        out_dir=out_dir,
        expected_n=2,
        metric_tolerance=1e-12,
        allow_validation_problems=False,
    ))

    assert summary["horizon_seconds"].tolist() == [5, 15, 30, 60, 300]
    assert summary.loc[summary["horizon_seconds"] == 30, "source"].item() == "existing_30s_headline"
    assert (out_dir / "as_horizon_summary.csv").exists()
    assert (out_dir / "as_horizon_summary.md").exists()
    assert (out_dir / "as_horizon_audit_note.md").exists()
    assert (out_dir / "as_horizon_common_sample_summary.csv").exists()
    assert (out_dir / "tab_as_horizon_robustness.tex").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["validation_problems"] == []
    assert set(manifest["run_roots"]) == {"5", "15", "30", "60", "300"}
    assert manifest["workers_by_horizon"]["5"] == 8


@pytest.mark.unit
def test_aggregate_horizons_rejects_mismatched_horizon_metadata(artifact_dir) -> None:
    run_root = artifact_dir / "runs"
    prefix = "as_h"
    baseline = artifact_dir / "baseline_30"
    out_dir = artifact_dir / "summary"

    _write_fake_run(baseline, 30, baseline=True)
    for horizon in (5, 15, 60, 300):
        metadata_horizon = 999 if horizon == 60 else horizon
        _write_fake_run(
            run_root / asr._horizon_run_id(prefix, horizon),
            metadata_horizon,
        )

    with pytest.raises(RuntimeError, match="horizon metadata"):
        asr.aggregate_horizons(SimpleNamespace(
            run_root=run_root,
            run_id_prefix=prefix,
            baseline_run=baseline,
            out_dir=out_dir,
            expected_n=2,
            metric_tolerance=1e-12,
            allow_validation_problems=False,
        ))


@pytest.mark.unit
def test_aggregate_horizons_rejects_sample_drift_by_default(artifact_dir) -> None:
    run_root = artifact_dir / "runs"
    prefix = "as_h"
    baseline = artifact_dir / "baseline_30"
    out_dir = artifact_dir / "summary"

    _write_fake_run(baseline, 30, baseline=True)
    for horizon in (5, 15, 60, 300):
        _write_fake_run(
            run_root / asr._horizon_run_id(prefix, horizon),
            horizon,
            net_alpha=-0.121 if horizon == 5 else -0.12,
            fill_rates=(0.6, 0.7, 0.8) if horizon == 5 else (0.6, 0.7),
        )

    with pytest.raises(RuntimeError, match="5s n 3 != baseline 2"):
        asr.aggregate_horizons(SimpleNamespace(
            run_root=run_root,
            run_id_prefix=prefix,
            baseline_run=baseline,
            out_dir=out_dir,
            expected_n=2,
            metric_tolerance=1e-12,
            allow_validation_problems=False,
        ))


@pytest.mark.unit
def test_aggregate_horizons_can_record_noncritical_sample_drift_warning(artifact_dir) -> None:
    run_root = artifact_dir / "runs"
    prefix = "as_h"
    baseline = artifact_dir / "baseline_30"
    out_dir = artifact_dir / "summary"

    _write_fake_run(baseline, 30, baseline=True)
    for horizon in (5, 15, 60, 300):
        _write_fake_run(
            run_root / asr._horizon_run_id(prefix, horizon),
            horizon,
            net_alpha=-0.121 if horizon == 5 else -0.12,
            fill_rates=(0.6, 0.7, 0.8) if horizon == 5 else (0.6, 0.7),
        )

    summary = asr.aggregate_horizons(SimpleNamespace(
        run_root=run_root,
        run_id_prefix=prefix,
        baseline_run=baseline,
        out_dir=out_dir,
        expected_n=2,
        metric_tolerance=1e-12,
        allow_noncritical_sample_drift=True,
        max_row_count_drift=1,
        max_net_alpha_drift_bps=0.002,
        max_fill_rate_drift=0.1,
        allow_validation_problems=False,
    ))

    assert summary["horizon_seconds"].tolist() == [5, 15, 30, 60, 300]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_warnings"
    assert manifest["validation_problems"] == []
    assert "as_horizon_audit_note.md" in manifest["outputs"]
    assert "as_horizon_common_sample_summary.csv" in manifest["outputs"]
    assert (out_dir / "as_horizon_audit_note.md").exists()
    common = pd.read_csv(out_dir / "as_horizon_common_sample_summary.csv")
    assert common["n"].unique().tolist() == [2]
    assert any("5s n 3 != baseline 2" in item for item in manifest["validation_warnings"])
    h5 = [row for row in manifest["drift_audit"] if row["horizon_seconds"] == 5][0]
    assert h5["n_delta_from_30"] == 1


@pytest.mark.unit
def test_aggregate_horizons_can_fill_missing_primary_rows_from_supplements(artifact_dir) -> None:
    run_root = artifact_dir / "runs"
    supplement_root = artifact_dir / "supplements"
    prefix = "as_h"
    baseline = artifact_dir / "baseline_30"
    out_dir = artifact_dir / "summary"

    _write_fake_run(baseline, 30, baseline=True, fill_rates=(0.6, 0.7))
    _write_fake_run(
        run_root / asr._horizon_run_id(prefix, 5),
        5,
        fill_rates=(0.6,),
    )
    _write_fake_run(
        supplement_root / "as_horizon_supplement_h005_20190102_MSFT",
        5,
        fill_rates=(0.7,),
        order_start=1,
    )
    for horizon in (15, 60, 300):
        _write_fake_run(
            run_root / asr._horizon_run_id(prefix, horizon),
            horizon,
            fill_rates=(0.6, 0.7),
        )

    summary = asr.aggregate_horizons(SimpleNamespace(
        run_root=run_root,
        run_id_prefix=prefix,
        baseline_run=baseline,
        supplement_root=supplement_root,
        out_dir=out_dir,
        expected_n=2,
        metric_tolerance=1e-12,
        allow_validation_problems=False,
    ))

    assert summary.loc[summary["horizon_seconds"] == 5, "n"].item() == 2
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["supplement_root"] == str(supplement_root)
    assert manifest["common_sample_n"] == 2
    h5_supplements = manifest["supplement_audit"]["5"]
    assert len(h5_supplements) == 1
    assert h5_supplements[0]["added_rows"] == 1
    assert h5_supplements[0]["added_order_ids"] == ["order-1"]
