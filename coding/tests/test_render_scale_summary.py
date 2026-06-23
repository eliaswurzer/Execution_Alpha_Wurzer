"""Tests for the read-only scale summary renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.runners import render_scale_summary as rss


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_volume_db(path: Path) -> None:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE daily_volume (
            Ticker VARCHAR,
            Date DATE,
            Is_Witching_Day BOOLEAN,
            Pre_Market_Val DOUBLE,
            Open_Auction_Val DOUBLE,
            Morning_30m_Val DOUBLE,
            Mid_Day_Val DOUBLE,
            Afternoon_30m_Val DOUBLE,
            Close_Auction_Val DOUBLE,
            Post_Market_Val DOUBLE,
            Official_Close_Marker_Val DOUBLE,
            Official_Close_Marker_Rows BIGINT,
            Total_Daily_Val DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO daily_volume VALUES
        ('AAPL', DATE '2018-07-02', FALSE, 1, 2, 3, 4, 5, 20, 6, 21, 2, 100)
        """
    )
    con.execute(
        """
        CREATE TABLE daily_volume_skipped (
            Ticker VARCHAR,
            Date DATE,
            Reason VARCHAR,
            Detail VARCHAR,
            Source_Path VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO daily_volume_skipped VALUES
        ('__NO_PARQUET_FILES__', DATE '2019-05-13', 'no_parquet_files', 'missing day', 'root/20190513')
        """
    )
    con.close()


def _make_bundle(root: Path) -> dict[str, Path]:
    run_root = root / "run"
    processed_root = root / "processed"
    audit_root = root / "audit"
    fill_artifacts = root / "fill_model"
    volume_db = root / "volume.duckdb"

    _write_json(run_root / "run_status.json", {
        "status": "running",
        "current_step": "simulation",
        "updated_at": "2026-06-11T18:00:00",
    })
    _write_json(run_root / "metadata" / "run_config.json", {
        "run_id": "synthetic_queue",
        "start": "2018-07-02",
        "end": "2018-07-03",
        "fill_specification": "tape_replay_queue",
        "universe": "sp500",
        "workers": 2,
        "tier_policy": "calibrated_plus_fallback",
        "artifacts": str(fill_artifacts),
    })
    _write_json(run_root / "metadata" / "simulation_config.json", {
        "fingerprint": "fp123",
        "schema_version": "headline_master_panel_v1",
        "simulation_source_sha256": "abc123",
        "feature_policy": "causal_features_v2",
        "trade_condition_policy": "trade_conditions_v1",
        "pool_backend": "thread",
    })
    pd.DataFrame([{
        "schema_version": "headline_master_panel_v1",
        "fingerprint": "fp123",
        "date": "2018-07-02",
        "status": "complete",
        "eligible_symbol_days": 1,
        "successful_symbol_days": 1,
        "failed_symbol_days": 0,
        "rows": 2,
        "runtime_seconds": 1.5,
        "shard_path": str(run_root / "panel_shards" / "date=2018-07-02" / "panel.parquet"),
    }]).to_csv(run_root / "metadata" / "simulation_manifest.csv", index=False)
    pd.DataFrame(columns=["symbol", "date", "reason"]).to_csv(
        run_root / "metadata" / "simulation_failures.csv", index=False,
    )

    _write_json(run_root / "cache" / "expected_vc_manifest.json", {
        "history_dates": 1,
        "history_rows": 1,
        "expected_vc_rows": 1,
        "fingerprint": "vcfp",
    })
    pd.DataFrame([{
        "symbol": "AAPL",
        "date": pd.Timestamp("2018-07-02"),
        "vc_shares": 1000.0,
        "vc_source": "close_trade",
        "close_price_source": "close_trade",
        "close_trade_volume": 1000.0,
        "close_trade_rows": 1,
        "official_close_marker_volume": 1100.0,
        "official_close_marker_rows": 1,
        "official_close_marker_fallback_volume": 0.0,
    }]).to_parquet(run_root / "cache" / "vc_history.parquet", index=False)

    panel_dir = run_root / "panel_shards" / "date=2018-07-02"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "order_id": "AAPL_20180702_parent",
            "symbol": "AAPL",
            "date": pd.Timestamp("2018-07-02"),
            "strategy": "S0_MOC",
            "qty_intended": 100,
            "qty_filled_passive": 0,
            "qty_filled_moc": 100,
            "close_trade_volume": 1000.0,
            "close_trade_rows": 1,
            "official_close_marker_volume": 1100.0,
            "official_close_marker_rows": 1,
            "official_close_marker_fallback_volume": 0.0,
            "close_price_source": "close_trade",
            "close_volume_source": "close_trade",
        },
        {
            "order_id": "AAPL_20180702_parent",
            "symbol": "AAPL",
            "date": pd.Timestamp("2018-07-02"),
            "strategy": "S3_FULL",
            "qty_intended": 100,
            "qty_filled_passive": 80,
            "qty_filled_moc": 20,
            "close_trade_volume": 1000.0,
            "close_trade_rows": 1,
            "official_close_marker_volume": 1100.0,
            "official_close_marker_rows": 1,
            "official_close_marker_fallback_volume": 0.0,
            "close_price_source": "close_trade",
            "close_volume_source": "close_trade",
        },
    ]).to_parquet(panel_dir / "panel.parquet", index=False)

    _write_json(processed_root / "20180702" / "qc" / "trade_qc_summary.json", {
        "date": "20180702",
        "kind": "trades",
        "total_read_rows": 1000,
        "total_symbol_filter_rows": 100,
        "total_kept_rows": 90,
        "symbols_written": 1,
        "trade_condition_policy_version": "trade_conditions_v1",
        "qc_counts": {
            "policy_input_rows": 100,
            "kept_opening_auction_condition": 1,
            "kept_closing_auction_condition": 2,
        },
    })
    _write_json(audit_root / "date_side_audit_summary.json", {
        "dates": 1,
        "complete_dates": 1,
        "active_membership_symbol_days": 1,
        "active_membership_complete_symbol_days": 1,
        "active_membership_coverage": 1.0,
        "excluded_dates": ["20190513"],
    })
    _write_json(fill_artifacts / "calibration_manifest.json", {
        "status": "complete",
        "feature_policy": "causal_features_v2",
        "coverage": 1.0,
        "n_critical_failures": 0,
        "n_event_rows": 10,
        "n_daily_feature_rows": 2,
        "n_as_rows": 10,
        "xgb_survival_status": "complete",
        "km_status": "complete",
    })
    pd.DataFrame([{"model": "cox", "metric": "auc", "value": 0.7}]).to_csv(
        fill_artifacts / "validation.csv",
        index=False,
    )
    _make_volume_db(volume_db)

    return {
        "run_root": run_root,
        "processed_root": processed_root,
        "audit_root": audit_root,
        "fill_artifacts": fill_artifacts,
        "volume_db": volume_db,
    }


@pytest.mark.unit
def test_render_scale_summary_outputs_and_deduplicates_parent_orders(artifact_dir) -> None:
    paths = _make_bundle(artifact_dir / "scale_summary")

    manifest = rss.render(
        paths["run_root"],
        processed_root=paths["processed_root"],
        audit_root=paths["audit_root"],
        volume_db=paths["volume_db"],
        fill_artifacts=paths["fill_artifacts"],
        allow_incomplete=True,
    )

    assert manifest["draft"] is True
    for output in manifest["outputs"].values():
        assert Path(output).exists()

    metrics = {(r["section"], r["metric"]): r["value"] for r in manifest["metrics"]}
    assert metrics[("Preprocessing", "total_read_rows")] == 1000
    assert metrics[("Data audit", "active_membership_coverage")] == 1.0
    assert metrics[("Volume DB", "close_auction_share_of_total")] == 0.2
    assert metrics[("Calibration", "n_event_rows")] == 10
    assert metrics[("Expected VC cache", "vc_shares")] == 1000.0
    assert metrics[("Simulation panel", "strategy_order_intended_shares")] == 200
    assert metrics[("Simulation panel", "parent_order_intended_shares")] == 100
    assert metrics[("Simulation panel", "distinct_symbol_day_close_trade_volume")] == 1000.0


@pytest.mark.unit
def test_render_scale_summary_rejects_incomplete_without_flag(artifact_dir) -> None:
    paths = _make_bundle(artifact_dir / "scale_summary")

    with pytest.raises(rss.ScaleSummaryError, match="allow-incomplete"):
        rss.render(
            paths["run_root"],
            processed_root=paths["processed_root"],
            audit_root=paths["audit_root"],
            volume_db=paths["volume_db"],
            fill_artifacts=paths["fill_artifacts"],
        )
