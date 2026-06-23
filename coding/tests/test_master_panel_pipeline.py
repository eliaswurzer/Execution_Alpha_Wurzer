from __future__ import annotations

import datetime as dt
import importlib
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.runners import master_panel
from analysis.runners import run_all_hypotheses
from analysis.simulation.parent_orders import rolling_expected_vc
from analysis.strategies.base import MarketState
from analysis.strategies.time_adaptive import TimeAdaptiveStrategy
from analysis.utils.adaptive_pool import AdaptivePool
from analysis.utils.symbols import canonical_symbol


@pytest.mark.unit
def test_canonical_symbol_collapses_file_safe_aliases() -> None:
    assert canonical_symbol("BRK_B") == "BRK B"
    assert canonical_symbol("brk.b") == "BRK B"
    assert canonical_symbol("GOOG L") == "GOOG L"


@pytest.mark.unit
def test_adaptive_pool_bounds_inflight_without_fixed_timeout(monkeypatch) -> None:
    monkeypatch.setenv("THESIS_POOL_BACKEND", "thread")
    monkeypatch.setattr(
        "analysis.utils.adaptive_pool._resources_available",
        lambda *_: True,
    )
    active = 0
    max_active = 0
    lock = threading.Lock()

    def work(value):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return value

    with AdaptivePool(
        max_workers=2,
        max_in_flight=4,
        poll_interval=0.001,
    ) as pool:
        futures = [pool.submit(work, i) for i in range(8)]
        assert [future.result() for future in futures] == list(range(8))
    assert max_active <= 2


@pytest.mark.unit
def test_adaptive_pool_waits_through_resource_pause(monkeypatch) -> None:
    monkeypatch.setenv("THESIS_POOL_BACKEND", "thread")
    checks = {"count": 0}

    def resources_available(*_):
        checks["count"] += 1
        return checks["count"] >= 4

    monkeypatch.setattr(
        "analysis.utils.adaptive_pool._resources_available",
        resources_available,
    )
    started = time.perf_counter()
    with AdaptivePool(
        max_workers=1,
        max_in_flight=1,
        poll_interval=0.01,
    ) as pool:
        assert pool.submit(lambda: 7).result() == 7
    assert checks["count"] == 4
    assert time.perf_counter() - started >= 0.02


@pytest.mark.unit
def test_expected_vc_cache_matches_rolling_calculation(
    monkeypatch, artifact_dir,
) -> None:
    dates = [dt.date(2018, 1, day) for day in range(2, 9)]
    history = pd.DataFrame({
        "symbol": ["AAPL"] * len(dates),
        "date": dates,
        "vc_shares": [100, 200, 300, 400, 500, 600, 700],
    })
    monkeypatch.setattr(master_panel._common, "_eval_dates", lambda *_: dates)
    calls = {"count": 0}

    def load_history(shard_dates, *_, **__):
        calls["count"] += 1
        mask = history["date"].isin(shard_dates)
        return history.loc[mask].reset_index(drop=True)

    monkeypatch.setattr(master_panel._common, "_vc_history", load_history)
    expected = rolling_expected_vc(history)
    first = master_panel._load_or_build_expected_vc(
        dates, ["AAPL"], artifact_dir / "cache", workers=1, resume=True, shard_dir=artifact_dir / "cache" / "vc_history_shards",
    )
    second = master_panel._load_or_build_expected_vc(
        dates, ["AAPL"], artifact_dir / "cache", workers=1, resume=True, shard_dir=artifact_dir / "cache" / "vc_history_shards",
    )
    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(second, expected)
    # One shard build per history date on the first pass; the second pass
    # resumes entirely from the cached manifest.
    assert calls["count"] == len(dates)


@pytest.mark.unit
def test_time_adaptive_searchsorted_matches_legacy_lookup() -> None:
    times = pd.to_datetime([
        "2018-02-01 15:00:00",
        "2018-02-01 15:05:00",
        "2018-02-01 15:10:00",
    ])
    rv = pd.Series([1.0, 2.0, 4.0], index=times)
    state = MarketState(
        symbol="AAPL",
        date=dt.date(2018, 2, 1),
        nbbo=pd.DataFrame(),
        trades=pd.DataFrame(),
        close_price=100.0,
        close_volume=1_000.0,
        ofi=pd.DataFrame(),
        rv=rv,
        imbalance=pd.DataFrame(),
        rv_times=times.values.astype("int64"),
    )
    strategy = TimeAdaptiveStrategy()
    for lookup_time in pd.to_datetime([
        "2018-02-01 14:59:59",
        "2018-02-01 15:00:00",
        "2018-02-01 15:07:00",
        "2018-02-01 15:20:00",
    ]):
        legacy = rv.loc[rv.index <= lookup_time]
        expected = 1.0 if legacy.empty else float(legacy.iloc[-1]) / 2.0
        assert np.isclose(strategy._vol_scalar(lookup_time, state, 2.0), expected)


def _fake_panel(date: dt.date, symbol: str, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        rows.append({
            "order_id": f"{date}_{symbol}_B_01_BUY",
            "symbol": symbol,
            "date": date,
            "side": "BUY",
            "strategy": strategy,
            "window": "B",
            "qty_intended": 100,
            "qty_filled_passive": 50 if strategy != "S0_MOC" else 0,
            "qty_filled_moc": 50 if strategy != "S0_MOC" else 100,
            "vwap_passive": 99.9 if strategy != "S0_MOC" else float("nan"),
            "close_price": 100.0,
            "avg_fill_price": 99.95 if strategy != "S0_MOC" else 100.0,
            "fill_rate": 0.5 if strategy != "S0_MOC" else 0.0,
            "adverse_selection_bps": 0.0,
            "size_frac": 0.01,
        })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_date_job_isolates_one_symbol_failure(monkeypatch, artifact_dir) -> None:
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)

    def simulate(symbol, date, parents, strategies, **kwargs):
        if symbol == "BAD":
            raise RuntimeError("synthetic failure")
        return _fake_panel(date, symbol, strategies)

    monkeypatch.setattr(master_panel, "simulate_symbol_day", simulate)
    date = dt.date(2018, 2, 1)
    result = master_panel._simulate_date_job(
        date,
        [
            {"symbol": "GOOD", "tier": 1, "expected_vc": 10_000},
            {"symbol": "BAD", "tier": 1, "expected_vc": 10_000},
        ],
        ["S0_MOC", "S1_STATIC"],
        (0.01,),
        "tape_replay",
        artifact_dir / "shards",
        "fingerprint",
    )
    assert result["status"] == "partial"
    assert result["successful_symbol_days"] == 1
    assert result["failed_symbol_days"] == 1
    assert result["failures"][0]["reason"] == "compute_error"
    assert (artifact_dir / "shards/date=2018-02-01/panel.parquet").exists()


@pytest.mark.unit
def test_date_job_passes_km_dispatcher(monkeypatch, artifact_dir) -> None:
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)
    worker_model = object()
    monkeypatch.setattr(master_panel._common, "_worker_model", worker_model)
    seen: dict[str, object] = {}

    def simulate(symbol, date, parents, strategies, **kwargs):
        seen["fill_model"] = kwargs.get("fill_model")
        seen["km_model"] = kwargs.get("km_model")
        seen["fill_specification"] = kwargs.get("fill_specification")
        return _fake_panel(date, symbol, strategies)

    monkeypatch.setattr(master_panel, "simulate_symbol_day", simulate)
    date = dt.date(2018, 2, 1)

    result = master_panel._simulate_date_job(
        date,
        [{"symbol": "AAPL", "tier": 1, "expected_vc": 10_000}],
        ["S0_MOC", "S1_STATIC"],
        (0.01,),
        "km",
        artifact_dir / "km_shards",
        "fingerprint",
    )

    assert result["status"] == "complete"
    assert seen == {
        "fill_model": worker_model,
        "km_model": worker_model,
        "fill_specification": "km",
    }


@pytest.mark.unit
def test_shard_validation_rejects_corruption(artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    shard_root = artifact_dir / "shards"
    panel_path, manifest_path = master_panel._shard_paths(shard_root, date)
    panel = master_panel.attach_moc_differential_columns(
        master_panel.attach_alpha_columns(
            _fake_panel(date, "AAPL", ["S0_MOC", "S1_STATIC"]),
        )
    )
    master_panel._write_parquet_atomic(panel, panel_path)
    manifest = {
        "schema_version": master_panel.SCHEMA_VERSION,
        "fingerprint": "abc",
        "sha256": master_panel._sha256(panel_path),
        "rows": len(panel),
    }
    master_panel._write_json_atomic(manifest_path, manifest)
    assert master_panel.validate_shard(
        shard_root, date, "abc", ["S0_MOC", "S1_STATIC"],
    ) is not None

    panel_path.write_bytes(b"corrupt")
    assert master_panel.validate_shard(
        shard_root, date, "abc", ["S0_MOC", "S1_STATIC"],
    ) is None


@pytest.mark.unit
def test_shard_validation_rejects_nonfinite_required_metrics(artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    shard_root = artifact_dir / "nan_metric_shards"
    panel_path, manifest_path = master_panel._shard_paths(shard_root, date)
    panel = master_panel.attach_moc_differential_columns(
        master_panel.attach_alpha_columns(
            _fake_panel(date, "AAPL", ["S0_MOC", "S1_STATIC"]),
        )
    )
    panel.loc[panel["strategy"] == "S1_STATIC", "net_alpha_bps"] = np.nan
    master_panel._write_parquet_atomic(panel, panel_path)
    master_panel._write_json_atomic(manifest_path, {
        "schema_version": master_panel.SCHEMA_VERSION,
        "fingerprint": "abc",
        "sha256": master_panel._sha256(panel_path),
        "rows": len(panel),
    })

    assert master_panel.validate_shard(
        shard_root, date, "abc", ["S0_MOC", "S1_STATIC"],
    ) is None


@pytest.mark.unit
def test_shard_validation_rejects_old_schema(artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    shard_root = artifact_dir / "old_schema_shards"
    panel_path, manifest_path = master_panel._shard_paths(shard_root, date)
    panel = _fake_panel(date, "AAPL", ["S0_MOC"])
    master_panel._write_parquet_atomic(panel, panel_path)
    master_panel._write_json_atomic(manifest_path, {
        "schema_version": "obsolete",
        "fingerprint": "abc",
        "sha256": master_panel._sha256(panel_path),
        "rows": len(panel),
    })

    assert master_panel.validate_shard(
        shard_root, date, "abc", ["S0_MOC"],
    ) is None


@pytest.mark.integration
def test_master_panel_resume_skips_valid_day(monkeypatch, artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    membership = pd.DataFrame([{"date": date, "symbol": "AAPL"}])
    tiers = pd.DataFrame([{"symbol": "AAPL", "tier": 1}])
    expected = pd.DataFrame([{
        "symbol": "AAPL", "date": date, "expected_vc": 10_000.0,
    }])

    monkeypatch.setattr(master_panel._common, "_eval_dates", lambda *_: [date])
    monkeypatch.setattr(
        master_panel, "_canonical_membership_panel", lambda *_, **__: membership,
    )
    monkeypatch.setattr(
        master_panel._common, "_load_artifacts",
        lambda *_, **__: (None, tiers, pd.DataFrame()),
    )
    monkeypatch.setattr(
        master_panel, "_load_or_build_expected_vc", lambda *_, **__: expected,
    )
    monkeypatch.setattr(master_panel, "_artifact_signature", lambda *_: {})
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)

    calls = {"count": 0}

    def fake_job(date, specs, strategies, sizes, fill_spec, shard_root, fingerprint, windows_map=None):
        calls["count"] += 1
        panel = _fake_panel(date, "AAPL", strategies)
        panel = master_panel.attach_alpha_columns(panel)
        panel = master_panel.attach_moc_differential_columns(panel)
        panel_path, manifest_path = master_panel._shard_paths(shard_root, date)
        master_panel._write_parquet_atomic(panel, panel_path)
        manifest = {
            "schema_version": master_panel.SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "date": date,
            "status": "complete",
            "eligible_symbol_days": 1,
            "successful_symbol_days": 1,
            "failed_symbol_days": 0,
            "rows": len(panel),
            "strategies": strategies,
            "sha256": master_panel._sha256(panel_path),
            "runtime_seconds": 0.01,
        }
        master_panel._write_json_atomic(manifest_path, manifest)
        return {**manifest, "failures": [], "shard_path": str(panel_path)}

    monkeypatch.setattr(master_panel, "_simulate_date_job", fake_job)
    kwargs = dict(
        strategies=["S0_MOC", "S1_STATIC"],
        start=date,
        end=date,
        artifacts_dir=artifact_dir,
        run_root=artifact_dir / "run",
        vc_shard_dir=artifact_dir / "run" / "vc_shards",
        symbols=["AAPL"],
        workers=1,
        resume=True,
        min_eligible_coverage=1.0,
        min_index_coverage=1.0,
    )
    first = master_panel.run_master_panel(**kwargs)
    second = master_panel.run_master_panel(**kwargs)
    assert first["status"] == second["status"] == "complete"
    assert calls["count"] == 1
    simulation_config = json.loads(
        (
            artifact_dir / "run/metadata/simulation_config.json"
        ).read_text(encoding="utf-8")
    )
    assert simulation_config["as_horizon_seconds"] == cfg.AS_HORIZON_SECONDS


@pytest.mark.integration
def test_master_panel_fallback_tier_for_missing_calibration_symbol(
    monkeypatch, artifact_dir,
) -> None:
    date = dt.date(2018, 3, 19)
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    membership = pd.DataFrame([{"date": date, "symbol": "MSCI"}])
    tiers = pd.DataFrame([{"symbol": "AAPL", "tier": 1}])
    expected = pd.DataFrame([{
        "symbol": "MSCI", "date": date, "expected_vc": 10_000.0,
    }])

    monkeypatch.setattr(master_panel._common, "_eval_dates", lambda *_: [date])
    monkeypatch.setattr(
        master_panel, "_canonical_membership_panel", lambda *_, **__: membership,
    )
    monkeypatch.setattr(
        master_panel._common, "_load_artifacts",
        lambda *_, **__: (None, tiers, pd.DataFrame()),
    )
    monkeypatch.setattr(
        master_panel, "_load_or_build_expected_vc", lambda *_, **__: expected,
    )
    monkeypatch.setattr(master_panel, "_artifact_signature", lambda *_: {})
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)

    calls = {"count": 0}

    def fake_job(date, specs, strategies, sizes, fill_spec, shard_root, fingerprint, windows_map=None):
        calls["count"] += 1
        assert specs == [{"symbol": "MSCI", "tier": 3, "expected_vc": 10_000.0}]
        panel = master_panel.attach_moc_differential_columns(
            master_panel.attach_alpha_columns(_fake_panel(date, "MSCI", strategies))
        )
        panel_path, manifest_path = master_panel._shard_paths(shard_root, date)
        master_panel._write_parquet_atomic(panel, panel_path)
        manifest = {
            "schema_version": master_panel.SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "date": date,
            "status": "complete",
            "eligible_symbol_days": 1,
            "successful_symbol_days": 1,
            "failed_symbol_days": 0,
            "rows": len(panel),
            "strategies": strategies,
            "sha256": master_panel._sha256(panel_path),
            "runtime_seconds": 0.01,
        }
        master_panel._write_json_atomic(manifest_path, manifest)
        return {**manifest, "failures": [], "shard_path": str(panel_path)}

    monkeypatch.setattr(master_panel, "_simulate_date_job", fake_job)
    summary = master_panel.run_master_panel(
        strategies=["S0_MOC", "S1_STATIC"],
        start=date,
        end=date,
        artifacts_dir=artifact_dir,
        run_root=artifact_dir / "fallback_run",
        vc_shard_dir=artifact_dir / "fallback_run" / "vc_shards",
        symbols=["MSCI"],
        workers=1,
        resume=True,
        min_eligible_coverage=1.0,
        min_index_coverage=1.0,
        tier_policy=master_panel.TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    )

    assert summary["status"] == "complete"
    assert summary["tier_fallback_symbols"] == 1
    assert summary["tier_fallback_symbol_days"] == 1
    assert calls["count"] == 1
    audit = pd.read_csv(
        artifact_dir / "fallback_run/metadata/liquidity_tier_audit.csv",
    )
    assert audit.to_dict("records") == [{
        "symbol": "MSCI",
        "tier": 3,
        "tier_source": "fallback_missing_calibration",
        "reason": "data_complete_symbol_absent_from_calibration_tier_map",
    }]


@pytest.mark.integration
def test_master_panel_calibrated_only_keeps_missing_tier_failure(
    monkeypatch, artifact_dir,
) -> None:
    date = dt.date(2018, 3, 19)
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    membership = pd.DataFrame([{"date": date, "symbol": "MSCI"}])
    tiers = pd.DataFrame([{"symbol": "AAPL", "tier": 1}])
    expected = pd.DataFrame([{
        "symbol": "MSCI", "date": date, "expected_vc": 10_000.0,
    }])

    monkeypatch.setattr(master_panel._common, "_eval_dates", lambda *_: [date])
    monkeypatch.setattr(
        master_panel, "_canonical_membership_panel", lambda *_, **__: membership,
    )
    monkeypatch.setattr(
        master_panel._common, "_load_artifacts",
        lambda *_, **__: (None, tiers, pd.DataFrame()),
    )
    monkeypatch.setattr(
        master_panel, "_load_or_build_expected_vc", lambda *_, **__: expected,
    )
    monkeypatch.setattr(master_panel, "_artifact_signature", lambda *_: {})
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)

    with pytest.raises(RuntimeError, match="Master-panel QC failed"):
        master_panel.run_master_panel(
            strategies=["S0_MOC"],
            start=date,
            end=date,
            artifacts_dir=artifact_dir,
            run_root=artifact_dir / "strict_tier_run",
        vc_shard_dir=artifact_dir / "strict_tier_run" / "vc_shards",
            symbols=["MSCI"],
            workers=1,
            min_eligible_coverage=1.0,
            min_index_coverage=1.0,
            tier_policy=master_panel.TIER_POLICY_CALIBRATED_ONLY,
        )

    failures = pd.read_csv(
        artifact_dir / "strict_tier_run/metadata/simulation_failures.csv",
    )
    assert failures["reason"].tolist() == ["missing_tier"]
    audit = pd.read_csv(
        artifact_dir / "strict_tier_run/metadata/liquidity_tier_audit.csv",
    )
    assert audit.loc[0, "tier_source"] == "missing"


@pytest.mark.unit
def test_tier_policy_participates_in_fingerprint() -> None:
    strict = master_panel._fingerprint({
        "liquidity_tier_policy": master_panel._tier_policy_version(
            master_panel.TIER_POLICY_CALIBRATED_ONLY,
        ),
    })
    fallback = master_panel._fingerprint({
        "liquidity_tier_policy": master_panel._tier_policy_version(
            master_panel.TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
        ),
    })
    assert strict != fallback


@pytest.mark.unit
def test_as_horizon_env_override_and_default_restore(monkeypatch) -> None:
    try:
        monkeypatch.setenv("THESIS_AS_HORIZON_SECONDS", "15")
        importlib.reload(cfg)
        assert cfg.AS_HORIZON_SECONDS == 15
        assert cfg.AS_HEADLINE_HORIZON_SECONDS == 30
        assert cfg.AS_HORIZON_GRID_SECONDS == (5, 15, 30, 60, 300)
    finally:
        monkeypatch.delenv("THESIS_AS_HORIZON_SECONDS", raising=False)
        importlib.reload(cfg)
    assert cfg.AS_HORIZON_SECONDS == cfg.AS_HEADLINE_HORIZON_SECONDS == 30


@pytest.mark.unit
def test_as_horizon_participates_in_fingerprint() -> None:
    h5 = master_panel._fingerprint({"as_horizon_seconds": 5})
    h30 = master_panel._fingerprint({"as_horizon_seconds": 30})
    assert h5 != h30


@pytest.mark.unit
def test_km_artifacts_participate_in_fingerprint(artifact_dir) -> None:
    (artifact_dir / "symbol_tier_map.csv").write_text("symbol,tier\nAAPL,1\n", encoding="utf-8")
    (artifact_dir / "km_symbol_tier_map.csv").write_text("symbol,tier\nAAPL,1\n", encoding="utf-8")
    km_path = artifact_dir / "km_tier_1.pkl"
    km_path.write_bytes(b"km-v1")

    first = master_panel._artifact_signature(artifact_dir, "km")
    km_path.write_bytes(b"km-v2")
    second = master_panel._artifact_signature(artifact_dir, "km")

    assert "km_tier_1.pkl" in first
    assert "km_symbol_tier_map.csv" in first
    assert first["km_tier_1.pkl"] != second["km_tier_1.pkl"]


@pytest.mark.integration
def test_missing_expected_vc_fails_qc_without_manifest_crash(
    monkeypatch, artifact_dir,
) -> None:
    date = dt.date(2018, 2, 1)
    dummy = artifact_dir / "dummy.parquet"
    dummy.write_bytes(b"x")
    membership = pd.DataFrame([{"date": date, "symbol": "AAPL"}])
    tiers = pd.DataFrame([{"symbol": "AAPL", "tier": 1}])

    monkeypatch.setattr(master_panel._common, "_eval_dates", lambda *_: [date])
    monkeypatch.setattr(
        master_panel, "_canonical_membership_panel", lambda *_, **__: membership,
    )
    monkeypatch.setattr(
        master_panel._common, "_load_artifacts",
        lambda *_, **__: (None, tiers, pd.DataFrame()),
    )
    monkeypatch.setattr(
        master_panel,
        "_load_or_build_expected_vc",
        lambda *_, **__: pd.DataFrame(
            columns=["symbol", "date", "expected_vc"],
        ),
    )
    monkeypatch.setattr(master_panel, "_artifact_signature", lambda *_: {})
    monkeypatch.setattr(master_panel, "parquet_available", lambda *_: True)

    with pytest.raises(RuntimeError, match="Master-panel QC failed"):
        master_panel.run_master_panel(
            strategies=["S0_MOC"],
            start=date,
            end=date,
            artifacts_dir=artifact_dir,
            run_root=artifact_dir / "missing_evc_run",
        vc_shard_dir=artifact_dir / "missing_evc_run" / "vc_shards",
            symbols=["AAPL"],
            workers=1,
            min_eligible_coverage=1.0,
            min_index_coverage=1.0,
        )

    summary = json.loads(
        (
            artifact_dir
            / "missing_evc_run/metadata/simulation_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["eligible_symbol_days"] == 1
    assert summary["successful_symbol_days"] == 0
    assert summary["eligible_coverage"] == 0.0


@pytest.mark.unit
def test_materialize_panel_filters_strategies(artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    panel_path, _ = master_panel._shard_paths(artifact_dir / "shards", date)
    master_panel._write_parquet_atomic(
        master_panel.attach_moc_differential_columns(
            master_panel.attach_alpha_columns(
                _fake_panel(date, "AAPL", ["S0_MOC", "S1_STATIC", "S3_FULL"]),
            )
        ),
        panel_path,
    )
    out = artifact_dir / "h2_panel.parquet"
    master_panel.materialize_panel(
        artifact_dir / "shards", ["S0_MOC", "S3_FULL"], out,
    )
    materialized = pd.read_parquet(out)
    assert set(materialized["strategy"]) == {"S0_MOC", "S3_FULL"}


@pytest.mark.unit
def test_materialize_panel_rejects_nonfinite_required_metrics(artifact_dir) -> None:
    date = dt.date(2018, 2, 1)
    panel_path, _ = master_panel._shard_paths(artifact_dir / "nan_shards", date)
    panel = master_panel.attach_moc_differential_columns(
        master_panel.attach_alpha_columns(
            _fake_panel(date, "AAPL", ["S0_MOC", "S1_STATIC"]),
        )
    )
    panel.loc[panel["strategy"] == "S1_STATIC", "alpha_bps"] = np.nan
    master_panel._write_parquet_atomic(panel, panel_path)

    with pytest.raises(ValueError, match="non-finite required metrics"):
        master_panel.materialize_panel(
            artifact_dir / "nan_shards", ["S0_MOC", "S1_STATIC"],
            artifact_dir / "bad_panel.parquet",
        )


@pytest.mark.unit
def test_hypothesis_resume_requires_matching_simulation_fingerprint(
    artifact_dir,
) -> None:
    out_dir = artifact_dir / "h1"
    out_dir.mkdir(parents=True)
    panel = master_panel.attach_moc_differential_columns(
        master_panel.attach_alpha_columns(
            _fake_panel(
                dt.date(2018, 2, 1),
                "AAPL",
                run_all_hypotheses.HYPOTHESIS_STRATEGIES["h1"],
            )
        )
    )
    panel.to_parquet(out_dir / "h1_panel.parquet", index=False)
    for filename in run_all_hypotheses.HYPOTHESIS_REQUIRED["h1"][1:]:
        (out_dir / filename).write_text("value\n1\n", encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "simulation_fingerprint": "old",
    }), encoding="utf-8")

    assert run_all_hypotheses._hypothesis_complete(
        "h1", out_dir, "old",
    )
    assert not run_all_hypotheses._hypothesis_complete(
        "h1", out_dir, "new",
    )
