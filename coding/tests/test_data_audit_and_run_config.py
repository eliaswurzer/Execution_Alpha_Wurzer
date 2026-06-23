from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from analysis.runners.validate_run_config import validate_config


PREPROCESSING_ROOT = Path(__file__).resolve().parents[1] / "preprocessing"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PREPROCESSING_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ps = _load_module("preprocessing_status_for_tests", "preprocessing_status.py")
audit_data_availability = _load_module("audit_data_availability_for_tests", "audit_data_availability.py")
preprocess_taq = _load_module("preprocess_taq_for_tests", "preprocess_taq.py")
build_conservative_union = _load_module("build_conservative_sp500_union_for_tests", "build_conservative_sp500_union.py")


def _raw(path: Path, side: str, date: str) -> Path:
    prefix = "EQY_US_ALL_TRADE_" if side == "Trade" else "EQY_US_ALL_NBBO_"
    out = path / side / f"{prefix}{date}.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    return out


def _write_manifest(date_root: Path, rows: list[tuple[str, int, int]]) -> None:
    lines = ["symbol,trade_rows,nbbo_rows"]
    lines.extend(f"{symbol},{trade_rows},{nbbo_rows}" for symbol, trade_rows, nbbo_rows in rows)
    (date_root / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trade_qc(date_root: Path, payload: dict | None = None) -> None:
    qc = date_root / "qc" / "trade_qc_summary.json"
    qc.parent.mkdir(parents=True, exist_ok=True)
    qc.write_text(json.dumps(payload or {"status": "ok"}), encoding="utf-8")


def _write_coverage(out_root: Path, date: str, expected_symbols: int = 1) -> None:
    coverage = out_root / "coverage" / f"{date}_summary.json"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(json.dumps({"expected_symbols": expected_symbols}), encoding="utf-8")


@pytest.mark.unit
def test_raw_discovery_handles_trade_and_nbbo_across_roots(artifact_dir: Path) -> None:
    root_a = artifact_dir / "raw_a"
    root_b = artifact_dir / "raw_b"
    _raw(root_a, "Trade", "20190102")
    _raw(root_b, "NBBO", "20190102")

    discovered = ps.discover_raw_date_sides([root_b, root_a])

    assert discovered["20190102"]["trade"] is True
    assert discovered["20190102"]["nbbo"] is True
    assert len(discovered["20190102"]["trade_paths"]) == 1
    assert len(discovered["20190102"]["nbbo_paths"]) == 1


@pytest.mark.unit
def test_side_status_requires_manifest_consistency_and_trade_qc(artifact_dir: Path) -> None:
    out_root = artifact_dir / "processed"
    date_root = out_root / "20190102"
    (date_root / "trades").mkdir(parents=True)
    (date_root / "trades" / "AAPL.parquet").touch()
    _write_manifest(date_root, [("AAPL", 10, 0)])

    status = ps.date_side_status(out_root, "20190102", ["AAPL"], raw_trade=True)

    assert status.status == ps.STATUS_QC_PROBLEM
    assert status.trade_done is False

    _write_trade_qc(date_root)
    status = ps.date_side_status(out_root, "20190102", ["AAPL"], raw_trade=True)
    assert status.status == ps.STATUS_TRADE_ONLY
    assert status.trade_done is True

    (date_root / "trades" / "MSFT.parquet").touch()
    status = ps.date_side_status(out_root, "20190102", ["AAPL"], raw_trade=True)
    assert status.status == ps.STATUS_MANIFEST_INCONSISTENT


@pytest.mark.unit
def test_complete_date_and_missing_symbols_are_reported(artifact_dir: Path) -> None:
    out_root = artifact_dir / "processed"
    date = "20190102"
    date_root = out_root / date
    (date_root / "trades").mkdir(parents=True)
    (date_root / "nbbo").mkdir(parents=True)
    (date_root / "trades" / "AAPL.parquet").touch()
    (date_root / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(date_root, [("AAPL", 10, 20)])
    _write_trade_qc(date_root)
    _write_coverage(out_root, date, expected_symbols=2)

    status = ps.date_side_status(out_root, date, ["AAPL", "MSFT"], raw_trade=True, raw_nbbo=True)

    assert status.complete_done is True
    assert status.status == ps.STATUS_COMPLETE_WITH_MISSING
    assert status.missing_trade_symbols == ["MSFT"]
    assert status.missing_nbbo_symbols == ["MSFT"]


@pytest.mark.unit
def test_audit_outputs_todo_for_newly_available_counterpart(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    out_dir = artifact_dir / "audit"
    symbols = ["AAPL"]
    date = "20190102"
    _raw(raw_root, "Trade", date)
    _raw(raw_root, "NBBO", date)

    date_root = processed_root / date
    (date_root / "trades").mkdir(parents=True)
    (date_root / "trades" / "AAPL.parquet").touch()
    _write_manifest(date_root, [("AAPL", 10, 0)])
    _write_trade_qc(date_root)

    audit_rows, todo_rows, missing_rows, summary = audit_data_availability.build_audit(
        raw_roots=[raw_root],
        processed_roots=[processed_root],
        symbols=symbols,
        start=date,
        end=date,
    )

    assert summary["dates"] == 1
    assert summary["raw_trade_unprocessed_dates"] == 0
    assert summary["raw_nbbo_unprocessed_dates"] == 1
    assert audit_rows[0]["status"] == ps.STATUS_TRADE_ONLY
    assert todo_rows == [
        {
            "date": date,
            "side": "nbbo",
            "status": ps.STATUS_TRADE_ONLY,
            "processed_root": str(processed_root),
            "raw_paths": str(raw_root / "NBBO" / f"EQY_US_ALL_NBBO_{date}.gz"),
            "reason": "raw_nbbo_available_but_nbbo_side_not_done",
        }
    ]
    assert missing_rows == [{
        "date": date,
        "side": "nbbo",
        "symbol": "AAPL",
        "status": ps.STATUS_TRADE_ONLY,
        "processed_root": str(processed_root),
    }]

    audit_data_availability._write_csv(out_dir / "preprocessing_todo.csv", todo_rows, list(todo_rows[0]))
    assert (out_dir / "preprocessing_todo.csv").exists()


def _valid_config(root: Path) -> dict:
    data_root = root / "data"
    volume_db = root / "volume.duckdb"
    membership_root = root / "membership"
    symbols_file = membership_root / "symbols.txt"
    data_root.mkdir(parents=True)
    membership_root.mkdir(parents=True)
    volume_db.touch()
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    return {
        "run_id_prefix": "final_test",
        "artifact_root": str(root / "artifacts"),
        "data_roots": {"2018": [str(data_root)]},
        "volume_db": str(volume_db),
        "membership_root": str(membership_root),
        "universe": "sp500",
        "symbols_file": str(symbols_file),
        "calibration": {
            "warmup_start": "2018-01-02",
            "rolling_min_train_days": 60,
            "rolling_lookback_days": 120,
            "workers": 1,
        },
        "evaluation": {
            "start": "2018-04-02",
            "end": "2019-12-31",
            "workers": 2,
            "fill_specification": "tape_replay",
            "tier_policy": "calibrated_only",
        },
        "value_model": {
            "enabled": True,
            "strategy": "S5_VALUE_AWARE_XGB",
            "policy": "rolling_value_model_v2",
            "xgb_device": "cpu",
            "target": "target_net_alpha_vs_moc_bps",
            "offset_grid_bps": [0.0, 1.0],
            "min_expected_alpha_bps": 0.0,
        },
    }


@pytest.mark.unit
def test_validate_run_config_accepts_locked_shape(artifact_dir: Path) -> None:
    errors = validate_config(_valid_config(artifact_dir))
    assert errors == []


@pytest.mark.unit
def test_validate_run_config_rejects_icloud_and_incomplete_value_model(artifact_dir: Path) -> None:
    config = _valid_config(artifact_dir)
    config["artifact_root"] = "C:\\Users\\example\\iCloud" + "Drive\\Documents\\bad"
    del config["value_model"]["offset_grid_bps"]

    errors = validate_config(config, allow_missing_paths=True)

    assert any("not iCloud" in error for error in errors)
    assert any("offset_grid_bps" in error for error in errors)


@pytest.mark.unit
def test_validate_run_config_rejects_stale_value_model_policy(artifact_dir: Path) -> None:
    config = _valid_config(artifact_dir)
    config["value_model"]["policy"] = "rolling_value_model_v1"

    errors = validate_config(config)

    assert any("VALUE_MODEL_POLICY_VERSION" in error for error in errors)



@pytest.mark.unit
def test_side_status_covers_nbbo_only_failed_qc_and_missing_coverage(artifact_dir: Path) -> None:
    out_root = artifact_dir / "processed_more_states"

    nbbo_date = "20190103"
    nbbo_root = out_root / nbbo_date
    (nbbo_root / "nbbo").mkdir(parents=True)
    (nbbo_root / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(nbbo_root, [("AAPL", 0, 20)])
    status = ps.date_side_status(out_root, nbbo_date, ["AAPL"], raw_nbbo=True)
    assert status.status == ps.STATUS_NBBO_ONLY
    assert status.trade_done is False
    assert status.nbbo_done is True

    qc_date = "20190104"
    qc_root = out_root / qc_date
    (qc_root / "trades").mkdir(parents=True)
    (qc_root / "trades" / "AAPL.parquet").touch()
    _write_manifest(qc_root, [("AAPL", 10, 0)])
    _write_trade_qc(qc_root, {"status": "failed", "reason": "synthetic"})
    status = ps.date_side_status(out_root, qc_date, ["AAPL"], raw_trade=True)
    assert status.status == ps.STATUS_QC_PROBLEM
    assert status.trade_done is False
    assert status.trade_qc_status == "failed"

    no_coverage_date = "20190105"
    full_root = out_root / no_coverage_date
    (full_root / "trades").mkdir(parents=True)
    (full_root / "nbbo").mkdir(parents=True)
    (full_root / "trades" / "AAPL.parquet").touch()
    (full_root / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(full_root, [("AAPL", 10, 20)])
    _write_trade_qc(full_root)
    status = ps.date_side_status(out_root, no_coverage_date, ["AAPL"], raw_trade=True, raw_nbbo=True)
    assert status.trade_done is True
    assert status.nbbo_done is True
    assert status.complete_done is False
    assert status.coverage_exists is False
    assert status.status == ps.STATUS_UNPROCESSED


@pytest.mark.unit
def test_audit_distinguishes_parquet_availability_from_side_done(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_qc"
    processed_root = artifact_dir / "processed_qc"
    date = "20190106"
    _raw(raw_root, "Trade", date)
    date_root = processed_root / date
    (date_root / "trades").mkdir(parents=True)
    (date_root / "trades" / "AAPL.parquet").touch()
    _write_manifest(date_root, [("AAPL", 10, 0)])

    audit_rows, todo_rows, _, summary = audit_data_availability.build_audit(
        raw_roots=[raw_root],
        processed_roots=[processed_root],
        symbols=["AAPL"],
        start=date,
        end=date,
    )

    row = audit_rows[0]
    assert row["trade_parquet_available"] is True
    assert row["trade_side_done"] is False
    assert row["trade_qc_status"] == "missing"
    assert row["status"] == ps.STATUS_QC_PROBLEM
    assert todo_rows[0]["side"] == "trade"
    assert summary["qc_problem_dates"] == 1


@pytest.mark.unit
def test_audit_chooses_best_processed_root_for_overlapping_dates(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_overlap"
    weak_root = artifact_dir / "processed_a_trade_only"
    best_root = artifact_dir / "processed_b_complete"
    date = "20190107"
    _raw(raw_root, "Trade", date)
    _raw(raw_root, "NBBO", date)

    weak_date = weak_root / date
    (weak_date / "trades").mkdir(parents=True)
    (weak_date / "trades" / "AAPL.parquet").touch()
    _write_manifest(weak_date, [("AAPL", 10, 0)])
    _write_trade_qc(weak_date)

    best_date = best_root / date
    (best_date / "trades").mkdir(parents=True)
    (best_date / "nbbo").mkdir(parents=True)
    (best_date / "trades" / "AAPL.parquet").touch()
    (best_date / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(best_date, [("AAPL", 10, 20)])
    _write_trade_qc(best_date)
    _write_coverage(best_root, date, expected_symbols=1)

    audit_rows, todo_rows, missing_rows, summary = audit_data_availability.build_audit(
        raw_roots=[raw_root],
        processed_roots=[weak_root, best_root],
        symbols=["AAPL"],
        start=date,
        end=date,
    )

    assert audit_rows[0]["processed_root"] == str(best_root)
    assert audit_rows[0]["status"] == ps.STATUS_COMPLETE
    assert todo_rows == []
    assert missing_rows == []
    assert summary["complete_dates"] == 1


@pytest.mark.unit
def test_audit_outputs_todo_for_missing_trade_counterpart(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_trade_late"
    processed_root = artifact_dir / "processed_trade_late"
    date = "20190108"
    _raw(raw_root, "Trade", date)
    _raw(raw_root, "NBBO", date)
    date_root = processed_root / date
    (date_root / "nbbo").mkdir(parents=True)
    (date_root / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(date_root, [("AAPL", 0, 20)])

    audit_rows, todo_rows, _, summary = audit_data_availability.build_audit(
        raw_roots=[raw_root],
        processed_roots=[processed_root],
        symbols=["AAPL"],
        start=date,
        end=date,
    )

    assert audit_rows[0]["status"] == ps.STATUS_NBBO_ONLY
    assert audit_rows[0]["suggested_processed_side"] == "trade"
    assert todo_rows == [{
        "date": date,
        "side": "trade",
        "status": ps.STATUS_NBBO_ONLY,
        "processed_root": str(processed_root),
        "raw_paths": str(raw_root / "Trade" / f"EQY_US_ALL_TRADE_{date}.gz"),
        "reason": "raw_trade_available_but_trade_side_not_done",
    }]
    assert summary["raw_trade_unprocessed_dates"] == 1
    assert summary["raw_nbbo_unprocessed_dates"] == 0



@pytest.mark.unit
def test_validate_run_config_rejects_unordered_windows_and_strict_missing_paths(artifact_dir: Path) -> None:
    config = _valid_config(artifact_dir)
    config["volume_db"] = str(artifact_dir / "does_not_exist.duckdb")
    config["evaluation"]["start"] = "2020-01-01"
    config["evaluation"]["end"] = "2019-12-31"

    errors = validate_config(config, allow_missing_paths=False)

    assert any("volume_db does not exist" in error for error in errors)
    assert any("evaluation.start must not be after evaluation.end" in error for error in errors)


@pytest.mark.unit
def test_raw_discovery_and_resolver_handle_flat_suffixed_gz_files(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "flat_raw"
    raw_root.mkdir(parents=True)
    date = "20181016"
    trade_path = raw_root / f"EQY_US_ALL_TRADE_{date}-002.gz"
    nbbo_path = raw_root / f"EQY_US_ALL_NBBO_{date}-001.gz"
    trade_path.touch()
    nbbo_path.touch()

    discovered = ps.discover_raw_date_sides([raw_root])

    assert discovered[date]["trade"] is True
    assert discovered[date]["nbbo"] is True
    assert discovered[date]["trade_paths"] == [str(trade_path)]
    assert discovered[date]["nbbo_paths"] == [str(nbbo_path)]
    assert preprocess_taq._resolve_path(str(raw_root), "EQY_US_ALL_TRADE", date) == str(trade_path)
    assert preprocess_taq._resolve_path(str(raw_root), "EQY_US_ALL_NBBO", date) == str(nbbo_path)


@pytest.mark.unit
def test_conservative_union_builder_includes_manual_old_and_new_tickers(artifact_dir: Path) -> None:
    ref_dir = artifact_dir / "membership_ref"
    ref_dir.mkdir(parents=True)

    symbols, metadata = build_conservative_union.build(
        ref_dir, [], __import__("datetime").date(2018, 1, 2), __import__("datetime").date(2019, 12, 31),
    )

    for symbol in ["PCLN", "BKNG", "HCN", "WELL", "CBG", "CBRE", "LUK", "JEF"]:
        assert symbol in symbols
    assert metadata["policy"] == "conservative_union_v1"
    assert metadata["over_inclusion_intentional"] is True


@pytest.mark.unit
def test_audit_reports_union_and_active_membership_coverage_separately(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_union_active"
    processed_root = artifact_dir / "processed_union_active"
    date = "20190110"
    _raw(raw_root, "Trade", date)
    _raw(raw_root, "NBBO", date)
    date_root = processed_root / date
    (date_root / "trades").mkdir(parents=True)
    (date_root / "nbbo").mkdir(parents=True)
    for symbol in ["AAPL", "MSFT"]:
        (date_root / "trades" / f"{symbol}.parquet").touch()
        (date_root / "nbbo" / f"{symbol}.parquet").touch()
    _write_manifest(date_root, [("AAPL", 10, 20), ("MSFT", 10, 20)])
    _write_trade_qc(date_root)
    _write_coverage(processed_root, date, expected_symbols=3)
    membership = artifact_dir / "membership.csv"
    membership.write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,AAPL,2019-01-01,2019-12-31\n",
        encoding="utf-8",
    )

    audit_rows, _, _, summary = audit_data_availability.build_audit(
        raw_roots=[raw_root],
        processed_roots=[processed_root],
        symbols=["AAPL", "MSFT", "ZZZ"],
        start=date,
        end=date,
        membership_file=membership,
    )

    row = audit_rows[0]
    assert row["status"] == ps.STATUS_COMPLETE_WITH_MISSING
    assert row["active_membership_symbols"] == 1
    assert row["active_complete_symbols"] == 1
    assert row["extra_preprocessed_complete_symbols"] == 1
    assert summary["union_coverage"] == pytest.approx(2 / 3)
    assert summary["active_membership_coverage"] == pytest.approx(1.0)
