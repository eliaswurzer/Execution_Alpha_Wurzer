from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PREPROCESSING_ROOT = Path(__file__).resolve().parents[1] / "preprocessing"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "preprocess_active_membership_gaps_for_tests",
        PREPROCESSING_ROOT / "preprocess_active_membership_gaps.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "date",
        "raw_trade_gz_available",
        "raw_nbbo_gz_available",
        "raw_trade_paths",
        "raw_nbbo_paths",
        "processed_root",
        "status",
        "trade_side_done",
        "nbbo_side_done",
        "trade_qc_status",
        "active_missing_trade_count",
        "active_missing_nbbo_count",
        "active_missing_trade_symbols",
        "active_missing_nbbo_symbols",
        "missing_trade_count",
        "missing_nbbo_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_audit_dir(artifact_dir: Path, raw_root: Path, processed_root: Path, rows: list[dict[str, str]]) -> Path:
    audit_dir = artifact_dir / "audit"
    _write_json(audit_dir / "date_side_audit_summary.json", {"raw_roots": [str(raw_root)]})
    _write_audit(audit_dir / "date_side_audit.csv", rows)
    processed_root.mkdir(parents=True, exist_ok=True)
    return audit_dir


@pytest.mark.unit
def test_trade_raw_present_missing_active_symbols_schedule_trade_only(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw root"
    processed_root = artifact_dir / "processed"
    date = "20190102"
    raw_file = raw_root / "Trade" / f"EQY_US_ALL_TRADE_{date}.gz"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    audit_dir = _base_audit_dir(artifact_dir, raw_root, processed_root, [{
        "date": date,
        "raw_trade_gz_available": "True",
        "raw_nbbo_gz_available": "False",
        "raw_trade_paths": str(raw_file),
        "raw_nbbo_paths": "",
        "processed_root": str(processed_root),
        "status": "complete_with_missing_symbols",
        "trade_side_done": "True",
        "nbbo_side_done": "True",
        "trade_qc_status": "ok",
        "active_missing_trade_count": "2",
        "active_missing_nbbo_count": "0",
        "active_missing_trade_symbols": '["ABC", "XYZ"]',
        "active_missing_nbbo_symbols": "[]",
        "missing_trade_count": "2",
        "missing_nbbo_count": "0",
    }])

    jobs, raw_missing = runner.build_jobs(audit_dir)

    assert raw_missing == []
    assert len(jobs) == 1
    assert jobs[0].side == "trade"
    assert jobs[0].symbols == ("ABC", "XYZ")
    assert jobs[0].raw_root == raw_root


@pytest.mark.unit
def test_nbbo_raw_missing_active_gap_goes_to_raw_checklist(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    audit_dir = _base_audit_dir(artifact_dir, raw_root, processed_root, [{
        "date": "20190711",
        "raw_trade_gz_available": "False",
        "raw_nbbo_gz_available": "False",
        "raw_trade_paths": "",
        "raw_nbbo_paths": "",
        "processed_root": str(processed_root),
        "status": "unprocessed",
        "trade_side_done": "False",
        "nbbo_side_done": "False",
        "trade_qc_status": "missing",
        "active_missing_trade_count": "0",
        "active_missing_nbbo_count": "3",
        "active_missing_trade_symbols": "[]",
        "active_missing_nbbo_symbols": '["A", "B", "C"]',
        "missing_trade_count": "0",
        "missing_nbbo_count": "20",
    }])

    jobs, raw_missing = runner.build_jobs(audit_dir)

    assert jobs == []
    assert len(raw_missing) == 1
    assert raw_missing[0]["side"] == "nbbo"
    assert raw_missing[0]["expected_raw_filename"] == "EQY_US_ALL_NBBO_20190711.gz"


@pytest.mark.unit
def test_union_only_gap_does_not_schedule_supplement(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    date = "20190103"
    raw_file = raw_root / "NBBO" / f"EQY_US_ALL_NBBO_{date}.gz"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    audit_dir = _base_audit_dir(artifact_dir, raw_root, processed_root, [{
        "date": date,
        "raw_trade_gz_available": "False",
        "raw_nbbo_gz_available": "True",
        "raw_trade_paths": "",
        "raw_nbbo_paths": str(raw_file),
        "processed_root": str(processed_root),
        "status": "complete_with_missing_symbols",
        "trade_side_done": "True",
        "nbbo_side_done": "True",
        "trade_qc_status": "ok",
        "active_missing_trade_count": "0",
        "active_missing_nbbo_count": "0",
        "active_missing_trade_symbols": "[]",
        "active_missing_nbbo_symbols": "[]",
        "missing_trade_count": "4",
        "missing_nbbo_count": "4",
    }])

    jobs, raw_missing = runner.build_jobs(audit_dir)

    assert jobs == []
    assert raw_missing == []


@pytest.mark.unit
def test_existing_parquet_symbols_are_not_requested_again(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    date = "20190104"
    raw_file = raw_root / "Trade" / f"EQY_US_ALL_TRADE_{date}.gz"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    existing = processed_root / date / "trades"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "ABC.parquet").write_text("already there", encoding="utf-8")
    audit_dir = _base_audit_dir(artifact_dir, raw_root, processed_root, [{
        "date": date,
        "raw_trade_gz_available": "True",
        "raw_nbbo_gz_available": "False",
        "raw_trade_paths": str(raw_file),
        "raw_nbbo_paths": "",
        "processed_root": str(processed_root),
        "status": "complete_with_missing_symbols",
        "trade_side_done": "True",
        "nbbo_side_done": "True",
        "trade_qc_status": "ok",
        "active_missing_trade_count": "2",
        "active_missing_nbbo_count": "0",
        "active_missing_trade_symbols": '["ABC", "XYZ"]',
        "active_missing_nbbo_symbols": "[]",
        "missing_trade_count": "2",
        "missing_nbbo_count": "0",
    }])

    jobs, _ = runner.build_jobs(audit_dir)

    assert len(jobs) == 1
    assert jobs[0].symbols == ("XYZ",)


@pytest.mark.unit
def test_command_uses_side_specific_symbol_file_and_skip_flag(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    raw_file = raw_root / "NBBO" / "EQY_US_ALL_NBBO_20190105.gz"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    job = runner.SupplementJob(
        date="20190105",
        side="nbbo",
        raw_root=raw_root,
        raw_path=raw_file,
        processed_root=processed_root,
        symbols=("BRK B", "GOOG L"),
        status_before="complete_with_missing_symbols",
        trade_done_before="True",
        nbbo_done_before="True",
        qc_status="ok",
    )

    command = runner.build_command(
        job,
        out_dir=artifact_dir / "run",
        chunksize=123,
        schema="slim",
        trade_filter_policy="preprocessing",
    )

    assert "--nbbo-symbols-file" in command
    assert "--skip-trades" in command
    assert "--skip-nbbo" not in command
    symbol_file = Path(command[command.index("--nbbo-symbols-file") + 1])
    assert symbol_file.read_text(encoding="utf-8").splitlines() == ["BRK B", "GOOG L"]


@pytest.mark.unit
def test_limit_only_limits_jobs_not_raw_missing_checklist(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw"
    processed_root = artifact_dir / "processed"
    raw_file = raw_root / "Trade" / "EQY_US_ALL_TRADE_20190102.gz"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    audit_dir = _base_audit_dir(artifact_dir, raw_root, processed_root, [
        {
            "date": "20190102",
            "raw_trade_gz_available": "True",
            "raw_nbbo_gz_available": "False",
            "raw_trade_paths": str(raw_file),
            "raw_nbbo_paths": "",
            "processed_root": str(processed_root),
            "status": "complete_with_missing_symbols",
            "trade_side_done": "True",
            "nbbo_side_done": "False",
            "trade_qc_status": "ok",
            "active_missing_trade_count": "1",
            "active_missing_nbbo_count": "0",
            "active_missing_trade_symbols": '["ABC"]',
            "active_missing_nbbo_symbols": "[]",
            "missing_trade_count": "1",
            "missing_nbbo_count": "0",
        },
        {
            "date": "20190103",
            "raw_trade_gz_available": "False",
            "raw_nbbo_gz_available": "False",
            "raw_trade_paths": "",
            "raw_nbbo_paths": "",
            "processed_root": str(processed_root),
            "status": "unprocessed",
            "trade_side_done": "False",
            "nbbo_side_done": "False",
            "trade_qc_status": "missing",
            "active_missing_trade_count": "0",
            "active_missing_nbbo_count": "1",
            "active_missing_trade_symbols": "[]",
            "active_missing_nbbo_symbols": '["XYZ"]',
            "missing_trade_count": "0",
            "missing_nbbo_count": "1",
        },
    ])

    jobs, raw_missing = runner.build_jobs(audit_dir, limit=1)

    assert len(jobs) == 1
    assert len(raw_missing) == 1
    assert raw_missing[0]["date"] == "20190103"
