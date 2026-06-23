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
        "render_data_quality_audit_report_for_tests",
        PREPROCESSING_ROOT / "render_data_quality_audit_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reporter = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _base_summary(**overrides) -> dict:
    payload = {
        "created_at": "2026-06-08T15:00:00",
        "start": "20180102",
        "end": "20191231",
        "dates": 500,
        "complete_dates": 500,
        "partial_dates": 0,
        "todo_rows": 0,
        "union_coverage": 0.91,
        "active_membership_coverage": 1.0,
        "raw_trade_dates": 500,
        "raw_nbbo_dates": 500,
        "manifest_inconsistent_dates": 0,
        "qc_problem_dates": 0,
        "active_membership_missing_trade_symbol_days": 0,
        "active_membership_missing_nbbo_symbol_days": 0,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_clean_audit_report_produces_go_status(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "clean_audit"
    _write_json(audit_dir / "date_side_audit_summary.json", _base_summary())

    out = reporter.render_report(audit_dir)
    text = out.read_text(encoding="utf-8")

    assert "| Active membership quality | GO |" in text
    assert "| Final data build | READY |" in text
    assert "No hard blockers were found" in text


@pytest.mark.unit
def test_manifest_and_qc_issues_produce_no_go(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "blocked_audit"
    _write_json(
        audit_dir / "date_side_audit_summary.json",
        _base_summary(manifest_inconsistent_dates=1, qc_problem_dates=2),
    )

    text = reporter.build_report(audit_dir)

    assert "| Active membership quality | NO-GO |" in text
    assert "`manifest_inconsistent_dates` = `1`" in text
    assert "`qc_problem_dates` = `2`" in text


@pytest.mark.unit
def test_active_membership_missing_symbols_are_blockers(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "active_gap_audit"
    _write_json(
        audit_dir / "date_side_audit_summary.json",
        _base_summary(
            active_membership_coverage=0.99,
            active_membership_missing_trade_symbol_days=3,
            active_membership_missing_nbbo_symbol_days=1,
        ),
    )
    _write_csv(audit_dir / "date_side_audit.csv", [{
        "date": "20190102",
        "status": "complete_with_missing_symbols",
        "active_missing_trade_count": "3",
        "active_missing_nbbo_count": "1",
        "missing_trade_count": "10",
        "missing_nbbo_count": "5",
        "raw_trade_gz_available": "True",
        "raw_nbbo_gz_available": "True",
    }])

    text = reporter.build_report(audit_dir)

    assert "| Active membership quality | NO-GO |" in text
    assert "These rows are final-run blockers" in text
    assert "| 20190102 | complete_with_missing_symbols | 3 | 1 |" in text
    assert "Raw-Present Active-Member Supplement Queue" in text
    assert "| 2019-01 | nbbo | 1 | 1 |" in text
    assert "| 2019-01 | trade | 1 | 3 |" in text


@pytest.mark.unit
def test_conservative_union_gaps_are_nonblocking_without_active_gaps(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "union_gap_audit"
    _write_json(audit_dir / "date_side_audit_summary.json", _base_summary(union_coverage=0.8))
    _write_csv(audit_dir / "date_side_audit.csv", [{
        "date": "20190103",
        "status": "complete_with_missing_symbols",
        "active_missing_trade_count": "0",
        "active_missing_nbbo_count": "0",
        "missing_trade_count": "4",
        "missing_nbbo_count": "4",
        "raw_trade_gz_available": "True",
        "raw_nbbo_gz_available": "True",
    }])

    text = reporter.build_report(audit_dir)

    assert "| Active membership quality | GO |" in text
    assert "not automatically fatal" in text
    assert "| 20190103 | complete_with_missing_symbols | 4 | 4 |" in text


@pytest.mark.unit
def test_todo_rows_are_grouped_by_side_status_and_month(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "todo_audit"
    _write_json(audit_dir / "date_side_audit_summary.json", _base_summary(todo_rows=3))
    _write_csv(audit_dir / "preprocessing_todo.csv", [
        {"date": "20190102", "side": "trade", "status": "nbbo_only_processed"},
        {"date": "20190103", "side": "nbbo", "status": "trade_only_processed"},
        {"date": "20190201", "side": "nbbo", "status": "trade_only_processed"},
    ])

    text = reporter.build_report(audit_dir)

    assert "| Final data build | WAIT |" in text
    assert "| nbbo | 2 |" in text
    assert "| trade | 1 |" in text
    assert "| 2019-01 | nbbo | 1 |" in text
    assert "| 2019-02 | nbbo | 1 |" in text


@pytest.mark.unit
def test_raw_missing_active_rows_are_separated_from_supplement_queue(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "raw_missing_active_audit"
    _write_json(
        audit_dir / "date_side_audit_summary.json",
        _base_summary(
            active_membership_coverage=0.99,
            active_membership_missing_nbbo_symbol_days=2,
        ),
    )
    _write_csv(audit_dir / "date_side_audit.csv", [{
        "date": "20190711",
        "status": "unprocessed",
        "active_missing_trade_count": "0",
        "active_missing_nbbo_count": "2",
        "missing_trade_count": "0",
        "missing_nbbo_count": "20",
        "raw_trade_gz_available": "True",
        "raw_nbbo_gz_available": "False",
    }])

    text = reporter.build_report(audit_dir)

    assert "No raw-present active membership supplement rows were found." in text
    assert "Raw-Missing Active-Member Blockers" in text
    assert "| 20190711 | nbbo | 2 | EQY_US_ALL_NBBO |" in text


@pytest.mark.unit
def test_missing_optional_csvs_still_render_readable_report(artifact_dir: Path) -> None:
    audit_dir = artifact_dir / "summary_only_audit"
    _write_json(audit_dir / "date_side_audit_summary.json", _base_summary())

    text = reporter.build_report(audit_dir)

    assert "No preprocessing todo rows were found." in text
    assert "No active S&P membership Trade/NBBO symbol gaps were found." in text
    assert "No missing-symbol rows were found." in text
