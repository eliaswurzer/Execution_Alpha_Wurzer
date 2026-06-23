from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "preprocessing"
    / "preprocess_index_full_run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "preprocess_index_full_run_for_tests",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
preprocess_index_full_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocess_index_full_run)


def _write_manifest(path: Path, *, trade_rows: int, nbbo_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,trade_rows,nbbo_rows\n"
        f"AAPL,{trade_rows},{nbbo_rows}\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_available_side_status_becomes_complete_after_counterpart_arrives(
    artifact_dir,
) -> None:
    raw_root = artifact_dir / "raw"
    out_root = artifact_dir / "out"
    date = "20190102"
    trade_raw = raw_root / "Trade" / f"EQY_US_ALL_TRADE_{date}.gz"
    trade_raw.parent.mkdir(parents=True)
    trade_raw.touch()

    discovered = preprocess_index_full_run._discover_date_sides(raw_root)
    assert discovered == {date: {"trade": True, "nbbo": False}}

    date_root = out_root / date
    trade_path = date_root / "trades" / "AAPL.parquet"
    trade_path.parent.mkdir(parents=True)
    trade_path.touch()
    _write_manifest(
        date_root / "manifest.csv",
        trade_rows=100,
        nbbo_rows=0,
    )
    qc_path = date_root / "qc" / "trade_qc_summary.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text("{}", encoding="utf-8")

    partial = preprocess_index_full_run._date_side_status(out_root, date)
    assert partial["trade_done"] is True
    assert partial["nbbo_done"] is False
    assert partial["complete_done"] is False

    nbbo_raw = raw_root / "NBBO" / f"EQY_US_ALL_NBBO_{date}.gz"
    nbbo_raw.parent.mkdir(parents=True)
    nbbo_raw.touch()
    nbbo_path = date_root / "nbbo" / "AAPL.parquet"
    nbbo_path.parent.mkdir(parents=True)
    nbbo_path.touch()
    _write_manifest(
        date_root / "manifest.csv",
        trade_rows=100,
        nbbo_rows=200,
    )
    coverage_path = out_root / "coverage" / f"{date}_summary.json"
    coverage_path.parent.mkdir(parents=True)
    coverage_path.write_text(
        json.dumps({"expected_symbols": 1}),
        encoding="utf-8",
    )

    complete = preprocess_index_full_run._date_side_status(out_root, date)
    done, missing = preprocess_index_full_run._done_status(
        out_root, date, ["AAPL"],
    )
    assert complete["trade_done"] is True
    assert complete["nbbo_done"] is True
    assert complete["complete_done"] is True
    assert done is True
    assert missing == []


def _touch_processed_side(out_root: Path, date: str, *, trade: bool, nbbo: bool) -> None:
    date_root = out_root / date
    if trade:
        trade_path = date_root / "trades" / "AAPL.parquet"
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        trade_path.touch()
    if nbbo:
        nbbo_path = date_root / "nbbo" / "AAPL.parquet"
        nbbo_path.parent.mkdir(parents=True, exist_ok=True)
        nbbo_path.touch()
    _write_manifest(
        date_root / "manifest.csv",
        trade_rows=100 if trade else 0,
        nbbo_rows=200 if nbbo else 0,
    )
    if trade:
        qc_path = date_root / "qc" / "trade_qc_summary.json"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    if trade and nbbo:
        coverage_path = out_root / "coverage" / f"{date}_summary.json"
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(json.dumps({"expected_symbols": 1}), encoding="utf-8")


def _touch_raw(raw_root: Path, side: str, date: str) -> None:
    prefix = "EQY_US_ALL_TRADE_" if side == "Trade" else "EQY_US_ALL_NBBO_"
    raw_path = raw_root / side / f"{prefix}{date}.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch()


@pytest.mark.unit
def test_resume_selection_schedules_only_new_nbbo_side(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_nbbo_late"
    out_root = artifact_dir / "out_nbbo_late"
    date = "20190103"
    _touch_raw(raw_root, "Trade", date)
    _touch_raw(raw_root, "NBBO", date)
    _touch_processed_side(out_root, date, trade=True, nbbo=False)

    dates, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL"],
        mode="available",
    )

    assert dates == [date]
    assert skipped == 0
    assert len(todo) == 1
    assert todo[0]["processed_side"] == "nbbo"
    assert todo[0]["skip_trades"] is True
    assert todo[0]["skip_nbbo"] is False
    assert todo[0]["status_before"] == "trade_only_processed"


@pytest.mark.unit
def test_resume_selection_schedules_only_new_trade_side(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_trade_late"
    out_root = artifact_dir / "out_trade_late"
    date = "20190104"
    _touch_raw(raw_root, "Trade", date)
    _touch_raw(raw_root, "NBBO", date)
    _touch_processed_side(out_root, date, trade=False, nbbo=True)

    _, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL"],
        mode="available",
    )

    assert skipped == 0
    assert len(todo) == 1
    assert todo[0]["processed_side"] == "trade"
    assert todo[0]["skip_trades"] is False
    assert todo[0]["skip_nbbo"] is True
    assert todo[0]["status_before"] == "nbbo_only_processed"


@pytest.mark.unit
def test_resume_selection_skips_complete_without_overwrite_and_overwrites_both(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_complete"
    out_root = artifact_dir / "out_complete"
    date = "20190105"
    _touch_raw(raw_root, "Trade", date)
    _touch_raw(raw_root, "NBBO", date)
    _touch_processed_side(out_root, date, trade=True, nbbo=True)

    _, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL"],
        mode="available",
        overwrite=False,
    )
    assert todo == []
    assert skipped == 1

    _, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL"],
        mode="available",
        overwrite=True,
    )
    assert skipped == 0
    assert len(todo) == 1
    assert todo[0]["processed_side"] == "trade+nbbo"


@pytest.mark.unit
def test_resume_selection_reprocesses_manifest_inconsistent_side_only(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_bad_manifest"
    out_root = artifact_dir / "out_bad_manifest"
    date = "20190106"
    _touch_raw(raw_root, "Trade", date)
    _touch_raw(raw_root, "NBBO", date)
    date_root = out_root / date
    (date_root / "trades").mkdir(parents=True)
    (date_root / "nbbo").mkdir(parents=True)
    (date_root / "trades" / "AAPL.parquet").touch()
    (date_root / "trades" / "MSFT.parquet").touch()
    (date_root / "nbbo" / "AAPL.parquet").touch()
    _write_manifest(date_root / "manifest.csv", trade_rows=100, nbbo_rows=200)
    qc_path = date_root / "qc" / "trade_qc_summary.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    _, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL", "MSFT"],
        mode="available",
    )

    assert skipped == 0
    assert len(todo) == 1
    assert todo[0]["status_before"] == "manifest_inconsistent"
    assert todo[0]["processed_side"] == "trade"


@pytest.mark.unit
def test_dry_run_reports_todo_without_writing_run_summary(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_dry_run"
    out_root = artifact_dir / "out_dry_run"
    symbols_file = artifact_dir / "symbols.txt"
    date = "20190107"
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    _touch_raw(raw_root, "Trade", date)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--raw-root", str(raw_root),
            "--out-root", str(out_root),
            "--symbols-file", str(symbols_file),
            "--mode", "available",
            "--start", date,
            "--end", date,
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"[TODO] {date}: trade-only" in result.stdout
    assert not (out_root / "run_summary.csv").exists()


@pytest.mark.unit
def test_resume_selection_supplements_missing_symbols_without_overwrite(artifact_dir: Path) -> None:
    raw_root = artifact_dir / "raw_supplement"
    out_root = artifact_dir / "out_supplement"
    date = "20190109"
    _touch_raw(raw_root, "Trade", date)
    _touch_raw(raw_root, "NBBO", date)
    _touch_processed_side(out_root, date, trade=True, nbbo=True)

    _, todo, skipped = preprocess_index_full_run._select_preprocessing_todo(
        raw_root=raw_root,
        out_root=out_root,
        symbols=["AAPL", "MSFT"],
        mode="available",
        overwrite=False,
    )

    assert skipped == 0
    assert len(todo) == 1
    assert todo[0]["processed_side"] == "trade+nbbo"
    assert todo[0]["supplement_missing"] is True
    assert todo[0]["trade_symbols_override"] == ["MSFT"]
    assert todo[0]["nbbo_symbols_override"] == ["MSFT"]
    assert todo[0]["status_before"] == "complete_with_missing_symbols"
