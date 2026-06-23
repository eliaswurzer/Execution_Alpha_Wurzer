"""Tests for the processed-root consolidation script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.consolidate_processed_roots import (
    _ordered_sources,
    consolidate,
    repair_aux,
)


def _mk(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_sources(root: Path) -> tuple[Path, Path]:
    h1 = root / "sp500_preprocess_2018h1_streaming"
    h2 = root / "sp500_preprocess_2018h2_streaming"
    # H1-only date
    _mk(h1 / "20180301" / "trades" / "AAPL.parquet", b"a" * 10)
    _mk(h1 / "20180301" / "nbbo" / "AAPL.parquet", b"b" * 11)
    _mk(h1 / "20180301" / "qc" / "trade_qc_summary.json", b"{}")
    # Overlapping H2 date: h2 must win, h1 duplicate skipped, h1-only file taken
    _mk(h2 / "20180702" / "trades" / "AAPL.parquet", b"h2-version" * 3)
    _mk(h1 / "20180702" / "trades" / "AAPL.parquet", b"h1-version!!!")
    _mk(h1 / "20180702" / "trades" / "MSFT.parquet", b"h1-only" * 2)
    _mk(h2 / "20180702" / "nbbo" / "AAPL.parquet", b"n" * 5)
    _mk(h2 / "20180702" / "qc" / "trade_qc_summary.json", b"{}")
    return h1, h2


@pytest.mark.unit
def test_ordered_sources_prefers_h2_for_second_half() -> None:
    h1 = Path("sp500_preprocess_2018h1_streaming")
    h2 = Path("sp500_preprocess_2018h2_streaming")
    assert _ordered_sources("20180301", [h1, h2])[0] == h1
    assert _ordered_sources("20180702", [h1, h2])[0] == h2


@pytest.mark.unit
def test_dry_run_plans_without_moving(artifact_dir) -> None:
    h1, h2 = _build_sources(artifact_dir)
    target = artifact_dir / "consolidated"
    manifest = consolidate([h1, h2], target, dry_run=True)
    assert manifest["status"] == "dry_run"
    # 20180301: trades+nbbo+qc = 3; 20180702: h2 trades+nbbo+qc = 3, h1-only MSFT = 1
    assert manifest["totals"]["files"] == 7
    assert manifest["totals"]["duplicates_skipped"] == 1
    assert not (target / "20180301" / "trades" / "AAPL.parquet").exists()
    assert (h1 / "20180301" / "trades" / "AAPL.parquet").exists()


@pytest.mark.unit
def test_move_consolidates_with_precedence_and_verifies(artifact_dir) -> None:
    h1, h2 = _build_sources(artifact_dir)
    target = artifact_dir / "consolidated"
    manifest = consolidate([h1, h2], target)
    assert manifest["status"] == "complete"
    assert manifest["verify_problems"] == []
    # Winner content for the duplicate is the h2 version.
    winner = (target / "20180702" / "trades" / "AAPL.parquet").read_bytes()
    assert winner == b"h2-version" * 3
    # Losing duplicate stays in place in the h1 source tree.
    assert (h1 / "20180702" / "trades" / "AAPL.parquet").exists()
    # H1-only file was taken despite h2 winning the date.
    assert (target / "20180702" / "trades" / "MSFT.parquet").exists()
    # QC files travel with the date.
    assert (target / "20180301" / "qc" / "trade_qc_summary.json").exists()
    assert (target / "20180702" / "qc" / "trade_qc_summary.json").exists()
    # Moved files are gone from their sources.
    assert not (h1 / "20180301" / "trades" / "AAPL.parquet").exists()
    assert not (h2 / "20180702" / "trades" / "AAPL.parquet").exists()
    data = json.loads((target / "consolidation_manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "complete"


@pytest.mark.unit
def test_rerun_is_idempotent(artifact_dir) -> None:
    h1, h2 = _build_sources(artifact_dir)
    target = artifact_dir / "consolidated"
    consolidate([h1, h2], target)
    again = consolidate([h1, h2], target)
    # Only the losing h1 duplicate remains in the sources; it must never
    # overwrite the target and the rerun must finish clean with zero moves.
    assert again["status"] == "complete"
    assert again["totals"]["files"] == 0
    assert again["totals"]["conflict_skipped"] == 1
    winner = (target / "20180702" / "trades" / "AAPL.parquet").read_bytes()
    assert winner == b"h2-version" * 3


@pytest.mark.unit
def test_repair_aux_moves_coverage_and_merges_manifests(artifact_dir) -> None:
    h1, h2 = _build_sources(artifact_dir)
    # Root-level coverage summaries and per-date manifests in the sources.
    _mk(h1 / "coverage" / "20180301_summary.json", b"{}")
    _mk(h2 / "coverage" / "20180702_summary.json", b"{}")
    _mk(h1 / "20180301" / "manifest.csv",
        b"symbol,trade_rows,nbbo_rows\nAAPL,10,11\n")
    # Overlap date: winner (h2) manifest lists only AAPL; the h1-only MSFT
    # trade row lives in the losing manifest and must be merged in.
    _mk(h2 / "20180702" / "manifest.csv",
        b"symbol,trade_rows,nbbo_rows\nAAPL,30,5\n")
    _mk(h1 / "20180702" / "manifest.csv",
        b"symbol,trade_rows,nbbo_rows\nAAPL,13,0\nMSFT,14,0\n")
    target = artifact_dir / "consolidated"
    consolidate([h1, h2], target)
    stats = repair_aux([h1, h2], target)
    assert stats["manifest_problems"] == []
    assert (target / "coverage" / "20180301_summary.json").exists()
    assert (target / "coverage" / "20180702_summary.json").exists()
    merged = (target / "20180702" / "manifest.csv").read_text(encoding="utf-8")
    assert "MSFT,14,0" in merged          # loser-only trade row merged in
    assert "AAPL,30,5" in merged          # winner row kept for the duplicate
    assert stats["manifests_merged"] >= 1


@pytest.mark.unit
def test_membership_report_flags_missing_active(artifact_dir) -> None:
    h1, h2 = _build_sources(artifact_dir)
    membership = artifact_dir / "membership.csv"
    membership.write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,AAPL,2018-01-01,\n"
        "sp500,MSFT,2018-01-01,\n"
        "sp500,XYZ,2018-01-01,\n",
        encoding="utf-8",
    )
    target = artifact_dir / "consolidated"
    manifest = consolidate([h1, h2], target, membership_file=membership)
    summary = manifest["membership_summary"]
    assert summary["dates_checked"] == 2
    # XYZ is active but has no trade file on either date; MSFT missing on 0301.
    assert summary["dates_with_missing_active_trades"] == 2
    report = (target / "membership_check.csv").read_text(encoding="utf-8")
    assert "XYZ" in report
