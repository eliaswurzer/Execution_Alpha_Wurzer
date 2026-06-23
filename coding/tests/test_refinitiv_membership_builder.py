from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


PREPROCESSING_ROOT = Path(__file__).resolve().parents[1] / "preprocessing"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_sp500_refinitiv_membership_for_tests",
        PREPROCESSING_ROOT / "build_sp500_refinitiv_membership.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _active_symbols(rows: list[dict[str, str]], date: str) -> set[str]:
    d = builder.parse_date(date)
    return {
        row["symbol"]
        for row in rows
        if builder.parse_date(row["effective_from"]) <= d <= builder.parse_date(row["effective_to"])
    }


def _read_project_intervals() -> list[dict[str, str]]:
    path = REPO_ROOT / "reference" / "index_membership" / "sp500_membership_intervals.csv"
    if not path.exists():
        pytest.skip("licensed project membership intervals are not included in the public repo")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.unit
def test_ric_parser_handles_suffixes_share_classes_and_google() -> None:
    assert builder.ric_root_to_symbol("BRKb.N") == "BRK B"
    assert builder.ric_root_to_symbol("BFb.N") == "BF B"
    assert builder.ric_root_to_symbol("GOOGL.OQ") == "GOOG L"
    assert builder.ric_root_to_symbol("COMS.OQ^D10") == "COMS"


@pytest.mark.unit
def test_crosswalk_is_date_bounded(artifact_dir: Path) -> None:
    crosswalk = artifact_dir / "crosswalk.csv"
    _write_csv(crosswalk, [
        {
            "refinitiv_symbol": "META",
            "taq_symbol": "FB",
            "effective_from": "1900-01-01",
            "effective_to": "2022-06-08",
            "reason": "test",
            "source_note": "test",
        },
        {
            "refinitiv_symbol": "META",
            "taq_symbol": "META",
            "effective_from": "2022-06-09",
            "effective_to": "2099-12-31",
            "reason": "test",
            "source_note": "test",
        },
    ])
    rows = builder.load_crosswalk(crosswalk)
    assert builder.map_refinitiv_symbol("META", builder.parse_date("2019-01-02"), rows) == ("FB", "test")
    assert builder.map_refinitiv_symbol("META", builder.parse_date("2023-01-02"), rows) == ("META", "test")


@pytest.mark.unit
def test_refinitiv_builder_writes_intervals_diff_and_audit(artifact_dir: Path) -> None:
    refinitiv = artifact_dir / "constituents.csv"
    _write_csv(refinitiv, [
        {"Date": "2017-12-29", "RIC": "META.OQ"},
        {"Date": "2017-12-29", "RIC": "BRKb.N"},
        {"Date": "2018-01-03", "RIC": "META.OQ"},
        {"Date": "2018-01-03", "RIC": "GOOGL.OQ"},
        {"Date": "2018-01-05", "RIC": "ABC.N"},
        {"Date": "2018-01-05", "RIC": "GOOGL.OQ"},
    ])
    crosswalk = artifact_dir / "crosswalk.csv"
    _write_csv(crosswalk, [
        {
            "refinitiv_symbol": "META",
            "taq_symbol": "FB",
            "effective_from": "1900-01-01",
            "effective_to": "2022-06-08",
            "reason": "test_mapping",
            "source_note": "test",
        }
    ])
    previous = artifact_dir / "previous.csv"
    pd.DataFrame([{
        "index_id": "sp500",
        "symbol": "FB",
        "effective_from": "2018-01-02",
        "effective_to": "2018-01-04",
        "company_name": "",
        "sector": "",
        "listing_exchange": "",
        "source": "test",
        "source_note": "test",
    }]).to_csv(previous, index=False)
    out = artifact_dir / "sp500_membership_intervals.csv"
    audit = artifact_dir / "audit.json"
    diff = artifact_dir / "diff.csv"

    result = builder.build_refinitiv_intervals(
        refinitiv,
        crosswalk,
        artifact_dir / "missing_overrides.csv",
        out,
        audit,
        diff,
        previous_file=previous,
        start=builder.parse_date("2018-01-02"),
        end=builder.parse_date("2018-01-10"),
    )

    intervals = pd.read_csv(out)
    assert result["active_on_end"] == 2
    assert set(intervals["symbol"]) == {"ABC", "BRK B", "FB", "GOOG L"}
    assert (intervals["effective_to"] >= intervals["effective_from"]).all()
    assert audit.exists()
    assert diff.exists()


@pytest.mark.unit
def test_duplicate_mapped_rics_are_audited(artifact_dir: Path) -> None:
    refinitiv = artifact_dir / "constituents.csv"
    _write_csv(refinitiv, [
        {"Date": "2018-01-02", "RIC": "META.OQ"},
        {"Date": "2018-01-02", "RIC": "FB.OQ"},
    ])
    crosswalk = artifact_dir / "crosswalk.csv"
    _write_csv(crosswalk, [{
        "refinitiv_symbol": "META",
        "taq_symbol": "FB",
        "effective_from": "1900-01-01",
        "effective_to": "2022-06-08",
        "reason": "test_mapping",
        "source_note": "test",
    }])
    out = artifact_dir / "intervals.csv"
    audit = artifact_dir / "audit.json"
    diff = artifact_dir / "diff.csv"

    result = builder.build_refinitiv_intervals(
        refinitiv,
        crosswalk,
        artifact_dir / "missing_overrides.csv",
        out,
        audit,
        diff,
        previous_file=artifact_dir / "missing_previous.csv",
        start=builder.parse_date("2018-01-02"),
        end=builder.parse_date("2018-01-03"),
    )

    assert result["active_on_end"] == 1
    assert result["duplicate_mapped_ric_rows"][0]["taq_symbol"] == "FB"


@pytest.mark.unit
def test_unmapped_implausible_active_ric_fails_validation(artifact_dir: Path) -> None:
    refinitiv = artifact_dir / "constituents.csv"
    _write_csv(refinitiv, [
        {"Date": "2018-01-02", "RIC": "BAD$RIC.N"},
    ])
    crosswalk = artifact_dir / "crosswalk.csv"
    _write_csv(crosswalk, [{
        "refinitiv_symbol": "META",
        "taq_symbol": "FB",
        "effective_from": "1900-01-01",
        "effective_to": "2022-06-08",
        "reason": "test_mapping",
        "source_note": "test",
    }])

    with pytest.raises(ValueError, match="could not be mapped"):
        builder.build_refinitiv_intervals(
            refinitiv,
            crosswalk,
            artifact_dir / "missing_overrides.csv",
            artifact_dir / "intervals.csv",
            artifact_dir / "audit.json",
            artifact_dir / "diff.csv",
            previous_file=artifact_dir / "missing_previous.csv",
            start=builder.parse_date("2018-01-02"),
            end=builder.parse_date("2018-01-03"),
        )


@pytest.mark.unit
def test_interval_overrides_replace_and_drop_symbols(artifact_dir: Path) -> None:
    refinitiv = artifact_dir / "constituents.csv"
    _write_csv(refinitiv, [
        {"Date": "2018-01-02", "RIC": "BCR.N^L17"},
        {"Date": "2018-01-02", "RIC": "BKNG.OQ"},
        {"Date": "2018-03-07", "RIC": "BKNG.OQ"},
    ])
    crosswalk = artifact_dir / "crosswalk.csv"
    _write_csv(crosswalk, [{
        "refinitiv_symbol": "BKNG",
        "taq_symbol": "PCLN",
        "effective_from": "1900-01-01",
        "effective_to": "2018-02-26",
        "reason": "test_mapping",
        "source_note": "test",
    }])
    overrides = artifact_dir / "overrides.csv"
    _write_csv(overrides, [
        {
            "symbol": "BCR",
            "effective_from": "",
            "effective_to": "",
            "action": "drop",
            "reason": "test_drop",
            "source_note": "test",
        },
        {
            "symbol": "PCLN",
            "effective_from": "2018-01-02",
            "effective_to": "2018-02-26",
            "action": "replace",
            "reason": "test_replace",
            "source_note": "test",
        },
        {
            "symbol": "BKNG",
            "effective_from": "2018-02-27",
            "effective_to": "2018-12-31",
            "action": "replace",
            "reason": "test_replace",
            "source_note": "test",
        },
    ])
    result = builder.build_refinitiv_intervals(
        refinitiv,
        crosswalk,
        overrides,
        artifact_dir / "intervals.csv",
        artifact_dir / "audit.json",
        artifact_dir / "diff.csv",
        previous_file=artifact_dir / "missing_previous.csv",
        start=builder.parse_date("2018-01-02"),
        end=builder.parse_date("2018-12-31"),
    )

    intervals = pd.read_csv(artifact_dir / "intervals.csv")
    assert "BCR" not in set(intervals["symbol"])
    assert intervals.loc[intervals["symbol"] == "PCLN", "effective_to"].iloc[0] == "2018-02-26"
    assert intervals.loc[intervals["symbol"] == "BKNG", "effective_from"].iloc[0] == "2018-02-27"
    assert result["interval_override_rows"] == 3


@pytest.mark.unit
def test_project_interval_override_file_covers_known_raw_present_absent_cases() -> None:
    path = REPO_ROOT / "reference" / "index_membership" / "sp500_refinitiv_interval_overrides.csv"
    if not path.exists():
        pytest.skip("licensed project interval overrides are not included in the public repo")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["symbol"]: row for row in csv.DictReader(handle)}

    expected = {
        "BCR": ("drop", "", ""),
        "PCLN": ("replace", "2018-01-02", "2018-02-26"),
        "BKNG": ("replace", "2018-02-27", "2019-12-31"),
        "HCN": ("replace", "2018-01-02", "2018-02-27"),
        "WELL": ("replace", "2018-02-28", "2019-12-31"),
        "CBG": ("replace", "2018-01-02", "2018-03-19"),
        "CBRE": ("replace", "2018-03-20", "2019-12-31"),
        "LUK": ("replace", "2018-01-02", "2018-05-23"),
        "JEF": ("replace", "2018-05-24", "2019-09-25"),
        "PX": ("replace", "2018-01-02", "2018-10-30"),
        "LIN": ("replace", "2018-10-31", "2019-12-31"),
        "BHGE": ("replace", "2018-01-02", "2019-10-17"),
        "BKR": ("replace", "2019-10-18", "2019-12-31"),
        "HCP": ("replace", "2018-01-02", "2019-11-04"),
        "PEAK": ("replace", "2019-11-05", "2019-12-31"),
        "SYMC": ("replace", "2018-01-02", "2019-11-04"),
        "NLOK": ("replace", "2019-11-05", "2019-12-31"),
        "JEC": ("replace", "2018-01-02", "2019-12-09"),
        "J": ("replace", "2019-12-10", "2019-12-31"),
    }
    for symbol, (action, effective_from, effective_to) in expected.items():
        assert symbol in rows
        assert rows[symbol]["action"] == action
        assert rows[symbol]["effective_from"] == effective_from
        assert rows[symbol]["effective_to"] == effective_to


@pytest.mark.unit
def test_project_intervals_resolve_raw_present_absent_ticker_boundaries() -> None:
    intervals = _read_project_intervals()

    boundary_expectations = [
        ("2018-01-02", "BCR", False),
        ("2018-02-26", "PCLN", True),
        ("2018-02-27", "PCLN", False),
        ("2018-02-27", "BKNG", True),
        ("2018-02-27", "HCN", True),
        ("2018-02-28", "HCN", False),
        ("2018-02-28", "WELL", True),
        ("2018-03-19", "CBG", True),
        ("2018-03-20", "CBG", False),
        ("2018-03-20", "CBRE", True),
        ("2018-05-09", "CHX", False),
        ("2018-05-23", "LUK", True),
        ("2018-05-24", "LUK", False),
        ("2018-05-24", "JEF", True),
        ("2018-10-30", "PX", True),
        ("2018-10-31", "PX", False),
        ("2018-10-31", "LIN", True),
        ("2018-06-14", "TWX", True),
        ("2018-06-15", "TWX", False),
        ("2018-11-02", "CA", True),
        ("2018-11-05", "CA", False),
        ("2018-11-28", "AET", True),
        ("2018-11-29", "AET", False),
        ("2018-11-26", "COL", True),
        ("2018-11-27", "COL", False),
        ("2018-12-20", "ESRX", True),
        ("2018-12-21", "ESRX", False),
        ("2019-06-10", "BMS", True),
        ("2019-06-10", "AMCR", False),
        ("2019-06-11", "BMS", False),
        ("2019-06-11", "AMCR", True),
        # Acquisition-completion days without a closing session end membership
        # on the last day with a regular close (close-based tradability).
        ("2019-07-08", "RHT", True),
        ("2019-07-09", "RHT", False),
        ("2019-09-17", "TSS", True),
        ("2019-09-18", "TSS", False),
        ("2019-10-16", "BHGE", True),
        ("2019-10-17", "BHGE", True),
        ("2019-10-17", "BKR", False),
        ("2019-10-18", "BHGE", False),
        ("2019-10-18", "BKR", True),
        ("2019-11-04", "HCP", True),
        ("2019-11-05", "HCP", False),
        ("2019-11-05", "PEAK", True),
        ("2019-11-04", "SYMC", True),
        ("2019-11-05", "SYMC", False),
        ("2019-11-05", "NLOK", True),
        ("2019-12-09", "JEC", True),
        ("2019-12-10", "JEC", False),
        ("2019-12-10", "J", True),
    ]
    for date, symbol, should_be_active in boundary_expectations:
        active = _active_symbols(intervals, date)
        assert (symbol in active) is should_be_active, f"{symbol} active on {date}"
