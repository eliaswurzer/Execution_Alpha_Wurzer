from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data import index_universe, taq_loader


def _set_taq_root(monkeypatch: pytest.MonkeyPatch, artifact_dir):
    root = artifact_dir / "taq_root"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    return root


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _qc_summary(root, date: dt.date) -> None:
    path = root / date.strftime("%Y%m%d") / "qc" / "trade_qc_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "trade_filter_policy": cfg.EXPECTED_PREPROCESS_TRADE_FILTER_POLICY,
            "trade_condition_policy_version": cfg.TRADE_CONDITION_POLICY_VERSION,
        }),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_loader_resolves_all_supported_trade_layouts(monkeypatch, artifact_dir) -> None:
    root = _set_taq_root(monkeypatch, artifact_dir)
    date = dt.date(2018, 1, 2)
    ds = date.strftime("%Y%m%d")
    paths = {
        "new": root / "Trade" / ds / "AAPL.parquet",
        "pilot": root / "Trade" / ds / "trades" / "MSFT.parquet",
        "legacy": root / ds / "trades" / "NVDA.parquet",
    }
    for path in paths.values():
        _touch(path)

    assert taq_loader.trades_parquet_path(date, "AAPL") == paths["new"]
    assert taq_loader.trades_parquet_path(date, "MSFT") == paths["pilot"]
    assert taq_loader.trades_parquet_path(date, "NVDA") == paths["legacy"]


@pytest.mark.unit
def test_loader_list_dates_symbols_and_file_safe_alias(monkeypatch, artifact_dir) -> None:
    root = _set_taq_root(monkeypatch, artifact_dir)
    date = dt.date(2018, 1, 2)
    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "off")
    _touch(root / "20180102" / "trades" / "BRK_B.parquet")
    _touch(root / "Trade" / "20180103" / "trades" / "AAPL.parquet")

    assert taq_loader.list_dates(2018) == [dt.date(2018, 1, 2), dt.date(2018, 1, 3)]
    assert taq_loader.list_symbols(date) == ["BRK B"]
    assert taq_loader.trades_parquet_path(date, "BRK B").name == "BRK_B.parquet"


@pytest.mark.unit
def test_trade_qc_modes_off_warn_and_enforce(monkeypatch, artifact_dir) -> None:
    _set_taq_root(monkeypatch, artifact_dir)
    date = dt.date(2018, 1, 2)
    monkeypatch.setattr(taq_loader, "read_trade_qc_summary", lambda _date: None)
    taq_loader._QC_POLICY_WARNED_DATES.clear()

    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "off")
    taq_loader.ensure_trade_qc_policy(date)

    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "warn")
    taq_loader.ensure_trade_qc_policy(date)

    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "enforce")
    with pytest.raises(taq_loader.TradePolicyMismatchError):
        taq_loader.ensure_trade_qc_policy(date)


@pytest.mark.unit
def test_trade_qc_status_reads_expected_manifest(monkeypatch, artifact_dir) -> None:
    root = _set_taq_root(monkeypatch, artifact_dir)
    date = dt.date(2018, 1, 2)
    _qc_summary(root, date)

    assert taq_loader.trade_qc_policy_status(date) == (True, "ok")


def _write_membership(root, rows) -> None:
    path = root / "sp500_membership_intervals.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.unit
def test_index_membership_intervals_are_canonical_by_default(artifact_dir) -> None:
    root = artifact_dir / "membership"
    root.mkdir(parents=True, exist_ok=True)
    _write_membership(root, [
        {
            "index_id": "sp500",
            "symbol": "BRK B",
            "effective_from": "2018-01-01",
            "effective_to": "2018-01-31",
        },
        {
            "index_id": "sp500",
            "symbol": "AAPL",
            "effective_from": "2018-02-01",
            "effective_to": "2018-12-31",
        },
    ])

    panel = index_universe.build_index_universe_panel(
        "S&P500",
        [dt.date(2018, 1, 2), dt.date(2018, 2, 2)],
        root=root,
    )

    jan_symbols = set(panel.loc[panel["date"] == dt.date(2018, 1, 2), "symbol"])
    feb_symbols = set(panel.loc[panel["date"] == dt.date(2018, 2, 2), "symbol"])
    assert jan_symbols == {"BRK B"}
    assert feb_symbols == {"AAPL"}

    aliases = index_universe.build_index_universe_panel(
        "sp500",
        [dt.date(2018, 1, 2)],
        root=root,
        expand_aliases=True,
    )
    assert set(aliases["symbol"]) == {"BRK B", "BRK_B"}


@pytest.mark.unit
def test_index_membership_validation_errors(artifact_dir) -> None:
    root = artifact_dir / "membership_errors"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "index_id": "sp500",
        "symbol": "AAPL",
        "effective_from": "2018-01-01",
    }]).to_csv(root / "sp500_membership_intervals.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        index_universe.load_index_membership("sp500", root=root)

    _write_membership(root, [{
        "index_id": "nasdaq100",
        "symbol": "AAPL",
        "effective_from": "2018-01-01",
        "effective_to": "2018-01-31",
    }])
    with pytest.raises(ValueError, match="unexpected index_id"):
        index_universe.load_index_membership("sp500", root=root)

    _write_membership(root, [{
        "index_id": "sp500",
        "symbol": "AAPL",
        "effective_from": "2018-02-01",
        "effective_to": "2018-01-31",
    }])
    with pytest.raises(ValueError, match="effective_to < effective_from"):
        index_universe.load_index_membership("sp500", root=root)
