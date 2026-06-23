"""Regression tests for the consolidated-root date dedup bug and the
incremental expected-VC shard cache (see README_simulation_correctness_audit)."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from analysis import config as cfg
from analysis.data import taq_loader
from analysis.runners import (
    _common,
    audit_expected_vc_recovery,
    calibrate_fill_model,
    master_panel,
)
from analysis.simulation.parent_orders import rolling_expected_vc


def _shared_root(monkeypatch: pytest.MonkeyPatch, artifact_dir):
    """Consolidated layout: ONE root holding both 2018 and 2019 date dirs."""
    root = artifact_dir / "consolidated"
    for ds in ("20180102", "20180103", "20190103", "20191230"):
        (root / ds / "trades").mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2019, root)
    return root


@pytest.mark.unit
def test_list_dates_filters_by_year_on_shared_root(monkeypatch, artifact_dir) -> None:
    _shared_root(monkeypatch, artifact_dir)
    assert taq_loader.list_dates(2018) == [dt.date(2018, 1, 2), dt.date(2018, 1, 3)]
    assert taq_loader.list_dates(2019) == [dt.date(2019, 1, 3), dt.date(2019, 12, 30)]


@pytest.mark.unit
def test_eval_dates_unique_across_years_on_shared_root(monkeypatch, artifact_dir) -> None:
    _shared_root(monkeypatch, artifact_dir)
    dates = _common._eval_dates(dt.date(2018, 1, 1), dt.date(2019, 12, 31))
    assert dates == sorted(set(dates))
    assert dates == [
        dt.date(2018, 1, 2), dt.date(2018, 1, 3),
        dt.date(2019, 1, 3), dt.date(2019, 12, 30),
    ]


@pytest.mark.unit
def test_calibration_select_dates_uses_common_eval_dates(monkeypatch) -> None:
    sentinel = [dt.date(2018, 1, 3)]
    seen: dict[str, dt.date] = {}

    def fake_eval_dates(start, end):
        seen["start"] = start
        seen["end"] = end
        return sentinel

    monkeypatch.setattr(_common, "_eval_dates", fake_eval_dates)

    assert calibrate_fill_model._select_dates(
        dt.date(2018, 1, 1), dt.date(2019, 12, 31),
    ) == sentinel
    assert seen == {
        "start": dt.date(2018, 1, 1),
        "end": dt.date(2019, 12, 31),
    }


@pytest.mark.unit
def test_rolling_expected_vc_rejects_duplicate_symbol_days() -> None:
    history = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [dt.date(2018, 1, 2), dt.date(2018, 1, 2)],
        "vc_shares": [1000.0, 1000.0],
    })
    with pytest.raises(ValueError, match="duplicated"):
        rolling_expected_vc(history)


@pytest.mark.unit
def test_trade_qc_policy_status_is_memoised_per_date(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_summary(_date):
        calls["n"] += 1
        return {
            "trade_condition_policy_version": cfg.TRADE_CONDITION_POLICY_VERSION,
            "trade_filter_policy": cfg.EXPECTED_PREPROCESS_TRADE_FILTER_POLICY,
        }

    monkeypatch.setattr(taq_loader, "read_trade_qc_summary", _fake_summary)
    date = dt.date(2018, 1, 2)
    for _ in range(50):
        assert taq_loader.trade_qc_policy_status(date) == (True, "ok")
    assert calls["n"] == 1

    # A different date is a separate cache entry.
    taq_loader.trade_qc_policy_status(dt.date(2018, 1, 3))
    assert calls["n"] == 2

    # Clearing the memo forces a re-read.
    taq_loader.clear_trade_qc_policy_cache()
    taq_loader.trade_qc_policy_status(date)
    assert calls["n"] == 3


def _stub_vc_history(call_log: list):
    def _stub(dates, symbols, workers=1):
        assert len(dates) == 1
        call_log.append(dates[0])
        rows = []
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "date": dates[0],
                "vc_shares": 1000.0,
                "vc_source": "closing_trade",
                "close_price_source": "official_marker",
                "close_trade_volume": 1000.0,
                "close_trade_rows": 1,
                "official_close_marker_volume": 0.0,
                "official_close_marker_rows": 0,
                "official_close_marker_fallback_volume": 0.0,
            })
        return pd.DataFrame(rows)
    return _stub


def _identity_map_frame(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "target_symbol", "source_symbol", "effective_from", "effective_to",
        "mapping_type", "headline_allowed", "scale_factor", "source", "note",
    ]
    frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame.columns:
            frame[col] = ""
    frame = frame[columns].copy()
    frame["effective_from"] = pd.to_datetime(
        frame["effective_from"], errors="coerce",
    ).dt.date
    frame["effective_to"] = pd.to_datetime(
        frame["effective_to"], errors="coerce",
    ).dt.date
    frame["headline_allowed"] = frame["headline_allowed"].astype(bool)
    frame["scale_factor"] = pd.to_numeric(
        frame["scale_factor"], errors="coerce",
    ).fillna(1.0).astype(float)
    return frame


def _history_loader_from(
    history: pd.DataFrame,
    *,
    date_log: list | None = None,
    symbol_log: list | None = None,
):
    def _load(shard_dates, symbols, workers=1):
        if date_log is not None:
            date_log.extend(shard_dates)
        if symbol_log is not None:
            symbol_log.append(list(symbols))
        mask = history["date"].isin(shard_dates) & history["symbol"].isin(symbols)
        return history.loc[mask].reset_index(drop=True)
    return _load


@pytest.mark.unit
def test_expected_vc_shard_cache_resume_and_invalidation(monkeypatch, artifact_dir) -> None:
    hist_dates = [dt.date(2018, 1, 2), dt.date(2018, 1, 3), dt.date(2018, 1, 4)]
    eval_dates = hist_dates[-1:]
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: hist_dates)
    call_log: list = []
    monkeypatch.setattr(_common, "_vc_history", _stub_vc_history(call_log))

    cache_dir = artifact_dir / "cache"
    symbols = ["AAPL", "MSFT"]

    evc = master_panel._load_or_build_expected_vc(
        eval_dates, symbols, cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    assert call_log == hist_dates
    assert set(evc.columns) == {"symbol", "date", "expected_vc"}
    shard_dir = cache_dir / "vc_history_shards"
    assert sorted(p.name for p in shard_dir.glob("*.parquet")) == [
        "20180102.parquet", "20180103.parquet", "20180104.parquet",
    ]

    # Full resume: manifest fingerprint matches, nothing is rebuilt.
    call_log.clear()
    master_panel._load_or_build_expected_vc(
        eval_dates, symbols, cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    assert call_log == []

    # Partial resume: drop the top-level manifest and one shard; only the
    # missing shard is rebuilt from raw data.
    (cache_dir / "expected_vc_manifest.json").unlink()
    (shard_dir / "20180103.parquet").unlink()
    call_log.clear()
    master_panel._load_or_build_expected_vc(
        eval_dates, symbols, cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    assert call_log == [dt.date(2018, 1, 3)]

    # Changing the symbol set invalidates every shard fingerprint.
    (cache_dir / "expected_vc_manifest.json").unlink()
    call_log.clear()
    master_panel._load_or_build_expected_vc(
        eval_dates, ["AAPL"], cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    assert call_log == hist_dates

    # Shard manifests record the fingerprint they were built under.
    meta = json.loads((shard_dir / "20180102.json").read_text(encoding="utf-8"))
    assert meta["fingerprint"] == master_panel._vc_shard_fingerprint(["AAPL"])


@pytest.mark.unit
def test_expected_vc_uses_approved_predecessor_history(
    monkeypatch, artifact_dir,
) -> None:
    hist_dates = [dt.date(2018, 1, day) for day in range(2, 9)]
    eval_dates = [dt.date(2018, 1, 8)]
    history = pd.DataFrame([
        {
            "symbol": "OLD",
            "date": day,
            "vc_shares": float(100 * i),
            "vc_source": "closing_trade",
        }
        for i, day in enumerate(hist_dates[:-1], start=1)
    ] + [{
        "symbol": "NEW",
        "date": eval_dates[0],
        "vc_shares": 999.0,
        "vc_source": "closing_trade",
    }])
    identity_map = _identity_map_frame([{
        "target_symbol": "NEW",
        "source_symbol": "OLD",
        "effective_from": eval_dates[0],
        "effective_to": dt.date(2018, 12, 31),
        "mapping_type": "ticker_continuity",
        "headline_allowed": True,
        "scale_factor": 1.0,
        "source": "unit_test",
        "note": "Approved continuity mapping.",
    }])
    symbol_log: list[list[str]] = []
    monkeypatch.setattr(_common, "_eval_dates", lambda *_: hist_dates)
    monkeypatch.setattr(
        _common, "_vc_history",
        _history_loader_from(history, symbol_log=symbol_log),
    )
    monkeypatch.setattr(
        master_panel, "_load_expected_vc_identity_map", lambda path=None: identity_map,
    )
    monkeypatch.setattr(master_panel, "_identity_map_sha", lambda path=None: "map-sha")

    evc = master_panel._load_or_build_expected_vc(
        eval_dates, ["NEW"], artifact_dir / "cache", workers=1, resume=True,
        shard_dir=artifact_dir / "cache" / "vc_history_shards",
    )

    row = evc[(evc["symbol"] == "NEW") & (evc["date"] == eval_dates[0])].iloc[0]
    assert row["expected_vc"] == pytest.approx(350.0)
    assert all("OLD" in symbols for symbols in symbol_log)
    repaired = pd.read_parquet(
        artifact_dir / "cache" / "vc_history_for_expected_vc.parquet",
    )
    assert ((repaired["symbol"] == "NEW") & (repaired["date"] < eval_dates[0])).sum() == 6
    manifest = json.loads(
        (artifact_dir / "cache" / "expected_vc_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    assert manifest["expected_vc_policy_version"] == master_panel.EXPECTED_VC_POLICY_VERSION
    assert manifest["identity_map_sha256"] == "map-sha"
    assert manifest["predecessor_rows_used"] == 6


@pytest.mark.unit
def test_expected_vc_does_not_use_disallowed_predecessor_mapping(
    monkeypatch, artifact_dir,
) -> None:
    hist_dates = [dt.date(2018, 1, day) for day in range(2, 9)]
    eval_dates = [dt.date(2018, 1, 8)]
    history = pd.DataFrame([
        {
            "symbol": "OLD",
            "date": day,
            "vc_shares": float(100 * i),
            "vc_source": "closing_trade",
        }
        for i, day in enumerate(hist_dates[:-1], start=1)
    ] + [{
        "symbol": "NEW",
        "date": eval_dates[0],
        "vc_shares": 999.0,
        "vc_source": "closing_trade",
    }])
    identity_map = _identity_map_frame([{
        "target_symbol": "NEW",
        "source_symbol": "OLD",
        "effective_from": eval_dates[0],
        "effective_to": dt.date(2018, 12, 31),
        "mapping_type": "ambiguous_spin_or_split",
        "headline_allowed": False,
        "scale_factor": 1.0,
        "source": "unit_test",
        "note": "Disallowed without verified scaling.",
    }])
    symbol_log: list[list[str]] = []
    monkeypatch.setattr(_common, "_eval_dates", lambda *_: hist_dates)
    monkeypatch.setattr(
        _common, "_vc_history",
        _history_loader_from(history, symbol_log=symbol_log),
    )
    monkeypatch.setattr(
        master_panel, "_load_expected_vc_identity_map", lambda path=None: identity_map,
    )
    monkeypatch.setattr(master_panel, "_identity_map_sha", lambda path=None: "map-sha")

    evc = master_panel._load_or_build_expected_vc(
        eval_dates, ["NEW"], artifact_dir / "cache", workers=1, resume=True,
        shard_dir=artifact_dir / "cache" / "vc_history_shards",
    )

    row = evc[(evc["symbol"] == "NEW") & (evc["date"] == eval_dates[0])].iloc[0]
    assert pd.isna(row["expected_vc"])
    assert all("OLD" not in symbols for symbols in symbol_log)
    manifest = json.loads(
        (artifact_dir / "cache" / "expected_vc_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    assert manifest["predecessor_rows_used"] == 0
    assert manifest["ambiguous_rows_left_unrepaired"] == 1


@pytest.mark.unit
def test_expected_vc_manifest_changes_when_identity_map_sha_changes(
    monkeypatch, artifact_dir,
) -> None:
    hist_dates = [dt.date(2018, 1, day) for day in range(2, 9)]
    history = pd.DataFrame({
        "symbol": ["AAPL"] * len(hist_dates),
        "date": hist_dates,
        "vc_shares": [100, 200, 300, 400, 500, 600, 700],
        "vc_source": ["closing_trade"] * len(hist_dates),
    })
    identity_map = _identity_map_frame([])
    sha = {"value": "first-map"}
    monkeypatch.setattr(_common, "_eval_dates", lambda *_: hist_dates)
    monkeypatch.setattr(_common, "_vc_history", _history_loader_from(history))
    monkeypatch.setattr(
        master_panel, "_load_expected_vc_identity_map", lambda path=None: identity_map,
    )
    monkeypatch.setattr(
        master_panel, "_identity_map_sha", lambda path=None: sha["value"],
    )

    cache_dir = artifact_dir / "cache"
    master_panel._load_or_build_expected_vc(
        hist_dates, ["AAPL"], cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    first = json.loads(
        (cache_dir / "expected_vc_manifest.json").read_text(encoding="utf-8"),
    )
    sha["value"] = "second-map"
    master_panel._load_or_build_expected_vc(
        hist_dates, ["AAPL"], cache_dir, workers=1, resume=True, shard_dir=cache_dir / "vc_history_shards",
    )
    second = json.loads(
        (cache_dir / "expected_vc_manifest.json").read_text(encoding="utf-8"),
    )

    assert first["fingerprint"] != second["fingerprint"]
    assert second["identity_map_sha256"] == "second-map"


@pytest.mark.unit
def test_missing_expected_vc_recovery_audit_classifies_rows() -> None:
    failures = pd.DataFrame([
        {"date": "2018-01-08", "symbol": "SAME", "reason": "missing_expected_vc"},
        {"date": "2018-01-08", "symbol": "NEW", "reason": "missing_expected_vc"},
        {"date": "2018-01-08", "symbol": "AMBIG", "reason": "missing_expected_vc"},
        {"date": "2018-01-08", "symbol": "DEAD", "reason": "missing_expected_vc"},
        {"date": "2018-01-08", "symbol": "NONE", "reason": "missing_expected_vc"},
        {"date": "2018-01-08", "symbol": "IGN", "reason": "empty_after_filter"},
    ])
    prior_dates = [dt.date(2018, 1, day) for day in range(2, 8)]
    vc_history = pd.DataFrame([
        {"symbol": "SAME", "date": day, "vc_shares": 1000.0}
        for day in prior_dates
    ] + [
        {"symbol": "OLD", "date": day, "vc_shares": 2000.0}
        for day in prior_dates
    ])
    identity_map = _identity_map_frame([
        {
            "target_symbol": "NEW",
            "source_symbol": "OLD",
            "effective_from": dt.date(2018, 1, 8),
            "effective_to": dt.date(2018, 12, 31),
            "mapping_type": "ticker_continuity",
            "headline_allowed": True,
        },
        {
            "target_symbol": "AMBIG",
            "source_symbol": "AOLD",
            "effective_from": dt.date(2018, 1, 8),
            "effective_to": dt.date(2018, 12, 31),
            "mapping_type": "ambiguous_spin_or_split",
            "headline_allowed": False,
        },
    ])
    membership = pd.DataFrame([
        {
            "symbol": "DEAD",
            "effective_from": dt.date(2018, 1, 1),
            "effective_to": dt.date(2018, 1, 7),
        },
    ])

    audit = audit_expected_vc_recovery.classify_missing_expected_vc(
        failures, vc_history, identity_map, membership,
    )
    classes = dict(zip(audit["symbol"], audit["recovery_class"]))

    assert classes["SAME"] == "recoverable_same_symbol_history"
    assert classes["NEW"] == "recoverable_predecessor_mapping"
    assert classes["AMBIG"] == "ambiguous_corporate_action"
    assert classes["DEAD"] == "excluded_no_regular_close"
    assert classes["NONE"] == "unrecoverable_no_history"
    assert "IGN" not in classes
