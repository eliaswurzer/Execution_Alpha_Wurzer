"""Tests for the membership transition audit (delisting-day boundary check)."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from analysis import config as cfg
from preprocessing.audit_membership_transitions import audit


def _write_trades(root, ds: str, symbol: str, *, with_close: bool) -> None:
    times = [f"{ds[:4]}-{ds[4:6]}-{ds[6:]} 12:00:00"]
    conds = [""]
    if with_close:
        times.append(f"{ds[:4]}-{ds[4:6]}-{ds[6:]} 16:00:00")
        conds.append("6")
    frame = pd.DataFrame({
        "time": pd.to_datetime(times),
        "price": [100.0] * len(times),
        "volume": [500] * len(times),
        "sale_condition": conds,
        "correction": ["00"] * len(times),
    })
    path = root / ds / "trades" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.mark.unit
def test_transition_audit_flags_drop_without_close(monkeypatch, artifact_dir) -> None:
    root = artifact_dir / "taq"
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "off")
    days = ("20180102", "20180103", "20180104")
    # STAY trades the whole window with closes; GOODDROP ends 01-03 WITH a
    # close; BADDROP ends 01-03 WITHOUT a close (halted intraday); NEWADD
    # joins on 01-03 with full data.
    for ds in days:
        _write_trades(root, ds, "STAY", with_close=True)
    _write_trades(root, "20180102", "GOODDROP", with_close=True)
    _write_trades(root, "20180103", "GOODDROP", with_close=True)
    _write_trades(root, "20180102", "BADDROP", with_close=True)
    _write_trades(root, "20180103", "BADDROP", with_close=False)
    _write_trades(root, "20180103", "NEWADD", with_close=True)
    _write_trades(root, "20180104", "NEWADD", with_close=True)

    membership = artifact_dir / "membership.csv"
    membership.write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,STAY,2018-01-02,2018-01-04\n"
        "sp500,GOODDROP,2018-01-02,2018-01-03\n"
        "sp500,BADDROP,2018-01-02,2018-01-03\n"
        "sp500,NEWADD,2018-01-03,2018-01-04\n",
        encoding="utf-8",
    )
    out_dir = artifact_dir / "audit"
    summary = audit(
        membership, dt.date(2018, 1, 2), dt.date(2018, 1, 4), out_dir,
        count_band=(2, 4),
    )
    assert summary["drops_checked"] == 2
    assert summary["drop_problems"] == 1
    assert summary["drop_problem_symbols"] == ["BADDROP"]
    assert summary["adds_checked"] == 1
    assert summary["add_problems"] == 0
    assert summary["days_outside_count_band"] == []
    # Artifacts exist and the drop CSV carries the structured flag.
    drops = pd.read_csv(out_dir / "drop_audit.csv")
    flagged = drops[drops["problem"]]
    assert flagged["symbol"].tolist() == ["BADDROP"]
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_transition_audit_flags_count_band_violation(monkeypatch, artifact_dir) -> None:
    root = artifact_dir / "taq"
    monkeypatch.setitem(cfg.TAQ_PARQUET_DIR, 2018, root)
    monkeypatch.setattr(cfg, "TRADE_QC_POLICY_CHECK_MODE", "off")
    _write_trades(root, "20180102", "ONLY", with_close=True)
    membership = artifact_dir / "membership.csv"
    membership.write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,ONLY,2018-01-02,2018-01-02\n",
        encoding="utf-8",
    )
    summary = audit(
        membership, dt.date(2018, 1, 2), dt.date(2018, 1, 2),
        artifact_dir / "audit", count_band=(2, 3),
    )
    assert summary["days_outside_count_band"] == ["2018-01-02"]
