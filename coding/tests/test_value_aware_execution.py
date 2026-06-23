from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.fill_model.rolling import (
    assign_monthly_anchor,
    build_monthly_training_schedule,
    rolling_training_window,
)
from analysis.fill_model.value_model import (
    VALUE_MODEL_MANIFEST,
    VALUE_TARGET_COLUMN,
    SideTieredXGBValueModel,
    XGBValueModel,
    attach_value_labels,
    realized_candidate_value_bps,
    validate_value_model_manifest,
)
from analysis.reporting.static_posting_curve import posting_curve_summary
from analysis.strategies.value_aware import ValueAwareXGBStrategy
from analysis.strategies.base import MarketState


def _candidate_panel(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sides = np.where(np.arange(n) % 2 == 0, "BUY", "SELL")
    offsets = np.tile([0.0, 1.0, 2.0, 5.0], n // 4 + 1)[:n]
    frame = pd.DataFrame({
        "symbol": "AAPL",
        "date": dt.date(2018, 4, 2),
        "side": sides,
        "tier": np.where(np.arange(n) % 3 == 0, 1, 2),
        "sector": np.where(np.arange(n) % 2 == 0, "Information Technology", "Health Care"),
        "listing_exchange": np.where(np.arange(n) % 2 == 0, "NASDAQ", "NYSE"),
        "q0": 1000 + rng.normal(0, 10, n),
        "D0": 2000 + rng.normal(0, 10, n),
        "ofi_z": rng.normal(0, 1, n),
        "sigma": 0.001 + rng.random(n) * 0.001,
        "limit_offset_bps": offsets,
        "half_spread_bps": 1.0,
        "size_frac": 0.01,
        "time_to_cutoff_seconds": 1800 - np.arange(n),
        "close_price": 100.0,
        "limit_price": np.where(sides == "BUY", 99.95 - offsets * 0.001, 100.05 + offsets * 0.001),
        "event": np.where(offsets <= 1.0, 1, 0),
        "adverse_selection_bps": np.where(sides == "BUY", -0.5, 0.4),
    })
    for h in (10, 11, 12, 13, 14, 15):
        frame[f"tod_{h}"] = float(h == 15)
    return attach_value_labels(frame)


def test_candidate_value_buy_improvement_is_positive_after_common_moc_fee_cancel():
    value = realized_candidate_value_bps(
        side="BUY",
        close_price=100.0,
        passive_price=99.90,
        filled=True,
        adverse_selection_bps=0.0,
        size_frac=0.01,
    )
    assert value == pytest.approx(10.29)


def test_candidate_value_unfilled_attempt_matches_impact_penalty_only():
    value = realized_candidate_value_bps(
        side="SELL",
        close_price=100.0,
        passive_price=100.10,
        filled=False,
        adverse_selection_bps=0.0,
        size_frac=0.02,
    )
    assert value < 0.0


def test_rolling_training_window_excludes_anchor_date():
    dates = [dt.date(2018, 1, 2) + dt.timedelta(days=i) for i in range(90)]
    anchor = dates[70]
    window = rolling_training_window(anchor, dates, lookback_days=40, min_lookback_days=20)
    assert window is not None
    assert window.train_end < anchor
    assert window.n_train_dates == 40


def test_monthly_schedule_marks_early_period_as_warmup():
    dates = [dt.date(2018, 1, 2) + dt.timedelta(days=i) for i in range(130)]
    schedule = build_monthly_training_schedule(dates, lookback_days=60, min_lookback_days=60)
    assert "warmup_excluded" in set(schedule["status"])
    mapped = assign_monthly_anchor(dates, schedule)
    trainable = mapped[mapped["status"] == "mapped"]
    assert not trainable.empty
    assert (pd.to_datetime(trainable["anchor_date"]).dt.date <= trainable["date"]).all()


def test_xgb_value_model_save_load_predicts_finite_values(artifact_dir):
    pytest.importorskip("xgboost")
    panel = _candidate_panel(96)
    model = XGBValueModel(key="global").fit(panel, min_rows=20, n_estimators=5)
    preds = model.predict_frame(panel.head(5))
    assert np.isfinite(preds).all()
    model.save(artifact_dir)
    loaded = XGBValueModel.load(artifact_dir, "global")
    loaded_preds = loaded.predict_frame(panel.head(5))
    assert np.isfinite(loaded_preds).all()


def test_side_tiered_value_model_falls_back_to_side_or_global(artifact_dir):
    pytest.importorskip("xgboost")
    panel = _candidate_panel(96)
    model = SideTieredXGBValueModel().fit_panel(
        panel,
        min_rows_global=20,
        min_rows_side=20,
        min_rows_side_tier=20,
        n_estimators=5,
    )
    model.save(artifact_dir)
    loaded = SideTieredXGBValueModel.load(artifact_dir)
    candidates = panel.head(4).copy()
    candidates["tier"] = 99
    preds = loaded.predict_candidates(candidates)
    assert np.isfinite(preds).all()


def test_value_model_loader_rejects_stale_policy_before_deserializing(artifact_dir):
    manifest = {
        "policy": "rolling_value_model_v1",
        "model_type": "SideTieredXGBValueModel",
        "target_column": VALUE_TARGET_COLUMN,
        "model_keys": ["global"],
        "models": {"global": {"policy": "rolling_value_model_v1"}},
    }
    (artifact_dir / VALUE_MODEL_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    errors = validate_value_model_manifest(manifest, artifact_dir, require_files=False)
    assert any("policy mismatch" in error for error in errors)
    with pytest.raises(RuntimeError, match=cfg.VALUE_MODEL_POLICY_VERSION):
        SideTieredXGBValueModel.load(artifact_dir)


class _ConstantValueModel:
    def __init__(self, values):
        self.values = values

    def predict_candidates(self, candidates):
        return np.array(self.values[: len(candidates)], dtype=float)


def _market_state() -> MarketState:
    times = pd.to_datetime(["2018-04-02 15:30:00", "2018-04-02 15:30:30"])
    nbbo = pd.DataFrame({
        "time": times,
        "best_bid": [100.0, 100.0],
        "best_offer": [100.1, 100.1],
        "best_bid_size": [1000, 1000],
        "best_offer_size": [1200, 1200],
        "mid": [100.05, 100.05],
    })
    return MarketState(
        symbol="AAPL",
        date=dt.date(2018, 4, 2),
        nbbo=nbbo,
        trades=pd.DataFrame(),
        close_price=100.0,
        close_volume=1_000_000,
        ofi=pd.DataFrame(),
        rv=pd.Series(dtype=float),
        imbalance=pd.DataFrame(),
        nbbo_times=nbbo["time"].values.astype("int64"),
        nbbo_mid=nbbo[["time", "mid"]],
    )


def test_value_aware_strategy_selects_best_positive_offset():
    strategy = ValueAwareXGBStrategy(
        value_model=_ConstantValueModel([-1.0, 0.2, 0.8]),
        offset_grid_bps=(0.0, 1.0, 2.0),
        tier=1,
    )
    offset = strategy.limit_offset_bps(pd.Timestamp("2018-04-02 15:30:00"), "BUY", _market_state(), 0.0, 5.0)
    assert offset == pytest.approx(2.0)
    assert strategy.slice_size(pd.Timestamp("2018-04-02 15:30:00"), pd.Timestamp("2018-04-02 15:50:00"), 100, "BUY", _market_state()) > 0


def test_value_aware_strategy_skips_interval_when_all_values_nonpositive():
    strategy = ValueAwareXGBStrategy(
        value_model=_ConstantValueModel([-0.4, -0.1]),
        offset_grid_bps=(0.0, 1.0),
        tier=1,
    )
    t = pd.Timestamp("2018-04-02 15:30:00")
    strategy.limit_offset_bps(t, "SELL", _market_state(), 0.0, 5.0)
    assert strategy.slice_size(t, pd.Timestamp("2018-04-02 15:50:00"), 100, "SELL", _market_state()) == 0


def test_posting_curve_summary_bounds_fill_probability():
    panel = _candidate_panel(24)
    summary = posting_curve_summary(panel)
    assert not summary.empty
    assert summary["fill_probability"].between(0.0, 1.0).all()
    assert {"side", "tier", "limit_offset_bps", "mean_value_bps"}.issubset(summary.columns)
