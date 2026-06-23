from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.fill_model.value_model import VALUE_MODEL_MANIFEST, VALUE_TARGET_COLUMN
from analysis.fill_model import xgb_survival
from analysis.fill_model.state_vector import STATE_COLUMNS
from analysis.fill_model.tod_schedule import TODSchedule
from analysis.fill_model.validation import ValidationReport
from analysis.runners import _common
from analysis.runners import calibrate_fill_model as calib


def _artifact_dir(name: str) -> Path:
    # Anchor at the repo's coding/ root (like conftest.artifact_dir); a
    # relative path would create a stray coding/coding/ tree when pytest
    # runs from coding/, which .gitignore does not cover.
    coding_root = Path(__file__).resolve().parents[1]
    path = coding_root / "artifacts" / "test_xgb_survival_robustness" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_complete_manifest(path: Path) -> None:
    (path / "calibration_manifest.json").write_text(
        f'{{"status":"complete","feature_policy":"{cfg.FEATURE_POLICY_VERSION}"}}',
        encoding="utf-8",
    )


def _panel(symbol: str = "AAPL", n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "symbol": symbol,
            "date": dt.date(2018, 1, 2),
            "side": "BUY" if i % 2 == 0 else "SELL",
            "t0": pd.Timestamp("2018-01-02 10:00:00") + pd.Timedelta(seconds=30 * i),
            "duration": float(5 + (i % 25)),
            "event": int(i % 3 != 0),
            "as_bps": float(np.sin(i / 7.0)),
        }
        for j, col in enumerate(STATE_COLUMNS):
            row[col] = float((i % 11) / 10.0 + j * 0.01)
        rows.append(row)
    return pd.DataFrame(rows)


def _hazard_panel(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic events whose 30s fill hazard decreases with limit_offset_bps,
    so a correctly-calibrated model must track the offset->fill curve."""
    rng = np.random.default_rng(seed)
    offsets = (0.0, 1.0, 2.0, 5.0, 10.0)
    rows = []
    for i in range(n):
        off = offsets[i % len(offsets)]
        rate = float(np.exp(-3.0 - 0.5 * off + 0.3 * rng.normal()))
        t_fill = float(rng.exponential(1.0 / rate))
        row = {
            "symbol": "AAPL",
            "date": dt.date(2018, 1, 2),
            "side": "BUY",
            "t0": pd.Timestamp("2018-01-02 10:00:00") + pd.Timedelta(seconds=30 * i),
            "duration": float(min(t_fill, 30.0)),
            "event": int(t_fill <= 30.0),
        }
        for col in STATE_COLUMNS:
            row[col] = float(rng.normal())
        row["limit_offset_bps"] = off
        rows.append(row)
    return pd.DataFrame(rows)


def test_breslow_ties_and_censoring_match_manual_risk_sets() -> None:
    times, hazard, risk_center = xgb_survival._breslow(
        durations=np.array([1.0, 1.0, 2.0, 3.0]),
        events=np.array([1, 1, 0, 1]),
        risk_scores=np.zeros(4),
    )

    assert np.allclose(times, np.array([1.0, 3.0]))
    assert np.allclose(hazard, np.array([0.5, 1.5]))
    assert risk_center == 0.0


def test_xgb_survival_train_save_load_predict() -> None:
    out_dir = _artifact_dir("survival")
    panel = _panel()
    model = xgb_survival.XGBFillModel(tier=1).fit(
        panel,
        n_estimators=8,
        max_depth=2,
        min_child_weight=1,
        xgb_device="cpu",
    )

    p = model.fill_probability(30.0, panel.iloc[0])
    assert 0.0 <= p <= 1.0

    model.save(out_dir)
    loaded = xgb_survival.XGBFillModel.load(out_dir, tier=1)
    p_loaded = loaded.fill_probability(30.0, panel.iloc[1])
    assert 0.0 <= p_loaded <= 1.0


def test_xgb_load_rejects_legacy_breslow_metadata() -> None:
    import joblib

    out_dir = _artifact_dir("legacy_breslow")
    panel = _panel()
    model = xgb_survival.XGBFillModel(tier=1).fit(
        panel,
        n_estimators=8,
        max_depth=2,
        min_child_weight=1,
        xgb_device="cpu",
    )
    model.save(out_dir)
    meta_path = out_dir / "xgb_tier_1_breslow.pkl"
    meta = joblib.load(meta_path)
    meta.pop("risk_center", None)
    joblib.dump(meta, meta_path)

    with pytest.raises(ValueError, match="Legacy XGB Breslow artifact"):
        xgb_survival.XGBFillModel.load(out_dir, tier=1)


def test_xgb_dry_run_requires_xgb_artifacts_even_when_cox_exists(monkeypatch) -> None:
    out_dir = _artifact_dir("dry_run")
    pd.DataFrame({"symbol": ["AAPL"], "tier": [1]}).to_csv(
        out_dir / "symbol_tier_map.csv", index=False
    )
    _write_complete_manifest(out_dir)
    (out_dir / "cox_tier_1.pkl").write_bytes(b"not-used-by-this-check")
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: [start])

    ok = _common.validate_run(
        ["S1"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="xgb",
    )

    assert ok is False


def test_km_dry_run_requires_all_tier_artifacts(monkeypatch) -> None:
    out_dir = _artifact_dir("km_dry_run")
    pd.DataFrame({"symbol": ["AAPL", "MSFT"], "tier": [1, 2]}).to_csv(
        out_dir / "symbol_tier_map.csv", index=False,
    )
    pd.DataFrame({"symbol": ["AAPL", "MSFT"], "tier": [1, 2]}).to_csv(
        out_dir / "km_symbol_tier_map.csv", index=False,
    )
    _write_complete_manifest(out_dir)
    (out_dir / "km_tier_1.pkl").write_bytes(b"only-tier-one")
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: [start])

    ok = _common.validate_run(
        ["S1"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="km",
    )

    assert ok is False


def test_tape_replay_dry_run_does_not_require_cox(monkeypatch) -> None:
    out_dir = _artifact_dir("tape_dry_run")
    pd.DataFrame({"symbol": ["AAPL"], "tier": [1]}).to_csv(
        out_dir / "symbol_tier_map.csv", index=False,
    )
    _write_complete_manifest(out_dir)
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: [start])

    assert _common.validate_run(
        ["S1_STATIC"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="tape_replay",
    )


def test_s4_dry_run_requires_tod_artifact(monkeypatch) -> None:
    out_dir = _artifact_dir("s4_dry_run")
    pd.DataFrame({"symbol": ["AAPL"], "tier": [1]}).to_csv(
        out_dir / "symbol_tier_map.csv", index=False,
    )
    _write_complete_manifest(out_dir)
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: [start])

    assert not _common.validate_run(
        ["S4_TOD"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="tape_replay",
    )


def test_s5_dry_run_requires_current_value_model_policy(monkeypatch) -> None:
    out_dir = _artifact_dir("s5_dry_run")
    pd.DataFrame({"symbol": ["AAPL"], "tier": [1]}).to_csv(
        out_dir / "symbol_tier_map.csv", index=False,
    )
    _write_complete_manifest(out_dir)
    monkeypatch.setattr(_common, "_eval_dates", lambda start, end: [start])
    stale_manifest = {
        "policy": "rolling_value_model_v1",
        "model_type": "SideTieredXGBValueModel",
        "target_column": VALUE_TARGET_COLUMN,
        "model_keys": ["global"],
        "models": {"global": {"policy": "rolling_value_model_v1"}},
    }
    (out_dir / VALUE_MODEL_MANIFEST).write_text(
        json.dumps(stale_manifest),
        encoding="utf-8",
    )
    (out_dir / "xgb_value_global.ubj").write_bytes(b"placeholder")
    (out_dir / "xgb_value_global.pkl").write_bytes(b"placeholder")

    assert not _common.validate_run(
        ["S5_VALUE_AWARE_XGB"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="tape_replay",
    )

    current_manifest = {
        **stale_manifest,
        "policy": cfg.VALUE_MODEL_POLICY_VERSION,
        "models": {"global": {"policy": cfg.VALUE_MODEL_POLICY_VERSION}},
    }
    (out_dir / VALUE_MODEL_MANIFEST).write_text(
        json.dumps(current_manifest),
        encoding="utf-8",
    )

    assert _common.validate_run(
        ["S5_VALUE_AWARE_XGB"],
        dt.date(2018, 1, 2),
        dt.date(2018, 1, 2),
        out_dir,
        symbols=["AAPL"],
        fill_specification="tape_replay",
    )


def test_tiered_xgb_strict_mode_rejects_missing_tiers() -> None:
    tier_map = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "tier": [1, 2]})
    model = xgb_survival.TieredXGBFillModel()

    with pytest.raises(RuntimeError, match="missing tiers"):
        model.fit_panel(
            _panel("AAPL"),
            tier_map,
            strict=True,
            n_estimators=8,
            max_depth=2,
            min_child_weight=1,
            xgb_device="cpu",
        )


def test_tod_xgb_smoke_train_save_load_fraction() -> None:
    out_dir = _artifact_dir("tod")
    panel = _panel(n=90)
    tod = TODSchedule().calibrate(panel, xgb_device="cpu")

    assert tod.fitted
    tod.save(out_dir)
    loaded = TODSchedule.load(out_dir)
    frac = loaded.fraction(
        pd.Timestamp("2018-01-02 15:30:00"),
        intervals_remaining=4,
        state_vector=panel.iloc[0],
    )

    assert np.isfinite(frac)
    assert 0.0 <= frac <= 1.0


def test_xgb_fill_probability_batch_matches_per_row() -> None:
    """fill_probability must honour the cox_ph/kaplan_meier contract: a scalar
    for a single dict/Series and a per-row array for a DataFrame, with the batch
    result equal to row-by-row evaluation. Guards the survival() batch bug where
    inplace_predict(...)[0] silently returned only the first row."""
    panel = _hazard_panel()
    model = xgb_survival.XGBFillModel(tier=1).fit(
        panel, n_estimators=60, max_depth=3, min_child_weight=5, xgb_device="cpu",
    )
    cols = model.covariates
    sample = panel[cols].head(50).reset_index(drop=True)

    one = model.fill_probability(30.0, sample.iloc[0].to_dict())
    assert np.isscalar(one) and 0.0 <= float(one) <= 1.0

    batch = np.asarray(model.fill_probability(30.0, sample), dtype=float)
    assert batch.shape == (len(sample),)
    per_row = np.array([
        float(model.fill_probability(30.0, sample.iloc[i].to_dict()))
        for i in range(len(sample))
    ])
    assert np.allclose(batch, per_row, atol=1e-7)


def test_xgb_offset_calibration_tracks_observed() -> None:
    """A correctly-batched XGB must track the empirical offset->fill curve. Pre-fix
    this failed because the batch mean collapsed to a single arbitrary row."""
    panel = _hazard_panel()
    model = xgb_survival.XGBFillModel(tier=1).fit(
        panel, n_estimators=200, max_depth=4, min_child_weight=20, xgb_device="cpu",
    )
    cols = model.covariates
    H = 30.0
    preds: dict[float, float] = {}
    for off in sorted(panel["limit_offset_bps"].unique()):
        sub = panel[panel["limit_offset_bps"] == off]
        obs = float(((sub["event"] > 0) & (sub["duration"] <= H)).mean())
        pred = float(np.mean(np.asarray(model.fill_probability(H, sub[cols]), dtype=float)))
        assert abs(pred - obs) < 0.10, f"offset {off}: pred {pred:.3f} vs obs {obs:.3f}"
        preds[float(off)] = pred
    offs = sorted(preds)
    # Fills are far higher at the touch than deep in the book.
    assert preds[offs[0]] > preds[offs[-1]] + 0.2


def test_fill_validation_gate_handles_xgb_without_baseline() -> None:
    """The shared gate must skip the NaN base_fill_s0 (XGB has no explicit
    baseline) while still flagging poor AUC / mis-levelled mean prediction."""
    good = ValidationReport(
        brier=0.2, reliability=0.0, resolution=0.1, uncertainty=0.2,
        auc=0.72, n=1000, observed=0.30, mean_pred=0.29, base_fill_s0=float("nan"),
    )
    bad = ValidationReport(
        brier=0.3, reliability=0.1, resolution=0.0, uncertainty=0.2,
        auc=0.50, n=1000, observed=0.30, mean_pred=0.05, base_fill_s0=float("nan"),
    )
    assert calib._fill_validation_failures({1: good}) == {}
    failed = calib._fill_validation_failures({1: bad})
    assert 1 in failed
    assert any("auc" in r for r in failed[1])
    assert any("mean_pred" in r for r in failed[1])


def test_fill_validation_gate_allows_jensen_gap_baseline() -> None:
    """A healthy dispersed-covariate fit has a baseline-at-mean (base_fill_s0)
    well below the mean predicted fill (Jensen gap on exp). It must NOT be flagged
    just for that. This is the smoke scenario the relative-to-observed floor
    false-positived; only an absolute near-zero floor should flag a collapse."""
    healthy = ValidationReport(
        brier=0.2, reliability=0.0, resolution=0.1, uncertainty=0.21,
        auc=0.80, n=10000, observed=0.31, mean_pred=0.30, base_fill_s0=0.14,
    )
    assert calib._fill_validation_failures({1: healthy}) == {}
    collapsed = ValidationReport(
        brier=0.3, reliability=0.1, resolution=0.0, uncertainty=0.21,
        auc=0.80, n=10000, observed=0.31, mean_pred=0.30, base_fill_s0=1e-6,
    )
    assert 1 in calib._fill_validation_failures({1: collapsed})


def test_tod_inplace_prediction_matches_dmatrix_fallback() -> None:
    panel = _panel(n=90)
    tod = TODSchedule().calibrate(panel, xgb_device="cpu")
    timestamp = pd.Timestamp("2018-01-02 15:30:00")
    state = panel.iloc[0]
    inplace_value = tod.predict_as(timestamp, state)

    class DMatrixOnlyModel:
        def __init__(self, model):
            self._model = model

        def predict(self, matrix):
            return self._model.predict(matrix)

    original = tod._model
    tod._model = DMatrixOnlyModel(original)
    fallback_value = tod.predict_as(timestamp, state)

    assert np.isclose(inplace_value, fallback_value)
