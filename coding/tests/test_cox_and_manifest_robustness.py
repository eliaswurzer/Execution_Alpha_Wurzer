from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.fill_model.cox_ph import CoxFillModel
from analysis.fill_model.state_vector import STATE_COLUMNS
from analysis.fill_model.validation import ValidationReport
from analysis.runners import calibrate_fill_model as calib


def _cox_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic event panel with a genuine fill-time hazard so the Cox fit is
    well-calibrated and passes the baseline sanity gate. ``q0``/``ofi_z`` carry
    signal, ``D0`` is exactly collinear with ``q0`` (D0 = 2*q0, mirroring
    state_vector), and ``sigma`` is constant (dropped by the variance filter)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        xq = float(rng.normal())
        xo = float(rng.normal())
        # Hazard increasing in the covariates -> exponential time-to-fill, tuned
        # so the marginal 30s fill rate is roughly one half.
        rate = float(np.exp(-3.8 + 0.8 * xq + 0.5 * xo))
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
        row["q0"] = xq
        row["D0"] = 2.0 * xq  # exactly collinear with q0 (q0 = 0.5*D0)
        row["ofi_z"] = xo
        row["sigma"] = 0.0    # deliberately near-constant
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_cox_fit_drops_constant_covariates_and_survives_collinearity() -> None:
    model = CoxFillModel(tier=1).fit(_cox_panel(), penalizer=0.01)

    assert model.fitter is not None
    assert "sigma" not in model.covariates          # near-constant -> variance filter
    assert "D0" not in model.covariates             # redundant with q0 (D0 = 2*q0)
    assert "q0" in model.covariates                 # the retained member of the pair
    # First time-of-day dummy dropped as the reference category.
    assert f"tod_{cfg.TOD_HOUR_BINS[0]}" not in model.covariates
    p = model.fill_probability(30.0, _cox_panel().iloc[0])
    assert 0.0 <= p <= 1.0


@pytest.mark.unit
def test_cox_x_array_matches_xdf_bit_for_bit() -> None:
    """The numpy fast-path assembly must equal the pandas ``_xdf`` round-trip
    exactly, including the imputation edge cases (missing key and NaN map to the
    training median; inf passes through), so the optimisation is bit-parity."""
    model = CoxFillModel(tier=1).fit(_cox_panel(), penalizer=0.01)
    assert model._ensure_fast_path()
    cov = model.covariates
    base = {c: float(3.0 + i) for i, c in enumerate(cov)}
    cases = {
        "normal": dict(base),
        "missing_key": {k: v for k, v in base.items() if k != cov[0]},
        "nan_value": {**base, cov[1]: float("nan")},
        "inf_value": {**base, cov[2]: float("inf")},
        "extra_non_covariate_key": {**base, "not_a_covariate": 99.0},
    }
    for name, x in cases.items():
        fast = model._x_array(x)
        ref = model._xdf(x).to_numpy(dtype=float)
        assert np.array_equal(fast, ref), f"dict:{name}"
        s = pd.Series({k: v for k, v in x.items() if k in cov or k in base})
        assert np.array_equal(
            model._x_array(s), model._xdf(s).to_numpy(dtype=float),
        ), f"series:{name}"


@pytest.mark.unit
def test_cox_survival_bit_identical_to_xdf_path() -> None:
    """survival() through the new ``_x_array`` path returns exactly the same
    float as the previous ``_xdf``-based fast path."""
    model = CoxFillModel(tier=1).fit(_cox_panel(), penalizer=0.01)
    assert model._ensure_fast_path()
    x = {c: float(2.0 + i) for i, c in enumerate(model.covariates)}
    # Reference: the same fast-path math fed by the pandas-derived array, with
    # the standardization applied exactly as survival() now does on the fast path.
    X_ref = model._xdf(x).to_numpy(dtype=float)
    Z_ref = (X_ref - model._fast_scale_mean) / model._fast_scale_std
    if model._fast_winsor_z is not None:
        Z_ref = np.clip(Z_ref, -model._fast_winsor_z, model._fast_winsor_z)
    idx = int(np.searchsorted(model._fast_bs_times, 30.0, side="right")) - 1
    s0 = float(model._fast_bs_vals[idx]) if idx >= 0 else 1.0
    ref = float(np.power(s0, np.exp((Z_ref - model._fast_mean) @ model._fast_beta))[0])
    assert model.survival(30.0, x) == ref
    assert 0.0 <= model.fill_probability(30.0, x) <= 1.0


@pytest.mark.unit
def test_cox_fast_path_matches_lifelines_under_standardization() -> None:
    """The numpy fast path and the lifelines fallback must agree once the
    covariates are standardized; both apply the same persisted scaler."""
    model = CoxFillModel(tier=1).fit(_cox_panel(), penalizer=0.01)
    assert model._ensure_fast_path()
    for k in range(5):
        x = {c: float(1.0 + (k * 0.7 + i) % 11) for i, c in enumerate(model.covariates)}
        fast = model.survival(30.0, x)
        # Force the lifelines fallback by clearing and disabling the fast path.
        saved_beta, saved_failed = model._fast_beta, model._fast_path_failed
        model._fast_beta = None
        model._fast_path_failed = True
        slow = model.survival(30.0, x)
        model._fast_beta, model._fast_path_failed = saved_beta, saved_failed
        assert abs(float(fast) - float(slow)) < 1e-9, f"case {k}: {fast} vs {slow}"


@pytest.mark.unit
def test_cox_standardization_tames_tiny_scale_covariate() -> None:
    """A covariate on a tiny absolute scale (the realized-volatility pathology
    that blew up to a ~1e5 coefficient) must not blow up the fitted coefficient
    once covariates are z-scored before the Cox fit."""
    rng = np.random.default_rng(0)
    n = 600
    sig = rng.normal(size=n)
    rows = []
    for i in range(n):
        # Clean PH hazard driven by ``sig``, which enters only through the
        # tiny-scale covariate; produces a well-calibrated fit.
        rate = float(np.exp(-3.5 + 1.0 * sig[i]))
        t_fill = float(rng.exponential(1.0 / rate))
        rows.append({
            "symbol": "AAPL",
            "date": dt.date(2018, 1, 2),
            "side": "BUY",
            "t0": pd.Timestamp("2018-01-02 10:00:00"),
            "duration": float(min(t_fill, 30.0)),
            "event": int(t_fill <= 30.0),
            "tiny": 1e-4 * sig[i],  # informative but on a tiny absolute scale
            **{c: float(rng.normal()) for c in STATE_COLUMNS},
        })
    panel = pd.DataFrame(rows)
    model = CoxFillModel(tier=1, covariates=["tiny", *STATE_COLUMNS]).fit(panel, penalizer=0.01)
    coefs = np.abs(model.fitter.params_.to_numpy())
    # On the raw scale 'tiny' would draw a coefficient of order 1e4; with
    # standardization every coefficient stays bounded.
    assert float(coefs.max()) < 25.0
    assert model.scale_std.get("tiny", 1.0) < 1e-2


def _leverage_panel(n: int = 800, seed: int = 1) -> pd.DataFrame:
    """Panel whose fill signal lives in ``ofi_z`` while ``q0``/``D0`` carry a
    handful of extreme leverage points (the displayed-depth tail). Without
    winsorization these outliers blow up ``exp(beta'z)`` and collapse the
    Breslow baseline; with the |z|<=5 clip the fit stays well-levelled."""
    rng = np.random.default_rng(seed)
    xo = rng.normal(size=n)          # clean signal
    xq = rng.normal(size=n)          # heavy-tailed leverage / noise
    xq[:10] = rng.choice([-1.0, 1.0], 10) * rng.uniform(30.0, 60.0, 10)
    rows = []
    for i in range(n):
        rate = float(np.exp(-3.5 + 0.9 * xo[i]))
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
        row["ofi_z"] = float(xo[i])
        row["q0"] = float(xq[i])
        row["D0"] = 2.0 * float(xq[i])
        row["sigma"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_cox_winsorization_recovers_level_under_heavy_tails() -> None:
    """The baseline sanity gate would raise on a collapsed fit; with the |z|<=5
    winsorization the fit succeeds and the mean predicted fill matches the
    observed rate even though a few covariate rows are extreme outliers."""
    panel = _leverage_panel()
    model = CoxFillModel(tier=1).fit(panel)  # raises ValueError if degenerate
    assert model.winsor_z == 5.0

    h = 30.0
    xdf = panel[[c for c in model.covariates if c in panel.columns]].copy()
    pred = np.asarray(model.fill_probability(h, xdf), dtype=float)
    observed = float(((panel["event"] > 0) & (panel["duration"] <= h)).mean())
    assert abs(float(pred.mean()) - observed) < 0.15
    # Baseline must not have collapsed (implied baseline fill is substantial).
    assert model._ensure_fast_path()
    idx = int(np.searchsorted(model._fast_bs_times, h, side="right")) - 1
    base_fill = 1.0 - (float(model._fast_bs_vals[idx]) if idx >= 0 else 1.0)
    assert base_fill > 0.5 * observed


@pytest.mark.unit
def test_cox_validation_gate_flags_collapsed_baseline() -> None:
    good = ValidationReport(
        brier=0.21, reliability=0.0, resolution=0.1, uncertainty=0.21,
        auc=0.78, n=1000, observed=0.31, mean_pred=0.30, base_fill_s0=0.28,
    )
    collapsed = ValidationReport(
        brier=0.31, reliability=0.1, resolution=0.0, uncertainty=0.21,
        auc=0.44, n=1000, observed=0.31, mean_pred=0.002, base_fill_s0=1e-6,
    )
    # Non-Cox model: base_fill_s0 is NaN and must be skipped, not flagged.
    km_like = ValidationReport(
        brier=0.2, reliability=0.0, resolution=0.1, uncertainty=0.2,
        auc=0.70, n=100, observed=0.30, mean_pred=0.30, base_fill_s0=float("nan"),
    )

    assert calib._fill_validation_failures({1: good}) == {}
    assert calib._fill_validation_failures({1: km_like}) == {}
    failed = calib._fill_validation_failures({1: good, 2: collapsed})
    assert set(failed) == {2}
    assert any("base_fill_s0" in r for r in failed[2])
    assert any("auc" in r for r in failed[2])


@pytest.mark.unit
def test_km_level_validation_gate_flags_miscalibration() -> None:
    good = ValidationReport(
        brier=0.21, reliability=0.0, resolution=0.1, uncertainty=0.21,
        auc=0.50, n=1000, observed=0.31, mean_pred=0.30,
        base_fill_s0=float("nan"),
    )
    bad = ValidationReport(
        brier=0.31, reliability=0.1, resolution=0.0, uncertainty=0.21,
        auc=0.80, n=1000, observed=0.31, mean_pred=0.02,
        base_fill_s0=float("nan"),
    )

    assert calib._level_validation_failures({1: good}) == {}
    failed = calib._level_validation_failures({1: good, 2: bad})
    assert set(failed) == {2}
    assert any("mean_pred-observed" in r for r in failed[2])


@pytest.mark.unit
def test_required_tod_failure_marks_manifest_failed(artifact_dir) -> None:
    manifest = {"status": "pending_fit", "tod_required": True}
    path = artifact_dir / "calibration_manifest.json"

    with pytest.raises(RuntimeError, match="TOD schedule fit failed"):
        calib._record_tod_failure(
            manifest,
            path,
            RuntimeError("missing TOD model"),
            tod_required=True,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_tod_model_fit"
    assert payload["tod_status"] == "failed"
    assert "missing TOD model" in payload["tod_error"]


@pytest.mark.unit
def test_calibration_manifest_writer_accepts_numpy_scalars(artifact_dir) -> None:
    path = artifact_dir / "manifest.json"
    calib._write_manifest(
        path,
        {
            "status": "failed_model_fit",
            "missing_cox_tiers": [np.int64(1)],
            "coverage": np.float64(0.99),
            "date": dt.date(2018, 1, 2),
        },
    )

    text = path.read_text(encoding="utf-8")
    assert '"missing_cox_tiers": [\n    1\n  ]' in text
    assert '"coverage": 0.99' in text
