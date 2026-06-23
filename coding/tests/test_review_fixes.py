"""Regression tests for the 2026-06 code-review fixes.

Covers:
* XGB survival: margin-scale risk scores + Breslow centering (no double
  exponentiation) -> absolute survival probabilities are calibrated.
* Kaplan-Meier: vectorized ms-keyed estimator matches a brute-force
  reference; time-to-cutoff and limit-offset routing actually used at
  prediction time.
* Cox fast path: numpy survival evaluation matches lifelines'
  predict_survival_function exactly.
* Two-way clustering: CGM intersection term clusters on (symbol, date)
  cells, reducing to White only for singleton cells.
* Engine RNG: per-(day, strategy, order) streams are independent and
  reproducible.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from analysis.fill_model import xgb_survival
from analysis.fill_model.cox_ph import CoxFillModel
from analysis.fill_model.kaplan_meier import (
    KMFillModel,
    _bucket_for_seconds,
    _km_survival_function,
    _seconds_to_cutoff,
)
from analysis.fill_model.state_vector import STATE_COLUMNS
from analysis.inference.clustering import two_way_cluster_ols
from analysis.simulation.engine import _order_rng, _stable_hash_int


# ---------------------------------------------------------------------------
# Fix 1: XGB survival calibration
# ---------------------------------------------------------------------------

def _exponential_panel(n: int = 4000, mean_seconds: float = 20.0,
                       horizon: float = 30.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.exponential(mean_seconds, size=n)
    durations = np.minimum(raw, horizon)
    events = (raw <= horizon).astype(int)
    panel = pd.DataFrame({
        "symbol": "AAPL",
        "date": dt.date(2018, 1, 2),
        "side": "BUY",
        "t0": pd.Timestamp("2018-01-02 10:00:00"),
        "duration": np.maximum(durations, 1e-3),
        "event": events,
    })
    for j, col in enumerate(STATE_COLUMNS):
        panel[col] = rng.normal(size=n)  # uninformative covariates
    return panel


@pytest.mark.unit
def test_xgb_survival_calibrated_against_empirical_survival() -> None:
    """Under the old double exponentiation this is off by far more than 5pp."""
    panel = _exponential_panel()
    model = xgb_survival.XGBFillModel(tier=1).fit(
        panel, n_estimators=30, max_depth=2, min_child_weight=50,
        xgb_device="cpu",
    )
    h = 10.0
    empirical = float((panel["duration"] > h).mean())
    preds = np.array([
        model.survival(h, panel.iloc[i]) for i in range(0, len(panel), 10)
    ])
    assert abs(float(preds.mean()) - empirical) < 0.05


@pytest.mark.unit
def test_eval_breslow_decreasing_in_margin_and_centering_cancels() -> None:
    times = np.array([1.0, 2.0])
    vals = np.array([0.1, 0.3])
    s_low = xgb_survival._eval_breslow(times, vals, -1.0, 2.0, risk_center=0.0)
    s_high = xgb_survival._eval_breslow(times, vals, 1.0, 2.0, risk_center=0.0)
    assert s_high < s_low
    # Shifting both the score and the center by the same constant is a no-op
    # only if the baseline was computed with that center; here we verify the
    # algebra S = exp(-H0_c * exp(f - c)) directly.
    c = 3.0
    vals_centered = vals * np.exp(c)
    s_plain = xgb_survival._eval_breslow(times, vals, 0.5, 2.0, risk_center=0.0)
    s_centered = xgb_survival._eval_breslow(
        times, vals_centered, 0.5, 2.0, risk_center=c,
    )
    assert np.isclose(s_plain, s_centered)


# ---------------------------------------------------------------------------
# Fix 2d: KM estimator vs brute force (ties + float noise)
# ---------------------------------------------------------------------------

def _km_bruteforce(durations: np.ndarray, events: np.ndarray,
                   times: np.ndarray) -> np.ndarray:
    d = np.round(np.asarray(durations, dtype=float) * 1000.0).astype(np.int64)
    e = np.asarray(events, dtype=float)
    uniq = np.unique(d)
    s = 1.0
    surv_at: dict[int, float] = {}
    n_at_risk = len(d)
    for t in uniq:
        m = d == t
        deaths = float(e[m].sum())
        if n_at_risk > 0 and deaths > 0:
            s *= 1.0 - deaths / n_at_risk
        surv_at[int(t)] = s
        n_at_risk -= int(m.sum())
    out = []
    for q in np.asarray(times, dtype=float):
        keys = [t for t in uniq if t / 1000.0 <= q]
        out.append(surv_at[int(keys[-1])] if keys else 1.0)
    return np.array(out)


@pytest.mark.unit
def test_km_survival_function_matches_bruteforce_with_ties() -> None:
    rng = np.random.default_rng(1)
    base = np.repeat(rng.exponential(10.0, size=250), 2)  # exact ties
    noise = rng.uniform(-2e-7, 2e-7, size=base.shape)     # sub-ms float noise
    durations = np.minimum(base + noise, 30.0)
    events = (durations < 30.0).astype(int)
    times = np.array([1.0, 5.0, 10.0, 30.0])
    got = _km_survival_function(durations, events, times)
    want = _km_bruteforce(durations, events, times)
    assert np.allclose(got, want)
    assert np.all(np.diff(got) <= 1e-12)  # survival is non-increasing


# ---------------------------------------------------------------------------
# Fix 2a/2b: KM time-to-cutoff + offset routing at prediction time
# ---------------------------------------------------------------------------

def _km_routing_panel() -> pd.DataFrame:
    t0_early = pd.Timestamp("2018-01-02 10:00:00")   # >= 900s to MOC cutoff
    t0_late = pd.Timestamp("2018-01-02 15:49:45")    # 0-30s to MOC cutoff
    rows = []
    for _ in range(60):
        # Late-day at-touch: always fills fast.
        rows.append({"duration": 5.0, "event": 1, "t0": t0_late,
                     "limit_offset_bps": 0.0})
        # Early-day at-touch: never fills.
        rows.append({"duration": 30.0, "event": 0, "t0": t0_early,
                     "limit_offset_bps": 0.0})
        # Late-day deep offset: never fills.
        rows.append({"duration": 30.0, "event": 0, "t0": t0_late,
                     "limit_offset_bps": 10.0})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_km_routes_by_time_to_cutoff_and_offset() -> None:
    km = KMFillModel(tier=1).fit(_km_routing_panel())
    late = pd.Timestamp("2018-01-02 15:49:50")
    early = pd.Timestamp("2018-01-02 10:05:00")
    p_late_touch = km.fill_probability(30.0, {"t0": late, "limit_offset_bps": 0.0})
    p_early_touch = km.fill_probability(30.0, {"t0": early, "limit_offset_bps": 0.0})
    p_late_deep = km.fill_probability(30.0, {"t0": late, "limit_offset_bps": 10.0})
    assert p_late_touch > 0.9
    assert p_early_touch < 0.1
    assert p_late_deep < 0.1
    # Unseen offset snaps to the nearest calibrated grid value.
    p_late_snap = km.fill_probability(30.0, {"t0": late, "limit_offset_bps": 0.2})
    assert np.isclose(p_late_snap, p_late_touch)


@pytest.mark.unit
def test_km_time_bucket_uses_moc_cutoff() -> None:
    t0 = pd.Timestamp("2018-01-02 15:49:45")
    assert _seconds_to_cutoff(t0) == 15.0
    assert _bucket_for_seconds(_seconds_to_cutoff(t0)) == (0, 30)


@pytest.mark.unit
def test_km_save_load_roundtrip_preserves_offset_strata(artifact_dir) -> None:
    km = KMFillModel(tier=1).fit(_km_routing_panel())
    path = artifact_dir / "km_tier_1.pkl"
    km.save(path)
    loaded = KMFillModel.load(path)
    late = pd.Timestamp("2018-01-02 15:49:50")
    for offset in (0.0, 10.0):
        a = km.fill_probability(30.0, {"t0": late, "limit_offset_bps": offset})
        b = loaded.fill_probability(30.0, {"t0": late, "limit_offset_bps": offset})
        assert np.isclose(a, b)


@pytest.mark.unit
def test_cox_xdf_ignores_extra_t0_key() -> None:
    m = CoxFillModel(tier=1, covariates=["q0", "D0"])
    m.medians = {"q0": 1.0, "D0": 2.0}
    df = m._xdf({"q0": 5.0, "D0": 6.0, "t0": pd.Timestamp("2018-01-02 15:30:00")})
    assert list(df.columns) == ["q0", "D0"]
    assert df.iloc[0].tolist() == [5.0, 6.0]


# ---------------------------------------------------------------------------
# Fix 4: Cox fast path equals lifelines predict_survival_function
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cox_fast_path_matches_lifelines_predict() -> None:
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(3)
    n = 2000
    # Genuine fill-time hazard with the calibration censoring structure (non-fills
    # censored exactly at the 30s horizon), so the fit passes the baseline gate.
    # A brisk hazard + large n give a dense baseline step grid, so the fast-path
    # step lookup and lifelines' linear baseline interpolation agree tightly.
    xq = rng.normal(size=n)
    t_fill = rng.exponential(1.0 / np.exp(-2.0 + 0.5 * xq))
    panel = pd.DataFrame({
        "duration": np.minimum(t_fill, 30.0),
        "event": (t_fill <= 30.0).astype(int),
        "q0": xq,
        "D0": rng.normal(size=n),
    })
    m = CoxFillModel(tier=1, covariates=["q0", "D0"]).fit(panel)
    xdf = panel[["q0", "D0"]].head(50)
    assert m._ensure_fast_path()
    # The fitter is trained on standardized + winsorized covariates, so the
    # ground-truth lifelines call must standardize AND clip its input with the
    # same persisted scaler/cap (exactly what survival() does on both its paths).
    xz = m._xdf(xdf).copy()
    for c in m.covariates:
        xz[c] = (xz[c] - m.scale_mean.get(c, 0.0)) / m.scale_std.get(c, 1.0)
    if m.winsor_z is not None:
        xz[m.covariates] = xz[m.covariates].clip(-m.winsor_z, m.winsor_z)
    bs_max = float(m.fitter.baseline_survival_.index.max())
    for h in (5.0, 30.0):
        fast = np.asarray(m.survival(h, xdf), dtype=float)
        slow = m.fitter.predict_survival_function(
            xz, times=[h],
        ).iloc[0].to_numpy(dtype=float)
        # The fast path uses a right-continuous baseline step lookup (the
        # production/KM semantics); lifelines linearly interpolates the baseline
        # cumulative hazard between steps. They agree to machine precision at or
        # beyond the last baseline step (h=30) and to within the baseline step
        # granularity at interior times (h=5).
        tol = 1e-9 if h >= bs_max else 2e-3
        assert np.allclose(fast, slow, rtol=0, atol=tol), f"h={h}"
    # Scalar (dict) path agrees with the DataFrame path (1-row frames
    # collapse to a scalar by design).
    single = m.survival(30.0, panel.iloc[0][["q0", "D0"]].to_dict())
    batch = m.survival(30.0, panel[["q0", "D0"]].head(1))
    assert np.isclose(float(single), float(batch))


# ---------------------------------------------------------------------------
# Fix 5: two-way clustering intersection term
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_twoway_intersection_reduces_to_white_for_singleton_cells() -> None:
    rng = np.random.default_rng(4)
    n = 200
    y = rng.normal(size=n)
    X = np.ones((n, 1))
    sym = np.array([f"s{i % 20}" for i in range(n)])
    date = np.array([f"d{i // 20}" for i in range(n)])  # each (sym, date) unique
    res = two_way_cluster_ols(y, X, sym, date)
    lhs = res.se_cluster_twoway[0] ** 2
    rhs = res.se_cluster_sym[0] ** 2 + res.se_cluster_date[0] ** 2 - res.se_white[0] ** 2
    assert np.isclose(lhs, rhs, rtol=1e-10)


@pytest.mark.unit
def test_twoway_intersection_differs_from_white_for_multirow_cells() -> None:
    rng = np.random.default_rng(5)
    n_cells = 100
    reps = 5  # several orders per (symbol, date) cell
    cell_y = rng.normal(size=n_cells)
    y = np.repeat(cell_y, reps)  # within-cell correlation = 1
    X = np.ones((len(y), 1))
    sym = np.repeat([f"s{i % 10}" for i in range(n_cells)], reps)
    date = np.repeat([f"d{i // 10}" for i in range(n_cells)], reps)
    res = two_way_cluster_ols(y, X, sym, date)
    v_new = res.se_cluster_twoway[0] ** 2
    v_old = (
        res.se_cluster_sym[0] ** 2
        + res.se_cluster_date[0] ** 2
        - res.se_white[0] ** 2
    )
    # Perfectly correlated rows within a cell: the cell-clustered intersection
    # exceeds White, so the corrected two-way variance is strictly smaller
    # than the old White-based formula.
    assert v_new < v_old


# ---------------------------------------------------------------------------
# Fix 3: per-(day, strategy, order) RNG streams
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_order_rng_streams_are_independent_and_reproducible() -> None:
    day_a = _stable_hash_int("42|2018-01-02|AAPL")
    day_b = _stable_hash_int("42|2018-01-03|AAPL")
    day_c = _stable_hash_int("42|2018-01-02|MSFT")

    draws_a = _order_rng(day_a, "S1_STATIC", "oid1").random(8)
    draws_a_again = _order_rng(day_a, "S1_STATIC", "oid1").random(8)
    draws_other_day = _order_rng(day_b, "S1_STATIC", "oid1").random(8)
    draws_other_symbol = _order_rng(day_c, "S1_STATIC", "oid1").random(8)
    draws_other_strategy = _order_rng(day_a, "S2_TIME_ADAPTIVE", "oid1").random(8)
    draws_other_order = _order_rng(day_a, "S1_STATIC", "oid2").random(8)

    assert np.array_equal(draws_a, draws_a_again)
    for other in (draws_other_day, draws_other_symbol,
                  draws_other_strategy, draws_other_order):
        assert not np.array_equal(draws_a, other)
