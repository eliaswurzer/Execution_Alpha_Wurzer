"""Risk-Adjusted Execution-Alpha Ratio, Information Ratio, und Break-Even-Eta (Thesis §4.4.5).

* ``information_ratio``  -- IR = ᾱ / TES  (primary, dimensionless)
* ``raear(eta)``         -- Eq. 4.22: ᾱ_s - eta * TEV_s  (secondary; eta in bps⁻¹)
* ``break_even_eta``     -- Eq. 4.23: eta* = ᾱ_s / TEV_s
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TEV_EPS = 1e-12


def information_ratio(mean_alpha: float, tev: float) -> float:
    """Execution-alpha information ratio: IR = ᾱ / TES = ᾱ / sqrt(TEV).

    Dimensionless; the standard risk-adjusted statistic in the tracking-error
    management literature (Goodwin, 1998; Grinold & Kahn).
    Returns nan when tev <= 0 or mean_alpha is nan.
    """
    if np.isnan(mean_alpha) or tev <= _TEV_EPS:
        return float("nan")
    return float(mean_alpha / np.sqrt(tev))


def raear(mean_alpha: float, tev: float, eta: float) -> float:
    """Scalar RAEAR: ᾱ - eta * TEV.  eta carries units bps⁻¹."""
    if np.isnan(mean_alpha) or np.isnan(tev):
        return float("nan")
    return float(mean_alpha - eta * tev)


def break_even_eta(mean_alpha: float, tev: float) -> float:
    """Break-Even-Aversion eta* = ᾱ / TEV.  +inf wenn TEV = 0."""
    if np.isnan(mean_alpha) or np.isnan(tev):
        return float("nan")
    if abs(tev) <= _TEV_EPS:
        return float("inf") if mean_alpha > 0 else float("nan")
    return float(mean_alpha / tev)


def raear_panel(tev_per_strategy: pd.DataFrame, etas: list[float]) -> pd.DataFrame:
    """Tabelle mit Zeilen = Strategien, IR, TES, RAEAR fuer jede eta, und Break-Even.

    Erwartet ``mean_alpha`` und ``tev`` in ``tev_per_strategy``.
    Primaere Spalten: ``tes`` (Tracking-Error-Std, bps), ``ir`` (dimensionslos).
    Sekundaere Spalten: ``raear_eta_*`` und ``eta_star`` (eta in bps⁻¹).
    """
    out = tev_per_strategy[["strategy", "mean_alpha", "tev"]].copy()
    out["tes"] = np.sqrt(out["tev"].clip(lower=0))
    out["ir"] = [information_ratio(a, v) for a, v in zip(out["mean_alpha"], out["tev"])]
    for eta in etas:
        out[f"raear_eta_{eta:g}"] = [raear(a, v, eta) for a, v in zip(out["mean_alpha"], out["tev"])]
    out["eta_star"] = [break_even_eta(a, v) for a, v in zip(out["mean_alpha"], out["tev"])]
    return out
