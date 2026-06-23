"""Multiple-testing helpers used by the supplementary validation suite."""

from __future__ import annotations

from math import erf, sqrt
from typing import Iterable

import numpy as np
import pandas as pd


def two_sided_p_from_t(t_value: float) -> float:
    """Large-sample two-sided normal p-value."""
    if not np.isfinite(t_value):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(float(t_value)) / sqrt(2.0)))))


def one_sided_p_from_t(t_value: float, *, greater: bool = True) -> float:
    """Large-sample one-sided normal p-value.

    ``greater=True`` tests the registered alternative ``mu > 0`` (used for H1,
    H2a, and H2b); ``greater=False`` tests ``mu < 0``. When the estimate sign
    agrees with the alternative this equals half the two-sided p-value.
    """
    if not np.isfinite(t_value):
        return float("nan")
    upper = 1.0 - 0.5 * (1.0 + erf(float(t_value) / sqrt(2.0)))
    return float(upper if greater else 1.0 - upper)


def holm_step_down(p_values: Iterable[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values in original input order."""
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return out
    idx = np.where(valid)[0]
    vals = arr[idx]
    order = np.argsort(vals)
    m = len(vals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, original_pos in enumerate(order):
        candidate = (m - rank) * vals[original_pos]
        running_max = max(running_max, candidate)
        adjusted[original_pos] = min(1.0, running_max)
    out[idx] = adjusted
    return out


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values in original input order.

    Reported alongside Holm for exploratory families: Holm controls the
    family-wise error rate (the strict claim) while BH controls the false
    discovery rate, which is more informative when a family of subgroup tests is
    large and FWER control is conservative. NaNs are preserved.
    """
    arr = np.asarray(list(p_values), dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return out
    idx = np.where(valid)[0]
    vals = arr[idx]
    m = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    scaled = ranked * m / (np.arange(m) + 1.0)
    # Enforce monotonicity from the largest p downwards.
    running_min = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.minimum(running_min, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[idx] = restored
    return out


def attach_holm(
    frame: pd.DataFrame,
    *,
    p_col: str = "p_value",
    out_col: str = "p_holm",
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copy with Holm-adjusted p-values over the selected rows."""
    out = frame.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out
    selected = mask if mask is not None else pd.Series(True, index=out.index)
    selected = selected.reindex(out.index).fillna(False).astype(bool)
    if selected.any():
        out.loc[selected, out_col] = holm_step_down(out.loc[selected, p_col])
    return out


def attach_fdr(
    frame: pd.DataFrame,
    *,
    p_col: str = "p_value",
    out_col: str = "p_fdr_bh",
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copy with Benjamini-Hochberg FDR p-values over selected rows."""
    out = frame.copy()
    out[out_col] = np.nan
    if out.empty or p_col not in out.columns:
        return out
    selected = mask if mask is not None else pd.Series(True, index=out.index)
    selected = selected.reindex(out.index).fillna(False).astype(bool)
    if selected.any():
        out.loc[selected, out_col] = benjamini_hochberg(out.loc[selected, p_col])
    return out


def star_suffix(p_value: float) -> str:
    """Journal-style significance stars."""
    if not np.isfinite(p_value):
        return ""
    if p_value <= 0.01:
        return "$^{***}$"
    if p_value <= 0.05:
        return "$^{**}$"
    if p_value <= 0.10:
        return "$^{*}$"
    return ""

