"""
tests.py -- Primary t-test, Sub-Group-Tests, Placebo (Thesis §4.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

from .. import config as cfg
from .clustering import mean_with_twoway_se


def _two_sided_p(t: float) -> float:
    """Zweiseitiges p-value approx. Normalverteilung (n gross)."""
    if np.isnan(t):
        return float("nan")
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))


def _one_sided_p(t: float, greater: bool = True) -> float:
    """Einseitiges p-value approx. Normalverteilung.

    ``greater=True`` prueft die registrierte Alternative ``mu > 0`` (H1, H2a,
    H2b). Stimmt das Vorzeichen mit der Alternative ueberein, ist dies die
    Haelfte des zweiseitigen p-Werts.
    """
    if np.isnan(t):
        return float("nan")
    upper = 1.0 - 0.5 * (1.0 + erf(t / sqrt(2.0)))
    return float(upper if greater else 1.0 - upper)


@dataclass
class TTestResult:
    mean: float
    se: float
    t: float
    p_value: float
    n: int
    label: str = ""
    alternative: str = "two-sided"
    p_one_sided: float = float("nan")


# ---------------------------------------------------------------------------
# Primary-Test: H0: mean (alpha_net_{S3, window B} - alpha_net_{MOC}) = 0
# ---------------------------------------------------------------------------

def holm_step_down(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm-Bonferroni Step-Down P-Value-Korrektur.

    Liefert ein Array adjustierter P-Werte in der Reihenfolge der Eingabe.
    NaNs in der Eingabe bleiben NaN. Werte > 1 werden auf 1 geklippt.

    Anwendung: Family-wise Error-Rate Kontrolle innerhalb einer Subgroup-
    Familie (Tier-Quintile, Stress-Tage, etc.). Die *primary* Tests
    (H1, H2a, H2b, H3) werden bewusst nicht korrigiert; nur die
    explorativen Sub-Group-Familien.
    """
    arr = np.asarray(p_values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr.copy()
    valid = ~np.isnan(arr)
    out = np.full_like(arr, np.nan, dtype=float)
    if valid.sum() == 0:
        return out
    idx = np.where(valid)[0]
    p_valid = arr[idx]
    # order[i] = position in p_valid of the i-th smallest p-value.
    # adj_sorted[j] = adjusted p for position j in p_valid.
    # out[idx] maps adj_sorted back to original array positions.
    order = np.argsort(p_valid)
    m = len(p_valid)
    adj_sorted = np.empty(m)
    running_max = 0.0
    for i, j in enumerate(order):
        candidate = (m - i) * p_valid[j]
        running_max = max(running_max, candidate)
        adj_sorted[j] = min(running_max, 1.0)
    out[idx] = adj_sorted
    return out


def primary_ttest(
    results: pd.DataFrame,
    strategy: str = "S3_FULL",
    window: str = cfg.PRIMARY_WINDOW,
    alpha_col: str = "net_alpha_bps",
    benchmark: str = "S0_MOC",
    size_frac: float = cfg.PARENT_ORDER_PRIMARY_FRACTION,
    alternative: str = "greater",
) -> TTestResult:
    required = {"strategy", "window", "order_id", "symbol", "date", alpha_col}
    label = f"primary:{strategy}-{benchmark}:{window}:{size_frac:g}"
    if not required.issubset(results.columns):
        return TTestResult(float("nan"), float("nan"), float("nan"), float("nan"), 0,
                           label, alternative, float("nan"))

    sub = results[(results["window"] == window)].copy()
    if "size_frac" in sub.columns:
        sub = sub[np.isclose(sub["size_frac"].astype(float), size_frac)]

    a = sub[sub["strategy"] == strategy][["order_id", "symbol", "date", alpha_col]]
    b = sub[sub["strategy"] == benchmark][["order_id", alpha_col]].rename(
        columns={alpha_col: f"{alpha_col}_benchmark"}
    )
    paired = a.merge(b, on="order_id", how="inner")
    if paired.empty:
        return TTestResult(float("nan"), float("nan"), float("nan"), float("nan"), 0,
                           label, alternative, float("nan"))
    diff = paired[alpha_col] - paired[f"{alpha_col}_benchmark"]
    mean, se = mean_with_twoway_se(diff, paired["symbol"], paired["date"])
    t = mean / se if se is not None and se > 0 else float("nan")
    p_one = _one_sided_p(t, greater=alternative != "less") if alternative != "two-sided" else float("nan")
    return TTestResult(
        mean=mean, se=se, t=float(t), p_value=_two_sided_p(t), n=len(paired),
        label=label, alternative=alternative, p_one_sided=p_one,
    )


# ---------------------------------------------------------------------------
# Sub-Group Tests: per-tier, per-year, per-size
# ---------------------------------------------------------------------------

def subgroup_ttests(
    results: pd.DataFrame,
    alpha_col: str = "net_alpha_bps",
    by: str = "tier",
    strategy: str = "S3_FULL",
    apply_holm: bool = True,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Sub-Group t-Tests mit optionaler Holm-Step-Down-Korrektur.

    Die ``apply_holm``-Korrektur betrifft *nur* die Family der Sub-Group-Tests
    in dieser Tabelle. Primaer-Tests bleiben separat unkorrigiert. Der
    zweiseitige ``p_value`` bleibt fuer Rueckwaerts-Kompatibilitaet erhalten;
    ``p_one_sided`` ergaenzt die registrierte gerichtete Alternative.
    """
    sub = results[results["strategy"] == strategy]
    if sub.empty or by not in sub.columns:
        return pd.DataFrame()
    rows = []
    for level, grp in sub.groupby(by, dropna=True):
        mean, se = mean_with_twoway_se(grp[alpha_col], grp["symbol"], grp["date"])
        t = mean / se if se is not None and se > 0 else float("nan")
        rows.append({
            "group": by, "level": level, "mean": mean, "se": se,
            "t": t, "p_value": _two_sided_p(t),
            "p_one_sided": _one_sided_p(t, greater=alternative != "less"),
            "alternative": alternative, "n": len(grp),
        })
    out = pd.DataFrame(rows)
    if not out.empty and apply_holm:
        out["p_holm"] = holm_step_down(out["p_value"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Placebo (11:00-12:00 ET)
# ---------------------------------------------------------------------------

def placebo_flag() -> dict:
    """Marker fuer Runner, dass Evaluations-Fenster auf 11:00-12:00 gesetzt wird."""
    return {
        "start": cfg.PLACEBO_WINDOW_START,
        "end": cfg.PLACEBO_WINDOW_END,
        "note": "Placebo: Strategien auf 11:00-12:00 ET laufen lassen",
    }
