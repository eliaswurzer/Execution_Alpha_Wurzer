"""
alpha.py -- Execution Alpha Metriken gemaess Thesis §4.4.

* ``execution_alpha``  -- Eq. 4.18
* ``net_execution_alpha`` -- Eq. 4.19
* ``to_bps`` Hilfsfunktion

Alle Werte werden in *Basispunkten* gefuehrt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg


def side_sign(side: str) -> int:
    s = side.strip().upper()
    if s == "BUY":
        return 1
    if s == "SELL":
        return -1
    raise ValueError(f"Unknown side {side!r}")


def to_bps(value: float, reference_price: float) -> float:
    if reference_price == 0 or np.isnan(reference_price):
        return float("nan")
    return value / reference_price * 1e4


# ---------------------------------------------------------------------------
# Eq. 4.18
# ---------------------------------------------------------------------------

def execution_alpha_bps(
    close_price: float,
    vwap_passive: float,
    fill_rate: float,
    side: str,
) -> float:
    """Thesis Eq. 4.18::

        alpha = phi * side * (P_C - P_intraday) / P_C * 10000
    """
    if np.isnan(close_price) or close_price <= 0:
        return float("nan")
    if fill_rate <= 0 or np.isnan(vwap_passive):
        return 0.0
    s = side_sign(side)
    return float(fill_rate * s * (close_price - vwap_passive) / close_price * 1e4)


# ---------------------------------------------------------------------------
# Market-Impact (square-root) Komponente
# ---------------------------------------------------------------------------

def impact_bps(
    parent_size_pct: float,
    *,
    threshold: float = cfg.IMPACT_ACTIVATION_THRESHOLD,
    coef_bps: float = cfg.IMPACT_COEF_BPS,
) -> float:
    """Square-root Impact in Basispunkten (Almgren-Chriss-Stil).

    Inaktiv (= 0) solange ``parent_size_pct <= threshold``. Sonst::

        impact = coef_bps * sqrt(parent_size_pct)

    Beide Parameter koennen ueber ``cfg.IMPACT_*`` oder explizit gesetzt
    werden; das ermoeglicht Sensitivitaeten ohne Code-Aenderung.
    """
    if np.isnan(parent_size_pct) or parent_size_pct <= threshold:
        return 0.0
    return float(coef_bps * np.sqrt(parent_size_pct))


def break_even_impact_coef(
    alpha_net_no_impact_bps: float,
    parent_size_pct: float,
) -> float:
    """Break-even impact coefficient: the kappa at which net alpha hits zero.

    Inverts impact_bps: kappa_be = alpha_net_no_impact / sqrt(parent_size_pct).
    Returns nan when size <= 0 or alpha <= 0 (no break-even exists).
    """
    if parent_size_pct <= 0 or np.isnan(alpha_net_no_impact_bps):
        return float("nan")
    if alpha_net_no_impact_bps <= 0:
        return float("nan")
    return float(alpha_net_no_impact_bps / np.sqrt(parent_size_pct))


# ---------------------------------------------------------------------------
# Eq. 4.19 (mit fuenfter Komponente alpha_impact)
# ---------------------------------------------------------------------------

def adverse_selection_cost_bps(adverse_selection_signed_bps: float) -> float:
    """One-sided adverse-selection cost from side-signed post-fill drift.

    ``adverse_selection_signed_bps`` is positive when post-fill drift is
    favorable to the passive supplier and negative when it moves against the
    fill. Net alpha only charges the adverse part; the signed diagnostic stays
    available on the result row.
    """
    if np.isnan(adverse_selection_signed_bps):
        return 0.0
    return float(max(0.0, -adverse_selection_signed_bps))


def net_execution_alpha_bps(
    alpha_gross_bps: float,
    fill_rate: float,
    maker_rebate_bps: float = cfg.MAKER_REBATE_BPS,
    commission_bps: float = cfg.COMMISSION_BPS,
    impact_component_bps: float = 0.0,
) -> float:
    """Implementation-shortfall net alpha::

        alpha_net = alpha + phi * r_make - c_comm - impact

    ``alpha_gross_bps`` is the close-relative gross alpha of eq:alpha_def and
    already prices the entire post-fill mid drift, including the realized
    adverse-selection component over the AS horizon. Adverse selection is
    therefore reported as a decomposition diagnostic of the gross alpha
    (see :func:`adverse_selection_cost_bps`) and must NOT be charged again
    here; doing so double-counts the first ``AS_HORIZON_SECONDS`` of drift.
    Only the cash adjustments enter: the maker rebate on the filled fraction,
    the commission on the full parent, and the stylized self-impact term
    (``impact_component_bps``, 0 below the activation threshold).
    """
    if np.isnan(alpha_gross_bps):
        return float("nan")
    return float(
        alpha_gross_bps
        + fill_rate * maker_rebate_bps
        - commission_bps
        - impact_component_bps
    )


# ---------------------------------------------------------------------------
# DataFrame-Panel Helfer
# ---------------------------------------------------------------------------

def attach_alpha_columns(
    results: pd.DataFrame,
    maker_rebate_bps: float = cfg.MAKER_REBATE_BPS,
    commission_bps: float = cfg.COMMISSION_BPS,
    impact_threshold: float = cfg.IMPACT_ACTIVATION_THRESHOLD,
    impact_coef_bps: float = cfg.IMPACT_COEF_BPS,
) -> pd.DataFrame:
    """Ergaenzt ``results`` um Alpha-Spalten incl. ``impact_bps``.

    Erwartete Input-Spalten:
        ``close_price``, ``vwap_passive``, ``fill_rate``, ``side``,
        ``adverse_selection_bps`` (optional, default 0),
        ``size_frac`` (Parent-Order-Groesse als Anteil von E[V_C]; default 0).
    """
    df = results.copy()
    side_sign_vec = np.where(df["side"].str.upper() == "BUY", 1.0, -1.0)
    cp = df["close_price"].to_numpy(dtype=float)
    vwap = df["vwap_passive"].to_numpy(dtype=float)
    fr = df["fill_rate"].to_numpy(dtype=float)
    valid_close = (cp > 0) & np.isfinite(cp)
    has_passive_fill = valid_close & np.isfinite(vwap) & (fr > 0)
    alpha_raw = np.where(valid_close & (fr <= 0), 0.0, np.nan)
    alpha_raw = np.where(
        has_passive_fill,
        fr * side_sign_vec * (cp - vwap) / cp * 1e4,
        alpha_raw,
    )
    df["alpha_bps"] = alpha_raw

    if "adverse_selection_bps" in df.columns:
        as_col = df["adverse_selection_bps"].fillna(0.0)
    else:
        as_col = pd.Series(0.0, index=df.index)

    if "size_frac" in df.columns:
        size_col = df["size_frac"].fillna(0.0)
    else:
        size_col = pd.Series(0.0, index=df.index)

    if "strategy" in df.columns:
        passive_mask = df["strategy"].fillna("").astype(str) != "S0_MOC"
    else:
        passive_mask = pd.Series(True, index=df.index)

    df["impact_bps"] = [
        impact_bps(s, threshold=impact_threshold, coef_bps=impact_coef_bps)
        if is_passive else 0.0
        for s, is_passive in zip(size_col, passive_mask)
    ]
    # Diagnostic decomposition column: the one-sided adverse part of the
    # post-fill drift that is already embedded in the close-relative gross
    # alpha. It is reported, not deducted again (implementation shortfall).
    df["adverse_selection_cost_bps"] = [
        adverse_selection_cost_bps(ase) for ase in as_col
    ]

    df["net_alpha_bps"] = [
        net_execution_alpha_bps(
            a, fr, maker_rebate_bps, commission_bps, imp,
        )
        for a, fr, imp in zip(
            df["alpha_bps"], df["fill_rate"], df["impact_bps"],
        )
    ]
    return df


def attach_moc_differential_columns(
    results: pd.DataFrame,
    benchmark: str = "S0_MOC",
) -> pd.DataFrame:
    """Attach per-order gross and net alpha differentials versus MOC."""
    df = results.copy()
    required = {"order_id", "strategy", "alpha_bps", "net_alpha_bps"}
    if df.empty or not required.issubset(df.columns):
        return df

    moc = (
        df[df["strategy"] == benchmark][["order_id", "alpha_bps", "net_alpha_bps"]]
        .drop_duplicates(subset=["order_id"])
        .rename(columns={
            "alpha_bps": "moc_alpha_bps",
            "net_alpha_bps": "moc_net_alpha_bps",
        })
    )
    if moc.empty:
        df["moc_alpha_bps"] = np.nan
        df["moc_net_alpha_bps"] = np.nan
    else:
        df = df.merge(moc, on="order_id", how="left")
    df["alpha_vs_moc_bps"] = df["alpha_bps"] - df["moc_alpha_bps"]
    df["net_alpha_vs_moc_bps"] = df["net_alpha_bps"] - df["moc_net_alpha_bps"]
    return df
