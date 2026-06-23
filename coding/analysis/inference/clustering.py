"""
clustering.py -- Two-Way Cluster-Robust Standardfehler (Petersen 2009;
Cameron, Gelbach & Miller 2011).

Implementiert die ``symbol``- und ``date``-Clustering-Formel fuer die
Inferenz in Thesis §4.5.1::

    Var_twoway(beta) = Var_sym(beta) + Var_date(beta) - Var_intersect(beta)

wobei ``Var_intersect`` nach CGM auf den **(symbol x date)-Zellen** clustert.
Bei genau einer Beobachtung pro Zelle reduziert sich der Intersektions-Term
auf White/HC0; bei mehreren Zeilen pro Zelle (gepoolte Window/Size-Panels)
wuerde White die Within-Zell-Korrelation unter-subtrahieren.

Implementierung ist minimal und benoetigt keine statsmodels-Abhaengigkeit.
Fuer OLS mit Konstante ``y = X*beta + eps``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class OLSResult:
    coef: np.ndarray
    se_white: np.ndarray
    se_cluster_sym: np.ndarray
    se_cluster_date: np.ndarray
    se_cluster_twoway: np.ndarray
    names: list[str]
    n: int

    def tstat(self, which: str = "twoway") -> np.ndarray:
        se = {
            "white": self.se_white,
            "sym": self.se_cluster_sym,
            "date": self.se_cluster_date,
            "twoway": self.se_cluster_twoway,
        }[which]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.coef / se


def _cluster_cov(X: np.ndarray, resid: np.ndarray, cluster_ids: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Liu-White (1980) cluster-robust covariance."""
    u = X * resid[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cluster_ids):
        m = cluster_ids == c
        sc = u[m].sum(axis=0)
        meat += np.outer(sc, sc)
    return XtX_inv @ meat @ XtX_inv


def two_way_cluster_ols(
    y: np.ndarray,
    X: np.ndarray,
    cluster_sym: np.ndarray,
    cluster_date: np.ndarray,
    names: list[str] | None = None,
) -> OLSResult:
    """OLS mit White + symbol + date + two-way Cluster-Standardfehlern.

    Parameters
    ----------
    y : (n,) response.
    X : (n, k) Regressoren (incl. Intercept-Spalte).
    cluster_sym, cluster_date : (n,) Cluster-Ids.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if names is None:
        names = [f"x{i}" for i in range(k)]

    XtX = X.T @ X
    cond = np.linalg.cond(XtX)
    if cond > 1e10:
        log.warning(
            "two_way_cluster_ols: ill-conditioned X'X (cond=%.2e, n=%d) — "
            "SE estimates may be unreliable for thin subgroups",
            cond, n,
        )
    XtX_inv = np.linalg.pinv(XtX)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef

    # White / HC0 (reported separately; not the CGM intersection term)
    u = X * resid[:, None]
    V_white = XtX_inv @ (u.T @ u) @ XtX_inv

    V_sym = _cluster_cov(X, resid, cluster_sym, XtX_inv)
    V_date = _cluster_cov(X, resid, cluster_date, XtX_inv)
    # CGM intersection: cluster on (symbol, date) cells. With one observation
    # per cell this equals V_white exactly (each cell is a singleton cluster).
    inter_ids = pd.factorize(
        pd.Series(list(zip(cluster_sym, cluster_date)))
    )[0]
    V_inter = _cluster_cov(X, resid, inter_ids, XtX_inv)
    V_two = V_sym + V_date - V_inter

    # V_two can be non-PSD under thin panels; clamp diagonal and warn.
    diag_two = np.diag(V_two)
    n_clamped = int((diag_two < 0).sum())
    if n_clamped:
        log.warning(
            "two_way_cluster_ols: %d diagonal element(s) of V_two < 0 (min=%.2e); "
            "clamping to 0 — panel may be too thin for two-way clustering",
            n_clamped, float(diag_two.min()),
        )

    return OLSResult(
        coef=coef,
        se_white=np.sqrt(np.diag(V_white)),
        se_cluster_sym=np.sqrt(np.maximum(np.diag(V_sym), 0)),
        se_cluster_date=np.sqrt(np.maximum(np.diag(V_date), 0)),
        se_cluster_twoway=np.sqrt(np.maximum(diag_two, 0)),
        names=names,
        n=n,
    )


def assert_size_stratification_ok(
    panel: pd.DataFrame, size_col: str = "size_frac",
) -> dict:
    """Sanity-Check fuer size-stratifizierte Panels.

    Die Parent-Size ist eine **within-cell** Stratifikations-Variable (jede
    Zeile hat genau einen size_frac); sie ist *keine* Clustering-Dimension.
    Die two-way SE bleiben symbol x date geclustert. Diese Funktion verifiziert
    nur, dass

    * jeder ``order_id`` mit genau einem ``size_frac`` assoziiert ist, und
    * jede (symbol, date)-Cell mehrere ``size_frac``-Levels enthaelt.

    Bricht nicht ab; liefert eine Diagnose-Dict.
    """
    if size_col not in panel.columns or "order_id" not in panel.columns:
        return {"checked": False, "reason": "missing columns"}
    by_order = panel.groupby("order_id")[size_col].nunique()
    bad_orders = int((by_order > 1).sum())
    by_cell = panel.groupby(["symbol", "date"])[size_col].nunique()
    avg_levels = float(by_cell.mean()) if not by_cell.empty else 0.0
    return {
        "checked": True,
        "orders_with_multiple_sizes": bad_orders,  # erwartet 0
        "avg_size_levels_per_symbol_date_cell": avg_levels,
        "n_cells": int(by_cell.shape[0]),
        "n_orders": int(by_order.shape[0]),
    }


def mean_with_twoway_se(
    values: pd.Series,
    symbols: pd.Series,
    dates: pd.Series,
) -> tuple[float, float]:
    """Spezialfall: Mittelwert gegen 0 mit two-way SE.

    Praktisch: fittet ``values = mu + eps`` ueber nur einer Konstante.
    Liefert ``(mean, se_twoway)``.

    Fuer das Intercept-only-Modell ist der CGM-Schaetzer in geschlossener Form
    ``Var = (V_sym + V_date - V_inter) / n^2`` mit ``V_g = sum_g (sum_{i in g}
    u_i)^2``. Diese vektorisierte Form ist identisch zu ``two_way_cluster_ols``
    fuer ``X = 1`` (im Testsuite verifiziert), vermeidet aber die Python-Schleife
    ueber die ~``n`` Singleton-Intersektionszellen und ist dadurch um
    Groessenordnungen schneller auf grossen Panels.
    """
    df = pd.concat({"y": values, "s": symbols, "d": dates}, axis=1).dropna()
    if df.empty:
        return float("nan"), float("nan")
    y = df["y"].to_numpy(dtype=float)
    n = y.shape[0]
    mean = float(np.mean(y))
    u = y - mean
    sym_codes, sym_uni = pd.factorize(df["s"])
    date_codes, date_uni = pd.factorize(df["d"])
    inter_codes = pd.factorize(
        pd.Series(list(zip(sym_codes.tolist(), date_codes.tolist())))
    )[0]

    def _sq_sum(codes: np.ndarray, k: int) -> float:
        s = np.bincount(codes, weights=u, minlength=k)
        return float(np.dot(s, s))

    v_sym = _sq_sum(sym_codes, len(sym_uni))
    v_date = _sq_sum(date_codes, len(date_uni))
    v_inter = _sq_sum(inter_codes, int(inter_codes.max()) + 1 if n else 0)
    var = (v_sym + v_date - v_inter) / (n * n)
    se = float(np.sqrt(var)) if var > 0 else 0.0
    return mean, se
