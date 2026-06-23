"""Tracking-Error-Variance nach Thesis Eq. 4.21.

Zusaetzlich: Independence- vs. Perfect-Correlation-Upper-Bound fuer Portfolio-
Tracking-Error gemaess Thesis §4.4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tracking_error_variance(
    results: pd.DataFrame,
    alpha_col: str = "net_alpha_bps",
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    """TEV je Strategie.

    ``TEV_s = mean((alpha_{i,d,s} - mean_s)^2)``
    """
    grp = results.groupby(strategy_col)
    out = grp[alpha_col].agg(
        mean_alpha="mean",
        tev=lambda s: float(np.var(s.dropna(), ddof=1)),
        n="count",
    ).reset_index()
    return out


def portfolio_tracking_error(
    tev_per_strategy: pd.DataFrame,
    n_positions: int,
) -> pd.DataFrame:
    """Portfolio-TE unter Independence- und Perfect-Correlation-Bound.

    Independence:      TE_port = sqrt(TEV / N)
    Perfect corr.:     TE_port = sqrt(TEV)
    """
    if n_positions <= 0:
        raise ValueError(f"n_positions must be > 0, got {n_positions}")
    out = tev_per_strategy.copy()
    out["te_port_indep"] = np.sqrt(out["tev"].astype(float) / n_positions)
    out["te_port_perf_corr"] = np.sqrt(out["tev"].astype(float))
    return out
