"""Fill-Rate Aggregation (Thesis Eq. 4.20)."""

from __future__ import annotations

import pandas as pd


def per_strategy_fill_rate(results: pd.DataFrame) -> pd.DataFrame:
    """Unconditional + per-liquidity-tier fill-rate per strategy."""
    grp = results.groupby("strategy")
    out = grp["fill_rate"].agg(["mean", "median", "count"]).reset_index()
    return out


def per_tier_fill_rate(results: pd.DataFrame) -> pd.DataFrame:
    """Mean fill-rate by (strategy, tier)."""
    if "tier" not in results.columns:
        return pd.DataFrame()
    grp = results.groupby(["strategy", "tier"])
    return grp["fill_rate"].mean().reset_index()
