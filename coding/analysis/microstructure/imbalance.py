"""
imbalance.py -- Pre-cutoff closing-pressure proxy (Thesis section 5.4.5).

The official NYSE/Nasdaq auction-imbalance feed is not available in this data
extract. For S3_IMB/S3_FULL we therefore use a causal public proxy observed
before the MOC cutoff:

    IMB_proxy_t = (rolling_OFI(t - L, t)
                + best-depth drift versus the start of the lookback window)
                * NBBO_SIZE_SHARES_PER_LOT

The signal is measured in SHARES: NBBO sizes (and hence OFI contributions and
depth drift) are denominated in Daily-TAQ round lots, so the proxy is
converted with ``cfg.NBBO_SIZE_SHARES_PER_LOT`` before being returned.
Without that conversion the S3 scaling by expected closing-auction volume
(shares) would dampen the factor ~100x and neutralise f_IMB. S3 scales it by
expected closing-auction volume, not full-day ADV, so the factor can move
limit prices at realistic pre-close magnitudes. Positive values indicate
buy-side closing pressure.

THESIS_DEVIATION: the true auction feed contains indicative clearing prices and
auction-order quantities. This proxy uses only NBBO and OFI. It is intentionally
separate from cfg.AUCTION_IMBALANCE_START, which remains the official
dissemination timestamp used for reporting/subgroups.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .. import config as cfg
from .ofi import event_level_ofi


def compute_auction_imbalance_proxy(
    nbbo: pd.DataFrame,
    start_time: _dt.time | None = None,
    lookback_seconds: int | None = None,
) -> pd.DataFrame:
    """Return a causal pre-cutoff closing-pressure proxy.

    Parameters
    ----------
    nbbo:
        Single symbol-day NBBO with ``time``, ``best_bid``, ``best_offer``,
        ``best_bid_size`` and ``best_offer_size``.
    start_time:
        Earliest timestamp at which the proxy is exposed to the strategy.
        Default: ``cfg.IMBALANCE_PROXY_START``.
    lookback_seconds:
        Rolling lookback length. Default:
        ``cfg.IMBALANCE_PROXY_LOOKBACK_SECONDS``.

    Returns
    -------
    DataFrame with ``time``, ``imb_shares`` and ``imb_sign``.
    """
    cols = ["time", "imb_shares", "imb_sign"]
    if nbbo.empty:
        return pd.DataFrame(columns=cols)

    required = {"time", "best_bid", "best_offer", "best_bid_size", "best_offer_size"}
    if not required.issubset(nbbo.columns):
        return pd.DataFrame(columns=cols)

    start = start_time or cfg.IMBALANCE_PROXY_START
    lookback = int(lookback_seconds or cfg.IMBALANCE_PROXY_LOOKBACK_SECONDS)
    if lookback <= 0:
        raise ValueError("lookback_seconds must be positive")

    df = nbbo.sort_values("time").reset_index(drop=True).copy()
    df["_ofi"] = event_level_ofi(df).reset_index(drop=True).to_numpy(dtype=float)

    window = df[df["time"].dt.time >= start].reset_index(drop=True)
    if window.empty:
        return pd.DataFrame(columns=cols)

    rolling_ofi = (
        window.set_index("time")["_ofi"]
        .rolling(f"{lookback}s", min_periods=1)
        .sum()
        .to_numpy(dtype=float)
    )

    times = window["time"].values.astype("int64")
    start_ts = pd.Timestamp.combine(window["time"].iloc[0].date(), start)
    start_ns = int(start_ts.value)
    lookback_ns = lookback * 1_000_000_000

    depth = (
        window["best_bid_size"].astype(float).to_numpy()
        - window["best_offer_size"].astype(float).to_numpy()
    )
    baseline_ns = np.maximum(times - lookback_ns, start_ns)
    baseline_idx = np.searchsorted(times, baseline_ns, side="left")
    baseline_idx = np.clip(baseline_idx, 0, len(depth) - 1)
    depth_drift = depth - depth[baseline_idx]

    # OFI contributions and depth drift are in round lots (Daily TAQ NBBO
    # convention); convert to shares so the column name and the S3 scaling
    # by expected closing-auction volume (shares) are unit-consistent.
    imb = (rolling_ofi + depth_drift) * float(cfg.NBBO_SIZE_SHARES_PER_LOT)
    out = pd.DataFrame({
        "time": window["time"].values,
        "imb_shares": imb,
    })
    out["imb_sign"] = np.sign(out["imb_shares"]).fillna(0).astype(int)
    return out
