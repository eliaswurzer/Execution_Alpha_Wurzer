"""Tick-grid helpers shared by calibration and simulation."""

from __future__ import annotations

import numpy as np

from .. import config as cfg


def snap_limit_to_tick(limit_price: float, side: str) -> float:
    """Snap a model limit price onto the penny grid in the passive direction."""
    if not cfg.SNAP_LIMIT_TO_TICK or not np.isfinite(limit_price) or limit_price < 1.0:
        return float(limit_price)
    ticks = limit_price / cfg.TICK_SIZE
    if str(side).upper() == "BUY":
        return float(np.floor(ticks + 1e-4) * cfg.TICK_SIZE)
    return float(np.ceil(ticks - 1e-4) * cfg.TICK_SIZE)


def effective_limit_offset_bps(touch_price: float, limit_price: float, side: str) -> float:
    """Distance from the relevant touch to the placeable snapped limit."""
    touch = float(touch_price)
    limit = float(limit_price)
    if not np.isfinite(touch) or not np.isfinite(limit) or touch <= 0:
        return 0.0
    if str(side).upper() == "BUY":
        return float(max(0.0, (touch - limit) / touch * 1e4))
    return float(max(0.0, (limit - touch) / touch * 1e4))
