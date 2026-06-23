"""S3 -- Signal-Conditioned Strategy (Thesis §4.3.5, Eq. 4.12-4.15).

``delta^{(3)}(t) = delta_max * g(t) * h(sigma_t) * f_OFI(t) * f_IMB(t)``

plus Volatilitaets-Regime-Switch (x1.2 / x0.8) auf ``delta_max`` (Thesis Eq.
nach 4.14).

Diese Datei stellt drei Modi bereit, die als separate Strategien registriert
werden:

* ``mode='ofi'``  -- nur ``f_OFI`` plus Vol-Regime auf S2
* ``mode='imb'``  -- nur ``f_IMB`` plus Vol-Regime auf S2
* ``mode='full'`` -- beide Faktoren plus Vol-Regime (Thesis-konform)

Damit lassen sich H2a (OFI marginal), H2b (IMB marginal) und die Interaktion
sauber dekomponieren.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .. import config as cfg
from .base import MarketState
from .time_adaptive import TimeAdaptiveStrategy


SignalMode = Literal["ofi", "imb", "full"]


class SignalConditionedStrategy(TimeAdaptiveStrategy):
    """S3-Strategie mit konfigurierbarem Signal-Mode."""

    def __init__(
        self,
        *,
        mode: SignalMode = "full",
        kappa: float = 0.5,
        lambda_imb: float = 1.0,
        ofi_scale: float = 1.0,
        adv_shares: float = 1.0,
        imbalance_scale_shares: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("ofi", "imb", "full"):
            raise ValueError(f"Unknown signal mode: {mode!r}")
        self.mode = mode
        self.kappa = kappa
        self.lambda_imb = lambda_imb
        self.ofi_scale = ofi_scale
        self.adv_shares = adv_shares
        self.imbalance_scale_shares = imbalance_scale_shares

    # Strategy-IDs werden vom Registry vergeben; ``name`` dient nur als
    # Default-Anzeige fuer Reports falls keine ID hinterlegt ist.
    @property
    def name(self) -> str:  # type: ignore[override]
        return {
            "ofi": "S3_OFI",
            "imb": "S3_IMB",
            "full": "S3_FULL",
        }[self.mode]

    # ---- Signal-Faktoren -------------------------------------------------

    def _f_ofi(self, t: pd.Timestamp, side: str, state: MarketState) -> float:
        """Thesis Eq. 4.14: ``max(1 - kappa * tanh(OFI/OFI_scale), f_min)``."""
        if state.ofi is None or state.ofi.empty:
            return 1.0
        idx = int(np.searchsorted(state.ofi_times, t.value, side="right")) - 1
        if idx < 0:
            return 1.0
        ofi_col = "ofi_zscore" if "ofi_zscore" in state.ofi.columns else "ofi"
        ofi_val = float(state.ofi[ofi_col].iloc[idx])
        if side == "SELL":
            ofi_val = -ofi_val
        arg = ofi_val / max(self.ofi_scale, 1e-9)
        factor = 1.0 - self.kappa * float(np.tanh(arg))
        return float(max(factor, cfg.F_OFI_MIN))

    def _f_imb(self, t: pd.Timestamp, side: str, state: MarketState) -> float:
        """Thesis Eq. 4.15 adapted to a causal pre-cutoff proxy.

        The proxy is scaled by expected closing-auction volume when available.
        ``cfg.AUCTION_IMBALANCE_START`` is not used here; that timestamp is the
        official feed dissemination time and belongs to reporting/subgroups.
        """
        if state.imbalance is None or state.imbalance.empty:
            return 1.0
        idx = int(np.searchsorted(state.imbalance_times, t.value, side="right")) - 1
        if idx < 0:
            return 1.0
        imb_shares = float(state.imbalance["imb_shares"].iloc[idx])
        if side == "SELL":
            imb_shares = -imb_shares
        scale = self.imbalance_scale_shares or self.adv_shares
        if scale <= 0:
            return 1.0
        arg = -imb_shares / scale
        raw = 1.0 - self.lambda_imb * arg
        return float(np.clip(raw, cfg.F_IMB_MIN, cfg.F_IMB_MAX))

    def _vol_regime_multiplier(
        self, t: pd.Timestamp, state: MarketState, sigma_bar: float,
    ) -> float:
        if state.rv is None or state.rv.empty or sigma_bar <= 0:
            return 1.0
        idx = int(np.searchsorted(state.rv_times, t.value, side="right")) - 1
        if idx < 0:
            return 1.0
        sigma_t = float(state.rv.iloc[idx])
        if sigma_t > cfg.VOL_REGIME_THRESHOLD * sigma_bar:
            return cfg.VOL_REGIME_HIGH_MULTIPLIER
        return cfg.VOL_REGIME_LOW_MULTIPLIER

    # ---- override --------------------------------------------------------

    def limit_offset_bps(
        self,
        t: pd.Timestamp,
        side: str,
        state: MarketState,
        sigma_bar: float,
        delta_max_bps: float,
    ) -> float:
        base = super().limit_offset_bps(t, side, state, sigma_bar, delta_max_bps)
        regime = self._vol_regime_multiplier(t, state, sigma_bar)

        f_ofi = self._f_ofi(t, side, state) if self.mode in ("ofi", "full") else 1.0
        f_imb = self._f_imb(t, side, state) if self.mode in ("imb", "full") else 1.0

        return base * regime * f_ofi * f_imb


# ---------------------------------------------------------------------------
# Bequeme Konstruktoren fuer das Registry
# ---------------------------------------------------------------------------

class SignalConditionedOFI(SignalConditionedStrategy):
    """S3 mit nur dem OFI-Faktor (plus Vol-Regime)."""

    def __init__(self, **kwargs):
        super().__init__(mode="ofi", **kwargs)


class SignalConditionedIMB(SignalConditionedStrategy):
    """S3 mit nur dem Imbalance-Faktor (plus Vol-Regime)."""

    def __init__(self, **kwargs):
        super().__init__(mode="imb", **kwargs)


class SignalConditionedFull(SignalConditionedStrategy):
    """S3 mit beiden Faktoren (Thesis-konform)."""

    def __init__(self, **kwargs):
        super().__init__(mode="full", **kwargs)
