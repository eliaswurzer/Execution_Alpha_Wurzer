"""Execution-Strategien S0/S1/S2/S3 gemaess Thesis §4.3.

S3 wird in drei Varianten angeboten -- OFI, IMB, FULL -- damit die Runner die
Beitraege der einzelnen Signal-Faktoren zerlegen koennen.
"""

from .base import ExecutionStrategy, FillResult, MarketState
from .moc import MOCStrategy
from .static_passive import StaticPassiveStrategy
from .time_adaptive import TimeAdaptiveStrategy
from .signal_conditioned import (
    SignalConditionedFull,
    SignalConditionedIMB,
    SignalConditionedOFI,
    SignalConditionedStrategy,
)
from .registry import STRATEGY_REGISTRY, get_strategy
from .value_aware import ValueAwareXGBStrategy

__all__ = [
    "ExecutionStrategy",
    "FillResult",
    "MarketState",
    "MOCStrategy",
    "StaticPassiveStrategy",
    "TimeAdaptiveStrategy",
    "SignalConditionedStrategy",
    "SignalConditionedOFI",
    "SignalConditionedIMB",
    "SignalConditionedFull",
    "ValueAwareXGBStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
