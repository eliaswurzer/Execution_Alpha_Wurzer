"""Registry: String-Lookup fuer die Strategien.

Drei S3-Varianten werden separat registriert, sodass die Runner H2a (OFI),
H2b (IMB) und die Interaktion direkt anfordern koennen.
"""

from __future__ import annotations

from .moc import MOCStrategy
from .optimal_schedule import OptimalScheduleStrategy
from .signal_conditioned import (
    SignalConditionedFull,
    SignalConditionedIMB,
    SignalConditionedOFI,
    SignalConditionedStrategy,
)
from .static_passive import StaticPassiveStrategy
from .time_adaptive import TimeAdaptiveStrategy
from .value_aware import ValueAwareXGBStrategy


STRATEGY_REGISTRY = {
    "S0_MOC": MOCStrategy,
    "S1_STATIC": StaticPassiveStrategy,
    "S2_TIME_ADAPTIVE": TimeAdaptiveStrategy,
    "S3_OFI": SignalConditionedOFI,
    "S3_IMB": SignalConditionedIMB,
    "S3_FULL": SignalConditionedFull,
    # Backward-compat: ``S3_SIGNAL`` zeigt auf die Full-Variante
    "S3_SIGNAL": SignalConditionedFull,
    # S4: Learned optimal quantity schedule (XGBoost-predicted AS minimization)
    "S4_TOD": OptimalScheduleStrategy,
    "S5_VALUE_AWARE_XGB": ValueAwareXGBStrategy,
}


def get_strategy(name: str, **kwargs):
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy {name}. Known: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)


__all__ = [
    "STRATEGY_REGISTRY",
    "get_strategy",
    "MOCStrategy",
    "StaticPassiveStrategy",
    "TimeAdaptiveStrategy",
    "SignalConditionedStrategy",
    "SignalConditionedOFI",
    "SignalConditionedIMB",
    "SignalConditionedFull",
    "ValueAwareXGBStrategy",
]
