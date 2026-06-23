"""
kaplan_meier.py -- Empirische Kaplan-Meier Fill-Frequenzen, stratifiziert
nach Liquiditaets-Tier, Time-to-MOC-Cutoff-Bucket und Limit-Offset (gesnappt auf
``cfg.FILL_MODEL_OFFSET_GRID_BPS``) auf dem Calibration-Sample.

Exponiert dieselbe ``fill_probability``-/``survival``-API wie
``cox_ph.CoxFillModel``, sodass die Simulation-Engine via Config-Flag
zwischen den Spezifikationen umschalten kann.

Time-to-Cutoff-Buckets sind (ungebremst) ``[0-30, 30-90, 90-300, 300-900,
>=900]`` Sekunden. Default-Buckets in ``KM_TIME_TO_CUTOFF_BUCKETS``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..utils.symbols import expand_symbol_to_tier

log = logging.getLogger(__name__)


KM_TIME_TO_CUTOFF_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 30),
    (30, 90),
    (90, 300),
    (300, 900),
    (900, 24 * 3600),
)
# Backward-compatible alias for older diagnostics/imports. Semantics are now
# time-to-MOC-cutoff, not time-to-closing-auction.
KM_TIME_TO_CLOSE_BUCKETS = KM_TIME_TO_CUTOFF_BUCKETS


def _bucket_for_seconds(secs: float) -> tuple[int, int]:
    for lo, hi in KM_TIME_TO_CUTOFF_BUCKETS:
        if lo <= secs < hi:
            return (lo, hi)
    return KM_TIME_TO_CUTOFF_BUCKETS[-1]


def _snap_offset_bps(value: float | None) -> float:
    """Naechstgelegener Wert auf dem Kalibrierungs-Offset-Grid.

    Die Strategien posten kontinuierliche Offsets; das KM-Modell ist auf dem
    diskreten ``cfg.FILL_MODEL_OFFSET_GRID_BPS`` stratifiziert.
    """
    grid = np.asarray(cfg.FILL_MODEL_OFFSET_GRID_BPS, dtype=float)
    if value is None or not np.isfinite(value):
        return float(grid[0])
    v = max(0.0, float(value))
    return float(grid[int(np.argmin(np.abs(grid - v)))])


def _seconds_to_cutoff(t0: pd.Timestamp) -> float:
    end = pd.Timestamp.combine(t0.date(), cfg.MOC_CUTOFF)
    return float(max(0.0, (end - t0).total_seconds()))


# ---------------------------------------------------------------------------
# Empirisches Survival via Kaplan-Meier-Estimator
# ---------------------------------------------------------------------------

def _km_survival_function(
    durations: np.ndarray, events: np.ndarray, times: np.ndarray,
) -> np.ndarray:
    """KM-Estimator S(t) ausgewertet an ``times``.

    Durations werden einmalig auf Millisekunden gerundet, sodass Dedup,
    Death-Counts und Risk-Set-Dekremente dieselben Zeit-Keys verwenden
    (float-Noise aus Timestamp-Differenzen kann keine Ties aufspalten).
    Vektorisiert: O(n log n) statt O(unique * n).
    """
    times_arr = np.asarray(times, dtype=float)
    if len(durations) == 0:
        return np.ones_like(times_arr, dtype=float)
    d_ms = np.round(np.asarray(durations, dtype=float) * 1000.0).astype(np.int64)
    e = np.asarray(events, dtype=float)
    uniq_ms, inverse, counts = np.unique(d_ms, return_inverse=True, return_counts=True)
    deaths = np.bincount(inverse, weights=e, minlength=len(uniq_ms))
    n = len(d_ms)
    # Risk set just before each unique time: everyone not yet removed.
    at_risk = n - np.concatenate(([0], np.cumsum(counts)[:-1]))
    factors = 1.0 - np.where(at_risk > 0, deaths / at_risk, 0.0)
    surv = np.cumprod(factors)
    # Step-function lookup: S = 1 before the first observed duration.
    key_times = uniq_ms.astype(float) / 1000.0
    idx = np.searchsorted(key_times, times_arr, side="right") - 1
    out = np.ones_like(times_arr, dtype=float)
    mask = idx >= 0
    out[mask] = surv[idx[mask]]
    return out


# ---------------------------------------------------------------------------
# Modell-Wrapper -- API-kompatibel zu CoxFillModel
# ---------------------------------------------------------------------------

_KM_TS_GRID = np.array(
    [1, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600], dtype=float,
)


@dataclass
class KMFillModel:
    """Empirisches KM-Modell fuer einen Liquiditaets-Tier.

    Speichert pro (time-to-cutoff-bucket, offset-bps) ein KM-Step-Funktions-
    Sample plus eine offset-gepoolte Fallback-Kurve je ttc-Bucket. Die
    Offset-Stratifikation ist notwendig, damit Strategien, die sich nur in
    ihrer Offset-Politik unterscheiden (S1/S2/S3), unterschiedliche
    Fill-Dynamik sehen.
    """

    tier: int
    survival_table: dict[tuple[int, int, float], pd.Series] = field(default_factory=dict)
    # Offset-gepoolte Kurven je ttc-Bucket (Fallback fuer fehlende Zellen).
    pooled_table: dict[tuple[int, int], pd.Series] = field(default_factory=dict)

    # ---- training --------------------------------------------------------

    def fit(self, panel: pd.DataFrame) -> "KMFillModel":
        """``panel`` braucht Spalten ``duration``, ``event``, ``t0`` und
        (optional) ``limit_offset_bps``.

        Time-to-Cutoff-Bucket wird aus ``t0`` abgeleitet; der Offset wird auf
        das Kalibrierungs-Grid gesnappt.
        """
        if panel.empty:
            return self
        df = panel.copy()
        df["_ttc"] = df["t0"].apply(lambda ts: _seconds_to_cutoff(pd.Timestamp(ts)))
        df["_bucket"] = df["_ttc"].apply(_bucket_for_seconds)
        if "limit_offset_bps" in df.columns:
            df["_offset"] = df["limit_offset_bps"].astype(float).apply(_snap_offset_bps)
        else:
            df["_offset"] = _snap_offset_bps(0.0)
        for (bucket, offset), grp in df.groupby(["_bucket", "_offset"]):
            durations = grp["duration"].astype(float).to_numpy()
            events = grp["event"].astype(int).to_numpy()
            sf = _km_survival_function(durations, events, _KM_TS_GRID)
            key = (int(bucket[0]), int(bucket[1]), float(offset))
            self.survival_table[key] = pd.Series(sf, index=_KM_TS_GRID)
        for bucket, grp in df.groupby("_bucket"):
            durations = grp["duration"].astype(float).to_numpy()
            events = grp["event"].astype(int).to_numpy()
            sf = _km_survival_function(durations, events, _KM_TS_GRID)
            self.pooled_table[(int(bucket[0]), int(bucket[1]))] = pd.Series(sf, index=_KM_TS_GRID)
        return self

    # ---- prediction (api-kompatibel zu CoxFillModel) ---------------------

    def survival(self, horizon_seconds: float, x) -> float | np.ndarray:
        if not self.survival_table and not self.pooled_table:
            raise RuntimeError("KMFillModel nicht gefittet")
        # ``x`` darf ein dict, Series oder DataFrame sein. ``t0`` kommt im
        # Simulations-Pfad aus strategies/base.py (sv["t0"] = t); ohne t0
        # faellt der Lookup auf den 0-30s-Bucket zurueck.
        if isinstance(x, (dict, pd.Series)):
            return self._lookup(
                horizon_seconds, x.get("t0"), x.get("limit_offset_bps"),
            )
        if isinstance(x, pd.DataFrame):
            t0s = x["t0"].tolist() if "t0" in x.columns else [None] * len(x)
            offs = (
                x["limit_offset_bps"].tolist()
                if "limit_offset_bps" in x.columns else [None] * len(x)
            )
            return np.array([
                self._lookup(horizon_seconds, t, o) for t, o in zip(t0s, offs)
            ])
        return self._lookup(horizon_seconds, None, None)

    def fill_probability(self, horizon_seconds: float, x) -> float | np.ndarray:
        s = self.survival(horizon_seconds, x)
        return 1.0 - s if np.isscalar(s) else 1.0 - np.asarray(s)

    def _lookup(self, horizon_seconds: float, t0, offset_bps=None) -> float:
        if t0 is None or pd.isna(t0):
            bucket = KM_TIME_TO_CUTOFF_BUCKETS[0]
        else:
            ttc = _seconds_to_cutoff(pd.Timestamp(t0))
            bucket = _bucket_for_seconds(ttc)
        offset = _snap_offset_bps(
            None if offset_bps is None or pd.isna(offset_bps) else float(offset_bps)
        )
        sf = self.survival_table.get((int(bucket[0]), int(bucket[1]), offset))
        if sf is None or sf.empty:
            sf = self.pooled_table.get((int(bucket[0]), int(bucket[1])))
        if sf is None or sf.empty:
            return 1.0
        idx = np.searchsorted(sf.index.to_numpy(), horizon_seconds, side="right") - 1
        idx = max(idx, 0)
        return float(sf.iloc[idx])

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({
            "tier": self.tier,
            "survival_table": {
                k: v.to_dict() for k, v in self.survival_table.items()
            },
            "pooled_table": {
                k: v.to_dict() for k, v in self.pooled_table.items()
            },
        }, path)

    @classmethod
    def load(cls, path: Path) -> "KMFillModel":
        import joblib
        payload = joblib.load(path)
        st: dict[tuple[int, int, float], pd.Series] = {}
        pooled: dict[tuple[int, int], pd.Series] = {}
        for k, v in payload["survival_table"].items():
            key = tuple(k)
            if len(key) == 2:
                # Artefakt aus der Vor-Offset-Stratifikation: als gepoolte
                # Kurve weiterverwenden.
                pooled[(int(key[0]), int(key[1]))] = pd.Series(v)
            else:
                st[(int(key[0]), int(key[1]), float(key[2]))] = pd.Series(v)
        for k, v in payload.get("pooled_table", {}).items():
            key = tuple(k)
            pooled[(int(key[0]), int(key[1]))] = pd.Series(v)
        return cls(tier=int(payload["tier"]), survival_table=st, pooled_table=pooled)


# ---------------------------------------------------------------------------
# Multi-Tier Container
# ---------------------------------------------------------------------------

@dataclass
class TieredKMFillModel:
    """Mirror von ``TieredFillModel`` -- liefert per Symbol ein KM-Modell."""

    models: dict[int, KMFillModel] = field(default_factory=dict)
    symbol_to_tier: dict[str, int] = field(default_factory=dict)

    def fit_panel(self, panel: pd.DataFrame, symbol_tier_map: pd.DataFrame) -> "TieredKMFillModel":
        self.symbol_to_tier = expand_symbol_to_tier(
            dict(zip(symbol_tier_map["symbol"], symbol_tier_map["tier"].astype(int)))
        )
        panel = panel.copy()
        panel["tier"] = panel["symbol"].map(self.symbol_to_tier)
        panel = panel.dropna(subset=["tier"])
        panel["tier"] = panel["tier"].astype(int)
        for tier, grp in panel.groupby("tier"):
            try:
                self.models[int(tier)] = KMFillModel(tier=int(tier)).fit(grp)
            except Exception as e:
                log.error("KM tier %d skipped: %s", tier, e)
        return self

    def for_symbol(self, symbol: str) -> KMFillModel:
        tier = self.symbol_to_tier.get(symbol)
        if tier is None or tier not in self.models:
            raise KeyError(f"No KM fill-model for symbol {symbol}")
        return self.models[tier]

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        for tier, m in self.models.items():
            m.save(dirpath / f"km_tier_{tier}.pkl")
        pd.DataFrame(list(self.symbol_to_tier.items()),
                     columns=["symbol", "tier"]).to_csv(
            dirpath / "km_symbol_tier_map.csv", index=False,
        )

    @classmethod
    def load(cls, dirpath: Path) -> "TieredKMFillModel":
        models = {}
        for pkl in sorted(dirpath.glob("km_tier_*.pkl")):
            m = KMFillModel.load(pkl)
            models[m.tier] = m
        mapping_path = dirpath / "km_symbol_tier_map.csv"
        if mapping_path.exists():
            mapping = pd.read_csv(mapping_path)
            mapping_dict = expand_symbol_to_tier(
                dict(zip(mapping["symbol"], mapping["tier"].astype(int)))
            )
        else:
            # Fallback: nimm das mapping aus dem Cox-PH Run wenn vorhanden
            cox_map = dirpath / "symbol_tier_map.csv"
            if cox_map.exists():
                mapping = pd.read_csv(cox_map)
                mapping_dict = expand_symbol_to_tier(
                    dict(zip(mapping["symbol"], mapping["tier"].astype(int)))
                )
            else:
                mapping_dict = {}
        return cls(models=models, symbol_to_tier=mapping_dict)
