"""
tod_schedule.py -- Time-of-Day Optimal Posting Schedule (Strategy S4).

Trains an XGBoost regressor on the presample to predict expected adverse
selection (AS) per fill as a function of time-of-day and current market state.

Sign convention (matches strategies/base.py realized AS): y is the side-signed
post-fill mid-quote drift in bps. POSITIVE y = drift in our favor after the
fill (good); NEGATIVE y = price moved against us (adverse, bad). At execution
time the strategy therefore posts proportionally MORE in intervals where the
predicted AS is HIGH relative to the presample mean, and sticks to the TWAP
baseline when conditions are average or worse.

Schedule logic:
    base_fraction = 1 / intervals_remaining  (TWAP baseline)
    favorability = clip((predicted_AS - AS_mean_signed) / AS_std, 0, 1)
    slice_fraction = min(1.0, base_fraction * (1 + favorability))

GPU training: set device="cuda" in XGBoost params (RTX 5070 Ti, ~1 s per tier).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from .state_vector import STATE_COLUMNS
from .xgb_survival import resolve_xgb_device

log = logging.getLogger(__name__)


# Features used for the AS regressor: state vector + time-of-day encoding
_TOD_FEATURES = STATE_COLUMNS + ["tod_sin", "tod_cos", "time_to_close_frac"]


def _tod_feature_values(t: pd.Timestamp) -> dict[str, float]:
    ts = pd.Timestamp(t)
    seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
    day_seconds = 24 * 3600
    close_seconds = cfg.RTH_CLOSE.hour * 3600 + cfg.RTH_CLOSE.minute * 60
    return {
        "tod_sin": float(np.sin(2 * np.pi * seconds / day_seconds)),
        "tod_cos": float(np.cos(2 * np.pi * seconds / day_seconds)),
        "time_to_close_frac": float(np.clip(
            (close_seconds - seconds) / (close_seconds - 9 * 3600), 0, 1,
        )),
    }


def _add_tod_features(df: pd.DataFrame, t0_col: str = "t0") -> pd.DataFrame:
    """Add time-of-day sine/cosine encoding and time-to-close fraction."""
    out = df.copy()
    if t0_col in out.columns:
        t = pd.to_datetime(out[t0_col])
        seconds_since_midnight = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
        day_seconds = 24 * 3600
        out["tod_sin"] = np.sin(2 * np.pi * seconds_since_midnight / day_seconds)
        out["tod_cos"] = np.cos(2 * np.pi * seconds_since_midnight / day_seconds)
        close_seconds = cfg.RTH_CLOSE.hour * 3600 + cfg.RTH_CLOSE.minute * 60
        out["time_to_close_frac"] = np.clip(
            (close_seconds - seconds_since_midnight) / (close_seconds - 9 * 3600), 0, 1
        )
    else:
        out["tod_sin"] = 0.0
        out["tod_cos"] = 1.0
        out["time_to_close_frac"] = 0.5
    return out


@dataclass
class TODSchedule:
    """XGBoost-based time-of-day posting schedule.

    Predicts expected adverse selection per fill from state vector +
    time-of-day features, then uses the prediction to upweight posting
    in low-AS intervals.
    """

    _model: object | None = field(default=None, repr=False)
    _as_mean: float = 0.0          # presample mean |AS| (scale diagnostic)
    _as_mean_signed: float = 0.0   # presample mean signed AS (favorability ref)
    _as_std: float = 1.0
    _feature_cols: list[str] = field(default_factory=list)

    # ---- training --------------------------------------------------------

    def calibrate(
        self,
        event_panel_with_as: pd.DataFrame,
        *,
        xgb_device: str = "cpu",
        random_state: int = cfg.DEFAULT_SEED,
    ) -> "TODSchedule":
        """Fit from the presample event panel augmented with realized per-fill AS.

        Required columns:
            STATE_COLUMNS + ['as_bps', 'event', 'side'] + t0 (timestamp)
        Only filled events (event == 1) are used for training.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        df = event_panel_with_as.copy()
        df = _add_tod_features(df, t0_col="t0")
        filled = df[df["event"] == 1].dropna(subset=["as_bps"])

        if len(filled) < 50:
            log.warning("TODSchedule: fewer than 50 fill events for training (%d)", len(filled))
            return self

        feature_cols = [c for c in _TOD_FEATURES if c in filled.columns]
        self._feature_cols = feature_cols

        X = filled[feature_cols].fillna(0.0).astype(np.float32).values
        y = filled["as_bps"].astype(np.float32).values  # signed AS per fill

        self._as_mean = float(np.mean(np.abs(y)))
        self._as_mean_signed = float(np.mean(y))
        self._as_std = float(np.std(y)) or 1.0

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 20,
            "tree_method": "hist",
            "device": resolve_xgb_device(xgb_device),
            "seed": int(random_state),
            "nthread": 1,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        self._model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

        # Quick in-sample diagnostics
        pred = self._model.predict(dtrain)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        log.info(
            "TODSchedule fitted: n=%d mean_|AS|=%.2f bps RMSE=%.2f bps",
            len(filled), self._as_mean, rmse,
        )
        return self

    # ---- prediction ------------------------------------------------------

    def predict_as(self, t: pd.Timestamp, state_vector) -> float:
        """Predict expected signed AS for a fill at time t with given state.

        Fallbacks return the presample signed mean, i.e. a neutral prediction
        that yields favorability 0 (pure TWAP) in :meth:`fraction`.
        """
        if self._model is None:
            return self._as_mean_signed

        try:
            import xgboost as xgb
        except ImportError:
            return self._as_mean_signed

        row: dict[str, float] = {}
        if isinstance(state_vector, dict):
            row.update(state_vector)
        elif isinstance(state_vector, pd.Series):
            row.update(state_vector.to_dict())

        row.update(_tod_feature_values(t))
        df = pd.DataFrame([row])
        for c in self._feature_cols:
            if c not in df.columns:
                log.debug("TODSchedule.predict_as: missing feature '%s' — filling with 0.0", c)
                df[c] = 0.0

        X = df[self._feature_cols].fillna(0.0).astype(np.float32).values
        if hasattr(self._model, "inplace_predict"):
            return float(self._model.inplace_predict(X)[0])
        dm = xgb.DMatrix(X, feature_names=self._feature_cols)
        return float(self._model.predict(dm)[0])

    def fraction(
        self,
        t: pd.Timestamp,
        intervals_remaining: int,
        state_vector,
    ) -> float:
        """Return fraction of qty_remaining to post at interval t.

        Base is TWAP (1/intervals_remaining). Signed AS is POSITIVE when the
        post-fill drift favors us, so posting is scaled up when the predicted
        AS exceeds the presample signed mean (favorable window) and stays at
        the TWAP base when conditions are average or adverse. Capped at 1.0
        (can never post more than what remains).
        """
        base = 1.0 / max(1, intervals_remaining)
        if self._model is None:
            return base

        predicted_as = self.predict_as(t, state_vector)
        # favorability in [0, 1]: 0 = at-average or adverse, 1 = one std (or
        # more) better-than-average expected post-fill drift.
        favorability = float(np.clip(
            (predicted_as - self._as_mean_signed) / max(1e-6, self._as_std),
            0.0, 1.0,
        ))
        # Boost posting fraction by up to 2× when conditions are very favorable
        return min(1.0, base * (1.0 + favorability))

    # ---- persistence -----------------------------------------------------

    def save(self, dirpath: Path) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_model(str(dirpath / "tod_schedule_xgb.ubj"))
        import joblib
        joblib.dump({
            "as_mean": self._as_mean,
            "as_mean_signed": self._as_mean_signed,
            "as_std": self._as_std,
            "feature_cols": self._feature_cols,
        }, dirpath / "tod_schedule_meta.pkl")

    @classmethod
    def load(cls, dirpath: Path) -> "TODSchedule":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required")
        dirpath = Path(dirpath)
        obj = cls()
        ubj = dirpath / "tod_schedule_xgb.ubj"
        if ubj.exists():
            booster = xgb.Booster()
            booster.load_model(str(ubj))
            obj._model = booster
        meta = dirpath / "tod_schedule_meta.pkl"
        if meta.exists():
            import joblib
            d = joblib.load(meta)
            obj._as_mean = d.get("as_mean", 0.0)
            # Artifacts written before the sign fix lack the signed mean; 0.0
            # keeps favorability conservative until re-calibration.
            obj._as_mean_signed = d.get("as_mean_signed", 0.0)
            obj._as_std = d.get("as_std", 1.0)
            obj._feature_cols = d.get("feature_cols", [])
        return obj

    @property
    def fitted(self) -> bool:
        return self._model is not None
