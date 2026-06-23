"""
xgb_survival.py -- Gradient-Boosted Survival Fill Model (XGBoost alternative to Cox-PH).

Same interface as CoxFillModel / TieredFillModel so it can be swapped in
without any changes to the simulation engine.

Model: XGBoost with objective="survival:cox" fits the same proportional-hazards
likelihood as Cox-PH but uses a tree ensemble f(X) instead of a linear predictor.
A Breslow baseline hazard is estimated on the training data to convert risk scores
into absolute survival probabilities.

Artifacts saved to:
    xgb_tier_<n>.ubj   -- XGBoost binary model
    xgb_tier_<n>_breslow.pkl  -- (event_times, cum_hazard) numpy arrays
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..utils.symbols import expand_symbol_to_tier
from .state_vector import STATE_COLUMNS

log = logging.getLogger(__name__)


def resolve_xgb_device(requested: str = "cpu") -> str:
    """Resolve an explicit XGBoost device request.

    CPU is the default for reproducibility. ``auto`` probes CUDA once at the
    caller boundary; callers should pass the resolved value into per-tier fits.
    """
    requested = (requested or "cpu").strip().lower()
    if requested in {"cpu", "cuda"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported XGBoost device {requested!r}; use cpu, cuda, or auto")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Breslow baseline hazard
# ---------------------------------------------------------------------------

def _breslow(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Breslow cumulative baseline hazard.

    ``risk_scores`` must be log-risk margins f(X) (XGBoost ``output_margin=True``),
    NOT the hazard-ratio scale exp(f) that ``predict`` returns by default.

    Returns (unique_event_times, cumulative_baseline_hazard, risk_center).
    ``risk_center`` is the centering constant subtracted from the margins for
    numerical stability; the same constant must be subtracted from margins at
    evaluation time (see ``_eval_breslow``), otherwise the hazard is off by a
    factor exp(risk_center).
    """
    risk_center = float(risk_scores.max())
    exp_scores = np.exp(risk_scores - risk_center)  # numerical stability
    order = np.argsort(durations)
    sorted_dur = durations[order]
    sorted_ev = events[order].astype(bool)
    sorted_exp = exp_scores[order]

    # Reverse cumulative sum = denominator (sum of exp-scores still at risk)
    rev_cumsum = np.cumsum(sorted_exp[::-1])[::-1]

    ev_times = sorted_dur[sorted_ev]
    ev_risk_sums = rev_cumsum[sorted_ev]

    unique_times = np.unique(ev_times)
    cum_haz = np.empty(len(unique_times))
    running = 0.0
    for i, t in enumerate(unique_times):
        mask_t = ev_times == t
        n_ev = float(mask_t.sum())
        # Breslow risk set at time t = sum of exp-scores over ALL observations
        # still at risk (duration >= t). In the reverse-cumsum, that is the value
        # at the FIRST tied position (the largest), NOT the mean across tied
        # positions — the mean understates the risk set and inflates the hazard.
        risk_sum = float(ev_risk_sums[mask_t].max())
        running += n_ev / max(risk_sum, 1e-12)
        cum_haz[i] = running
    return unique_times, cum_haz, risk_center


def _eval_breslow(
    h0_times: np.ndarray,
    h0_vals: np.ndarray,
    risk_score: float,
    horizon: float,
    risk_center: float = 0.0,
) -> float:
    """S(horizon | x) = exp(-H0(horizon) * exp(f(x) - risk_center)).

    ``risk_score`` must be the log-risk margin f(x); ``risk_center`` is the
    centering constant used when the Breslow baseline was computed, so the
    centering cancels exactly: H0_centered * exp(f - c) == H0_true * exp(f).
    """
    idx = np.searchsorted(h0_times, horizon, side="right") - 1
    h0 = h0_vals[idx] if idx >= 0 else 0.0
    risk_score_clipped = float(np.clip(risk_score - risk_center, -50.0, 50.0))
    return float(np.exp(-h0 * np.exp(risk_score_clipped)))


def _eval_breslow_batch(
    h0_times: np.ndarray,
    h0_vals: np.ndarray,
    risk_scores: np.ndarray,
    horizon: float,
    risk_center: float = 0.0,
) -> np.ndarray:
    """Vectorized ``_eval_breslow`` over an array of margins at a single horizon.

    The baseline lookup is a single scalar (one horizon), so only the margins
    vary; identical math to ``_eval_breslow`` applied row-wise.
    """
    idx = int(np.searchsorted(h0_times, horizon, side="right")) - 1
    h0 = float(h0_vals[idx]) if idx >= 0 else 0.0
    rs = np.clip(np.asarray(risk_scores, dtype=float) - risk_center, -50.0, 50.0)
    return np.exp(-h0 * np.exp(rs))


# ---------------------------------------------------------------------------
# Per-tier model
# ---------------------------------------------------------------------------

@dataclass
class XGBFillModel:
    """XGBoost survival fill model for one liquidity tier."""

    tier: int
    covariates: list[str] = field(default_factory=lambda: list(STATE_COLUMNS))
    _booster: object | None = field(default=None, repr=False)          # xgb.Booster
    _h0_times: np.ndarray | None = field(default=None, repr=False)
    _h0_vals: np.ndarray | None = field(default=None, repr=False)
    # Centering constant subtracted from training margins when the Breslow
    # baseline was computed; must be subtracted again at predict time.
    _risk_center: float = field(default=0.0, repr=False)
    # Per-covariate training medians for NaN imputation at predict time
    # (matches training imputation — avoids train/serve skew, thesis B2).
    _medians: dict = field(default_factory=dict, repr=False)
    # Best boosting iteration from early stopping; (0, n) iteration range used
    # for all predictions so deployment scores match the early-stopped model.
    _iteration_range: tuple = field(default=(0, 0), repr=False)

    def fit(
        self,
        panel: pd.DataFrame,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 20,
        xgb_device: str = "cpu",
        random_state: int = cfg.DEFAULT_SEED,
    ) -> "XGBFillModel":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        cols = [c for c in self.covariates if c in panel.columns]
        if not cols:
            raise ValueError(f"Tier {self.tier}: no usable XGB covariates in panel")
        df = panel[cols + ["duration", "event"]].copy()
        # Capture medians on the raw (pre-imputation) training data, then impute.
        medians: dict[str, float] = {}
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            med = float(df[c].median())
            if not np.isfinite(med):
                med = 0.0
            medians[c] = med
            df[c] = df[c].fillna(med)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df["event"] = pd.to_numeric(df["event"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["duration", "event"])
        df = df[df["duration"] > 0]
        df["event"] = (df["event"] > 0).astype(np.int8)

        if len(df) < 50 or df["event"].sum() < 10:
            raise ValueError(
                f"Tier {self.tier}: insufficient events (n={len(df)}, k={int(df['event'].sum())})"
            )

        X = df[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Tier {self.tier}: non-finite values remain in XGB features")
        # XGBoost survival:cox expects label = duration, with negative values for censored
        y = df["duration"].to_numpy(dtype=np.float32)
        event = df["event"].to_numpy(dtype=np.float32)
        if not np.isfinite(y).all() or not np.isfinite(event).all():
            raise ValueError(f"Tier {self.tier}: non-finite duration/event values")
        # Censored observations: negate duration (XGBoost convention for survival:cox)
        y_xgb = np.where(event.astype(bool), y, -y)
        device = resolve_xgb_device(xgb_device)

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "max_depth": max_depth,
            "eta": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "tree_method": "hist",
            "device": device,
            "seed": int(random_state),
            "nthread": 1,       # CPU threads per worker (GPU handles compute)
            "verbosity": 0,
        }

        # Hold out a validation split for early stopping (thesis C3/C4). Use a
        # deterministic random split; require enough validation events, else
        # fall back to a fixed-round fit without early stopping.
        rng = np.random.default_rng(random_state)
        n = len(df)
        perm = rng.permutation(n)
        n_valid = int(0.2 * n)
        valid_idx = perm[:n_valid]
        train_idx = perm[n_valid:]
        enough_valid = (
            n_valid >= 50
            and int(event[valid_idx].sum()) >= 5
            and int(event[train_idx].sum()) >= 10
        )

        if enough_valid:
            dtr = xgb.DMatrix(X[train_idx], label=y_xgb[train_idx], feature_names=cols)
            dva = xgb.DMatrix(X[valid_idx], label=y_xgb[valid_idx], feature_names=cols)
            booster = xgb.train(
                params, dtr,
                num_boost_round=n_estimators,
                evals=[(dtr, "train"), (dva, "valid")],
                early_stopping_rounds=max(10, n_estimators // 10),
                verbose_eval=False,
            )
            best_it = getattr(booster, "best_iteration", None)
            self._iteration_range = (0, int(best_it) + 1) if best_it is not None else (0, 0)
            # Breslow baseline on the TRAIN portion, using the early-stopped trees.
            # output_margin=True yields the log-risk f(x); the default predict
            # output for survival:cox is the hazard ratio exp(f(x)), which must
            # NOT be fed into the Breslow estimator (double exponentiation).
            dtr_full = xgb.DMatrix(X[train_idx], feature_names=cols)
            train_scores = booster.predict(
                dtr_full, iteration_range=self._iteration_range, output_margin=True,
            )
            bres_dur = df["duration"].to_numpy()[train_idx]
            bres_ev = df["event"].to_numpy()[train_idx]
        else:
            dtrain = xgb.DMatrix(X, label=y_xgb, feature_names=cols)
            booster = xgb.train(
                params, dtrain,
                num_boost_round=n_estimators,
                verbose_eval=False,
            )
            self._iteration_range = (0, 0)  # use all trees
            train_scores = booster.predict(
                dtrain, iteration_range=self._iteration_range, output_margin=True,
            )
            bres_dur = df["duration"].to_numpy()
            bres_ev = df["event"].to_numpy()

        self._booster = booster
        self.covariates = cols
        self._medians = medians

        # Compute Breslow baseline (train portion, early-stopped trees)
        self._h0_times, self._h0_vals, self._risk_center = _breslow(
            durations=bres_dur,
            events=bres_ev,
            risk_scores=train_scores,
        )

        log.info(
            "XGBoost fitted tier %d: n=%d k=%d features=%d device=%s",
            self.tier, len(df), int(df["event"].sum()), len(cols), device,
        )
        return self

    def _med(self, c: str) -> float:
        return float(self._medians.get(c, 0.0))

    def _to_xarray(self, x) -> np.ndarray:
        # Impute missing / NaN with persisted training medians (fallback 0.0)
        # so prediction matches the imputation used at fit time (thesis B2).
        if isinstance(x, dict):
            row = [float(x.get(c, self._med(c))) for c in self.covariates]
        elif isinstance(x, pd.Series):
            row = [float(x.get(c, self._med(c))) for c in self.covariates]
        elif isinstance(x, pd.DataFrame):
            out = x.reindex(columns=self.covariates)
            for c in self.covariates:
                out[c] = out[c].fillna(self._med(c))
            return out.to_numpy(dtype=np.float32)
        else:
            row = [float(v) for v in x]
        row = [m if not np.isfinite(v) else v
               for v, m in zip(row, (self._med(c) for c in self.covariates))]
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def survival(self, horizon_seconds: float, x) -> float | np.ndarray:
        """S(h|X). Scalar for a single dict/Series/row, per-row ``np.ndarray`` for
        a multi-row DataFrame/2D input (matches the cox_ph / kaplan_meier API)."""
        if self._booster is None or self._h0_times is None or self._h0_vals is None:
            raise RuntimeError("XGBFillModel not fitted")
        arr = self._to_xarray(x)  # shape (N, k); single dict/Series -> (1, k)
        # Margin scale (log-risk f(x)) is required: default survival:cox output
        # is the hazard ratio exp(f), which would be double-exponentiated in
        # _eval_breslow. inplace_predict avoids a per-call DMatrix allocation.
        try:
            scores = self._booster.inplace_predict(
                arr, iteration_range=self._iteration_range, predict_type="margin",
            )
        except (AttributeError, TypeError):
            import xgboost as xgb
            dm = xgb.DMatrix(arr, feature_names=self.covariates)
            scores = self._booster.predict(
                dm, iteration_range=self._iteration_range, output_margin=True,
            )
        scores = np.asarray(scores, dtype=float).ravel()
        values = _eval_breslow_batch(
            self._h0_times, self._h0_vals, scores, horizon_seconds,
            risk_center=self._risk_center,
        )
        return float(values[0]) if values.shape[0] == 1 else values

    def fill_probability(self, horizon_seconds: float, x) -> float | np.ndarray:
        s = self.survival(horizon_seconds, x)
        return 1.0 - s if np.isscalar(s) else 1.0 - np.asarray(s)

    def save(self, dirpath: Path, suffix: str = "") -> None:
        if self._booster is None or self._h0_times is None or self._h0_vals is None:
            raise RuntimeError("XGBFillModel not fitted")
        dirpath.mkdir(parents=True, exist_ok=True)
        tag = f"xgb_tier_{self.tier}{suffix}"
        import joblib
        self._booster.save_model(str(dirpath / f"{tag}.ubj"))
        joblib.dump({
            "tier": self.tier,
            "covariates": self.covariates,
            "h0_times": self._h0_times,
            "h0_vals": self._h0_vals,
            "medians": self._medians,
            "iteration_range": self._iteration_range,
            "risk_center": self._risk_center,
        }, dirpath / f"{tag}_breslow.pkl")

    @classmethod
    def load(cls, dirpath: Path, tier: int, suffix: str = "") -> "XGBFillModel":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required")
        tag = f"xgb_tier_{tier}{suffix}"
        booster = xgb.Booster()
        booster.load_model(str(dirpath / f"{tag}.ubj"))
        import joblib
        meta = joblib.load(dirpath / f"{tag}_breslow.pkl")
        required_meta = {"risk_center", "medians", "iteration_range"}
        missing_meta = sorted(required_meta - set(meta))
        if missing_meta:
            raise ValueError(
                f"Legacy XGB Breslow artifact {tag}_breslow.pkl is missing "
                f"{missing_meta}; re-run fill-model calibration."
            )
        obj = cls(tier=tier, covariates=meta["covariates"])
        obj._booster = booster
        obj._h0_times = meta["h0_times"]
        obj._h0_vals = meta["h0_vals"]
        obj._medians = meta["medians"]
        obj._iteration_range = tuple(meta["iteration_range"])
        obj._risk_center = float(meta["risk_center"])
        return obj


# ---------------------------------------------------------------------------
# Multi-tier container — same API as TieredFillModel
# ---------------------------------------------------------------------------

@dataclass
class TieredXGBFillModel:
    """Container: tier -> XGBFillModel. Drop-in replacement for TieredFillModel."""

    models: dict[int, XGBFillModel] = field(default_factory=dict)
    symbol_to_tier: dict[str, int] = field(default_factory=dict)

    def fit_panel(
        self,
        panel: pd.DataFrame,
        symbol_tier_map: pd.DataFrame,
        *,
        strict: bool = False,
        **fit_kwargs,
    ) -> "TieredXGBFillModel":
        self.symbol_to_tier = expand_symbol_to_tier(
            dict(zip(symbol_tier_map["symbol"], symbol_tier_map["tier"].astype(int)))
        )
        expected_tiers = set(symbol_tier_map["tier"].dropna().astype(int).unique())
        panel = panel.copy()
        panel["tier"] = panel["symbol"].map(self.symbol_to_tier)
        panel = panel.dropna(subset=["tier"])
        panel["tier"] = panel["tier"].astype(int)
        resolved_device = resolve_xgb_device(fit_kwargs.pop("xgb_device", "cpu"))
        fit_kwargs["xgb_device"] = resolved_device

        for tier, grp in panel.groupby("tier"):
            try:
                m = XGBFillModel(tier=int(tier)).fit(grp, **fit_kwargs)
                self.models[int(tier)] = m
            except ValueError as e:
                log.warning("Tier %d XGB skipped (insufficient data): %s", tier, e)
            except Exception as e:
                log.error("Tier %d XGB failed unexpectedly: %s", tier, e, exc_info=True)
                raise
        missing_tiers = sorted(expected_tiers - set(self.models))
        if strict and missing_tiers:
            raise RuntimeError(f"XGB fit missing tiers: {missing_tiers}")
        return self

    def for_symbol(self, symbol: str) -> XGBFillModel:
        tier = self.symbol_to_tier.get(symbol)
        if tier is None or tier not in self.models:
            raise KeyError(f"No XGB fill-model for symbol {symbol!r}")
        return self.models[tier]

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        for tier, m in self.models.items():
            m.save(dirpath)

    @classmethod
    def load(cls, dirpath: Path) -> "TieredXGBFillModel":
        dirpath = Path(dirpath)
        models = {}
        for ubj in sorted(dirpath.glob("xgb_tier_*.ubj")):
            # Extract tier number from filename xgb_tier_<n>.ubj
            tier = int(ubj.stem.split("_")[2])
            models[tier] = XGBFillModel.load(dirpath, tier)
        if not models:
            raise FileNotFoundError(f"No XGB model files found in {dirpath}")
        mapping = pd.read_csv(dirpath / "symbol_tier_map.csv")
        mapping_dict = expand_symbol_to_tier(
            dict(zip(mapping["symbol"], mapping["tier"].astype(int)))
        )
        return cls(models=models, symbol_to_tier=mapping_dict)
