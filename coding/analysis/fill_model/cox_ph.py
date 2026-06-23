"""
cox_ph.py -- Cox Proportional-Hazards Fill-Modell (Thesis §4.2.3).

Schaetzt ``lambda(h | X) = lambda_0(h) * exp(beta' X)`` per Partial-Likelihood
mit ``lifelines.CoxPHFitter``. Ein Modell wird **pro Liquiditaets-Tier**
kalibriert.

Interface:

* ``CoxFillModel.fit(panel)`` -- nimmt Panel aus ``state_vector.build_event_panel``
* ``CoxFillModel.survival(horizon, X)`` -- S(h | X)
* ``CoxFillModel.fill_probability(horizon, X)`` -- 1 - S(h|X)
* ``save(path) / load(path)`` -- Pickle der trainierten Koeffizienten

Die Event-Duration ist in Sekunden. Horizont in Sekunden.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..utils.symbols import expand_symbol_to_tier
from .state_vector import STATE_COLUMNS

log = logging.getLogger(__name__)

# On the standardized (unit-variance) covariate scale a well-behaved Cox fit has
# coefficients of order one. A larger magnitude signals quasi-separation that
# lifelines converged to without raising, which the penalizer escalation in
# ``CoxFillModel.fit`` treats as a soft failure and escalates the penalizer.
_COEF_SANITY_MAX = 25.0

# Winsorization cap on the standardized covariate scale, applied identically at
# fit and predict time. Heavy-tailed covariates (displayed depth reaching
# |z|>100, realized vol |z|~65) otherwise produce a handful of astronomical
# ``exp(beta'z)`` partial hazards that dominate the Breslow risk-set sum and
# collapse the baseline survival to ~1 (i.e. ~zero fill) in liquid tiers.
# Clipping the standardized inputs at |z|<=5 removes that leverage without
# discarding signal; the cap is persisted per model for train/serve parity.
_WINSOR_Z = 5.0

# Covariates dropped from the Cox design before fitting. They stay in
# STATE_COLUMNS so KM/XGB are unaffected. ``D0`` is exactly redundant with
# ``q0`` (q0 = 0.5*D0, see state_vector.py), and the first time-of-day dummy is
# dropped as the reference category because the full ToD one-hot set sums to one
# over the sampled hours (rank-deficient). Both feed the singular Hessian /
# convergence failures observed in the liquid tiers.
_COX_DROP_COVARIATES = ("D0", f"tod_{cfg.TOD_HOUR_BINS[0]}")

# Baseline-collapse / calibration sanity-gate thresholds, evaluated in-sample on
# the (standardized, winsorized) training panel at the fill horizon. The primary
# signal is |mean_pred - observed|; the baseline-fill check is an absolute floor
# (a genuinely collapsed Breslow baseline has 1 - S0(h) ~ 0). A relative-to-
# observed floor would false-positive on healthy dispersed-covariate fits because
# the baseline-at-mean sits below the mean predicted fill (Jensen gap on exp).
_CALIB_ABS_TOL = 0.10              # |mean_pred - observed| beyond this is degenerate
_FIT_BASELINE_FLOOR = 0.01         # implied baseline fill this close to zero has collapsed


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@dataclass
class CoxFillModel:
    """Cox-PH Fill-Modell fuer einen Liquiditaets-Tier.

    Attributes
    ----------
    tier : int
        Liquiditaets-Tier (1 = engste Spreads).
    fitter : lifelines.CoxPHFitter
        Trainiertes lifelines-Objekt. ``None`` bis ``fit`` aufgerufen wurde.
    covariates : list[str]
        Spaltenreihenfolge des State-Vektors.
    """

    tier: int
    covariates: list[str] = field(default_factory=lambda: list(STATE_COLUMNS))
    fitter: object | None = None  # lifelines.CoxPHFitter
    # Per-covariate training medians, persisted so that prediction-time NaN
    # imputation matches training (avoids train/serve skew — see thesis A2).
    medians: dict[str, float] = field(default_factory=dict)
    # Schoenfeld proportional-hazards test result (per-covariate p-values +
    # global p-value), populated at fit time as a diagnostic (thesis B1).
    ph_test: dict[str, float] = field(default_factory=dict)
    # Per-covariate standardization (z-score) applied before the Cox fit and
    # re-applied identically at prediction time, persisted so there is no
    # train/serve skew. Empty for legacy models, which then predict unscaled.
    scale_mean: dict[str, float] = field(default_factory=dict)
    scale_std: dict[str, float] = field(default_factory=dict)
    # Winsorization cap on the standardized scale; persisted so predictions clip
    # identically to training. ``None`` for legacy models (then no clipping).
    winsor_z: float | None = None
    # Fast-path prediction cache (rebuilt lazily; not persisted). Holds the
    # fitted coefficients, lifelines' covariate centering means, and the
    # baseline survival step function so S(h|x) = S0(h)^exp(beta'(z - mean))
    # can be evaluated without lifelines' per-call DataFrame machinery, where
    # z is the standardized covariate row.
    _fast_beta: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_mean: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_bs_times: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_bs_vals: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_scale_mean: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_scale_std: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fast_winsor_z: float | None = field(default=None, repr=False, compare=False)
    _fast_path_failed: bool = field(default=False, repr=False, compare=False)

    # ---- training --------------------------------------------------------

    def fit(self, panel: pd.DataFrame, penalizer: float = 0.01) -> "CoxFillModel":
        """Schaetzt Cox-PH auf dem gelieferten Event-Panel.

        ``panel`` muss Spalten ``duration``, ``event`` und die Covariates
        enthalten. NaNs in Covariates werden mit dem Spalten-Median imputiert;
        diese Mediane werden gespeichert und beim Predicten identisch
        wiederverwendet (kein Train/Serve-Skew).
        """
        from lifelines import CoxPHFitter  # local import (optional dependency)
        from lifelines.exceptions import ConvergenceWarning

        cols = [
            c for c in self.covariates
            if c in panel.columns and c not in _COX_DROP_COVARIATES
        ]
        missing = (set(self.covariates) - set(_COX_DROP_COVARIATES)) - set(cols)
        if missing:
            log.warning("Tier %d panel missing covariates %s", self.tier, missing)

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
            if df[c].isna().any():
                df[c] = df[c].fillna(med)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df["event"] = pd.to_numeric(df["event"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["duration", "event"])
        df = df[df["duration"] > 0]
        df["event"] = (df["event"] > 0).astype(np.int8)

        usable_cols: list[str] = []
        dropped_cols: list[str] = []
        for c in cols:
            series = pd.to_numeric(df[c], errors="coerce")
            variance = float(series.var(ddof=0)) if len(series) else 0.0
            if not np.isfinite(variance) or variance <= 1e-12 or series.nunique(dropna=True) <= 1:
                dropped_cols.append(c)
            else:
                usable_cols.append(c)
        if dropped_cols:
            log.warning(
                "Tier %d dropping near-constant Cox covariates: %s",
                self.tier, dropped_cols,
            )
        cols = usable_cols
        if not cols:
            raise ValueError(f"Tier {self.tier}: no usable Cox covariates after variance filter")
        df = df[cols + ["duration", "event"]]

        if len(df) < 50 or df["event"].sum() < 10:
            raise ValueError(
                f"Tier {self.tier}: insufficient events "
                f"(n={len(df)}, k={df['event'].sum()})"
            )

        # Standardize covariates (z-score) before fitting. Cox coefficients are
        # scale-dependent, so a covariate with a tiny absolute scale (e.g. the
        # realized-volatility feature in liquid tiers) otherwise forces a
        # blown-up coefficient under weak penalization (quasi-separation). With
        # unit-variance inputs the penalizer regularizes every covariate fairly.
        # The scaler is persisted and re-applied identically at prediction time.
        scale_mean = {c: float(df[c].mean()) for c in cols}
        scale_std = {c: float(df[c].std(ddof=0)) for c in cols}
        for c in cols:
            if not np.isfinite(scale_std[c]) or scale_std[c] < 1e-9:
                scale_std[c] = 1.0  # effectively constant; leave unscaled
        df_fit = df.copy()
        for c in cols:
            df_fit[c] = (df[c] - scale_mean[c]) / scale_std[c]
        # Winsorize the standardized covariates to remove the high-leverage tails
        # that otherwise dominate the Breslow risk-set sum and collapse the
        # baseline survival (see _WINSOR_Z). lifelines' centering mean is then
        # the mean of the clipped inputs, so prediction-time clipping is exact.
        df_fit[cols] = df_fit[cols].clip(-_WINSOR_Z, _WINSOR_Z)

        penalizers = [float(penalizer)]
        for candidate in (0.05, 0.1, 0.5, 1.0):
            if candidate > penalizers[-1]:
                penalizers.append(candidate)
        last_exc: Exception | None = None
        cph = None
        for pen in penalizers:
            try:
                candidate = CoxPHFitter(penalizer=pen)
                # Treat lifelines convergence warnings as fit failures so the
                # penalizer escalates rather than silently accepting a model that
                # lifelines flagged as non-converged.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    candidate.fit(df_fit, duration_col="duration", event_col="event", show_progress=False)
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Tier %d Cox-PH failed with penalizer %.4g: %s",
                    self.tier, pen, exc,
                )
                continue
            # Sanity gate: on the standardized scale a well-behaved fit has
            # coefficients of order one. A magnitude in the tens still signals
            # quasi-separation that lifelines converged to without raising, so
            # escalate the penalizer rather than accept a degenerate model.
            max_coef = (
                float(np.max(np.abs(candidate.params_.to_numpy())))
                if len(candidate.params_) else 0.0
            )
            if max_coef > _COEF_SANITY_MAX and pen < penalizers[-1]:
                last_exc = RuntimeError(
                    f"standardized |coef|max={max_coef:.1f} > {_COEF_SANITY_MAX}"
                )
                log.warning(
                    "Tier %d Cox-PH penalizer %.4g gave |coef|max=%.1f; escalating",
                    self.tier, pen, max_coef,
                )
                continue
            # Baseline / calibration sanity gate: a fit whose Breslow baseline has
            # collapsed predicts ~zero fill for typical orders even though the
            # empirical rate is substantial (the liquid-tier pathology). This is
            # not caught by the |coef| gate (coefficients can stay modest while a
            # few high-leverage rows crush the baseline). Escalate the penalizer;
            # if even the strongest penalizer stays degenerate, fail the fit.
            h = float(cfg.FILL_MODEL_HORIZON_SECONDS)
            observed, mean_pred, base_fill = self._fit_calibration_stats(
                candidate, df_fit, cols, h,
            )
            collapsed = (
                abs(mean_pred - observed) > _CALIB_ABS_TOL
                or base_fill < _FIT_BASELINE_FLOOR
            )
            if collapsed:
                last_exc = RuntimeError(
                    f"baseline collapse at horizon {h:.0f}s: observed={observed:.3f} "
                    f"mean_pred={mean_pred:.3f} base_fill={base_fill:.3f}"
                )
                if pen < penalizers[-1]:
                    log.warning(
                        "Tier %d Cox-PH penalizer %.4g degenerate "
                        "(observed=%.3f mean_pred=%.3f base_fill=%.3f); escalating",
                        self.tier, pen, observed, mean_pred, base_fill,
                    )
                    continue
                log.error(
                    "Tier %d Cox-PH still degenerate at strongest penalizer %.4g "
                    "(observed=%.3f mean_pred=%.3f base_fill=%.3f); failing",
                    self.tier, pen, observed, mean_pred, base_fill,
                )
                break  # cph stays None -> ValueError below
            cph = candidate
            if pen != float(penalizer):
                log.warning(
                    "Tier %d Cox-PH required stronger penalizer %.4g",
                    self.tier, pen,
                )
            break
        if cph is None:
            raise ValueError(f"Tier {self.tier}: Cox-PH fit failed after retries: {last_exc}")
        self.fitter = cph
        self.covariates = cols
        self.medians = {c: medians[c] for c in cols}
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.winsor_z = _WINSOR_Z
        # Invalidate any fast-path cache from a previous fit on this object.
        self._fast_beta = None
        self._fast_mean = None
        self._fast_bs_times = None
        self._fast_bs_vals = None
        self._fast_scale_mean = None
        self._fast_scale_std = None
        self._fast_winsor_z = None
        self._fast_path_failed = False
        self.ph_test = self._run_ph_test(cph, df_fit)
        log.info("Cox-PH fitted on tier %d: n=%d k=%d concordance=%.3f ph_global_p=%.3g",
                 self.tier, len(df), int(df["event"].sum()), cph.concordance_index_,
                 self.ph_test.get("_global", float("nan")))
        return self

    @staticmethod
    def _run_ph_test(cph, df: pd.DataFrame) -> dict[str, float]:
        """Schoenfeld proportional-hazards test (thesis B1).

        Returns a dict ``{covariate: p_value, '_global': global_p}``. Low
        p-values flag covariates whose effect is not constant over the fill
        horizon, i.e. a violation of the PH assumption. Diagnostic only; the
        fit is not altered.
        """
        try:
            from lifelines.statistics import proportional_hazard_test
            res = proportional_hazard_test(cph, df, time_transform="rank")
            summ = res.summary
            out = {str(idx): float(row["p"]) for idx, row in summ.iterrows()}
            # Global p: Bonferroni-corrected minimum across covariates. The raw
            # minimum (kept as _min_p) is anti-conservative as a global test.
            if out:
                min_p = float(min(out.values()))
                out["_min_p"] = min_p
                out["_global"] = float(min(1.0, len(summ) * min_p))
            else:
                out["_global"] = float("nan")
            return out
        except Exception as exc:  # pragma: no cover - diagnostic best-effort
            log.warning("PH test skipped: %s", exc)
            return {}

    @staticmethod
    def _fit_calibration_stats(cph, df_fit: pd.DataFrame, cols: list[str], h: float) -> tuple[float, float, float]:
        """In-sample calibration stats for the baseline sanity gate.

        Returns ``(observed_fill_rate, mean_predicted_fill, baseline_fill)`` at
        horizon ``h`` using lifelines' centered parameterisation
        ``S(h|x) = S0(h) ** exp(beta'(z - norm_mean))``, evaluated on the already
        standardized + winsorized ``df_fit`` so it matches the served model. A
        collapsed Breslow baseline shows up as ``baseline_fill`` ~ 0 and
        ``mean_predicted_fill`` far below ``observed_fill_rate``.
        """
        bs = cph.baseline_survival_
        bst = bs.index.to_numpy(dtype=float)
        bsv = bs.iloc[:, 0].to_numpy(dtype=float)
        idx = int(np.searchsorted(bst, h, side="right")) - 1
        s0 = float(bsv[idx]) if idx >= 0 else 1.0
        beta = cph.params_.reindex(cols).to_numpy(dtype=float)
        mean = cph._norm_mean.reindex(cols).to_numpy(dtype=float)
        Z = df_fit[cols].to_numpy(dtype=float)
        ph = np.exp((Z - mean) @ beta)
        pred = 1.0 - np.power(s0, ph)
        observed = float(
            ((df_fit["event"].to_numpy() > 0)
             & (df_fit["duration"].to_numpy(dtype=float) <= h)).mean()
        )
        return observed, float(np.nanmean(pred)), float(1.0 - s0)

    # ---- prediction ------------------------------------------------------

    def _xdf(self, x: dict[str, float] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(x, dict):
            df = pd.DataFrame([x])
        elif isinstance(x, pd.Series):
            df = pd.DataFrame([x.to_dict()])
        else:
            df = x.copy()
        # Impute missing / NaN with the persisted training medians (fallback 0.0)
        # so prediction matches the imputation used at fit time.
        for c in self.covariates:
            fill_val = float(self.medians.get(c, 0.0))
            if c not in df.columns:
                df[c] = fill_val
            df[c] = df[c].astype(float).fillna(fill_val)
        return df[self.covariates]

    def _x_array(self, x: dict[str, float] | pd.Series | pd.DataFrame) -> np.ndarray:
        """Covariate matrix for the numpy survival fast path.

        For ``dict``/``Series`` inputs (the simulation hot path), the covariate
        row is assembled directly without constructing a one-row pandas
        DataFrame, which dominates the per-call cost of ``_xdf`` (a pandas
        round-trip of ~2 ms versus a few microseconds for the numpy math).
        The imputation matches ``_xdf`` exactly so the result is bit-identical
        to ``_xdf(x).to_numpy(dtype=float)`` (asserted in the test suite):
        missing keys and NaN values map to the persisted training median, and
        non-finite ``inf`` passes through unchanged (``_xdf`` only ``fillna``s
        NaN, not inf). DataFrame and other inputs fall back to ``_xdf``.
        """
        if isinstance(x, (dict, pd.Series)):
            row = np.empty(len(self.covariates), dtype=float)
            for i, c in enumerate(self.covariates):
                med = float(self.medians.get(c, 0.0))
                v = float(x.get(c, med))
                row[i] = med if v != v else v  # NaN -> median; inf passes through
            return row.reshape(1, -1)
        return self._xdf(x).to_numpy(dtype=float)

    def _ensure_fast_path(self) -> bool:
        """Build (once) the numpy survival fast path from the fitted model.

        Replicates lifelines' centered parameterisation
        ``S(t|x) = S0(t) ** exp(beta'(x - x_mean))`` with a step-function
        lookup on ``baseline_survival_``. Equivalence with
        ``predict_survival_function`` is asserted in the test suite. Falls
        back to lifelines on any structural mismatch (e.g. future lifelines
        versions renaming ``_norm_mean``).
        """
        if self._fast_beta is not None:
            return True
        if self._fast_path_failed:
            return False
        try:
            beta = self.fitter.params_.reindex(self.covariates).to_numpy(dtype=float)
            mean = self.fitter._norm_mean.reindex(self.covariates).to_numpy(dtype=float)
            bs = self.fitter.baseline_survival_
            bs_times = bs.index.to_numpy(dtype=float)
            bs_vals = bs.iloc[:, 0].to_numpy(dtype=float)
            if (np.isnan(beta).any() or np.isnan(mean).any()
                    or np.isnan(bs_vals).any() or len(bs_times) == 0):
                raise ValueError("fast-path arrays contain NaN/empty")
        except Exception as exc:
            log.warning(
                "Cox fast path unavailable (tier %d); using lifelines predict: %s",
                self.tier, exc,
            )
            self._fast_path_failed = True
            return False
        self._fast_beta = beta
        self._fast_mean = mean
        self._fast_bs_times = bs_times
        self._fast_bs_vals = bs_vals
        # Standardization arrays aligned to ``covariates`` (identity for legacy
        # unscaled models, so they keep predicting on the raw covariate scale).
        self._fast_scale_mean = np.array(
            [float(self.scale_mean.get(c, 0.0)) for c in self.covariates], dtype=float
        )
        self._fast_scale_std = np.array(
            [float(self.scale_std.get(c, 1.0)) for c in self.covariates], dtype=float
        )
        self._fast_winsor_z = self.winsor_z
        return True

    def survival(self, horizon_seconds: float, x) -> float | np.ndarray:
        """S(h|X) -- Ueberlebenswahrscheinlichkeit (noch nicht gefillt)."""
        if self.fitter is None:
            raise RuntimeError("Cox-PH nicht gefittet")
        if self._ensure_fast_path():
            # Hot path: assemble the (raw) covariate row directly without a
            # per-call pandas DataFrame, standardize it with the stored scaler,
            # then evaluate S(h|x) = s0 ** exp(beta'(z - mean)) with z the
            # standardized row (mean/std are identity for legacy models).
            X = self._x_array(x)
            Z = (X - self._fast_scale_mean) / self._fast_scale_std
            if self._fast_winsor_z is not None:
                Z = np.clip(Z, -self._fast_winsor_z, self._fast_winsor_z)
            idx = int(np.searchsorted(self._fast_bs_times, horizon_seconds, side="right")) - 1
            s0 = float(self._fast_bs_vals[idx]) if idx >= 0 else 1.0
            partial_hazard = np.exp((Z - self._fast_mean) @ self._fast_beta)
            values = np.power(s0, partial_hazard)
            return float(values[0]) if len(values) == 1 else values
        # Fallback (no fast path): lifelines needs a DataFrame and the times
        # grid. The fitter was trained on standardized covariates, so apply the
        # same scaler before predicting.
        xz = self._xdf(x).copy()
        for c in self.covariates:
            sm = float(self.scale_mean.get(c, 0.0))
            ss = float(self.scale_std.get(c, 1.0))
            xz[c] = (xz[c] - sm) / ss
        if self.winsor_z is not None:
            xz[self.covariates] = xz[self.covariates].clip(-self.winsor_z, self.winsor_z)
        sf = self.fitter.predict_survival_function(xz, times=[horizon_seconds])
        # lifelines liefert einen DataFrame mit Zeit im Index, Spalten = rows
        values = sf.iloc[0].to_numpy()
        return float(values[0]) if len(values) == 1 else values

    def fill_probability(self, horizon_seconds: float, x) -> float | np.ndarray:
        """F(h|X) = 1 - S(h|X)."""
        s = self.survival(horizon_seconds, x)
        return 1.0 - s if np.isscalar(s) else 1.0 - np.asarray(s)

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({
            "tier": self.tier,
            "covariates": self.covariates,
            "fitter": self.fitter,
            "medians": self.medians,
            "ph_test": self.ph_test,
            "scale_mean": self.scale_mean,
            "scale_std": self.scale_std,
            "winsor_z": self.winsor_z,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "CoxFillModel":
        import joblib
        payload = joblib.load(path)
        return cls(
            tier=payload["tier"],
            covariates=payload["covariates"],
            fitter=payload["fitter"],
            medians=payload.get("medians", {}),
            ph_test=payload.get("ph_test", {}),
            scale_mean=payload.get("scale_mean", {}),
            scale_std=payload.get("scale_std", {}),
            winsor_z=payload.get("winsor_z", None),
        )


# ---------------------------------------------------------------------------
# Multi-Tier Container
# ---------------------------------------------------------------------------

@dataclass
class TieredFillModel:
    """Container: tier -> CoxFillModel. Dispatch per Symbol-Tier-Mapping."""

    models: dict[int, CoxFillModel] = field(default_factory=dict)
    symbol_to_tier: dict[str, int] = field(default_factory=dict)

    def fit_panel(self, panel: pd.DataFrame, symbol_tier_map: pd.DataFrame) -> "TieredFillModel":
        """Schaetzt je Tier ein Modell.

        ``symbol_tier_map`` braucht Spalten ``symbol`` und ``tier``.
        ``panel`` braucht Spalte ``symbol`` plus State-Vector + duration/event.
        """
        self.symbol_to_tier = expand_symbol_to_tier(
            dict(zip(symbol_tier_map["symbol"], symbol_tier_map["tier"].astype(int)))
        )
        panel = panel.copy()
        panel["tier"] = panel["symbol"].map(self.symbol_to_tier)
        panel = panel.dropna(subset=["tier"])
        panel["tier"] = panel["tier"].astype(int)

        for tier, grp in panel.groupby("tier"):
            try:
                m = CoxFillModel(tier=int(tier)).fit(grp)
                self.models[int(tier)] = m
            except ValueError as e:
                log.error("Tier %d skipped: %s", tier, e)
        return self

    def for_symbol(self, symbol: str) -> CoxFillModel:
        tier = self.symbol_to_tier.get(symbol)
        if tier is None or tier not in self.models:
            raise KeyError(f"No fill-model available for symbol {symbol}")
        return self.models[tier]

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        for tier, m in self.models.items():
            m.save(dirpath / f"cox_tier_{tier}.pkl")
        pd.DataFrame(list(self.symbol_to_tier.items()), columns=["symbol", "tier"]).to_csv(
            dirpath / "symbol_tier_map.csv", index=False
        )

    @classmethod
    def load(cls, dirpath: Path) -> "TieredFillModel":
        models = {}
        for pkl in sorted(dirpath.glob("cox_tier_*.pkl")):
            m = CoxFillModel.load(pkl)
            models[m.tier] = m
        mapping = pd.read_csv(dirpath / "symbol_tier_map.csv")
        mapping_dict = expand_symbol_to_tier(
            dict(zip(mapping["symbol"], mapping["tier"].astype(int)))
        )
        return cls(models=models, symbol_to_tier=mapping_dict)
