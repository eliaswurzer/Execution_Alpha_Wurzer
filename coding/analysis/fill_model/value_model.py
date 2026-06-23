"""Value-aware XGBoost models for passive execution decisions.

The existing survival models estimate fill probabilities. This module adds a
separate research layer that predicts realized net execution alpha versus MOC
for a candidate passive posting decision. It is intentionally optional so the
headline S0-S4 pipeline remains comparable to earlier runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..metrics.alpha import (
    execution_alpha_bps,
    impact_bps,
    net_execution_alpha_bps,
)
from .state_vector import STATE_COLUMNS

log = logging.getLogger(__name__)

VALUE_TARGET_COLUMN = "target_net_alpha_vs_moc_bps"
VALUE_MODEL_MANIFEST = "value_model_manifest.json"
VALUE_MODEL_GLOBAL_KEY = "global"
VALUE_MODEL_FEATURE_BASE = list(STATE_COLUMNS) + [
    "side_buy",
    "tier",
    "size_frac",
    "time_to_cutoff_seconds",
]
VALUE_MODEL_CATEGORICAL_COLUMNS = ["sector", "listing_exchange"]


def validate_value_model_manifest(
    manifest: dict,
    dirpath: Path | None = None,
    *,
    require_files: bool = True,
) -> list[str]:
    """Return validation errors for an S5 value-model manifest.

    The check is deliberately lightweight and does not deserialize XGBoost
    objects. It is used by dry-run validation and by the strict loader before
    any model file is opened.
    """
    errors: list[str] = []
    policy = manifest.get("policy")
    if policy != cfg.VALUE_MODEL_POLICY_VERSION:
        errors.append(
            f"policy mismatch: got {policy!r}, expected {cfg.VALUE_MODEL_POLICY_VERSION!r}"
        )
    target = manifest.get("target_column")
    if target != VALUE_TARGET_COLUMN:
        errors.append(
            f"target_column mismatch: got {target!r}, expected {VALUE_TARGET_COLUMN!r}"
        )
    model_type = manifest.get("model_type")
    if model_type not in {None, "SideTieredXGBValueModel"}:
        errors.append(f"unsupported model_type: {model_type!r}")
    keys = manifest.get("model_keys")
    if not isinstance(keys, list) or not keys:
        errors.append("model_keys must be a non-empty list")
        keys = []
    models = manifest.get("models")
    if isinstance(models, dict):
        for key, meta in models.items():
            if isinstance(meta, dict) and meta.get("policy") != cfg.VALUE_MODEL_POLICY_VERSION:
                errors.append(
                    f"model {key!r} policy mismatch: got {meta.get('policy')!r}, "
                    f"expected {cfg.VALUE_MODEL_POLICY_VERSION!r}"
                )
    if require_files and dirpath is not None:
        root = Path(dirpath)
        for key in keys:
            for suffix in ("ubj", "pkl"):
                path = root / f"xgb_value_{key}.{suffix}"
                if not path.exists():
                    errors.append(f"missing value-model file: {path.name}")
    return errors


def _clean_side(side: str) -> str:
    s = str(side).strip().upper()
    if s not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported side {side!r}")
    return s


def realized_candidate_value_bps(
    *,
    side: str,
    close_price: float,
    passive_price: float,
    filled: bool,
    adverse_selection_bps: float = 0.0,
    size_frac: float = cfg.PARENT_ORDER_PRIMARY_FRACTION,
    maker_rebate_bps: float = cfg.MAKER_REBATE_BPS,
    commission_bps: float = cfg.COMMISSION_BPS,
) -> float:
    """Return candidate net alpha versus pure MOC in basis points.

    A candidate represents an attempted passive slice. If it fills, the label
    is the close-relative passive price improvement plus the maker rebate
    under the same implementation-shortfall convention as the panel metrics;
    realized post-fill drift (including adverse selection) is already priced
    by the close-relative gross term. If it does not fill, the residual routes
    to MOC and the candidate value is the passive-strategy impact penalty, if
    active, relative to pure MOC. The ``adverse_selection_bps`` argument is
    retained for label diagnostics but no longer enters the value.
    """
    side = _clean_side(side)
    fill_rate = 1.0 if bool(filled) else 0.0
    vwap = float(passive_price) if fill_rate > 0 else float("nan")
    gross = execution_alpha_bps(float(close_price), vwap, fill_rate, side)
    passive_net = net_execution_alpha_bps(
        gross,
        fill_rate,
        maker_rebate_bps=maker_rebate_bps,
        commission_bps=commission_bps,
        impact_component_bps=impact_bps(float(size_frac)),
    )
    moc_net = net_execution_alpha_bps(
        0.0,
        0.0,
        maker_rebate_bps=maker_rebate_bps,
        commission_bps=commission_bps,
        impact_component_bps=0.0,
    )
    return float(passive_net - moc_net)


def candidate_value_label(row: pd.Series | dict) -> float:
    """Row-oriented wrapper for candidate panels."""
    data = row if isinstance(row, dict) else row.to_dict()
    return realized_candidate_value_bps(
        side=data["side"],
        close_price=float(data["close_price"]),
        passive_price=float(data["limit_price"]),
        filled=bool(data.get("event", data.get("filled", False))),
        adverse_selection_bps=float(data.get("adverse_selection_bps", 0.0) or 0.0),
        size_frac=float(data.get("size_frac", cfg.PARENT_ORDER_PRIMARY_FRACTION) or 0.0),
    )


def attach_value_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the value target used by the XGB execution-value model."""
    if frame.empty:
        out = frame.copy()
        out[VALUE_TARGET_COLUMN] = pd.Series(dtype=float)
        return out
    required = {"side", "close_price", "limit_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate panel missing columns: {sorted(missing)}")
    out = frame.copy()
    out[VALUE_TARGET_COLUMN] = [candidate_value_label(row) for _, row in out.iterrows()]
    return out


def _normalise_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    medians: dict[str, float] | None = None,
    categorical_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, list[str]]]:
    df = frame.copy()
    if "side_buy" not in df.columns:
        if "side" not in df.columns:
            raise ValueError("Value-model features require side or side_buy")
        df["side_buy"] = (df["side"].astype(str).str.upper() == "BUY").astype(float)
    if "tier" not in df.columns:
        df["tier"] = 0
    if "size_frac" not in df.columns:
        df["size_frac"] = cfg.PARENT_ORDER_PRIMARY_FRACTION
    if "time_to_cutoff_seconds" not in df.columns:
        df["time_to_cutoff_seconds"] = 0.0

    levels = categorical_levels or {}
    if categorical_levels is None:
        levels = {}
        for col in VALUE_MODEL_CATEGORICAL_COLUMNS:
            if col in df.columns:
                vals = sorted(str(x) for x in df[col].fillna("__MISSING__").unique())
            else:
                vals = ["__MISSING__"]
            levels[col] = vals

    pieces: list[pd.DataFrame] = []
    base_cols = list(VALUE_MODEL_FEATURE_BASE)
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0.0
    pieces.append(df[base_cols].copy())
    for col in VALUE_MODEL_CATEGORICAL_COLUMNS:
        if col in df.columns:
            vals = df[col].fillna("__MISSING__").astype(str)
        else:
            vals = pd.Series("__MISSING__", index=df.index)
        for level in levels.get(col, ["__MISSING__"]):
            safe = level.replace(" ", "_").replace("/", "_").replace("-", "_")
            pieces.append(pd.DataFrame({f"{col}__{safe}": (vals == level).astype(float)}))
    X = pd.concat(pieces, axis=1)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
    else:
        feature_columns = list(X.columns)

    computed_medians: dict[str, float] = {}
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if medians is None:
            median = float(X[col].median())
            if not np.isfinite(median):
                median = 0.0
            computed_medians[col] = median
        else:
            computed_medians[col] = float(medians.get(col, 0.0))
        X[col] = X[col].fillna(computed_medians[col])
    return X, feature_columns, computed_medians, levels


@dataclass
class XGBValueModel:
    """XGBoost regressor for expected net alpha versus MOC."""

    key: str = VALUE_MODEL_GLOBAL_KEY
    target_column: str = VALUE_TARGET_COLUMN
    feature_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _booster: object | None = field(default=None, repr=False)

    def fit(
        self,
        panel: pd.DataFrame,
        *,
        n_estimators: int = 120,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        min_child_weight: int = 5,
        min_rows: int = cfg.VALUE_MODEL_MIN_ROWS_GLOBAL,
        random_state: int = cfg.DEFAULT_SEED,
        xgb_device: str = "cpu",
    ) -> "XGBValueModel":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost required for XGBValueModel") from exc
        if self.target_column not in panel.columns:
            raise ValueError(f"Panel missing target column {self.target_column!r}")
        df = panel.copy()
        y = pd.to_numeric(df[self.target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = y.notna()
        df = df.loc[valid].reset_index(drop=True)
        y = y.loc[valid].to_numpy(dtype=np.float32)
        if len(df) < int(min_rows):
            raise ValueError(f"{self.key}: insufficient rows for value model: {len(df)}")
        if float(np.nanstd(y)) < cfg.VALUE_MODEL_MIN_TARGET_STD_BPS:
            raise ValueError(f"{self.key}: target has near-zero variance")
        X, cols, med, levels = _normalise_feature_frame(df)
        arr = X.to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all() or not np.isfinite(y).all():
            raise ValueError(f"{self.key}: non-finite training data")
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": int(max_depth),
            "eta": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "min_child_weight": int(min_child_weight),
            "tree_method": "hist",
            "device": str(xgb_device or "cpu"),
            "seed": int(random_state),
            "nthread": 1,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(arr, label=y, feature_names=cols)
        self._booster = xgb.train(params, dtrain, num_boost_round=int(n_estimators), verbose_eval=False)
        self.feature_columns = cols
        self.medians = med
        self.categorical_levels = levels
        self.metadata = {
            "key": self.key,
            "rows": int(len(df)),
            "target_mean_bps": float(np.mean(y)),
            "target_std_bps": float(np.std(y)),
            "policy": cfg.VALUE_MODEL_POLICY_VERSION,
        }
        log.info(
            "Fitted XGB value model %s: n=%d mean=%.4f std=%.4f",
            self.key,
            len(df),
            self.metadata["target_mean_bps"],
            self.metadata["target_std_bps"],
        )
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError(f"XGBValueModel {self.key!r} is not fitted")
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost required for XGBValueModel") from exc
        X, _, _, _ = _normalise_feature_frame(
            frame,
            feature_columns=self.feature_columns,
            medians=self.medians,
            categorical_levels=self.categorical_levels,
        )
        dm = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=self.feature_columns)
        return self._booster.predict(dm).astype(float)

    def save(self, dirpath: Path) -> None:
        if self._booster is None:
            raise RuntimeError("Cannot save an unfitted XGBValueModel")
        dirpath.mkdir(parents=True, exist_ok=True)
        import joblib
        self._booster.save_model(str(dirpath / f"xgb_value_{self.key}.ubj"))
        joblib.dump({
            "key": self.key,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "medians": self.medians,
            "categorical_levels": self.categorical_levels,
            "metadata": self.metadata,
        }, dirpath / f"xgb_value_{self.key}.pkl")

    @classmethod
    def load(cls, dirpath: Path, key: str) -> "XGBValueModel":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost required for XGBValueModel") from exc
        import joblib
        dirpath = Path(dirpath)
        meta = joblib.load(dirpath / f"xgb_value_{key}.pkl")
        booster = xgb.Booster()
        booster.load_model(str(dirpath / f"xgb_value_{key}.ubj"))
        obj = cls(key=key, target_column=meta.get("target_column", VALUE_TARGET_COLUMN))
        obj.feature_columns = list(meta["feature_columns"])
        obj.medians = dict(meta.get("medians", {}))
        obj.categorical_levels = dict(meta.get("categorical_levels", {}))
        obj.metadata = dict(meta.get("metadata", {}))
        obj._booster = booster
        return obj


def _model_key(side: str | None = None, tier: int | None = None) -> str:
    if side is None:
        return VALUE_MODEL_GLOBAL_KEY
    side = _clean_side(side)
    if tier is None:
        return f"side_{side}"
    return f"side_{side}_tier_{int(tier)}"


@dataclass
class SideTieredXGBValueModel:
    """Hierarchical value model with side-tier, side, and global fallback."""

    models: dict[str, XGBValueModel] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    def fit_panel(
        self,
        panel: pd.DataFrame,
        *,
        min_rows_global: int = cfg.VALUE_MODEL_MIN_ROWS_GLOBAL,
        min_rows_side: int = cfg.VALUE_MODEL_MIN_ROWS_SIDE,
        min_rows_side_tier: int = cfg.VALUE_MODEL_MIN_ROWS_SIDE_TIER,
        **fit_kwargs,
    ) -> "SideTieredXGBValueModel":
        if panel.empty:
            raise ValueError("Cannot fit value model on an empty panel")
        df = panel.copy()
        if "side" not in df.columns or "tier" not in df.columns:
            raise ValueError("Value model panel requires side and tier columns")
        df["side"] = df["side"].astype(str).str.upper()
        df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(0).astype(int)

        fitted: dict[str, dict] = {}
        global_model = XGBValueModel(key=VALUE_MODEL_GLOBAL_KEY).fit(
            df, min_rows=min_rows_global, **fit_kwargs,
        )
        self.models[VALUE_MODEL_GLOBAL_KEY] = global_model
        fitted[VALUE_MODEL_GLOBAL_KEY] = global_model.metadata

        for side, grp in df.groupby("side"):
            key = _model_key(side)
            if len(grp) >= min_rows_side:
                try:
                    model = XGBValueModel(key=key).fit(grp, min_rows=min_rows_side, **fit_kwargs)
                    self.models[key] = model
                    fitted[key] = model.metadata
                except ValueError as exc:
                    log.warning("Skipping side value model %s: %s", key, exc)
            for tier, tgrp in grp.groupby("tier"):
                tkey = _model_key(side, int(tier))
                if len(tgrp) < min_rows_side_tier:
                    continue
                try:
                    model = XGBValueModel(key=tkey).fit(tgrp, min_rows=min_rows_side_tier, **fit_kwargs)
                    self.models[tkey] = model
                    fitted[tkey] = model.metadata
                except ValueError as exc:
                    log.warning("Skipping side-tier value model %s: %s", tkey, exc)

        self.manifest = {
            "policy": cfg.VALUE_MODEL_POLICY_VERSION,
            "model_type": "SideTieredXGBValueModel",
            "target_column": VALUE_TARGET_COLUMN,
            "model_keys": sorted(self.models),
            "models": fitted,
        }
        return self

    def _resolve_key(self, side: str, tier: int | None) -> str:
        chain = []
        if tier is not None:
            chain.append(_model_key(side, int(tier)))
        chain.append(_model_key(side))
        chain.append(VALUE_MODEL_GLOBAL_KEY)
        for key in chain:
            if key in self.models:
                return key
        raise KeyError("No value model is available")

    def predict_candidates(self, candidates: pd.DataFrame) -> np.ndarray:
        if candidates.empty:
            return np.array([], dtype=float)
        if "side" not in candidates.columns:
            raise ValueError("Value-model candidates require a side column")
        df = candidates.copy().reset_index(drop=True)
        if "tier" not in df.columns:
            df["tier"] = pd.NA
        out = np.empty(len(df), dtype=float)
        groups = df.groupby([df["side"].astype(str).str.upper(), df["tier"]], dropna=False).groups
        for (side, tier), idx in groups.items():
            tier_value = None if pd.isna(tier) else int(tier)
            key = self._resolve_key(str(side), tier_value)
            positions = list(idx)
            out[positions] = self.models[key].predict_frame(df.iloc[positions])
        return out

    def save(self, dirpath: Path) -> None:
        if not self.models:
            raise RuntimeError("Cannot save an empty SideTieredXGBValueModel")
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        for model in self.models.values():
            model.save(dirpath)
        manifest = dict(self.manifest)
        manifest["policy"] = cfg.VALUE_MODEL_POLICY_VERSION
        manifest["target_column"] = VALUE_TARGET_COLUMN
        manifest["model_keys"] = sorted(self.models)
        (dirpath / VALUE_MODEL_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        dirpath: Path,
        *,
        require_current_policy: bool = True,
    ) -> "SideTieredXGBValueModel":
        dirpath = Path(dirpath)
        manifest_path = dirpath / VALUE_MODEL_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing value model manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if require_current_policy:
            errors = validate_value_model_manifest(manifest, dirpath)
            if errors:
                raise RuntimeError(
                    f"Invalid or stale value model manifest {manifest_path}: "
                    + "; ".join(errors)
                )
        models = {
            key: XGBValueModel.load(dirpath, key)
            for key in manifest.get("model_keys", [])
        }
        if not models:
            raise FileNotFoundError(f"No value model artifacts listed in {manifest_path}")
        return cls(models=models, manifest=manifest)
