"""Out-of-sample fill-model calibration diagnostics."""

from __future__ import annotations

import concurrent.futures as futures
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config as cfg
from analysis.fill_model.cox_ph import TieredFillModel
from analysis.fill_model.kaplan_meier import TieredKMFillModel
from analysis.fill_model.xgb_survival import TieredXGBFillModel
from analysis.runners.calibrate_fill_model import (
    _canonical_symbol,
    _clean_event_panel_for_fit,
    _process_symbol_day,
    _read_shards,
    _safe_symbol,
)

from . import config as st_cfg


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_many(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_fingerprint(
    *,
    headline_run: Path,
    h1_panel_sha256: str,
    days_per_quarter: int,
    event_sample_per_symbol_day: int | None,
) -> str:
    payload = {
        "policy": st_cfg.OOS_CACHE_POLICY_VERSION,
        "headline_run": str(Path(headline_run)),
        "h1_panel_sha256": h1_panel_sha256,
        "days_per_quarter": int(days_per_quarter),
        "event_sample_per_symbol_day": (
            None if event_sample_per_symbol_day is None
            else int(event_sample_per_symbol_day)
        ),
        "feature_policy": cfg.FEATURE_POLICY_VERSION,
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "fill_model_horizon_seconds": int(cfg.FILL_MODEL_HORIZON_SECONDS),
        "refresh_seconds": int(cfg.REFRESH_SECONDS_DEFAULT),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_oos_manifest(
    out_dir: Path,
    *,
    fingerprint: str,
    headline_run: Path,
    h1_panel_sha256: str,
    selected_dates: list[dt.date],
    status_path: Path,
    status: pd.DataFrame,
    event_paths: list[Path],
    days_per_quarter: int,
    event_sample_per_symbol_day: int | None,
    expected_pairs: int,
) -> None:
    payload = {
        "policy": st_cfg.OOS_CACHE_POLICY_VERSION,
        "fingerprint": fingerprint,
        "headline_run": str(Path(headline_run)),
        "h1_panel_sha256": h1_panel_sha256,
        "feature_policy": cfg.FEATURE_POLICY_VERSION,
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "days_per_quarter": int(days_per_quarter),
        "event_sample_per_symbol_day": (
            None if event_sample_per_symbol_day is None
            else int(event_sample_per_symbol_day)
        ),
        "selected_dates": [d.isoformat() for d in selected_dates],
        "expected_pairs": int(expected_pairs),
        "status_sha256": _sha256_file(status_path) if status_path.exists() else None,
        "status_rows": int(len(status)),
        "ok_rows": int((status.get("status", pd.Series(dtype=str)) == "ok").sum()),
        "event_shards": int(len(event_paths)),
        "event_shards_sha256": _sha256_many(event_paths) if event_paths else None,
    }
    (Path(out_dir) / "oos_event_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def select_stratified_oos_dates(
    dates: pd.Series | list[str] | list[dt.date],
    *,
    days_per_quarter: int = st_cfg.OOS_DAYS_PER_QUARTER,
) -> list[dt.date]:
    """Select evenly spaced evaluation dates within each calendar quarter."""
    if days_per_quarter <= 0:
        return []
    ser = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    if ser.empty:
        return []
    frame = pd.DataFrame({"date": sorted(ser.dt.date.unique())})
    frame["period"] = pd.PeriodIndex(pd.to_datetime(frame["date"]), freq="Q")
    selected: list[dt.date] = []
    for _, grp in frame.groupby("period", sort=True):
        vals = list(grp["date"])
        if len(vals) <= days_per_quarter:
            selected.extend(vals)
            continue
        pos = np.linspace(0, len(vals) - 1, days_per_quarter)
        idx = sorted({int(round(x)) for x in pos})
        # Rounding can collide for small groups; fill from the left if needed.
        cursor = 0
        while len(idx) < days_per_quarter and cursor < len(vals):
            if cursor not in idx:
                idx.append(cursor)
            cursor += 1
        selected.extend(vals[i] for i in sorted(idx[:days_per_quarter]))
    return selected


def _symbol_date_pairs(headline_panel: pd.DataFrame, dates: list[dt.date]) -> list[tuple[dt.date, str]]:
    panel = headline_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.date
    panel = panel[panel["date"].isin(set(dates))]
    if "strategy" in panel.columns:
        panel = panel[panel["strategy"] == "S0_MOC"]
    pairs = (
        panel[["date", "symbol"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["date", "symbol"])
    )
    return [(d, str(s)) for d, s in pairs.itertuples(index=False, name=None)]


def _existing_oos_status_row(shard_root: Path, d: dt.date, sym: str) -> dict | None:
    sym = _canonical_symbol(sym)
    ds = d.strftime("%Y%m%d")
    safe = _safe_symbol(sym)
    event_path = shard_root / "events" / ds / f"{safe}.parquet"
    if not event_path.exists():
        return None
    daily_path = shard_root / "daily" / ds / f"{safe}.parquet"
    as_path = shard_root / "as" / ds / f"{safe}.parquet"
    return {
        "date": d,
        "symbol": sym,
        "status": "ok",
        "reason": "existing_shard",
        "event_path": str(event_path),
        "daily_path": str(daily_path) if daily_path.exists() else "",
        "as_path": str(as_path) if as_path.exists() else "",
    }


def _checkpoint_status(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def build_oos_event_panel(
    *,
    headline_run: Path = st_cfg.HEADLINE_RUN,
    out_dir: Path = st_cfg.OUTPUT_DIR,
    days_per_quarter: int = st_cfg.OOS_DAYS_PER_QUARTER,
    workers: int = 1,
    event_sample_per_symbol_day: int | None = st_cfg.OOS_EVENT_SAMPLE_PER_SYMBOL_DAY,
    force: bool = False,
) -> tuple[pd.DataFrame, list[dt.date], pd.DataFrame]:
    """Build or reuse a stratified OOS event panel from evaluation dates."""
    out_dir = Path(out_dir)
    shard_root = out_dir / "oos_event_shards"
    status_path = out_dir / "oos_event_status.csv"
    manifest_path = out_dir / "oos_event_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    h1_path = Path(headline_run) / "hypotheses" / "h1" / "h1_panel.parquet"
    h1_sha = _sha256_file(h1_path)
    panel = pd.read_parquet(h1_path, columns=["date", "symbol", "strategy"])
    selected_dates = select_stratified_oos_dates(panel["date"], days_per_quarter=days_per_quarter)
    pairs = _symbol_date_pairs(panel, selected_dates)
    fingerprint = _cache_fingerprint(
        headline_run=headline_run,
        h1_panel_sha256=h1_sha,
        days_per_quarter=days_per_quarter,
        event_sample_per_symbol_day=event_sample_per_symbol_day,
    )

    reuse_status = False
    if status_path.exists() and manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reuse_status = (
                manifest.get("policy") == st_cfg.OOS_CACHE_POLICY_VERSION
                and manifest.get("fingerprint") == fingerprint
                and int(manifest.get("status_rows", -1)) >= len(pairs)
                and int(manifest.get("expected_pairs", len(pairs))) == len(pairs)
            )
        except (OSError, ValueError):
            reuse_status = False

    if reuse_status:
        status = pd.read_csv(status_path)
    else:
        rows: list[dict] = []
        pending: list[tuple[dt.date, str]] = []
        if force:
            pending = pairs
        else:
            for d, sym in pairs:
                existing = _existing_oos_status_row(shard_root, d, sym)
                if existing is None:
                    pending.append((d, sym))
                else:
                    rows.append(existing)
            _checkpoint_status(status_path, rows)
        if workers and workers > 1:
            with futures.ProcessPoolExecutor(max_workers=int(workers)) as pool:
                futs = [
                    pool.submit(
                        _process_symbol_day,
                        d,
                        sym,
                        shard_root,
                        event_sample_per_symbol_day,
                        None,
                        cfg.REFRESH_SECONDS_DEFAULT,
                        60,
                    )
                    for d, sym in pending
                ]
                for i, fut in enumerate(futures.as_completed(futs), start=1):
                    rows.append(fut.result())
                    if i % 500 == 0:
                        _checkpoint_status(status_path, rows)
        else:
            for i, (d, sym) in enumerate(pending, start=1):
                rows.append(
                    _process_symbol_day(
                        d,
                        sym,
                        shard_root,
                        event_sample_per_symbol_day,
                        None,
                        cfg.REFRESH_SECONDS_DEFAULT,
                        60,
                    )
                )
                if i % 500 == 0:
                    _checkpoint_status(status_path, rows)
        status = pd.DataFrame(rows)
        status.to_csv(status_path, index=False)

    ok = status[status.get("status", "") == "ok"].copy()
    event_paths = [
        Path(str(p))
        for p in ok.get("event_path", pd.Series(dtype=str)).dropna()
        if Path(str(p)).exists()
    ]
    _write_oos_manifest(
        out_dir,
        fingerprint=fingerprint,
        headline_run=headline_run,
        h1_panel_sha256=h1_sha,
        selected_dates=selected_dates,
        status_path=status_path,
        status=status,
        event_paths=event_paths,
        days_per_quarter=days_per_quarter,
        event_sample_per_symbol_day=event_sample_per_symbol_day,
        expected_pairs=len(pairs),
    )
    if not event_paths:
        return pd.DataFrame(), selected_dates, status
    events = _read_shards([str(path) for path in event_paths])
    events = _clean_event_panel_for_fit(events)
    return events, selected_dates, status


def _brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bs = float(np.mean((y_prob - y_true) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    base = float(np.mean(y_true))
    unc = base * (1.0 - base)
    rel = 0.0
    res = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        weight = float(mask.mean())
        pred = float(np.mean(y_prob[mask]))
        obs = float(np.mean(y_true[mask]))
        rel += weight * (pred - obs) ** 2
        res += weight * (obs - base) ** 2
    return bs, rel, res, unc


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and s[order[end + 1]] == s[order[pos]]:
            end += 1
        mean_rank = 0.5 * (pos + end) + 1.0
        ranks[order[pos:end + 1]] = mean_rank
        pos = end + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    sum_ranks_pos = float(ranks[y == 1].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_models(
    fill_model_dir: Path = st_cfg.FILL_MODEL_DIR,
    *,
    model_specs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    fill_model_dir = Path(fill_model_dir)
    allowed = set(st_cfg.MODEL_SPECS if model_specs is None else model_specs)
    models: dict[str, object] = {}
    if "cox" in allowed and any(fill_model_dir.glob("cox_tier_*.pkl")):
        models["cox"] = TieredFillModel.load(fill_model_dir)
    if "km" in allowed and any(fill_model_dir.glob("km_tier_*.pkl")):
        models["km"] = TieredKMFillModel.load(fill_model_dir)
    if "xgb" in allowed and any(fill_model_dir.glob("xgb_tier_*.ubj")):
        models["xgb"] = TieredXGBFillModel.load(fill_model_dir)
    return models


def _predict_for_tier(model_name: str, tier_model, horizon_seconds: int, grp: pd.DataFrame) -> np.ndarray:
    return np.asarray(tier_model.fill_probability(horizon_seconds, grp), dtype=float)


def score_event_panel(
    event_panel: pd.DataFrame,
    models: dict[str, object] | None = None,
    *,
    fill_model_dir: Path = st_cfg.FILL_MODEL_DIR,
    horizon_seconds: int = cfg.FILL_MODEL_HORIZON_SECONDS,
    model_specs: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Score Cox, KM and XGB on one OOS event panel."""
    if event_panel.empty:
        return pd.DataFrame()
    models = models or load_models(fill_model_dir, model_specs=model_specs)
    rows: list[dict] = []
    for model_name, model in models.items():
        panel = event_panel.copy()
        panel["tier"] = panel["symbol"].map(model.symbol_to_tier)
        panel = panel.dropna(subset=["tier"])
        if panel.empty:
            continue
        panel["tier"] = panel["tier"].astype(int)
        for tier, grp in panel.groupby("tier", sort=True):
            tier = int(tier)
            tier_model = model.models.get(tier)
            if tier_model is None:
                continue
            y_true = (
                pd.to_numeric(grp["event"], errors="coerce").fillna(0).to_numpy(dtype=int)
                & (pd.to_numeric(grp["duration"], errors="coerce").to_numpy(dtype=float) <= horizon_seconds)
            ).astype(int)
            y_prob = _predict_for_tier(model_name, tier_model, horizon_seconds, grp)
            y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
            valid = np.isfinite(y_prob)
            if not valid.any():
                continue
            y_true = y_true[valid]
            y_prob = y_prob[valid]
            brier, rel, res, unc = _brier_decomposition(y_true, y_prob)
            rows.append({
                "model": model_name,
                "tier": tier,
                "n": int(len(y_true)),
                "observed_fill_rate": float(np.mean(y_true)),
                "mean_predicted_probability": float(np.mean(y_prob)),
                "absolute_calibration_error": float(abs(np.mean(y_prob) - np.mean(y_true))),
                "brier": brier,
                "reliability": rel,
                "resolution": res,
                "uncertainty": unc,
                "auc": _auc_roc(y_true, y_prob),
            })
    return pd.DataFrame(rows)
