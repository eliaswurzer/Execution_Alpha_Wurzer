"""Resumable daily-shard simulation for headline hypothesis runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.adv_spread_buckets import load_adv_spread_bucket_map
from ..data.index_universe import build_index_universe_panel
from ..data.taq_loader import (
    TradePolicyMismatchError,
    parquet_available,
)
from ..metrics.alpha import attach_alpha_columns, attach_moc_differential_columns
from ..simulation.engine import simulate_symbol_day
from ..simulation.parent_orders import build_parent_orders, rolling_expected_vc
from ..utils.adaptive_pool import AdaptivePool
from ..utils.symbols import canonical_symbol, expand_symbol_to_tier
from . import _common

log = logging.getLogger(__name__)

SCHEMA_VERSION = "headline_master_panel_v1"
EXPECTED_VC_POLICY_VERSION = "expected_vc_identity_repair_v1"
EXPECTED_VC_IDENTITY_MAP = (
    Path(__file__).resolve().parents[3]
    / "reference"
    / "index_membership"
    / "expected_vc_identity_map.csv"
)
ADV_SPREAD_BUCKET_COLUMNS = ("adv_bucket", "spread_bucket", "adv_spread_bucket")
TIER_POLICY_CALIBRATED_ONLY = "calibrated_only"
TIER_POLICY_CALIBRATED_PLUS_FALLBACK = "calibrated_plus_fallback"
LIQUIDITY_TIER_POLICY_VERSION = "calibrated_plus_conservative_fallback_v1"
FALLBACK_LIQUIDITY_TIER = 3
TIER_POLICY_CHOICES = {
    TIER_POLICY_CALIBRATED_ONLY,
    TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
}
CRITICAL_REASONS = {"compute_error", "policy_mismatch", "dtype_error"}
REQUIRED_PANEL_COLUMNS = {
    "order_id", "symbol", "date", "strategy", "size_frac",
    "net_alpha_bps", "net_alpha_vs_moc_bps",
}
REQUIRED_METRIC_COLUMNS = {
    "alpha_bps", "net_alpha_bps", "net_alpha_vs_moc_bps",
    "fill_rate", "impact_bps",
}
SIMULATION_SOURCE_DIRS = {
    "data",
    "fill_model",
    "metrics",
    "microstructure",
    "simulation",
    "strategies",
    "utils",
}
SIMULATION_SOURCE_FILES = {
    "config.py",
    "runners/_common.py",
    "runners/master_panel.py",
}
FAILURE_COLUMNS = [
    "date", "symbol", "reason", "detail", "critical",
]
STATUS_COLUMNS = [
    "date", "status", "eligible_symbol_days", "successful_symbol_days",
    "failed_symbol_days", "rows", "runtime_seconds", "resumed",
]


def _json_default(value):
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def _atomic_replace(tmp: Path, path: Path, attempts: int = 6, base_delay: float = 0.2) -> None:
    """``tmp.replace(path)`` with retry on transient Windows lock errors.

    On Windows ``os.replace`` can raise ``PermissionError`` (WinError 5/32) when
    the destination is momentarily held by another handle, an antivirus scan, or
    the search indexer. These are transient, so retry with a short backoff before
    giving up. Avoids spurious critical ``compute_error`` failures that would fail
    the master-panel QC gate on an otherwise healthy run.
    """
    import time
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _atomic_replace(tmp, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp, index=False)
    _atomic_replace(tmp, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    _atomic_replace(tmp, path)


def _status_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.sort_values("date").reset_index(drop=True)


def _failure_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=FAILURE_COLUMNS)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonfinite_metric_counts(
    panel: pd.DataFrame,
    columns: Iterable[str] = REQUIRED_METRIC_COLUMNS,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in columns:
        if col not in panel.columns:
            counts[col] = len(panel)
            continue
        values = pd.to_numeric(panel[col], errors="coerce")
        bad = values.isna() | ~np.isfinite(values)
        n_bad = int(bad.sum())
        if n_bad:
            counts[col] = n_bad
    return counts


def _raise_on_bad_metrics(panel: pd.DataFrame, context: str) -> None:
    bad = _nonfinite_metric_counts(panel)
    if bad:
        raise ValueError(f"{context}: non-finite required metrics: {bad}")


def _identity_map_sha(path: Path | None = None) -> str | None:
    path = path or EXPECTED_VC_IDENTITY_MAP
    return _sha256(path) if path.exists() else None


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_expected_vc_identity_map(
    path: Path | None = None,
) -> pd.DataFrame:
    path = path or EXPECTED_VC_IDENTITY_MAP
    columns = [
        "target_symbol", "source_symbol", "effective_from", "effective_to",
        "mapping_type", "headline_allowed", "scale_factor", "source", "note",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Expected-VC identity map {path} missing columns: {sorted(missing)}"
        )
    frame = frame[columns].copy()
    frame["target_symbol"] = frame["target_symbol"].map(canonical_symbol)
    frame["source_symbol"] = frame["source_symbol"].map(
        lambda value: canonical_symbol(value) if str(value).strip() else ""
    )
    frame["effective_from"] = pd.to_datetime(
        frame["effective_from"], errors="coerce",
    ).dt.date
    frame["effective_to"] = pd.to_datetime(
        frame["effective_to"], errors="coerce",
    ).dt.date
    frame["headline_allowed"] = frame["headline_allowed"].map(_truthy)
    frame["scale_factor"] = pd.to_numeric(
        frame["scale_factor"], errors="coerce",
    ).fillna(1.0).astype(float)
    return frame


def _source_symbols_for_expected_vc(
    identity_map: pd.DataFrame,
    target_symbols: list[str],
) -> list[str]:
    if identity_map.empty:
        return []
    targets = {canonical_symbol(symbol) for symbol in target_symbols}
    allowed = identity_map[
        identity_map["headline_allowed"]
        & identity_map["target_symbol"].isin(targets)
        & identity_map["source_symbol"].astype(bool)
    ]
    return sorted({str(symbol) for symbol in allowed["source_symbol"]})


def _vc_history_for_expected_vc(
    vc_history: pd.DataFrame,
    identity_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if vc_history.empty:
        ambiguous = (
            int((~identity_map["headline_allowed"]).sum())
            if "headline_allowed" in identity_map.columns else 0
        )
        return vc_history.copy(), {
            "predecessor_rows_used": 0,
            "predecessor_rows_skipped_duplicate": 0,
            "same_symbol_preindex_rows_used": 0,
            "ambiguous_rows_left_unrepaired": ambiguous,
        }

    base = vc_history.copy()
    base["symbol"] = base["symbol"].map(canonical_symbol)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    if identity_map.empty:
        return base, {
            "predecessor_rows_used": 0,
            "predecessor_rows_skipped_duplicate": 0,
            "same_symbol_preindex_rows_used": 0,
            "ambiguous_rows_left_unrepaired": 0,
        }

    augmented_parts = [base]
    mapped_parts: list[pd.DataFrame] = []
    for row in identity_map.itertuples(index=False):
        if not bool(row.headline_allowed) or not str(row.source_symbol):
            continue
        if pd.isna(row.effective_from):
            continue
        source = str(row.source_symbol)
        target = str(row.target_symbol)
        source_rows = base[
            (base["symbol"] == source)
            & (pd.to_datetime(base["date"]).dt.date < row.effective_from)
        ].copy()
        if source_rows.empty:
            continue
        source_rows["symbol"] = target
        scale = float(row.scale_factor)
        for col in (
            "vc_shares", "close_trade_volume",
            "official_close_marker_volume",
            "official_close_marker_fallback_volume",
        ):
            if col in source_rows.columns:
                source_rows[col] = pd.to_numeric(
                    source_rows[col], errors="coerce",
                ) * scale
        if "vc_source" in source_rows.columns:
            source_base = source_rows["vc_source"].astype(str)
        else:
            source_base = pd.Series("unknown", index=source_rows.index)
        source_rows["vc_source"] = (
            source_base + f"|expected_vc_identity:{source}->{target}"
        )
        mapped_parts.append(source_rows)
    if mapped_parts:
        augmented_parts.extend(mapped_parts)

    augmented = pd.concat(augmented_parts, ignore_index=True)
    before = len(augmented)
    augmented = augmented.drop_duplicates(["symbol", "date"], keep="first")
    skipped = before - len(augmented)
    mapped_rows = sum(len(part) for part in mapped_parts)
    return augmented, {
        "predecessor_rows_used": int(mapped_rows - skipped),
        "predecessor_rows_skipped_duplicate": int(skipped),
        "same_symbol_preindex_rows_used": 0,
        "ambiguous_rows_left_unrepaired": int((~identity_map["headline_allowed"]).sum()),
    }


def _artifact_signature(artifacts_dir: Path, fill_specification: str) -> dict:
    names = ["symbol_tier_map.csv", "glosten_as.csv"]
    if fill_specification == "xgb":
        names.extend(p.name for p in sorted(artifacts_dir.glob("xgb_tier_*")))
    elif fill_specification == "cox":
        names.extend(p.name for p in sorted(artifacts_dir.glob("cox_tier_*.pkl")))
    elif fill_specification == "km":
        names.extend(p.name for p in sorted(artifacts_dir.glob("km_tier_*.pkl")))
        names.append("km_symbol_tier_map.csv")
    tod = ["tod_schedule_xgb.ubj", "tod_schedule_meta.pkl"]
    names.extend(name for name in tod if (artifacts_dir / name).exists())
    value_artifacts = ["value_model_manifest.json"]
    value_artifacts.extend(p.name for p in sorted(artifacts_dir.glob("xgb_value_*")))
    names.extend(name for name in value_artifacts if (artifacts_dir / name).exists())
    out = {}
    for name in sorted(set(names)):
        path = artifacts_dir / name
        if path.exists():
            out[name] = _sha256(path)
    return out


def _tier_policy_version(tier_policy: str) -> str:
    if tier_policy == TIER_POLICY_CALIBRATED_ONLY:
        return "calibrated_only_v1"
    if tier_policy == TIER_POLICY_CALIBRATED_PLUS_FALLBACK:
        return LIQUIDITY_TIER_POLICY_VERSION
    raise ValueError(
        f"Unsupported tier policy {tier_policy!r}; "
        f"expected one of {sorted(TIER_POLICY_CHOICES)}"
    )


def _canonical_tier_lookup(tier_map: pd.DataFrame) -> dict[str, int]:
    if not {"symbol", "tier"}.issubset(tier_map.columns):
        raise ValueError("symbol_tier_map.csv must contain symbol and tier columns")
    base: dict[str, int] = {}
    for row in tier_map[["symbol", "tier"]].dropna().itertuples(index=False):
        symbol = canonical_symbol(row.symbol)
        if symbol not in base:
            base[symbol] = int(row.tier)
    return expand_symbol_to_tier(base)


def _tier_audit_frame(rows: dict[str, dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "tier", "tier_source", "reason"],
        )
    frame = pd.DataFrame(rows.values())
    return frame.sort_values("symbol").reset_index(drop=True)


def _simulation_source_signature() -> str:
    analysis_root = Path(__file__).resolve().parents[1]
    paths: list[Path] = []
    for rel in SIMULATION_SOURCE_FILES:
        path = analysis_root / rel
        if path.exists():
            paths.append(path)
    for dirname in SIMULATION_SOURCE_DIRS:
        root = analysis_root / dirname
        if root.exists():
            paths.extend(root.rglob("*.py"))
    paths = sorted(set(paths))

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(analysis_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_fingerprint(
    dates: list[dt.date],
    symbols: list[str],
    *,
    identity_map_sha256: str | None = None,
) -> str:
    return _fingerprint({
        "schema": "expected_vc_v2",
        "expected_vc_policy_version": EXPECTED_VC_POLICY_VERSION,
        "identity_map_sha256": identity_map_sha256,
        "dates": [d.isoformat() for d in dates],
        "symbols": symbols,
        "taq_roots": {str(y): str(p) for y, p in cfg.TAQ_PARQUET_DIR.items()},
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "trade_qc_mode": cfg.TRADE_QC_POLICY_CHECK_MODE,
        "lookback_days": 20,
    })


# Output schema of _common._load_vc_one; shards with zero rows keep these
# columns so concatenation and downstream consumers stay schema-stable.
VC_HISTORY_COLUMNS = [
    "symbol", "date", "vc_shares", "vc_source", "close_price_source",
    "close_trade_volume", "close_trade_rows",
    "official_close_marker_volume", "official_close_marker_rows",
    "official_close_marker_fallback_volume",
]


def _vc_shard_fingerprint(symbols: list[str]) -> str:
    return _fingerprint({
        "schema": "vc_history_shard_v1",
        "symbols": symbols,
        "taq_roots": {str(y): str(p) for y, p in cfg.TAQ_PARQUET_DIR.items()},
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "trade_qc_mode": cfg.TRADE_QC_POLICY_CHECK_MODE,
    })


def _load_or_build_vc_shard(
    date: dt.date,
    symbols: list[str],
    shard_dir: Path,
    shard_fingerprint: str,
    *,
    workers: int,
) -> pd.DataFrame:
    name = date.strftime("%Y%m%d")
    shard_path = shard_dir / f"{name}.parquet"
    meta_path = shard_dir / f"{name}.json"
    if shard_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == shard_fingerprint:
                return pd.read_parquet(shard_path)
        except (OSError, ValueError, KeyError):
            pass
    frame = _common._vc_history([date], symbols, workers=workers)
    if frame.empty:
        frame = pd.DataFrame(columns=VC_HISTORY_COLUMNS)
    _write_parquet_atomic(frame, shard_path)
    _write_json_atomic(meta_path, {
        "fingerprint": shard_fingerprint,
        "date": name,
        "rows": len(frame),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    return frame


def _default_vc_shard_dir() -> Path:
    """Shared per-date vc_history shard cache, reused across run ids.

    Shard validity is governed entirely by the per-shard fingerprint
    (symbol set, TAQ roots, trade policy, QC mode), so different runs with
    the same inputs can safely share the expensive raw-tape extraction
    instead of rebuilding ~400 shards per run id.
    """
    return cfg.ARTIFACTS_DIR / "cache" / "vc_history_shards"


def _load_or_build_expected_vc(
    dates: list[dt.date],
    symbols: list[str],
    cache_dir: Path,
    *,
    workers: int,
    resume: bool,
    shard_dir: Path | None = None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vc_path = cache_dir / "vc_history.parquet"
    repaired_vc_path = cache_dir / "vc_history_for_expected_vc.parquet"
    evc_path = cache_dir / "expected_vc.parquet"
    meta_path = cache_dir / "expected_vc_manifest.json"
    identity_map = _load_expected_vc_identity_map()
    identity_map_sha = _identity_map_sha()
    target_symbols = sorted({canonical_symbol(symbol) for symbol in symbols})
    history_symbols = sorted(
        set(target_symbols)
        | set(_source_symbols_for_expected_vc(identity_map, target_symbols))
    )
    fingerprint = _cache_fingerprint(
        dates, target_symbols, identity_map_sha256=identity_map_sha,
    )

    if (
        resume
        and meta_path.exists()
        and vc_path.exists()
        and repaired_vc_path.exists()
        and evc_path.exists()
    ):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                return pd.read_parquet(evc_path)
        except (OSError, ValueError, KeyError):
            pass

    hist_dates = _common._eval_dates(
        dates[0] - dt.timedelta(days=40), dates[-1],
    )
    log.info(
        "Building expected-VC cache: %d history dates x %d symbols",
        len(hist_dates), len(history_symbols),
    )
    shard_dir = Path(shard_dir) if shard_dir is not None else _default_vc_shard_dir()
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_fingerprint = _vc_shard_fingerprint(history_symbols)
    frames: list[pd.DataFrame] = []
    started = time.perf_counter()
    for i, hist_date in enumerate(hist_dates, start=1):
        frames.append(_load_or_build_vc_shard(
            hist_date, history_symbols, shard_dir, shard_fingerprint, workers=workers,
        ))
        if i % 20 == 0 or i == len(hist_dates):
            elapsed = time.perf_counter() - started
            log.info(
                "expected-VC shards: %d/%d dates (%.1f s elapsed, %.2f s/date)",
                i, len(hist_dates), elapsed, elapsed / i,
            )
    non_empty = [f for f in frames if not f.empty]
    vc_history = (
        pd.concat(non_empty, ignore_index=True)
        if non_empty else pd.DataFrame(columns=VC_HISTORY_COLUMNS)
    )
    vc_history_for_expected_vc, repair_meta = _vc_history_for_expected_vc(
        vc_history, identity_map,
    )
    if not vc_history.empty:
        loaded = vc_history.copy()
        loaded["symbol"] = loaded["symbol"].map(canonical_symbol)
        loaded["date"] = pd.to_datetime(loaded["date"]).dt.date
        repair_meta["same_symbol_preindex_rows_used"] = int(
            (
                loaded["symbol"].isin(target_symbols)
                & (loaded["date"] < dates[0])
            ).sum()
        )
    expected_vc_all = rolling_expected_vc(vc_history_for_expected_vc)
    expected_vc = expected_vc_all[
        expected_vc_all["symbol"].map(canonical_symbol).isin(target_symbols)
    ].reset_index(drop=True)
    _write_parquet_atomic(vc_history, vc_path)
    _write_parquet_atomic(vc_history_for_expected_vc, repaired_vc_path)
    _write_parquet_atomic(expected_vc, evc_path)
    _write_json_atomic(meta_path, {
        "fingerprint": fingerprint,
        "shard_fingerprint": shard_fingerprint,
        "shard_dir": str(shard_dir),
        "expected_vc_policy_version": EXPECTED_VC_POLICY_VERSION,
        "identity_map_path": str(EXPECTED_VC_IDENTITY_MAP),
        "identity_map_sha256": identity_map_sha,
        "history_dates": len(hist_dates),
        "history_rows": len(vc_history),
        "history_symbols": len(history_symbols),
        "target_symbols": len(target_symbols),
        "vc_history_for_expected_vc_rows": len(vc_history_for_expected_vc),
        "expected_vc_rows": len(expected_vc),
        "expected_vc_repair_counts": repair_meta,
        "predecessor_rows_used": repair_meta["predecessor_rows_used"],
        "same_symbol_preindex_rows_used": repair_meta["same_symbol_preindex_rows_used"],
        "ambiguous_rows_left_unrepaired": repair_meta["ambiguous_rows_left_unrepaired"],
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    return expected_vc


def _shard_paths(shard_root: Path, date: dt.date) -> tuple[Path, Path]:
    day_dir = shard_root / f"date={date.isoformat()}"
    return day_dir / "panel.parquet", day_dir / "manifest.json"


def validate_shard(
    shard_root: Path,
    date: dt.date,
    run_fingerprint: str,
    strategies: Iterable[str],
) -> dict | None:
    panel_path, manifest_path = _shard_paths(shard_root, date)
    if not panel_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return None
        if manifest.get("fingerprint") != run_fingerprint:
            return None
        if manifest.get("sha256") != _sha256(panel_path):
            return None
        check = pd.read_parquet(
            panel_path,
            columns=sorted({
                "date", "symbol", "strategy", "order_id",
                *REQUIRED_METRIC_COLUMNS,
            }),
        )
        if check.empty or set(check["strategy"].unique()) != set(strategies):
            return None
        if {pd.Timestamp(x).date() for x in check["date"].unique()} != {date}:
            return None
        if int(manifest.get("rows", -1)) != len(check):
            return None
        if _nonfinite_metric_counts(check):
            return None
        return manifest
    except Exception:
        return None


def _simulate_date_job(
    date: dt.date,
    symbol_specs: list[dict],
    strategies: list[str],
    size_fractions: tuple[float, ...],
    fill_specification: str,
    shard_root: Path,
    run_fingerprint: str,
    windows_map: dict | None = None,
) -> dict:
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    tier_lookup: dict[str, int] = {}

    for spec in symbol_specs:
        symbol = canonical_symbol(spec["symbol"])
        tier = int(spec["tier"])
        tier_lookup[symbol] = tier
        # Data completeness was already established by the precheck that
        # built these specs; a file vanishing in between surfaces through
        # the engine's FileNotFoundError path as a structured
        # ``missing_parquet`` skip, so no per-symbol stat calls here.
        try:
            parents = build_parent_orders(
                symbol,
                date,
                float(spec["expected_vc"]),
                size_fractions=size_fractions,
                windows=windows_map,
            )
            if parents.empty:
                failures.append({
                    "date": date,
                    "symbol": symbol,
                    "reason": "no_parent_orders",
                    "detail": "expected_vc produced no positive parent quantity",
                    "critical": False,
                })
                continue
            skip_reason: dict = {}
            result = simulate_symbol_day(
                symbol,
                date,
                parents,
                strategies,
                fill_model=_common._worker_model,
                delta_max_bps_by_tier=cfg.DELTA_MAX_BPS,
                tier=tier,
                fill_specification=fill_specification,
                km_model=(
                    _common._worker_model
                    if fill_specification == "km" else None
                ),
                tod_schedule=_common._worker_tod,
                value_model=_common._worker_value_model,
                sector=spec.get("sector", ""),
                listing_exchange=spec.get("listing_exchange", ""),
                skip_reason_out=skip_reason,
            )
            for col in ADV_SPREAD_BUCKET_COLUMNS:
                if col in spec:
                    result[col] = spec[col]
            if result.empty:
                failures.append({
                    "date": date,
                    "symbol": symbol,
                    "reason": skip_reason.get(
                        "reason", "empty_after_filter_or_missing_auction",
                    ),
                    "detail": "",
                    "critical": False,
                })
                continue
            frames.append(result)
        except TradePolicyMismatchError as exc:
            failures.append({
                "date": date,
                "symbol": symbol,
                "reason": "policy_mismatch",
                "detail": str(exc),
                "critical": True,
            })
        except (TypeError, ValueError) as exc:
            failures.append({
                "date": date,
                "symbol": symbol,
                "reason": "dtype_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "critical": True,
            })
        except Exception as exc:
            failures.append({
                "date": date,
                "symbol": symbol,
                "reason": "compute_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "critical": True,
            })

    if not frames:
        return {
            "date": date,
            "status": "failed",
            "eligible_symbol_days": len(symbol_specs),
            "successful_symbol_days": 0,
            "failed_symbol_days": len(failures),
            "rows": 0,
            "runtime_seconds": time.perf_counter() - started,
            "failures": failures,
        }

    panel = pd.concat(frames, ignore_index=True)
    panel = attach_alpha_columns(panel)
    panel = attach_moc_differential_columns(panel)
    panel["symbol"] = panel["symbol"].map(canonical_symbol)
    panel["tier"] = panel["symbol"].map(tier_lookup)
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    panel["is_headline_size"] = np.isclose(
        panel["size_frac"].astype(float), cfg.PARENT_ORDER_PRIMARY_FRACTION,
    )
    _raise_on_bad_metrics(panel, f"date shard {date}")

    panel_path, manifest_path = _shard_paths(shard_root, date)
    _write_parquet_atomic(panel, panel_path)
    checksum = _sha256(panel_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": run_fingerprint,
        "date": date,
        "status": "partial" if failures else "complete",
        "eligible_symbol_days": len(symbol_specs),
        "successful_symbol_days": int(panel["symbol"].nunique()),
        "failed_symbol_days": len(failures),
        "rows": len(panel),
        "strategies": strategies,
        "sha256": checksum,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "failures": failures, "shard_path": str(panel_path)}


def _load_preprocessing_universe(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        canonical_symbol(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _canonical_membership_panel(
    dates: list[dt.date],
    *,
    symbols: list[str] | None,
    universe: str | None,
) -> pd.DataFrame:
    if symbols is not None:
        rows = [
            {"date": date, "symbol": canonical_symbol(symbol)}
            for date in dates
            for symbol in symbols
        ]
        return pd.DataFrame(rows).drop_duplicates(["date", "symbol"])
    if not universe:
        raise ValueError("run_master_panel requires symbols or universe")
    panel = build_index_universe_panel(universe, dates, expand_aliases=False)
    optional_cols = ["sector", "listing_exchange"]
    keep_cols = ["date", "symbol", *[c for c in optional_cols if c in panel.columns]]
    panel = panel[keep_cols].copy()
    panel["symbol"] = panel["symbol"].map(canonical_symbol)
    return panel.drop_duplicates(["date", "symbol"])


def run_master_panel(
    *,
    strategies: list[str],
    start: dt.date,
    end: dt.date,
    artifacts_dir: Path,
    run_root: Path,
    symbols: list[str] | None = None,
    universe: str | None = None,
    max_dates: int | None = None,
    workers: int = 1,
    fill_specification: str = "tape_replay_queue",
    size_fractions: tuple[float, ...] = (cfg.PARENT_ORDER_PRIMARY_FRACTION,),
    windows: tuple[str, ...] | None = None,
    vc_shard_dir: Path | None = None,
    resume: bool = True,
    min_eligible_coverage: float = 0.995,
    min_index_coverage: float = 0.95,
    tier_policy: str = TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    preprocessing_symbols_file: Path | None = None,
    adv_spread_bucket_map: Path | None = None,
) -> dict:
    windows_map: dict | None = None
    if windows is not None:
        unknown = sorted(set(windows) - set(cfg.EXECUTION_WINDOWS))
        if unknown:
            raise ValueError(
                f"Unknown execution windows {unknown}; "
                f"expected subset of {sorted(cfg.EXECUTION_WINDOWS)}"
            )
        windows_map = {
            name: cfg.EXECUTION_WINDOWS[name]
            for name in cfg.EXECUTION_WINDOWS if name in set(windows)
        }
    dates = _common._eval_dates(start, end)
    if max_dates is not None:
        dates = dates[:max_dates]
    if not dates:
        raise RuntimeError("No evaluation dates available")
    if len(dates) != len(set(dates)):
        dupes = sorted({d for d in dates if dates.count(d) > 1})[:5]
        raise RuntimeError(
            "Evaluation date list contains duplicates; duplicate date jobs "
            f"would race on the same panel shards. Examples: {dupes}"
        )

    shard_root = run_root / "panel_shards"
    cache_dir = run_root / "cache"
    metadata_dir = run_root / "metadata"
    for path in (shard_root, cache_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    membership = _canonical_membership_panel(
        dates, symbols=symbols, universe=universe,
    )
    canonical_symbols = sorted(membership["symbol"].unique())
    preprocessing_universe = _load_preprocessing_universe(preprocessing_symbols_file)
    missing_from_preprocessing_universe = (
        sorted(set(canonical_symbols) - preprocessing_universe)
        if preprocessing_universe else []
    )
    if missing_from_preprocessing_universe:
        raise RuntimeError(
            "Point-in-time membership contains symbols absent from the preprocessing universe: "
            + ", ".join(missing_from_preprocessing_universe[:20])
        )
    excluded_extra_preprocessed_symbols = (
        sorted(preprocessing_universe - set(canonical_symbols))
        if preprocessing_universe else []
    )
    _, tier_map, _ = _common._load_artifacts(
        artifacts_dir, fill_spec=fill_specification,
    )
    resolved_bucket_map_path = adv_spread_bucket_map
    if resolved_bucket_map_path is None:
        candidate = artifacts_dir / "symbol_adv_spread_bucket_map.csv"
        if candidate.exists():
            resolved_bucket_map_path = candidate
    bucket_map = load_adv_spread_bucket_map(resolved_bucket_map_path)
    bucket_lookup: dict[str, dict] = {}
    if not bucket_map.empty:
        for item in bucket_map.itertuples(index=False):
            symbol = canonical_symbol(getattr(item, "symbol"))
            bucket_lookup[symbol] = {
                "adv_bucket": getattr(item, "adv_bucket", pd.NA),
                "spread_bucket": getattr(item, "spread_bucket", pd.NA),
                "adv_spread_bucket": getattr(item, "adv_spread_bucket", "unassigned") or "unassigned",
            }
    tier_policy_version = _tier_policy_version(tier_policy)
    calibrated_tier_lookup = _canonical_tier_lookup(tier_map)
    tier_lookup = dict(calibrated_tier_lookup)
    tier_audit: dict[str, dict] = {}
    fallback_symbols: set[str] = set()
    fallback_symbol_days = 0
    expected_vc = _load_or_build_expected_vc(
        dates,
        canonical_symbols,
        cache_dir,
        workers=max(1, min(workers, 8)),
        resume=resume,
        shard_dir=vc_shard_dir,
    )
    evc_lookup = {
        (canonical_symbol(row.symbol), pd.Timestamp(row.date).date()):
        float(row.expected_vc)
        for row in expected_vc.itertuples(index=False)
        if pd.notna(row.expected_vc) and float(row.expected_vc) > 0
    }

    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "start": start,
        "end": end,
        "dates": dates,
        "strategies": strategies,
        "size_fractions": size_fractions,
        "windows": sorted(windows_map) if windows_map else None,
        "fill_specification": fill_specification,
        "universe": universe,
        "symbols": canonical_symbols,
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "feature_policy": cfg.FEATURE_POLICY_VERSION,
        "as_horizon_seconds": int(cfg.AS_HORIZON_SECONDS),
        "liquidity_tier_policy": tier_policy_version,
        "tier_policy_mode": tier_policy,
        "fallback_liquidity_tier": FALLBACK_LIQUIDITY_TIER,
        "expected_vc_policy_version": EXPECTED_VC_POLICY_VERSION,
        "expected_vc_identity_map_path": str(EXPECTED_VC_IDENTITY_MAP),
        "expected_vc_identity_map_sha256": _identity_map_sha(),
        "simulation_source_sha256": _simulation_source_signature(),
        "artifact_signature": _artifact_signature(
            artifacts_dir, fill_specification,
        ),
        "adv_spread_bucket_map": str(resolved_bucket_map_path) if resolved_bucket_map_path else None,
        "adv_spread_bucket_map_sha256": _sha256(resolved_bucket_map_path) if resolved_bucket_map_path and Path(resolved_bucket_map_path).exists() else None,
    }
    run_fingerprint = _fingerprint(fingerprint_payload)
    _write_json_atomic(metadata_dir / "simulation_config.json", {
        **fingerprint_payload,
        "fingerprint": run_fingerprint,
        "workers": workers,
    })

    precheck_failures: list[dict] = []
    jobs: dict[dt.date, list[dict]] = {date: [] for date in dates}
    data_complete = 0
    for row in membership.itertuples(index=False):
        date = pd.Timestamp(row.date).date()
        symbol = canonical_symbol(row.symbol)
        has_trade = parquet_available(date, symbol, "Trade")
        has_nbbo = parquet_available(date, symbol, "NBBO")
        if has_trade and has_nbbo:
            data_complete += 1
        else:
            precheck_failures.append({
                "date": date,
                "symbol": symbol,
                "reason": "missing_parquet",
                "detail": f"trade={has_trade};nbbo={has_nbbo}",
                "critical": False,
            })
            continue
        tier = tier_lookup.get(symbol)
        if tier is None:
            if tier_policy == TIER_POLICY_CALIBRATED_PLUS_FALLBACK:
                tier = FALLBACK_LIQUIDITY_TIER
                tier_lookup[symbol] = int(tier)
                fallback_symbols.add(symbol)
                fallback_symbol_days += 1
                tier_audit.setdefault(symbol, {
                    "symbol": symbol,
                    "tier": int(tier),
                    "tier_source": "fallback_missing_calibration",
                    "reason": "data_complete_symbol_absent_from_calibration_tier_map",
                })
            else:
                tier_audit.setdefault(symbol, {
                    "symbol": symbol,
                    "tier": pd.NA,
                    "tier_source": "missing",
                    "reason": "data_complete_symbol_absent_from_calibration_tier_map",
                })
                precheck_failures.append({
                    "date": date,
                    "symbol": symbol,
                    "reason": "missing_tier",
                    "detail": "",
                    "critical": False,
                })
                continue
        else:
            tier_audit.setdefault(symbol, {
                "symbol": symbol,
                "tier": int(tier),
                "tier_source": "calibrated",
                "reason": "present_in_calibration_tier_map",
            })
        expected = evc_lookup.get((symbol, date))
        if expected is None:
            precheck_failures.append({
                "date": date,
                "symbol": symbol,
                "reason": "missing_expected_vc",
                "detail": "",
                "critical": False,
            })
            continue
        spec = {
            "symbol": symbol,
            "tier": int(tier),
            "expected_vc": float(expected),
        }
        if bucket_lookup:
            spec.update(bucket_lookup.get(symbol, {
                "adv_bucket": pd.NA,
                "spread_bucket": pd.NA,
                "adv_spread_bucket": "unassigned",
            }))
        sector = getattr(row, "sector", "") or ""
        listing_exchange = getattr(row, "listing_exchange", "") or ""
        if sector:
            spec["sector"] = sector
        if listing_exchange:
            spec["listing_exchange"] = listing_exchange
        jobs[date].append(spec)

    completed_tier_map = _tier_audit_frame(tier_audit)
    _write_csv_atomic(completed_tier_map, metadata_dir / "liquidity_tier_audit.csv")
    _write_csv_atomic(
        completed_tier_map[["symbol", "tier", "tier_source"]],
        metadata_dir / "completed_symbol_tier_map.csv",
    )

    statuses: list[dict] = []
    failures = list(precheck_failures)
    pending: list[tuple[dt.date, list[dict]]] = []
    for date in dates:
        valid = validate_shard(
            shard_root, date, run_fingerprint, strategies,
        ) if resume else None
        if valid is not None:
            statuses.append({**valid, "resumed": True})
        elif jobs[date]:
            pending.append((date, jobs[date]))

    started = time.perf_counter()

    def record(result: dict) -> None:
        nonlocal statuses, failures
        day_failures = result.pop("failures", [])
        failures.extend(day_failures)
        statuses.append({**result, "resumed": False})
        status_df = _status_frame(statuses)
        failure_df = _failure_frame(failures)
        _write_csv_atomic(status_df, metadata_dir / "simulation_manifest.csv")
        _write_csv_atomic(failure_df, metadata_dir / "simulation_failures.csv")
        done = len(statuses)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(dates) - done) / rate if rate > 0 else float("nan")
        successful = int(status_df.get(
            "successful_symbol_days", pd.Series(dtype=float),
        ).fillna(0).sum())
        failed = len(failure_df)
        log.info(
            "Master-panel progress: %d/%d (%.1f%%), elapsed=%.1f min, "
            "eta=%.1f min, successful=%d, skipped_or_failed=%d, last=%s",
            done, len(dates), 100 * done / len(dates),
            elapsed / 60, eta / 60, successful, failed, result.get("date"),
        )

    pool_backend_used = "inline"
    if workers <= 1:
        _common._worker_init(artifacts_dir, fill_specification)
        for date, specs in pending:
            record(_simulate_date_job(
                date, specs, strategies, size_fractions, fill_specification,
                shard_root, run_fingerprint, windows_map,
            ))
    else:
        max_workers = max(1, min(int(workers), 16))
        with AdaptivePool(
            max_workers=max_workers,
            max_in_flight=max_workers * 2,
            initializer=_common._worker_init,
            initargs=(artifacts_dir, fill_specification),
        ) as pool:
            job_iter = iter(pending)
            inflight = {}

            def submit_next() -> bool:
                try:
                    date, specs = next(job_iter)
                except StopIteration:
                    return False
                future = pool.submit(
                    _simulate_date_job,
                    date, specs, strategies, size_fractions,
                    fill_specification, shard_root, run_fingerprint,
                    windows_map,
                )
                inflight[future] = date
                return True

            for _ in range(min(len(pending), max_workers * 2)):
                submit_next()
            while inflight:
                completed, _ = wait(
                    inflight, timeout=300.0, return_when=FIRST_COMPLETED,
                )
                if not completed:
                    elapsed = time.perf_counter() - started
                    log.info(
                        "Master-panel heartbeat: %d/%d dates complete, "
                        "%d date jobs in flight, elapsed=%.1f min",
                        len(statuses), len(dates), len(inflight), elapsed / 60,
                    )
                    continue
                for future in completed:
                    date = inflight.pop(future)
                    try:
                        record(future.result())
                    except Exception as exc:
                        result = {
                            "date": date,
                            "status": "failed",
                            "eligible_symbol_days": len(jobs[date]),
                            "successful_symbol_days": 0,
                            "failed_symbol_days": len(jobs[date]),
                            "rows": 0,
                            "runtime_seconds": 0.0,
                            "failures": [{
                                "date": date,
                                "symbol": "__DATE_JOB__",
                                "reason": "compute_error",
                                "detail": f"{type(exc).__name__}: {exc}",
                                "critical": True,
                            }],
                        }
                        record(result)
                    submit_next()
            pool_backend_used = pool.backend

    status_df = _status_frame(statuses)
    failure_df = _failure_frame(failures)
    _write_csv_atomic(status_df, metadata_dir / "simulation_manifest.csv")
    _write_csv_atomic(failure_df, metadata_dir / "simulation_failures.csv")

    # Every index symbol-day with both TAQ sides is expected to remain usable.
    # Missing tiers or expected-VC values therefore reduce eligible coverage
    # instead of silently shrinking its denominator.
    eligible = int(data_complete)
    successful = int(status_df.get(
        "successful_symbol_days", pd.Series(dtype=float),
    ).fillna(0).sum())
    index_total = int(len(membership))
    index_coverage = data_complete / max(index_total, 1)
    eligible_coverage = successful / max(eligible, 1)
    critical_count = int(
        failure_df.get("critical", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    all_dates_valid = (
        len(status_df) == len(dates)
        and not status_df.empty
        and not (status_df["status"] == "failed").any()
    )
    complete = (
        all_dates_valid
        and eligible_coverage >= min_eligible_coverage
        and index_coverage >= min_index_coverage
        and critical_count == 0
    )
    summary = {
        "status": "complete" if complete else "failed_qc",
        "fingerprint": run_fingerprint,
        "dates_expected": len(dates),
        "dates_with_valid_shards": len(status_df),
        "index_symbol_days": index_total,
        "data_complete_symbol_days": data_complete,
        "eligible_symbol_days": eligible,
        "successful_symbol_days": successful,
        "index_coverage": index_coverage,
        "eligible_coverage": eligible_coverage,
        "min_index_coverage": min_index_coverage,
        "min_eligible_coverage": min_eligible_coverage,
        "critical_failures": critical_count,
        "pool_backend": pool_backend_used,
        "workers": workers,
        "failure_reason_counts": (
            failure_df["reason"].value_counts().to_dict()
            if not failure_df.empty else {}
        ),
        "close_source_distribution": _close_source_distribution(shard_root),
        "liquidity_tier_policy": tier_policy_version,
        "tier_policy_mode": tier_policy,
        "tier_fallback_symbols": len(fallback_symbols),
        "tier_fallback_symbol_days": fallback_symbol_days,
        "tier_audit_path": str(metadata_dir / "liquidity_tier_audit.csv"),
        "completed_tier_map_path": str(metadata_dir / "completed_symbol_tier_map.csv"),
        "expected_vc_policy_version": EXPECTED_VC_POLICY_VERSION,
        "expected_vc_identity_map_path": str(EXPECTED_VC_IDENTITY_MAP),
        "expected_vc_identity_map_sha256": _identity_map_sha(),
        "adv_spread_bucket_map_path": str(resolved_bucket_map_path) if resolved_bucket_map_path else None,
        "adv_spread_bucketed_symbols": len(bucket_lookup),
        "preprocessing_symbols_file": str(preprocessing_symbols_file) if preprocessing_symbols_file else None,
        "preprocessing_universe_symbols": len(preprocessing_universe) if preprocessing_universe else None,
        "active_membership_symbol_days": index_total,
        "excluded_extra_preprocessed_symbols": len(excluded_extra_preprocessed_symbols),
        "missing_membership_symbols_from_preprocessing_universe": len(missing_from_preprocessing_universe),
        "shard_root": str(shard_root),
    }
    _write_json_atomic(metadata_dir / "simulation_summary.json", summary)
    if not complete:
        raise RuntimeError(f"Master-panel QC failed: {summary}")
    return summary


def _close_source_distribution(shard_root: Path) -> list[dict]:
    """In-panel distribution of closing-auction source metadata (per shard
    glob). Answers how often non-primary auction sources appear in the panel;
    symbol-days skipped for a missing auction are visible separately through
    failure_reason_counts."""
    import duckdb

    glob_path = str(shard_root / "date=*" / "panel.parquet").replace("\\", "/")
    try:
        frame = duckdb.sql(
            "SELECT close_price_source, close_volume_source, "
            "COUNT(*) AS panel_rows, "
            "COUNT(DISTINCT symbol || '|' || CAST(date AS VARCHAR)) AS symbol_days "
            f"FROM read_parquet('{glob_path}') "
            "GROUP BY 1, 2 ORDER BY symbol_days DESC"
        ).df()
    except Exception as exc:
        log.warning("close_source_distribution unavailable: %s", exc)
        return []
    return frame.to_dict("records")


def materialize_panel(
    shard_root: Path,
    strategies: Iterable[str],
    out_path: Path,
) -> Path:
    import duckdb

    glob_path = str(shard_root / "date=*" / "panel.parquet").replace("\\", "/")
    strategy_sql = ", ".join(
        "'" + str(strategy).replace("'", "''") + "'" for strategy in strategies
    )
    output = str(out_path).replace("\\", "/").replace("'", "''")
    source = glob_path.replace("'", "''")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT *
                  FROM read_parquet('{source}', union_by_name=true)
                 WHERE strategy IN ({strategy_sql})
            ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()
    validation_columns = sorted(REQUIRED_PANEL_COLUMNS | REQUIRED_METRIC_COLUMNS)
    panel = pd.read_parquet(out_path, columns=validation_columns)
    if panel.empty or not REQUIRED_PANEL_COLUMNS.issubset(panel.columns):
        raise RuntimeError(f"Materialized panel failed validation: {out_path}")
    _raise_on_bad_metrics(panel, f"Materialized panel {out_path}")
    missing = set(strategies) - set(panel["strategy"].unique())
    if missing:
        raise RuntimeError(f"Materialized panel missing strategies: {sorted(missing)}")
    return out_path


