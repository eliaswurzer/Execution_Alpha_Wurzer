"""
calibrate_fill_model.py -- Kalibriert das Cox-PH Fill-Modell sowie Glosten-AS
auf dem Pre-Sample (Jan-Jun 2018).

Aufruf::

    python -m analysis.runners.calibrate_fill_model --symbols AAPL MSFT AMZN --out artifacts/fill_model
    python -m analysis.runners.calibrate_fill_model --workers 4   # parallel (recommended)

Ergebnis:
* ``artifacts/fill_model/cox_tier_*.pkl``
* ``artifacts/fill_model/symbol_tier_map.csv``
* ``artifacts/fill_model/glosten_as.csv``
* ``artifacts/fill_model/validation.csv``
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, wait
from ..utils.adaptive_pool import AdaptivePool
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.features import compute_daily_features
from ..data.taq_loader import (
    TradePolicyMismatchError,
    filter_trades_near_quotes, filter_valid_quotes, filter_valid_trades,
    list_dates, list_symbols, load_symbol_day,
)
from ..data.index_universe import build_index_universe_panel
from ..data.adv_universe import assign_liquidity_tiers
from ..fill_model.adverse_selection import build_as_panel, fit_glosten_as
from ..fill_model.cox_ph import TieredFillModel
from ..fill_model.kaplan_meier import TieredKMFillModel
from ..fill_model.xgb_survival import TieredXGBFillModel, resolve_xgb_device
from ..fill_model.tod_schedule import TODSchedule
from ..fill_model.state_vector import STATE_COLUMNS, build_event_panel
from ..utils.symbols import canonical_symbol
from ..fill_model.validation import validate_tiered_model
from . import _common

log = logging.getLogger(__name__)


ALLOWED_SKIP_REASONS = {"missing_parquet", "empty_after_filter", "insufficient_quotes"}
CRITICAL_REASONS = {"compute_error", "timeout", "memory_guard", "dtype_error", "policy_mismatch"}

# Cox model validation gate (thesis fill-model correctness). A calibration whose
# Cox fit is degenerate in any tier (anti-discriminative, mis-levelled, or a
# collapsed Breslow baseline) must not be marked complete, because downstream
# runs would silently understate execution alpha (see the liquid-tier baseline
# collapse documented in README_simulation_correctness_audit.md).
_COX_VALIDATION_MIN_AUC = 0.55
_COX_VALIDATION_ABS_TOL = 0.10        # |mean_pred - observed| at the fill horizon
# Absolute floor on the implied baseline fill (1 - S0(h)) at the mean covariate.
# This catches a genuinely collapsed Breslow baseline (S0(h) ~ 1, base_fill ~ 0)
# without flagging healthy fits: the baseline-at-mean is legitimately well below
# the mean predicted fill when covariates are dispersed (Jensen gap on exp), so a
# relative-to-observed threshold produced false positives.
_COX_VALIDATION_BASELINE_FLOOR = 0.01


def _fill_validation_failures(reports) -> dict[int, list[str]]:
    """Per-tier fill-model validation-gate verdict (Cox or XGB).

    Returns ``{tier: [reasons]}`` for every tier whose fit is degenerate:
    anti-discriminative (AUC below threshold), mis-levelled (mean predicted fill
    far from observed), or a collapsed Breslow baseline (implied baseline fill
    far below observed). Empty dict means the calibration passes the gate. NaN
    metrics (e.g. base_fill_s0 for XGB, which has no explicit baseline) are
    skipped.
    """
    failed: dict[int, list[str]] = {}
    for tier, r in reports.items():
        reasons: list[str] = []
        if np.isfinite(r.auc) and r.auc < _COX_VALIDATION_MIN_AUC:
            reasons.append(f"auc={r.auc:.3f}<{_COX_VALIDATION_MIN_AUC}")
        if (np.isfinite(r.mean_pred) and np.isfinite(r.observed)
                and abs(r.mean_pred - r.observed) > _COX_VALIDATION_ABS_TOL):
            reasons.append(
                f"|mean_pred-observed|={abs(r.mean_pred - r.observed):.3f}"
                f">{_COX_VALIDATION_ABS_TOL}"
            )
        if np.isfinite(r.base_fill_s0) and r.base_fill_s0 < _COX_VALIDATION_BASELINE_FLOOR:
            reasons.append(
                f"base_fill_s0={r.base_fill_s0:.4f}<{_COX_VALIDATION_BASELINE_FLOOR}"
            )
        if reasons:
            failed[int(tier)] = reasons
    return failed


def _level_validation_failures(reports) -> dict[int, list[str]]:
    """Per-tier level-calibration gate for non-parametric robustness models."""
    failed: dict[int, list[str]] = {}
    for tier, r in reports.items():
        reasons: list[str] = []
        if not np.isfinite(r.observed) or not np.isfinite(r.mean_pred):
            reasons.append("observed/mean_pred is not finite")
        elif abs(r.mean_pred - r.observed) > _COX_VALIDATION_ABS_TOL:
            reasons.append(
                f"|mean_pred-observed|={abs(r.mean_pred - r.observed):.3f}"
                f">{_COX_VALIDATION_ABS_TOL}"
            )
        if reasons:
            failed[int(tier)] = reasons
    return failed


def _record_tod_failure(
    preliminary_manifest: dict,
    manifest_path: Path,
    error: Exception,
    *,
    tod_required: bool,
) -> None:
    preliminary_manifest["tod_status"] = "failed"
    preliminary_manifest["tod_error"] = str(error)
    if tod_required:
        preliminary_manifest["status"] = "failed_tod_model_fit"
        _write_manifest(manifest_path, preliminary_manifest)
        raise RuntimeError(f"TOD schedule fit failed: {error}") from error


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace(" ", "_").replace("/", "_")


def _canonical_symbol(symbol: str) -> str:
    return canonical_symbol(symbol)


def _dedupe_pairs(pairs: list[tuple[_dt.date, str]]) -> list[tuple[_dt.date, str]]:
    seen: set[tuple[_dt.date, str]] = set()
    out: list[tuple[_dt.date, str]] = []
    for d, sym in pairs:
        key = (d, _canonical_symbol(sym))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _wait_for_memory(
    *,
    ram_max_percent: float,
    ram_min_available_gb: float,
    max_wait_seconds: int,
) -> tuple[bool, str]:
    try:
        import psutil
    except ImportError:
        return True, "psutil_unavailable"

    deadline = time.monotonic() + max(0, int(max_wait_seconds))
    last_detail = ""
    while True:
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        last_detail = f"ram_percent={vm.percent:.1f};available_gb={available_gb:.2f}"
        if vm.percent <= ram_max_percent and available_gb >= ram_min_available_gb:
            return True, last_detail
        if time.monotonic() >= deadline:
            return False, last_detail
        time.sleep(5.0)


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8",
    )


def _downcast_event_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["symbol"] = out["symbol"].astype("string")
    out["side"] = out["side"].astype("category")
    out["t0"] = pd.to_datetime(out["t0"])
    out["date"] = pd.to_datetime(out["date"])
    for col in ["limit_price", "duration", "as_bps", *STATE_COLUMNS]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    if "event" in out.columns:
        out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(0).astype("int8")
    return out


def _downcast_as_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["symbol"] = out["symbol"].astype("string")
    out["side"] = out["side"].astype("category")
    out["t0"] = pd.to_datetime(out["t0"])
    out["date"] = pd.to_datetime(out["date"])
    if "dm_bps" in out.columns:
        out["dm_bps"] = pd.to_numeric(out["dm_bps"], errors="coerce").astype("float32")
    if "fill" in out.columns:
        out["fill"] = pd.to_numeric(out["fill"], errors="coerce").fillna(0).astype("int8")
    return out


def _downcast_daily_features(row: pd.Series | dict) -> pd.DataFrame:
    out = pd.DataFrame([dict(row)])
    out["symbol"] = out["symbol"].astype("string")
    out["date"] = pd.to_datetime(out["date"])
    for col in ["adv_shares", "adv_dollar", "avg_half_spread_bps", "rv_daily"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in ["n_trades", "n_quotes"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int32")
    return out


def _clean_event_panel_for_fit(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    out["duration"] = out["duration"].replace([float("inf"), float("-inf")], pd.NA)
    out["event"] = out["event"].replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["duration", "event"])
    out = out[out["duration"] > 0]
    out["event"] = out["event"].clip(lower=0, upper=1).astype("int8")
    for col in STATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([float("inf"), float("-inf")], pd.NA)
    if out.empty:
        raise RuntimeError("Event panel empty after dtype/finite-value cleanup")
    return out


def _read_shards(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths if p]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _status(
    d: _dt.date,
    sym: str,
    *,
    status: str,
    reason: str,
    started_at: float,
    detail: str = "",
    n_trades_raw: int = 0,
    n_nbbo_raw: int = 0,
    n_trades: int = 0,
    n_nbbo: int = 0,
    n_event_rows: int = 0,
    n_as_rows: int = 0,
    event_path: str = "",
    daily_path: str = "",
    as_path: str = "",
) -> dict:
    return {
        "date": d.isoformat(),
        "symbol": sym,
        "status": status,
        "reason": reason,
        "detail": str(detail)[:1000],
        "n_trades_raw": int(n_trades_raw),
        "n_nbbo_raw": int(n_nbbo_raw),
        "n_trades": int(n_trades),
        "n_nbbo": int(n_nbbo),
        "n_event_rows": int(n_event_rows),
        "n_as_rows": int(n_as_rows),
        "event_path": event_path,
        "daily_path": daily_path,
        "as_path": as_path,
        "runtime_seconds": round(time.perf_counter() - started_at, 3),
    }


def _attach_per_fill_as(evt: pd.DataFrame, nbbo: pd.DataFrame) -> pd.DataFrame:
    """Add 'as_bps' column to filled events (event==1) using post-fill mid-quote drift."""
    filled_mask = evt["event"] == 1
    if not filled_mask.any() or "t0" not in evt.columns or "duration" not in evt.columns:
        evt["as_bps"] = float("nan")
        return evt

    nbbo_mid = nbbo[["time", "mid"]].sort_values("time").reset_index(drop=True)
    horizon = pd.Timedelta(seconds=cfg.AS_HORIZON_SECONDS)

    # Fill timestamps
    t_fills = pd.to_datetime(evt.loc[filled_mask, "t0"]) + pd.to_timedelta(
        evt.loc[filled_mask, "duration"], unit="s"
    )
    t_afters = t_fills + horizon

    fills_df = pd.DataFrame({"time": t_fills.values, "orig_idx": evt.index[filled_mask]})
    afters_df = pd.DataFrame({"time": t_afters.values, "orig_idx": evt.index[filled_mask]})

    mid_at = pd.merge_asof(
        fills_df.sort_values("time"), nbbo_mid, on="time", direction="backward"
    ).rename(columns={"mid": "mid_at"}).sort_values("orig_idx")
    mid_after = pd.merge_asof(
        afters_df.sort_values("time"), nbbo_mid, on="time", direction="backward"
    ).rename(columns={"mid": "mid_after"}).sort_values("orig_idx")

    close_price = nbbo_mid["mid"].iloc[-1]
    side_signs = evt.loc[filled_mask, "side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(1.0).values
    mid_at_vals = mid_at["mid_at"].fillna(close_price).values
    mid_after_vals = mid_after["mid_after"].fillna(close_price).values

    as_vals = side_signs * (mid_after_vals - mid_at_vals) / max(close_price, 1.0) * 1e4

    evt = evt.copy()
    evt["as_bps"] = float("nan")
    evt.loc[filled_mask, "as_bps"] = as_vals
    return evt


def _select_dates(start: _dt.date, end: _dt.date) -> list[_dt.date]:
    return _common._eval_dates(start, end)


def _process_symbol_day(
    d: _dt.date,
    sym: str,
    shard_root: Path,
    event_sample_per_symbol_day: int | None = None,
    as_sample_per_symbol_day: int | None = None,
    event_sample_every_seconds: int = cfg.REFRESH_SECONDS_DEFAULT,
    as_sample_every_seconds: int = 60,
    ram_max_percent: float = 90.0,
    ram_min_available_gb: float = 2.0,
    memory_wait_seconds: int = 600,
) -> dict:
    """Load/process one (date, symbol) pair and write shard files.

    The worker returns a small status dict only. Large panels stay on disk.
    """
    started = time.perf_counter()
    sym = _canonical_symbol(sym)
    ds = d.strftime("%Y%m%d")
    safe = _safe_symbol(sym)
    memory_ok, memory_detail = _wait_for_memory(
        ram_max_percent=ram_max_percent,
        ram_min_available_gb=ram_min_available_gb,
        max_wait_seconds=memory_wait_seconds,
    )
    if not memory_ok:
        return _status(
            d, sym, status="failed", reason="memory_guard", started_at=started,
            detail=memory_detail,
        )

    try:
        trades, nbbo = load_symbol_day(d, sym)
    except TradePolicyMismatchError as exc:
        return _status(
            d, sym, status="failed", reason="policy_mismatch",
            started_at=started, detail=str(exc),
        )
    except FileNotFoundError:
        return _status(
            d, sym, status="skipped", reason="missing_parquet", started_at=started,
        )
    except MemoryError as exc:
        return _status(
            d, sym, status="failed", reason="memory_guard",
            started_at=started, detail=str(exc),
        )
    except (ValueError, TypeError) as exc:
        return _status(
            d, sym, status="failed", reason="dtype_error",
            started_at=started, detail=str(exc),
        )
    except Exception as exc:
        return _status(
            d, sym, status="failed", reason="compute_error",
            started_at=started, detail=str(exc),
        )

    n_trades_raw = len(trades)
    n_nbbo_raw = len(nbbo)
    try:
        trades = filter_valid_trades(trades)
        nbbo = filter_valid_quotes(nbbo)
        if nbbo.empty:
            return _status(
                d, sym, status="skipped", reason="insufficient_quotes",
                started_at=started, n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
            )
        trades = filter_trades_near_quotes(trades, nbbo)
        if trades.empty:
            return _status(
                d, sym, status="skipped", reason="empty_after_filter",
                started_at=started, n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
                n_nbbo=len(nbbo),
            )

        daily_row = compute_daily_features(trades, nbbo, sym, d)
        daily_df = _downcast_daily_features(daily_row)
        daily_path = shard_root / "daily" / ds / f"{safe}.parquet"
        _write_parquet_atomic(daily_df, daily_path)

        evt = build_event_panel(
            nbbo,
            trades,
            sym,
            d,
            sample_every_seconds=event_sample_every_seconds,
            max_rows=event_sample_per_symbol_day,
            sample_seed=_stable_sample_seed(d, sym, "event"),
        )
        if evt.empty:
            return _status(
                d, sym, status="skipped", reason="insufficient_quotes",
                started_at=started, n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
                n_trades=len(trades), n_nbbo=len(nbbo), daily_path=str(daily_path),
            )
        evt = _downcast_event_panel(_attach_per_fill_as(evt, nbbo))
        event_path = shard_root / "events" / ds / f"{safe}.parquet"
        _write_parquet_atomic(evt, event_path)

        asp = build_as_panel(
            nbbo,
            trades,
            sample_every_seconds=as_sample_every_seconds,
            max_rows=as_sample_per_symbol_day,
            sample_seed=_stable_sample_seed(d, sym, "as"),
        )
        as_path = ""
        if not asp.empty:
            asp = asp.copy()
            asp["symbol"] = sym
            asp["date"] = d
            asp = _downcast_as_panel(asp)
            as_file = shard_root / "as" / ds / f"{safe}.parquet"
            _write_parquet_atomic(asp, as_file)
            as_path = str(as_file)

        return _status(
            d,
            sym,
            status="ok",
            reason="ok",
            started_at=started,
            n_trades_raw=n_trades_raw,
            n_nbbo_raw=n_nbbo_raw,
            n_trades=len(trades),
            n_nbbo=len(nbbo),
            n_event_rows=len(evt),
            n_as_rows=len(asp) if not asp.empty else 0,
            event_path=str(event_path),
            daily_path=str(daily_path),
            as_path=as_path,
        )
    except MemoryError as exc:
        return _status(
            d, sym, status="failed", reason="memory_guard",
            started_at=started, detail=str(exc),
            n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
        )
    except (ValueError, TypeError) as exc:
        return _status(
            d, sym, status="failed", reason="dtype_error",
            started_at=started, detail=str(exc),
            n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
        )
    except Exception as exc:
        return _status(
            d, sym, status="failed", reason="compute_error",
            started_at=started, detail=str(exc),
            n_trades_raw=n_trades_raw, n_nbbo_raw=n_nbbo_raw,
        )
    finally:
        for name in ("trades", "nbbo", "evt", "asp", "daily_df"):
            if name in locals():
                del locals()[name]
        gc.collect()


def _stable_sample_seed(d: _dt.date, sym: str, label: str) -> int:
    payload = f"{d.isoformat()}|{sym}|{label}|{cfg.DEFAULT_SEED}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def run_calibration(
    symbols: list[str] | None,
    start: _dt.date = cfg.PRE_SAMPLE_START,
    end: _dt.date = cfg.PRE_SAMPLE_END,
    out_dir: Path = cfg.ARTIFACTS_DIR / "fill_model",
    max_days: int | None = None,
    workers: int = 1,
    universe: str | None = None,
    event_sample_per_symbol_day: int | None = None,
    as_sample_per_symbol_day: int | None = None,
    event_sample_every_seconds: int = cfg.REFRESH_SECONDS_DEFAULT,
    as_sample_every_seconds: int = 60,
    min_coverage: float = 0.95,
    fit_xgb_survival: bool = False,
    xgb_device: str = "cpu",
    ram_max_percent: float = 90.0,
    ram_min_available_gb: float = 2.0,
    memory_wait_seconds: int = 600,
    tod_required: bool = True,
) -> TieredFillModel:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_root = out_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    xgb_device_used = resolve_xgb_device(xgb_device)
    dates = _select_dates(start, end)
    if max_days is not None:
        dates = dates[:max_days]
    log.info("Calibration dates: %d (%s .. %s)", len(dates), dates[:1], dates[-1:] if dates else None)

    if symbols is None and universe:
        idx_panel = build_index_universe_panel(universe, dates)
        if idx_panel.empty:
            raise RuntimeError(f"Index universe {universe} produced no calibration symbol-days")
        pairs = [
            (pd.Timestamp(row.date).date(), str(row.symbol))
            for row in idx_panel[["date", "symbol"]].drop_duplicates().itertuples(index=False)
        ]
        symbols = sorted(idx_panel["symbol"].unique().tolist())
    elif symbols is None:
        if not dates:
            raise RuntimeError("No TAQ dates in pre-sample window")
        symbols = list_symbols(dates[0])[:50]
        pairs = [(d, sym) for d in dates for sym in symbols]
    else:
        pairs = [(d, sym) for d in dates for sym in symbols]
    pairs = _dedupe_pairs(pairs)
    symbols = sorted({sym for _, sym in pairs})
    log.info(
        "Symbol-day pairs after canonical de-duplication: %d  workers: %d",
        len(pairs), workers,
    )

    statuses: list[dict] = []

    if workers <= 1:
        for i, (d, sym) in enumerate(pairs, 1):
            result = _process_symbol_day(
                d,
                sym,
                shard_root,
                event_sample_per_symbol_day,
                as_sample_per_symbol_day,
                event_sample_every_seconds,
                as_sample_every_seconds,
                ram_max_percent,
                ram_min_available_gb,
                memory_wait_seconds,
            )
            statuses.append(result)
            if i % 38 == 0:
                log.info("  %d / %d pairs done", i, len(pairs))
    else:
        done = 0
        log_interval = max(1, len(pairs) // 10)
        pair_iter = iter(pairs)
        inflight = {}
        with AdaptivePool(max_workers=min(workers, 8)) as pool:
            def submit_next() -> bool:
                try:
                    d, sym = next(pair_iter)
                except StopIteration:
                    return False
                fut = pool.submit(
                    _process_symbol_day,
                    d,
                    sym,
                    shard_root,
                    event_sample_per_symbol_day,
                    as_sample_per_symbol_day,
                    event_sample_every_seconds,
                    as_sample_every_seconds,
                    ram_max_percent,
                    ram_min_available_gb,
                    memory_wait_seconds,
                )
                inflight[fut] = (d, sym)
                return True

            for _ in range(min(workers, len(pairs))):
                submit_next()
            while inflight:
                completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
                completed = {next(iter(completed))}
                for fut in completed:
                    d, sym = inflight.pop(fut)
                try:
                    statuses.append(fut.result())
                except Exception as exc:
                    d_sym = (d, sym)
                    log.warning("Calibration pair %s failed/timed out: %s — skipping", d_sym, exc)
                done += 1
                if done % log_interval == 0:
                    log.info("  %d / %d pairs done", done, len(pairs))
                submit_next()
                gc.collect()

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(out_dir / "calibration_status.csv", index=False)
    skips = status_df[status_df["status"] != "ok"].copy() if not status_df.empty else status_df
    skips.to_csv(out_dir / "calibration_skips.csv", index=False)
    ok_df = status_df[status_df["status"] == "ok"].copy()
    critical_df = status_df[status_df["reason"].isin(CRITICAL_REASONS)].copy()
    coverage = float(len(ok_df) / max(len(pairs), 1))
    preliminary_manifest = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe": universe,
        "symbols": symbols,
        "n_dates": len(dates),
        "n_symbols": len(symbols or []),
        "n_pairs": len(pairs),
        "n_ok_pairs": int(len(ok_df)),
        "n_skipped_pairs": int((status_df["status"] == "skipped").sum()) if not status_df.empty else 0,
        "n_failed_pairs": int((status_df["status"] == "failed").sum()) if not status_df.empty else 0,
        "coverage": coverage,
        "min_coverage": float(min_coverage),
        "critical_reasons": sorted(CRITICAL_REASONS),
        "allowed_skip_reasons": sorted(ALLOWED_SKIP_REASONS),
        "feature_policy": cfg.FEATURE_POLICY_VERSION,
        "trade_policy": cfg.TRADE_CONDITION_POLICY_VERSION,
        "status": "pending_fit",
        "event_sample_per_symbol_day": event_sample_per_symbol_day,
        "as_sample_per_symbol_day": as_sample_per_symbol_day,
        "event_sample_every_seconds": event_sample_every_seconds,
        "as_sample_every_seconds": as_sample_every_seconds,
        "workers": workers,
        "fit_xgb_survival": bool(fit_xgb_survival),
        "xgb_survival_status": "requested_pending" if fit_xgb_survival else "not_requested",
        "xgb_device_requested": xgb_device,
        "xgb_device_used": xgb_device_used,
        "xgb_tiers": [],
        "missing_xgb_tiers": [],
        "xgb_training_rows": 0,
        "tod_required": bool(tod_required),
        "tod_status": "requested_pending" if tod_required else "optional_pending",
        "ram_max_percent": ram_max_percent,
        "ram_min_available_gb": ram_min_available_gb,
    }
    if not critical_df.empty or coverage < min_coverage:
        preliminary_manifest["status"] = "failed_qc"
        preliminary_manifest["n_critical_failures"] = int(len(critical_df))
        _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
        raise RuntimeError(
            "Calibration QC failed before fit: "
            f"coverage={coverage:.3f}, critical_failures={len(critical_df)}"
        )

    event_paths = ok_df["event_path"].dropna().astype(str).tolist()
    daily_paths = ok_df["daily_path"].dropna().astype(str).tolist()
    as_paths = [p for p in ok_df["as_path"].dropna().astype(str).tolist() if p]
    event_panel = _read_shards(event_paths)
    daily_features = _read_shards(daily_paths)
    if event_panel.empty:
        raise RuntimeError("No event panel built -- check preprocessed parquet paths.")
    event_panel = _clean_event_panel_for_fit(event_panel)
    log.info(
        "Calibration rows: events=%d daily_features=%d as_panels=%d",
        len(event_panel), len(daily_features), int(ok_df["n_as_rows"].sum()),
    )

    # --- Liquidity tiers ---------------------------------------------------
    tier_map = assign_liquidity_tiers(daily_features, n_tiers=3)
    tier_map.to_csv(out_dir / "symbol_tier_map.csv", index=False)

    # --- Fit Cox-PH per tier ----------------------------------------------
    model = TieredFillModel()
    model.fit_panel(event_panel, tier_map)
    expected_tiers = set(tier_map["tier"].dropna().astype(int).unique())
    missing_tiers = sorted(expected_tiers - set(model.models))
    if missing_tiers:
        preliminary_manifest["status"] = "failed_model_fit"
        preliminary_manifest["missing_cox_tiers"] = missing_tiers
        _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
        raise RuntimeError(f"Cox fit missing tiers: {missing_tiers}")
    model.save(out_dir)

    # --- Kaplan-Meier fill model (non-parametric robustness spec) ----------
    # Produces km_tier_*.pkl + km_symbol_tier_map.csv so that
    # fill_specification="km" runs can load artifacts via validate_run.
    try:
        km_model = TieredKMFillModel()
        km_model.fit_panel(event_panel, tier_map)
        km_missing_tiers = sorted(expected_tiers - set(km_model.models))
        if km_missing_tiers:
            preliminary_manifest["status"] = "failed_km_model_fit"
            preliminary_manifest["km_status"] = "failed"
            preliminary_manifest["missing_km_tiers"] = km_missing_tiers
            _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
            raise RuntimeError(f"KM fit missing tiers: {km_missing_tiers}")
        km_model.save(out_dir)
        preliminary_manifest["km_status"] = "complete"
        preliminary_manifest["km_tiers"] = sorted(int(t) for t in km_model.models)
        preliminary_manifest["missing_km_tiers"] = km_missing_tiers
        log.info("Kaplan-Meier fill model saved to %s", out_dir)
    except Exception as e:
        if preliminary_manifest.get("status") != "failed_km_model_fit":
            preliminary_manifest["status"] = "failed_km_model_fit"
        preliminary_manifest["km_status"] = "failed"
        preliminary_manifest["km_error"] = str(e)
        _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
        log.error("Kaplan-Meier fill model training failed: %s", e, exc_info=True)
        raise RuntimeError(f"Kaplan-Meier fill model fit failed: {e}") from e

    # --- Optional XGBoost survival (parallel to Cox) -----------------------
    xgb_model = None
    if fit_xgb_survival:
        try:
            xgb_model = TieredXGBFillModel()
            xgb_model.fit_panel(
                event_panel,
                tier_map,
                strict=True,
                xgb_device=xgb_device_used,
                random_state=cfg.DEFAULT_SEED,
            )
            xgb_missing_tiers = sorted(expected_tiers - set(xgb_model.models))
            if xgb_missing_tiers:
                raise RuntimeError(f"XGB fit missing tiers: {xgb_missing_tiers}")
            xgb_model.save(out_dir)
            preliminary_manifest["xgb_survival_status"] = "complete"
            preliminary_manifest["xgb_tiers"] = sorted(int(t) for t in xgb_model.models)
            preliminary_manifest["missing_xgb_tiers"] = xgb_missing_tiers
            preliminary_manifest["xgb_training_rows"] = int(len(event_panel))
            log.info("XGBoost survival model saved to %s", out_dir)
        except Exception as e:
            preliminary_manifest["status"] = "failed_xgb_model_fit"
            preliminary_manifest["xgb_survival_status"] = "failed"
            preliminary_manifest["xgb_error"] = str(e)
            _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
            log.error("XGBoost survival training failed: %s", e, exc_info=True)
            raise RuntimeError(f"XGBoost survival fit failed: {e}") from e
    else:
        log.info("XGBoost survival model skipped; pass --fit-xgb-survival to enable")

    # --- Fit TOD Schedule (XGBoost regressor: state -> expected AS) --------
    try:
        tod = TODSchedule()
        tod.calibrate(event_panel, xgb_device=xgb_device_used, random_state=cfg.DEFAULT_SEED)
        tod.save(out_dir)
        preliminary_manifest["tod_status"] = "complete"
        log.info("TODSchedule fitted and saved to %s", out_dir)
    except (ValueError, ImportError, RuntimeError) as e:
        _record_tod_failure(
            preliminary_manifest,
            out_dir / "calibration_manifest.json",
            e,
            tod_required=tod_required,
        )
        log.warning("TODSchedule training skipped (optional): %s", e)
    except Exception as e:
        log.error("TODSchedule training failed unexpectedly: %s", e, exc_info=True)
        raise

    # --- Validation -------------------------------------------------------
    reports = validate_tiered_model(model, event_panel, horizon_seconds=cfg.FILL_MODEL_HORIZON_SECONDS)
    pd.DataFrame([
        {"tier": t, **vars(r)} for t, r in reports.items()
    ]).to_csv(out_dir / "validation.csv", index=False)

    # --- Cox validation gate ----------------------------------------------
    # Reject a degenerate Cox fit before any run can consume it. Mirrors the
    # per-fit baseline gate in CoxFillModel.fit but at the calibration level, so
    # a collapsed-baseline model can never reach status="complete".
    failed_validation = _fill_validation_failures(reports)
    if failed_validation:
        preliminary_manifest["status"] = "failed_model_validation"
        preliminary_manifest["failed_model_validation"] = {
            str(t): rs for t, rs in failed_validation.items()
        }
        _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
        log.error("Cox model validation gate failed: %s", failed_validation)
        raise RuntimeError(f"Cox model validation failed: {failed_validation}")

    # --- KM validation gate ------------------------------------------------
    km_reports = validate_tiered_model(
        km_model, event_panel, horizon_seconds=cfg.FILL_MODEL_HORIZON_SECONDS,
    )
    pd.DataFrame([
        {"tier": t, **vars(r)} for t, r in km_reports.items()
    ]).to_csv(out_dir / "km_validation.csv", index=False)
    km_failed = _level_validation_failures(km_reports)
    if km_failed:
        preliminary_manifest["status"] = "failed_km_model_validation"
        preliminary_manifest["failed_km_validation"] = {
            str(t): rs for t, rs in km_failed.items()
        }
        _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
        log.error("KM model validation gate failed: %s", km_failed)
        raise RuntimeError(f"KM model validation failed: {km_failed}")

    # --- XGB validation gate (same checks; base_fill_s0 is NaN -> auto-skipped) -
    # XGB has no explicit baseline, so only AUC and mean-pred-vs-observed apply.
    if xgb_model is not None:
        xgb_reports = validate_tiered_model(
            xgb_model, event_panel, horizon_seconds=cfg.FILL_MODEL_HORIZON_SECONDS,
        )
        pd.DataFrame([
            {"tier": t, **vars(r)} for t, r in xgb_reports.items()
        ]).to_csv(out_dir / "xgb_validation.csv", index=False)
        xgb_failed = _fill_validation_failures(xgb_reports)
        if xgb_failed:
            preliminary_manifest["status"] = "failed_model_validation"
            preliminary_manifest["failed_xgb_validation"] = {
                str(t): rs for t, rs in xgb_failed.items()
            }
            _write_manifest(out_dir / "calibration_manifest.json", preliminary_manifest)
            log.error("XGB model validation gate failed: %s", xgb_failed)
            raise RuntimeError(f"XGB model validation failed: {xgb_failed}")

    # --- Glosten AS -------------------------------------------------------
    if as_paths:
        as_panel = _read_shards(as_paths)
        as_model = fit_glosten_as(as_panel)
        pd.DataFrame([vars(as_model)]).to_csv(out_dir / "glosten_as.csv", index=False)

    manifest = {
        **preliminary_manifest,
        "status": "complete",
        "n_critical_failures": 0,
        "n_event_rows": int(len(event_panel)),
        "n_daily_feature_rows": int(len(daily_features)),
        "n_as_rows": int(ok_df["n_as_rows"].sum()),
    }
    _write_manifest(out_dir / "calibration_manifest.json", manifest)

    log.info("Calibration artifacts written to %s", out_dir)
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default=None,
                   help="Point-in-time index universe; ignored when --symbols is set")
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.PRE_SAMPLE_START)
    p.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.PRE_SAMPLE_END)
    p.add_argument("--out", type=Path, default=cfg.ARTIFACTS_DIR / "fill_model")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for symbol-day loading (default: 1; try 2 after smoke)")
    p.add_argument("--event-sample-per-symbol-day", type=int, default=None,
                   help="Deterministically sample at most N fill events per symbol-day")
    p.add_argument("--as-sample-per-symbol-day", type=int, default=None,
                   help="Deterministically sample at most N adverse-selection rows per symbol-day")
    p.add_argument("--event-sample-every-seconds", type=int,
                   default=cfg.REFRESH_SECONDS_DEFAULT,
                   help="Sampling grid for fill-event calibration candidates")
    p.add_argument("--as-sample-every-seconds", type=int, default=60,
                   help="Sampling grid for Glosten adverse-selection candidates")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="Minimum ok symbol-day share required before fitting")
    p.add_argument("--fit-xgb-survival", action="store_true",
                   help="Also fit optional XGBoost survival fill model")
    p.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="XGBoost device for survival/TOD models (default: cpu)")
    p.add_argument("--no-tod-required", action="store_true",
                   help="Allow calibration to complete without TOD artifacts; downstream runs must exclude S4_TOD.")
    p.add_argument("--ram-max-percent", type=float, default=90.0,
                   help="Pause/skip new symbol-day work above this system RAM percent")
    p.add_argument("--ram-min-available-gb", type=float, default=2.0,
                   help="Pause/skip new symbol-day work below this available RAM")
    p.add_argument("--memory-wait-seconds", type=int, default=600,
                   help="Maximum seconds a symbol-day waits for RAM headroom")
    args = p.parse_args()
    run_calibration(
        args.symbols, args.start, args.end, args.out, args.max_days,
        args.workers, args.universe,
        args.event_sample_per_symbol_day, args.as_sample_per_symbol_day,
        args.event_sample_every_seconds, args.as_sample_every_seconds,
        args.min_coverage, args.fit_xgb_survival, args.xgb_device,
        args.ram_max_percent, args.ram_min_available_gb,
        args.memory_wait_seconds, not args.no_tod_required,
    )


if __name__ == "__main__":
    main()
