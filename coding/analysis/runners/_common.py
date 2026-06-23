"""
_common.py -- Gemeinsame Build-Logik fuer die Hypothese-Runner.

Teilt die Schritte
    1. Lade Universum + Symbol-Tier-Map
    2. Lade Fill-Modell + Glosten-AS
    3. Baue Parent-Orders pro Symbol-Tag
    4. Rufe simulation.engine.simulate_symbol_day
    5. Attachen Metrik-Spalten
    6. Persistiere Long-Format Parquet

zwischen H1/H2/H3.

Zusaetzlich: ``rolling_window_panel`` fuer ueberlappende 6-Monats-Fenster.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from ..utils.adaptive_pool import AdaptivePool
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data import trade_conditions as tc
from ..data.taq_loader import (
    ensure_trade_qc_policy, extract_closing_auction_details, filter_valid_trades,
    list_dates, load_trades, trades_parquet_path,
)
from ..data.index_universe import build_index_universe_panel
from ..data.adv_universe import build_universe_panel
from ..fill_model.cox_ph import TieredFillModel
from ..metrics.alpha import attach_alpha_columns, attach_moc_differential_columns
from ..simulation.engine import simulate_symbol_day
from ..simulation.parent_orders import (
    build_parent_orders, rolling_expected_vc, same_day_vc_fallback,
)
from ..utils.symbols import canonical_symbol, expand_symbol_to_tier

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker model cache (initialised once per worker process)
# ---------------------------------------------------------------------------

_worker_model: TieredFillModel | None = None
_worker_tod = None  # TODSchedule, loaded once per worker if S4_TOD is in strategy set
_worker_value_model = None  # SideTieredXGBValueModel for optional S5


def _load_tod_schedule(artifacts_dir: Path, *, required: bool = False):
    from ..fill_model.tod_schedule import TODSchedule

    tod_path = Path(artifacts_dir) / "tod_schedule_xgb.ubj"
    if not tod_path.exists():
        if required:
            raise FileNotFoundError(
                f"S4_TOD requested but TOD schedule artifact is missing: {tod_path}"
            )
        return None
    tod = TODSchedule.load(artifacts_dir)
    if required and not tod.fitted:
        raise RuntimeError(f"S4_TOD artifact did not load a fitted model: {tod_path}")
    return tod


def _load_value_model(artifacts_dir: Path, *, required: bool = False):
    from ..fill_model.value_model import SideTieredXGBValueModel, VALUE_MODEL_MANIFEST

    manifest_path = Path(artifacts_dir) / VALUE_MODEL_MANIFEST
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError(
                f"S5_VALUE_AWARE_XGB requested but value model artifact is missing: {manifest_path}"
            )
        return None
    try:
        return SideTieredXGBValueModel.load(artifacts_dir)
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"S5_VALUE_AWARE_XGB requested but value model artifact is invalid: {exc}"
            ) from exc
        log.warning("Ignoring optional S5 value model artifact in %s: %s", artifacts_dir, exc)
        return None


def _worker_init(artifacts_dir: Path, fill_spec: str = "cox") -> None:
    global _worker_model, _worker_tod, _worker_value_model
    # Cap numpy/BLAS internal threads to 1 per worker to prevent CPU over-subscription
    # when 8 workers run simultaneously on 8 physical cores.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    if fill_spec == "xgb":
        from ..fill_model.xgb_survival import TieredXGBFillModel
        _worker_model = TieredXGBFillModel.load(artifacts_dir)
    elif fill_spec == "cox":
        _worker_model = TieredFillModel.load(artifacts_dir)
    elif fill_spec == "km":
        from ..fill_model.kaplan_meier import TieredKMFillModel
        _worker_model = TieredKMFillModel.load(artifacts_dir)
    else:
        _worker_model = None
    _worker_tod = _load_tod_schedule(artifacts_dir, required=False)
    _worker_value_model = _load_value_model(artifacts_dir, required=False)


# ---------------------------------------------------------------------------
# Module-level worker function (must be at top level to be picklable)
# ---------------------------------------------------------------------------

def _simulate_one(
    d: _dt.date,
    sym: str,
    strategies: list[str],
    tier: int,
    expected_vc: float,
    as_bps_buy: float,
    as_bps_sell: float,
    delta_max_bps: dict,
    size_fractions: tuple[float, ...],
    fill_specification: str = "tape_replay_queue",
) -> pd.DataFrame | None:
    """Simulate one (date, symbol) pair. Runs inside a worker process.
    Data loading happens inside simulate_symbol_day — no duplicate I/O here.
    """
    global _worker_model

    parents = build_parent_orders(sym, d, expected_vc, size_fractions=size_fractions)
    if parents.empty:
        return None

    res = simulate_symbol_day(
        sym, d, parents, strategies,
        fill_model=_worker_model,
        delta_max_bps_by_tier=delta_max_bps,
        tier=tier,
        fill_specification=fill_specification,
        km_model=_worker_model if fill_specification == "km" else None,
        tod_schedule=_worker_tod,
        value_model=_worker_value_model,
    )
    if res.empty:
        return None

    return res.copy()


# ---------------------------------------------------------------------------
# Helper: load vc for one (date, symbol) pair — used by threaded vc_history
# ---------------------------------------------------------------------------

def _load_vc_one(d: _dt.date, sym: str) -> dict | None:
    path = trades_parquet_path(d, sym)
    if not path.exists():
        return None
    ensure_trade_qc_policy(d)

    start = pd.Timestamp.combine(d, cfg.RTH_CLOSE)
    end = pd.Timestamp.combine(d, cfg.CLOSING_AUCTION_SEARCH_END)
    columns = ["time", "sale_condition", "volume", "price", "correction"]
    try:
        trades = pd.read_parquet(
            path,
            columns=columns,
            filters=[("time", ">=", start), ("time", "<=", end)],
        )
    except Exception:
        trades = pd.read_parquet(path, columns=columns)
        if not trades.empty:
            t = trades["time"].dt.time
            trades = trades[
                (t >= cfg.RTH_CLOSE)
                & (t <= cfg.CLOSING_AUCTION_SEARCH_END)
            ]
    if trades.empty:
        return None

    if "correction" in trades.columns:
        trades = trades[tc.valid_correction_mask(trades["correction"])]
    if "sale_condition" in trades.columns:
        bad_mask = tc.bad_sale_condition_mask(trades["sale_condition"], "evaluation")
        trades = trades[~bad_mask]
    trades = trades[(trades["price"] > 0) & (trades["volume"] > 0)]
    if trades.empty:
        return None

    cond = trades["sale_condition"].astype(str).fillna("")
    close_trade_mask = tc.contains_condition(cond, tc.CLOSING_TRADE_CONDITIONS)
    official_close_mask = tc.contains_condition(cond, tc.OFFICIAL_CLOSE_CONDITIONS)
    close_trade_rows = int(close_trade_mask.sum())
    official_marker_rows = int(official_close_mask.sum())
    if close_trade_rows == 0 and official_marker_rows == 0:
        return None

    close_trade_volume = float(trades.loc[close_trade_mask, "volume"].sum())
    official_marker_volume = float(trades.loc[official_close_mask, "volume"].sum())
    official_marker_fallback_volume = (
        float(trades.loc[official_close_mask, "volume"].max())
        if official_marker_rows
        else 0.0
    )
    price_rows = trades.loc[
        official_close_mask if official_marker_rows else close_trade_mask
    ]
    price = float(price_rows.sort_values("time")["price"].iloc[-1])
    price_source = "official_marker" if official_marker_rows else "closing_trade"
    if close_trade_volume > 0:
        volume = close_trade_volume
        volume_source = "closing_trade"
    elif official_marker_fallback_volume > 0:
        volume = official_marker_fallback_volume
        volume_source = "official_marker_fallback"
    else:
        return None
    if volume <= 0 or not np.isfinite(price):
        return None

    return {
        "symbol": sym,
        "date": d,
        "vc_shares": volume,
        "vc_source": volume_source,
        "close_price_source": price_source,
        "close_trade_volume": close_trade_volume,
        "close_trade_rows": close_trade_rows,
        "official_close_marker_volume": official_marker_volume,
        "official_close_marker_rows": official_marker_rows,
        "official_close_marker_fallback_volume": official_marker_fallback_volume,
    }


def _eval_dates(start: _dt.date, end: _dt.date) -> list[_dt.date]:
    out: set[_dt.date] = set()
    for y in range(start.year, end.year + 1):
        for d in list_dates(y):
            if start <= d <= end and d not in cfg.EXCLUDED_EVAL_DATES:
                out.add(d)
    return sorted(out)


def _load_artifacts(artifacts_dir: Path, fill_spec: str = "cox"):
    tier_map = pd.read_csv(artifacts_dir / "symbol_tier_map.csv")
    if fill_spec == "xgb":
        from ..fill_model.xgb_survival import TieredXGBFillModel
        model = TieredXGBFillModel.load(artifacts_dir)
        expected_tiers = set(tier_map["tier"].dropna().astype(int).unique())
        missing_tiers = sorted(expected_tiers - set(model.models))
        if missing_tiers:
            raise FileNotFoundError(
                f"XGB artifacts missing tiers {missing_tiers} in {artifacts_dir}"
            )
    elif fill_spec == "cox":
        model = TieredFillModel.load(artifacts_dir)
    elif fill_spec == "km":
        from ..fill_model.kaplan_meier import TieredKMFillModel
        model = TieredKMFillModel.load(artifacts_dir)
    else:
        model = None
    glosten_path = artifacts_dir / "glosten_as.csv"
    glosten = pd.read_csv(glosten_path) if glosten_path.exists() else pd.DataFrame()
    return model, tier_map, glosten


def _vc_history(
    dates: list[_dt.date],
    symbols: Iterable[str],
    workers: int = 1,
) -> pd.DataFrame:
    """Load closing-auction volumes for all (date, symbol) pairs.

    Uses ThreadPoolExecutor when workers > 1 (I/O-bound, benefits from threads).
    """
    sym_list = list(symbols)
    pairs = [(d, sym) for d in dates for sym in sym_list]

    if workers <= 1:
        rows = [_load_vc_one(d, sym) for d, sym in pairs]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pair_iter = iter(pairs)
            inflight = {}

            def _submit_next() -> bool:
                try:
                    d, sym = next(pair_iter)
                except StopIteration:
                    return False
                inflight[pool.submit(_load_vc_one, d, sym)] = (d, sym)
                return True

            for _ in range(min(len(pairs), max(1, workers * 4))):
                _submit_next()
            while inflight:
                completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in completed:
                    inflight.pop(fut)
                    rows.append(fut.result())
                    _submit_next()

    return pd.DataFrame([r for r in rows if r is not None])


def validate_run(
    strategies: list[str],
    start: _dt.date,
    end: _dt.date,
    artifacts_dir: Path,
    symbols: list[str] | None = None,
    universe: str | None = None,
    fill_specification: str = "tape_replay_queue",
) -> bool:
    """Dry-run validation: check artifacts and data paths, print work plan, return True if OK.

    Does not load data or run any simulation. Exits with a clear error message if
    required artifacts are missing so the user can fix config before a long run.
    """
    ok = True

    # Check fill model artifacts
    manifest_path = artifacts_dir / "calibration_manifest.json"
    if not manifest_path.exists():
        log.error("DRY-RUN: Missing calibration_manifest.json in %s", artifacts_dir)
        ok = False
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                log.error("DRY-RUN: Calibration manifest is not complete: %s", manifest.get("status"))
                ok = False
            if manifest.get("feature_policy") != cfg.FEATURE_POLICY_VERSION:
                log.error(
                    "DRY-RUN: Calibration feature_policy mismatch: got %s expected %s",
                    manifest.get("feature_policy"),
                    cfg.FEATURE_POLICY_VERSION,
                )
                ok = False
        except Exception as exc:
            log.error("DRY-RUN: Could not read calibration_manifest.json in %s: %s", artifacts_dir, exc)
            ok = False

    tier_map_path = artifacts_dir / "symbol_tier_map.csv"
    tier_map = None
    if not tier_map_path.exists():
        log.error("DRY-RUN: Missing symbol_tier_map.csv in %s", artifacts_dir)
        ok = False
    else:
        try:
            tier_map = pd.read_csv(tier_map_path)
        except Exception as exc:
            log.error("DRY-RUN: Could not read symbol_tier_map.csv in %s: %s", artifacts_dir, exc)
            ok = False

    if fill_specification == "xgb":
        xgb_files = sorted(artifacts_dir.glob("xgb_tier_*.ubj"))
        if not xgb_files:
            log.error("DRY-RUN: No XGB model files found in %s", artifacts_dir)
            ok = False
        xgb_tiers: set[int] = set()
        for ubj in xgb_files:
            try:
                tier = int(ubj.stem.split("_")[2])
            except Exception:
                log.error("DRY-RUN: Invalid XGB model filename: %s", ubj.name)
                ok = False
                continue
            xgb_tiers.add(tier)
            breslow = artifacts_dir / f"xgb_tier_{tier}_breslow.pkl"
            if not breslow.exists():
                log.error("DRY-RUN: Missing XGB Breslow artifact: %s", breslow)
                ok = False
        if tier_map is not None and "tier" in tier_map.columns:
            expected_tiers = set(tier_map["tier"].dropna().astype(int).unique())
            missing_tiers = sorted(expected_tiers - xgb_tiers)
            if missing_tiers:
                log.error("DRY-RUN: XGB artifacts missing tiers: %s", missing_tiers)
                ok = False
    elif fill_specification == "cox":
        cox_files = sorted(artifacts_dir.glob("cox_tier_*.pkl"))
        if not cox_files:
            log.error("DRY-RUN: No Cox fill model files found in %s", artifacts_dir)
            ok = False
        elif tier_map is not None and "tier" in tier_map.columns:
            cox_tiers: set[int] = set()
            for pkl in cox_files:
                try:
                    cox_tiers.add(int(pkl.stem.split("_")[2]))
                except Exception:
                    log.error("DRY-RUN: Invalid Cox model filename: %s", pkl.name)
                    ok = False
            expected_tiers = set(tier_map["tier"].dropna().astype(int).unique())
            missing_tiers = sorted(expected_tiers - cox_tiers)
            if missing_tiers:
                log.error("DRY-RUN: Cox artifacts missing tiers: %s", missing_tiers)
                ok = False
    elif fill_specification == "km":
        km_files = sorted(artifacts_dir.glob("km_tier_*.pkl"))
        if not km_files:
            log.error("DRY-RUN: No Kaplan-Meier model files found in %s", artifacts_dir)
            ok = False
        elif tier_map is not None and "tier" in tier_map.columns:
            km_tiers: set[int] = set()
            for pkl in km_files:
                try:
                    km_tiers.add(int(pkl.stem.split("_")[2]))
                except Exception:
                    log.error("DRY-RUN: Invalid KM model filename: %s", pkl.name)
                    ok = False
            expected_tiers = set(tier_map["tier"].dropna().astype(int).unique())
            missing_tiers = sorted(expected_tiers - km_tiers)
            if missing_tiers:
                log.error("DRY-RUN: KM artifacts missing tiers: %s", missing_tiers)
                ok = False
        if not (artifacts_dir / "km_symbol_tier_map.csv").exists():
            log.error("DRY-RUN: Missing KM symbol-tier map: %s", artifacts_dir / "km_symbol_tier_map.csv")
            ok = False

    if "S4_TOD" in strategies:
        tod_path = artifacts_dir / "tod_schedule_xgb.ubj"
        if not tod_path.exists():
            log.error("DRY-RUN: Missing S4 TOD artifact: %s", tod_path)
            ok = False
    if "S5_VALUE_AWARE_XGB" in strategies:
        from ..fill_model.value_model import (
            VALUE_MODEL_MANIFEST,
            validate_value_model_manifest,
        )
        value_manifest = artifacts_dir / VALUE_MODEL_MANIFEST
        if not value_manifest.exists():
            log.error("DRY-RUN: Missing S5 value model artifact: %s", value_manifest)
            ok = False
        else:
            try:
                manifest = json.loads(value_manifest.read_text(encoding="utf-8"))
                errors = validate_value_model_manifest(manifest, artifacts_dir)
                for error in errors:
                    log.error("DRY-RUN: Invalid S5 value model artifact: %s", error)
                if errors:
                    ok = False
            except Exception as exc:
                log.error("DRY-RUN: Could not read S5 value model artifact %s: %s", value_manifest, exc)
                ok = False

    # Check TAQ data availability
    dates = _eval_dates(start, end)
    if not dates:
        log.error("DRY-RUN: No TAQ dates found between %s and %s — check DATA_ROOT", start, end)
        ok = False

    # Summarise planned work
    n_syms = len(symbols) if symbols else (f"index:{universe}" if universe else "auto")
    log.info(
        "DRY-RUN plan: %d dates × %s symbols × %d strategies  fill=%s",
        len(dates), n_syms, len(strategies), fill_specification,
    )
    if universe and symbols is None:
        try:
            idx_panel = build_index_universe_panel(universe, dates)
            log.info(
                "DRY-RUN universe %s: %d symbol-days, %d unique symbols",
                universe, len(idx_panel), idx_panel["symbol"].nunique(),
            )
            if idx_panel.empty:
                ok = False
                log.error("DRY-RUN: index universe %s produced no symbol-days", universe)
        except Exception as exc:
            ok = False
            log.error("DRY-RUN: index universe %s unavailable: %s", universe, exc)
    log.info("DRY-RUN strategies: %s", strategies)
    log.info("DRY-RUN artifacts_dir: %s  [%s]", artifacts_dir, "OK" if ok else "MISSING FILES")
    if ok:
        log.info("DRY-RUN: validation passed — ready to run.")
    else:
        log.error("DRY-RUN: validation FAILED — fix errors above before running.")
    return ok


def run_panel(
    strategies: list[str],
    start: _dt.date,
    end: _dt.date,
    artifacts_dir: Path,
    out_path: Path,
    symbols: list[str] | None = None,
    universe: str | None = None,
    max_dates: int | None = None,
    top_n: int = cfg.UNIVERSE_TOP_N_DEFAULT,
    *,
    pilot_mode: bool = False,
    workers: int = 1,
    fill_specification: str = "tape_replay_queue",
    size_fractions: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Fuehrt simulation.engine ueber ein (date x symbol) Panel und speichert Parquet.

    workers > 1: parallelises both the vc_history loading (threads) and the
    simulation loop (processes, with model loaded once per worker via initializer).
    """
    model, tier_map, glosten = _load_artifacts(artifacts_dir, fill_spec=fill_specification)
    tier_lookup = expand_symbol_to_tier(
        dict(zip(tier_map["symbol"], tier_map["tier"].astype(int)))
    )
    selected_sizes = tuple(size_fractions or cfg.PARENT_ORDER_SIZE_FRACTIONS)
    tod_schedule = _load_tod_schedule(artifacts_dir, required="S4_TOD" in strategies)

    as_bps_buy = as_bps_sell = 0.0
    if not glosten.empty:
        as_bps_buy = abs(float(glosten.get("beta_buy", [0]).iloc[0] or 0.0))
        as_bps_sell = abs(float(glosten.get("beta_sell", [0]).iloc[0] or 0.0))

    dates = _eval_dates(start, end)
    if max_dates is not None:
        dates = dates[:max_dates]

    universe_panel = None
    if symbols is None:
        if pilot_mode:
            symbols = list(cfg.PILOT_UNIVERSE)
        elif universe:
            universe_panel = build_index_universe_panel(universe, dates)
            if universe_panel.empty:
                log.warning("Index universe %s produced no symbol-days.", universe)
                return pd.DataFrame()
            universe_panel = universe_panel.copy()
            universe_panel["symbol"] = universe_panel["symbol"].map(canonical_symbol)
            universe_panel = universe_panel.drop_duplicates(["date", "symbol"])
            symbols = sorted(universe_panel["symbol"].unique().tolist())
        else:
            adv_universe = build_universe_panel(start, end, n=top_n)
            adv_universe = adv_universe.merge(tier_map[["symbol"]], on="symbol")
            symbols = sorted(adv_universe["symbol"].unique().tolist())
    symbols = sorted({canonical_symbol(s) for s in symbols})

    log.info("run_panel: %d dates x %d symbols x %d strategies  workers=%d  fill=%s",
             len(dates), len(symbols), len(strategies), workers, fill_specification)
    log.info("run_panel parent sizes: %s", selected_sizes)

    # --- Build expected-VC lookup (parallelised I/O) -----------------------
    if pilot_mode:
        vc_hist = _vc_history(dates, symbols, workers=workers)
        evc = same_day_vc_fallback(vc_hist)
    else:
        hist_end = dates[-1] if dates else end
        hist_dates = _eval_dates(start - _dt.timedelta(days=40), hist_end)
        log.info("Loading vc_history: %d dates x %d symbols...", len(hist_dates), len(symbols))
        vc_hist = _vc_history(hist_dates, symbols, workers=workers)
        evc = rolling_expected_vc(vc_hist)

    # Build fast (symbol, date) → expected_vc lookup dict
    evc_lookup: dict[tuple, float] = {
        (row.symbol, row.date): row.expected_vc
        for row in evc.itertuples(index=False)
        if not pd.isna(row.expected_vc)
    }
    log.info("evc_lookup: %d entries", len(evc_lookup))

    # --- Build work list ---------------------------------------------------
    if universe_panel is not None:
        syms_by_date = {
            pd.Timestamp(d).date(): sorted(g["symbol"].unique().tolist())
            for d, g in universe_panel.groupby("date")
        }
    else:
        syms_by_date = {d: list(symbols) for d in dates}

    work_items = []
    for d in dates:
        for sym in syms_by_date.get(d, []):
            tier = tier_lookup.get(sym)
            if tier is None:
                continue
            expected_vc = evc_lookup.get((sym, d))
            if expected_vc is None or expected_vc <= 0:
                continue
            work_items.append((d, sym, strategies, tier, float(expected_vc),
                               as_bps_buy, as_bps_sell, cfg.DELTA_MAX_BPS,
                               selected_sizes,
                               fill_specification))

    log.info("Simulation work items: %d", len(work_items))

    # --- Run simulation ----------------------------------------------------
    all_rows: list[pd.DataFrame] = []
    done = 0

    if workers <= 1:
        for item in work_items:
            d, sym = item[0], item[1]
            parents = build_parent_orders(
                sym, d, item[4], size_fractions=item[8],
            )
            if parents.empty:
                continue
            res = simulate_symbol_day(
                sym, d, parents, item[2],
                fill_model=model,
                delta_max_bps_by_tier=item[7],
                tier=item[3],
                fill_specification=item[9],
                km_model=model if item[9] == "km" else None,
                tod_schedule=tod_schedule,
            )
            if res.empty:
                continue
            all_rows.append(res)
            done += 1
            if done % len(symbols) == 0:
                log.info("  %d / %d work items done", done, len(work_items))
    else:
        with AdaptivePool(
            max_workers=min(workers, 8),
            max_in_flight=max(2, min(workers, 8) * 2),
            initializer=_worker_init,
            initargs=(artifacts_dir, fill_specification),
        ) as pool:
            item_iter = iter(work_items)
            inflight = {}

            def _submit_next() -> bool:
                try:
                    item = next(item_iter)
                except StopIteration:
                    return False
                inflight[pool.submit(_simulate_one, *item)] = item
                return True

            for _ in range(min(len(work_items), max(2, workers * 2))):
                _submit_next()

            while inflight:
                completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in completed:
                    item = inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as exc:
                        log.warning(
                            "Work item (%s, %s) failed: %s - skipping",
                            item[0], item[1], exc,
                        )
                        res = None
                    if res is not None:
                        all_rows.append(res)
                    done += 1
                    if done % max(1, min(250, len(symbols))) == 0:
                        log.info("  %d / %d work items done", done, len(work_items))
                    _submit_next()

    if not all_rows:
        log.warning("No simulation results.")
        return pd.DataFrame()

    panel = pd.concat(all_rows, ignore_index=True)
    observed_pairs = {
        (str(row.symbol), pd.Timestamp(row.date).date())
        for row in panel[["symbol", "date"]].drop_duplicates().itertuples(index=False)
    }
    expected_pairs = {(str(item[1]), item[0]) for item in work_items}
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        log.warning(
            "Panel missing %d/%d expected symbol-day result(s); first missing: %s",
            len(missing_pairs), len(expected_pairs), missing_pairs[:10],
        )
    panel = attach_alpha_columns(panel)
    panel = attach_moc_differential_columns(panel)
    panel["tier"] = panel["symbol"].map(tier_lookup)
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    panel["is_headline_size"] = np.isclose(
        panel["size_frac"].astype(float), cfg.PARENT_ORDER_PRIMARY_FRACTION,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info("Panel written: %s (rows=%d)", out_path, len(panel))
    return panel


def require_headline_panel(
    panel: pd.DataFrame,
    context: str,
    *,
    require_moc: bool = False,
) -> None:
    """Raise if a headline runner accidentally receives mixed parent sizes."""
    if panel.empty or "size_frac" not in panel.columns:
        raise ValueError(f"{context}: headline panel has no size_frac rows")
    sizes = sorted(panel["size_frac"].dropna().astype(float).unique().tolist())
    if len(sizes) != 1 or not np.isclose(sizes[0], cfg.PARENT_ORDER_PRIMARY_FRACTION):
        raise ValueError(
            f"{context}: headline panel must contain only "
            f"{cfg.PARENT_ORDER_PRIMARY_FRACTION:g}; got {sizes}"
        )
    if require_moc and "strategy" in panel.columns and "S0_MOC" not in set(panel["strategy"]):
        raise ValueError(f"{context}: headline panel has no S0_MOC rows")


# ---------------------------------------------------------------------------
# Rolling-Window-Panel (P4.1)
# ---------------------------------------------------------------------------

def rolling_window_panel(
    panel: pd.DataFrame,
    *,
    window_months: int = 6,
    step_months: int = 1,
    alpha_col: str = "net_alpha_bps",
) -> pd.DataFrame:
    if panel.empty or "date" not in panel.columns:
        return pd.DataFrame()

    from ..inference.clustering import mean_with_twoway_se

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.dropna(subset=["date", alpha_col])
    if p.empty:
        return pd.DataFrame()

    start = p["date"].min().normalize()
    end = p["date"].max().normalize()
    rows = []
    cur = start
    win = pd.DateOffset(months=window_months)
    step = pd.DateOffset(months=step_months)
    while cur + win <= end + pd.Timedelta(days=1):
        w_start = cur
        w_end = cur + win
        sub = p[(p["date"] >= w_start) & (p["date"] < w_end)]
        if not sub.empty:
            for strat, grp in sub.groupby("strategy"):
                m, se = mean_with_twoway_se(
                    grp[alpha_col], grp["symbol"], grp["date"],
                )
                rows.append({
                    "window_start": w_start.date(),
                    "window_end": w_end.date(),
                    "strategy": strat,
                    "mean_alpha": m,
                    "clustered_se": se,
                    "n": int(len(grp)),
                })
        cur = cur + step
    return pd.DataFrame(rows)
