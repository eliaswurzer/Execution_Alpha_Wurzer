"""Benchmark thread/process worker configurations on a small real-data panel."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import pandas as pd

from .. import config as cfg
from ..data.index_universe import build_index_universe_panel
from ..data.taq_loader import nbbo_parquet_path, trades_parquet_path
from ..utils.adaptive_pool import AdaptivePool
from ..utils.symbols import canonical_symbol, expand_symbol_to_tier
from . import _common

STRATEGIES = [
    "S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
    "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD",
]


def _panel_hash(frames: list[pd.DataFrame]) -> str:
    if not frames:
        return ""
    panel = pd.concat(frames, ignore_index=True)
    cols = [
        "order_id", "strategy", "qty_intended", "qty_filled_passive",
        "qty_filled_moc", "avg_fill_price", "fill_rate",
    ]
    panel = panel[cols].sort_values(["order_id", "strategy"]).reset_index(drop=True)
    values = pd.util.hash_pandas_object(panel, index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _monitor(stop: threading.Event, samples: list[tuple[float, float]]) -> None:
    try:
        import psutil
    except ImportError:
        return
    while not stop.wait(0.2):
        samples.append((
            psutil.cpu_percent(interval=None),
            psutil.virtual_memory().percent,
        ))


def _run_candidate(
    items: list[tuple],
    artifacts_dir: Path,
    *,
    backend: str,
    workers: int,
) -> dict:
    os.environ["THESIS_POOL_BACKEND"] = backend
    frames: list[pd.DataFrame] = []
    errors = 0
    samples: list[tuple[float, float]] = []
    backend_used = backend
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with AdaptivePool(
            max_workers=workers,
            max_in_flight=workers * 2,
            cpu_max=1.0,
            ram_max=0.95,
            initializer=_common._worker_init,
            initargs=(artifacts_dir, "tape_replay_queue"),
        ) as pool:
            backend_used = pool.backend
            item_iter = iter(items)
            inflight = {}

            def submit_next() -> bool:
                try:
                    item = next(item_iter)
                except StopIteration:
                    return False
                inflight[pool.submit(_common._simulate_one, *item)] = item[:2]
                return True

            for _ in range(min(len(items), workers * 2)):
                submit_next()
            while inflight:
                completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in completed:
                    inflight.pop(future)
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            frames.append(result)
                    except Exception:
                        errors += 1
                    submit_next()
    finally:
        stop.set()
        monitor.join(timeout=2)
    runtime = time.perf_counter() - started
    return {
        "requested_backend": backend,
        "backend": backend_used,
        "workers": workers,
        "runtime_seconds": runtime,
        "items": len(items),
        "result_frames": len(frames),
        "errors": errors,
        "max_cpu_percent": max((x[0] for x in samples), default=float("nan")),
        "max_ram_percent": max((x[1] for x in samples), default=float("nan")),
        "result_hash": _panel_hash(frames),
    }


def benchmark(
    artifacts_dir: Path,
    out_dir: Path,
    *,
    universe: str = "sp500",
    start: dt.date = dt.date(2018, 2, 1),
    days: int = 2,
    symbol_count: int = 50,
) -> dict:
    dates = _common._eval_dates(start, start + dt.timedelta(days=10))[:days]
    membership = build_index_universe_panel(
        universe, dates, expand_aliases=False,
    )
    symbols = sorted({
        canonical_symbol(symbol) for symbol in membership["symbol"]
    })[:symbol_count]
    tier_map = pd.read_csv(artifacts_dir / "symbol_tier_map.csv")
    tier_lookup = expand_symbol_to_tier(
        dict(zip(tier_map["symbol"], tier_map["tier"].astype(int)))
    )
    items = []
    for date in dates:
        active = set(
            membership.loc[membership["date"] == date, "symbol"].map(canonical_symbol)
        )
        for symbol in symbols:
            if symbol not in active or symbol not in tier_lookup:
                continue
            if not trades_parquet_path(date, symbol).exists():
                continue
            if not nbbo_parquet_path(date, symbol).exists():
                continue
            items.append((
                date, symbol, STRATEGIES, int(tier_lookup[symbol]),
                1_000_000.0, 0.0, 0.0, cfg.DELTA_MAX_BPS,
                (cfg.PARENT_ORDER_PRIMARY_FRACTION,), "tape_replay_queue",
            ))
    if not items:
        raise RuntimeError("Benchmark found no usable real-data work items")

    results = []
    for backend in ("thread", "process"):
        for workers in (2, 4, 6):
            results.append(_run_candidate(
                items, artifacts_dir, backend=backend, workers=workers,
            ))
    baseline_hash = next(
        (row["result_hash"] for row in results if row["errors"] == 0), "",
    )
    baseline_frames = next(
        (row["result_frames"] for row in results if row["errors"] == 0), 0,
    )
    for row in results:
        row["equivalent"] = bool(
            row["errors"] == 0
            and row["result_frames"] == baseline_frames
            and row["result_frames"] > 0
            and row["result_hash"] == baseline_hash
        )
    valid = [row for row in results if row["equivalent"]]
    if not valid:
        raise RuntimeError(f"No valid worker benchmark candidate: {results}")
    selected = min(valid, key=lambda row: row["runtime_seconds"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "worker_benchmark.csv", index=False)
    payload = {
        "selected_backend": selected["backend"],
        "selected_workers": selected["workers"],
        "sample_dates": [date.isoformat() for date in dates],
        "sample_symbols": symbols,
        "items": len(items),
        "selected_result": selected,
    }
    (out_dir / "worker_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2018, 2, 1))
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--symbols", type=int, default=50)
    args = parser.parse_args()
    result = benchmark(
        args.artifacts,
        args.out,
        universe=args.universe,
        start=args.start,
        days=args.days,
        symbol_count=args.symbols,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
