"""
adaptive_pool.py -- Resource-aware parallel executor.

Replaces a fixed ProcessPoolExecutor with a submission loop that gates each
new task on current CPU, RAM, and GPU utilization.  Workers start immediately
when resources are available; submission pauses when any threshold is exceeded.

Hard cap: 8 concurrent workers (leaves headroom for interactive thesis work).

Usage
-----
    from concurrent.futures import as_completed
    from analysis.utils.adaptive_pool import AdaptivePool

    with AdaptivePool() as pool:
        futs = {pool.submit(fn, *args): args for args in work_items}
        for fut in as_completed(futs):
            result = fut.result()

The pool's ``initializer`` / ``initargs`` parameters mirror ProcessPoolExecutor
and are forwarded to the underlying pool on first use.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable

import psutil

log = logging.getLogger(__name__)


def _env_fraction(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r; using %.2f", name, raw, default)
        return default
    if val > 1.0:
        val = val / 100.0
    return min(1.0, max(0.01, val))

# ---------------------------------------------------------------------------
# GPU monitoring (optional — degrades gracefully if pynvml absent or no GPU)
# ---------------------------------------------------------------------------

_nvml_handle = None
_nvml_ok = False
_nvml_lock = threading.Lock()


def _init_nvml() -> None:
    global _nvml_handle, _nvml_ok
    with _nvml_lock:
        if _nvml_ok:
            return  # already initialised by another thread
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_ok = True
        except Exception:
            _nvml_ok = False


def _gpu_vram_fraction() -> float:
    """Return GPU VRAM used / total as a float in [0, 1]. Returns 0 if unavailable."""
    if not _nvml_ok:
        return 0.0
    try:
        import pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        return info.used / max(info.total, 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Resource check
# ---------------------------------------------------------------------------

def _resources_available(
    cpu_max: float,
    ram_max: float,
    gpu_max: float,
) -> bool:
    """Return True if all monitored resources are below their thresholds."""
    cpu = psutil.cpu_percent(interval=0.01) / 100.0  # 10ms sample — was 100ms
    ram = psutil.virtual_memory().percent / 100.0
    gpu = _gpu_vram_fraction()
    ok = cpu < cpu_max and ram < ram_max and gpu < gpu_max
    if not ok:
        log.debug(
            "Resource check: CPU=%.0f%% RAM=%.0f%% GPU=%.0f%% — waiting",
            cpu * 100, ram * 100, gpu * 100,
        )
    return ok


# ---------------------------------------------------------------------------
# AdaptivePool
# ---------------------------------------------------------------------------

class AdaptivePool:
    """ProcessPoolExecutor wrapper that throttles submission based on system load.

    Parameters
    ----------
    cpu_max : float
        Maximum CPU utilization fraction before pausing new submissions (default 0.80).
    ram_max : float
        Maximum RAM utilization fraction before pausing (default 0.85).
    gpu_max : float
        Maximum GPU VRAM fraction before pausing (default 0.90).
        Ignored if pynvml is unavailable or no GPU is present.
    poll_interval : float
        Seconds to wait between resource re-checks when throttled (default 2.0).
    max_workers : int
    Hard cap on concurrent workers (default 8).  Set
    ``THESIS_POOL_BACKEND=thread`` to use threads instead of processes on
    restricted Windows sessions where multiprocessing pipes are unavailable.
    initializer : callable | None
        Forwarded to ProcessPoolExecutor.
    initargs : tuple
        Forwarded to ProcessPoolExecutor.
    """

    def __init__(
        self,
        *,
        cpu_max: float | None = None,
        ram_max: float | None = None,
        gpu_max: float | None = None,
        poll_interval: float = 2.0,
        max_workers: int = 8,
        max_in_flight: int | None = None,
        submit_timeout_seconds: float | None = None,
        initializer: Callable | None = None,
        initargs: tuple = (),
    ) -> None:
        _init_nvml()
        self.cpu_max = _env_fraction("THESIS_POOL_CPU_MAX", 0.80) if cpu_max is None else cpu_max
        self.ram_max = _env_fraction("THESIS_POOL_RAM_MAX", 0.85) if ram_max is None else ram_max
        self.gpu_max = _env_fraction("THESIS_POOL_GPU_MAX", 0.90) if gpu_max is None else gpu_max
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self.max_in_flight = max(
            max_workers,
            int(max_in_flight if max_in_flight is not None else max_workers * 2),
        )
        self.submit_timeout_seconds = submit_timeout_seconds
        self._initializer = initializer
        self._initargs = initargs
        self.backend = os.environ.get("THESIS_POOL_BACKEND", "process").strip().lower()
        self._pool: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        self._sem = threading.BoundedSemaphore(self.max_in_flight)
        self._process_submit_ok = False

    def _make_pool(self) -> ProcessPoolExecutor | ThreadPoolExecutor:
        executor_cls = ThreadPoolExecutor if self.backend == "thread" else ProcessPoolExecutor
        return executor_cls(
            max_workers=self.max_workers,
            initializer=self._initializer,
            initargs=self._initargs,
        )

    def _fallback_to_threads(self, exc: BaseException) -> None:
        """Restricted Windows sessions cannot create multiprocessing pipes
        (PermissionError WinError 5); degrade to threads instead of dying."""
        log.warning(
            "Process pool unavailable (%s: %s); falling back to thread "
            "backend with %d workers. Set THESIS_POOL_BACKEND=thread to "
            "silence this fallback.",
            type(exc).__name__, exc, self.max_workers,
        )
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False)
            except Exception:
                pass
            self._pool = None
        self.backend = "thread"

    def _ensure_pool(self) -> ProcessPoolExecutor | ThreadPoolExecutor:
        if self._pool is None:
            try:
                self._pool = self._make_pool()
            except (PermissionError, OSError) as exc:
                if self.backend == "thread":
                    raise
                self._fallback_to_threads(exc)
                self._pool = self._make_pool()
        return self._pool

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """Wait until resources are available, then submit the task."""
        deadline = (
            time.monotonic() + float(self.submit_timeout_seconds)
            if self.submit_timeout_seconds is not None
            else None
        )
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    "AdaptivePool.submit timed out while waiting for a worker "
                    "slot or available resources"
                )
            acquired = self._sem.acquire(blocking=False)
            if acquired:
                if _resources_available(self.cpu_max, self.ram_max, self.gpu_max):
                    break  # slot acquired and resources free — submit
                else:
                    self._sem.release()  # return the slot, resources too busy
            time.sleep(self.poll_interval)

        pool = self._ensure_pool()

        def _release_on_done(fut: Future) -> None:
            self._sem.release()

        try:
            future = pool.submit(fn, *args, **kwargs)
        except (PermissionError, OSError) as exc:
            # Some platforms defer pipe creation until the first submit.
            if self.backend == "thread" or self._process_submit_ok:
                self._sem.release()
                raise
            self._fallback_to_threads(exc)
            try:
                future = self._ensure_pool().submit(fn, *args, **kwargs)
            except Exception:
                self._sem.release()
                raise
        except Exception:
            self._sem.release()
            raise
        if self.backend != "thread":
            self._process_submit_ok = True
        future.add_done_callback(_release_on_done)
        return future

    def map(
        self,
        fn: Callable,
        iterables: Iterable,
        *,
        timeout: float | None = None,
        chunksize: int = 1,
    ):
        """Submit all items with resource gating, yield results in order."""
        futs = [self.submit(fn, item) for item in iterables]
        for fut in futs:
            yield fut.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None

    def __enter__(self) -> "AdaptivePool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown(wait=True)
