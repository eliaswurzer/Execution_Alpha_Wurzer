"""Conservative overnight S&P 500 H1 calibration + hypothesis run.

Writes all run artifacts under ``coding/artifacts`` so the script does not
overwrite the external production artifact folders.  External TAQ parquet and
Volume-DuckDB paths are read-only inputs.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from .. import config as cfg


REPO_ROOT = Path(__file__).resolve().parents[3]
CODING_ROOT = REPO_ROOT / "coding"
RUN_STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"sp500_h1_overnight_{RUN_STAMP}"
ARTIFACT_BASE = Path(os.environ.get(
    "THESIS_ARTIFACTS_DIR",
    str(cfg.ARTIFACTS_DIR),
))
ARTIFACT_ROOT = ARTIFACT_BASE / RUN_ID
FILL_MODEL_DIR = ARTIFACT_ROOT / "fill_model"
RUN_ROOT = ARTIFACT_ROOT / "runs"
LOG_DIR = ARTIFACT_ROOT / "logs"

TAQ_ROOT = Path(os.environ.get("THESIS_TAQ_PARQUET_2018", str(cfg.TAQ_PARQUET_DIR[2018])))
VOLUME_DB = Path(os.environ.get("THESIS_VOLUME_DB", str(cfg.VOLUME_DB_PATH)))
EXTERNAL_RUN_ROOT = Path(os.environ.get("THESIS_RUN_ROOT", str(cfg.RUN_ROOT)))
CALIBRATION_SOURCE = Path(os.environ.get(
    "THESIS_CALIBRATION_SOURCE",
    str(cfg.ARTIFACTS_DIR / "fill_model"),
))

PRE_START = "2018-01-02"
PRE_END = "2018-01-31"
EVAL_START = "2018-02-01"
EVAL_END = "2018-06-29"
CALIBRATION_WORKERS = 1
EVALUATION_WORKERS = 2
EVENT_SAMPLE_PER_SYMBOL_DAY = 48
AS_SAMPLE_PER_SYMBOL_DAY = 48
EVENT_SAMPLE_EVERY_SECONDS = 30
AS_SAMPLE_EVERY_SECONDS = 30
CALIBRATION_MIN_COVERAGE = 0.95
CALIBRATION_RAM_MAX_PERCENT = 90.0
CALIBRATION_RAM_MIN_AVAILABLE_GB = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_complete(path: Path) -> bool:
    manifest_path = path / "calibration_manifest.json"
    required = [
        "symbol_tier_map.csv",
        "glosten_as.csv",
        "tod_schedule_xgb.ubj",
        "tod_schedule_meta.pkl",
    ]
    if not manifest_path.exists() or any(not (path / name).exists() for name in required):
        return False
    if not list(path.glob("cox_tier_*.pkl")):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("feature_policy") == cfg.FEATURE_POLICY_VERSION
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _import_calibration(source: Path, destination: Path) -> None:
    if not _calibration_complete(source):
        raise RuntimeError(f"Calibration source is not complete: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        path for path in source.iterdir()
        if path.is_file() and (
            path.name.startswith("cox_tier_")
            or path.name in {
                "calibration_manifest.json", "calibration_skips.csv",
                "calibration_status.csv", "glosten_as.csv",
                "symbol_tier_map.csv", "tod_schedule_meta.pkl",
                "tod_schedule_xgb.ubj", "validation.csv",
            }
        )
    ]
    hashes = {}
    for source_file in files:
        target = destination / source_file.name
        shutil.copy2(source_file, target)
        hashes[source_file.name] = _sha256(target)
    provenance = {
        "source": str(source),
        "imported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "evaluation_git_commit": _git_commit(),
        "source_manifest_sha256": _sha256(source / "calibration_manifest.json"),
        "configuration": {
            "taq_root": str(TAQ_ROOT),
            "volume_db": str(VOLUME_DB),
            "calibration_start": PRE_START,
            "calibration_end": PRE_END,
            "evaluation_start": EVAL_START,
            "evaluation_end": EVAL_END,
            "fill_specification": "tape_replay_queue",
            "feature_policy": cfg.FEATURE_POLICY_VERSION,
        },
        "files": hashes,
    }
    (destination / "calibration_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8",
    )


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODING_ROOT)
    env["THESIS_RUN_ROOT"] = str(EXTERNAL_RUN_ROOT)
    env["THESIS_TAQ_PARQUET_2018"] = str(TAQ_ROOT)
    env["THESIS_VOLUME_DB"] = str(VOLUME_DB)
    env["THESIS_ARTIFACTS_DIR"] = str(ARTIFACT_ROOT)
    env["THESIS_TRADE_QC_POLICY_CHECK"] = "enforce"
    env["THESIS_POOL_CPU_MAX"] = "0.80"
    env["THESIS_POOL_RAM_MAX"] = "0.85"
    env["THESIS_POOL_BACKEND"] = os.environ.get(
        "THESIS_SELECTED_POOL_BACKEND", "process",
    )
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def _write_status(status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "artifact_root": str(ARTIFACT_ROOT),
        **extra,
    }
    (ARTIFACT_ROOT / "run_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def _resource_monitor(stop: threading.Event) -> None:
    try:
        import psutil
    except ImportError:
        return
    path = ARTIFACT_ROOT / "resource_usage.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "cpu_percent", "ram_percent", "ram_available_gb"],
        )
        writer.writeheader()
        while not stop.is_set():
            vm = psutil.virtual_memory()
            writer.writerow({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "cpu_percent": psutil.cpu_percent(interval=1.0),
                "ram_percent": vm.percent,
                "ram_available_gb": round(vm.available / 1024**3, 3),
            })
            f.flush()
            stop.wait(59.0)


def _run_step(name: str, args: list[str]) -> None:
    log_path = LOG_DIR / f"{name}.log"
    _write_status("running", current_step=name, log=str(log_path))
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(" ".join(args) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            args,
            cwd=str(REPO_ROOT),
            env=_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-B", "-m", module, *args]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _volume_summary() -> dict:
    try:
        import duckdb
        con = duckdb.connect(str(VOLUME_DB), read_only=True)
        try:
            row = con.execute("""
                select count(*) as n_rows,
                       count(distinct Ticker) as n_symbols,
                       min(Date) as min_date,
                       max(Date) as max_date,
                       sum(case when Close_Auction_Val > 0 then 1 else 0 end) as close_positive_rows,
                       sum(case when Close_Auction_Val = 0 then 1 else 0 end) as close_zero_rows,
                       round(100 * sum(Close_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as close_share_pct,
                       round(100 * sum(Official_Close_Marker_Val) / nullif(sum(Total_Daily_Val), 0), 4) as marker_share_pct
                  from daily_volume
            """).fetchdf().iloc[0].to_dict()
            skips = con.execute(
                "select count(*) as n from daily_volume_skipped"
            ).fetchone()[0]
            row["volume_skips"] = int(skips)
            return row
        finally:
            con.close()
    except Exception as exc:
        return {"error": str(exc)}


def _write_executive_summary(status: str, error: str | None = None) -> None:
    run_dir = RUN_ROOT / RUN_ID
    cal_manifest = {}
    cal_path = FILL_MODEL_DIR / "calibration_manifest.json"
    if cal_path.exists():
        cal_manifest = json.loads(cal_path.read_text(encoding="utf-8"))
    validation = _safe_read_csv(FILL_MODEL_DIR / "validation.csv")
    glosten = _safe_read_csv(FILL_MODEL_DIR / "glosten_as.csv")
    h1_summary = _safe_read_csv(run_dir / "hypotheses" / "h1" / "h1_tev.csv")
    h1_primary = _safe_read_csv(run_dir / "hypotheses" / "h1" / "h1_primary_ttest.csv")
    h2 = _safe_read_csv(run_dir / "hypotheses" / "h2" / "h2_pooled.csv")
    h3 = _safe_read_csv(run_dir / "hypotheses" / "h3" / "h3_raear.csv")
    resource = _safe_read_csv(ARTIFACT_ROOT / "resource_usage.csv")
    volume = _volume_summary()
    worker_selection = {}
    selection_path = ARTIFACT_ROOT / "benchmark" / "worker_selection.json"
    if selection_path.exists():
        worker_selection = json.loads(selection_path.read_text(encoding="utf-8"))

    lines = [
        f"# Executive Summary: {RUN_ID}",
        "",
        f"- Status: `{status}`",
        f"- Artifact root: `{ARTIFACT_ROOT}`",
        f"- Run directory: `{run_dir}`",
        f"- Sample: calibration `{PRE_START}` to `{PRE_END}`, evaluation `{EVAL_START}` to `{EVAL_END}`",
        f"- Universe: point-in-time `sp500` membership",
        f"- Fill specification: `tape_replay_queue`",
        f"- Calibration workers: `{CALIBRATION_WORKERS}`",
        f"- Evaluation workers: `{worker_selection.get('selected_workers', EVALUATION_WORKERS)}`",
        f"- Evaluation backend: `{worker_selection.get('selected_backend', 'process')}`",
        "",
    ]
    if error:
        lines += ["## Error", "", error, ""]

    lines += [
        "## Calibration",
        "",
        f"- Status: `{cal_manifest.get('status', 'n/a')}`",
        f"- Pairs: `{cal_manifest.get('n_pairs', 'n/a')}`",
        f"- OK pairs: `{cal_manifest.get('n_ok_pairs', 'n/a')}`",
        f"- Coverage: `{cal_manifest.get('coverage', 'n/a')}`",
        f"- Event rows used: `{cal_manifest.get('n_event_rows', 'n/a')}`",
        f"- Daily feature rows: `{cal_manifest.get('n_daily_feature_rows', 'n/a')}`",
        f"- AS rows used: `{cal_manifest.get('n_as_rows', 'n/a')}`",
        f"- Event sample per symbol-day: `{cal_manifest.get('event_sample_per_symbol_day', 'n/a')}`",
        f"- Event grid seconds: `{cal_manifest.get('event_sample_every_seconds', 'n/a')}`",
        f"- AS grid seconds: `{cal_manifest.get('as_sample_every_seconds', 'n/a')}`",
        f"- XGB survival fitted: `{cal_manifest.get('fit_xgb_survival', 'n/a')}`",
        "",
    ]
    if not validation.empty:
        lines += ["### Fill-Model Validation", "", validation.to_markdown(index=False), ""]
    if not glosten.empty:
        lines += ["### Glosten AS", "", glosten.to_markdown(index=False), ""]

    lines += ["## Volume DB", "", pd.DataFrame([volume]).to_markdown(index=False), ""]

    if not h1_summary.empty:
        lines += ["## H1 Tracking Error / Alpha Summary", "", h1_summary.to_markdown(index=False), ""]
    if not h1_primary.empty:
        lines += ["## H1 Primary Test", "", h1_primary.to_markdown(index=False), ""]
    if not h2.empty:
        lines += ["## H2 Signal Decomposition", "", h2.to_markdown(index=False), ""]
    if not h3.empty:
        lines += ["## H3 RAEAR", "", h3.to_markdown(index=False), ""]
    if not resource.empty:
        lines += [
            "## Resource Envelope",
            "",
            f"- Max CPU observed: `{resource['cpu_percent'].max():.1f}%`",
            f"- Max RAM observed: `{resource['ram_percent'].max():.1f}%`",
            f"- Min RAM available: `{resource['ram_available_gb'].min():.2f} GB`",
            "",
        ]

    (ARTIFACT_ROOT / "EXECUTIVE_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def main() -> None:
    if (
        "iclouddrive" in str(ARTIFACT_ROOT).lower()
        and os.environ.get("THESIS_ALLOW_ICLOUD_ARTIFACTS") != "1"
    ):
        raise RuntimeError(
            "Refusing long-run artifacts inside a cloud-synced folder. Set "
            "THESIS_ARTIFACTS_DIR to a local path or explicitly set "
            "THESIS_ALLOW_ICLOUD_ARTIFACTS=1."
        )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    FILL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_status("starting")

    stop = threading.Event()
    monitor = threading.Thread(target=_resource_monitor, args=(stop,), daemon=True)
    monitor.start()

    try:
        if _calibration_complete(CALIBRATION_SOURCE):
            _write_status(
                "running",
                current_step="01_calibration_import",
                source=str(CALIBRATION_SOURCE),
            )
            _import_calibration(CALIBRATION_SOURCE, FILL_MODEL_DIR)
            (LOG_DIR / "01_calibration_import.log").write_text(
                f"Imported validated calibration from {CALIBRATION_SOURCE}\n",
                encoding="utf-8",
            )
        else:
            _run_step(
                "01_calibration",
                _python_module(
                    "analysis.runners.calibrate_fill_model",
                    "--universe", "sp500",
                    "--start", PRE_START,
                    "--end", PRE_END,
                    "--out", str(FILL_MODEL_DIR),
                    "--workers", str(CALIBRATION_WORKERS),
                    "--event-sample-per-symbol-day", str(EVENT_SAMPLE_PER_SYMBOL_DAY),
                    "--as-sample-per-symbol-day", str(AS_SAMPLE_PER_SYMBOL_DAY),
                    "--event-sample-every-seconds", str(EVENT_SAMPLE_EVERY_SECONDS),
                    "--as-sample-every-seconds", str(AS_SAMPLE_EVERY_SECONDS),
                    "--min-coverage", str(CALIBRATION_MIN_COVERAGE),
                    "--ram-max-percent", str(CALIBRATION_RAM_MAX_PERCENT),
                    "--ram-min-available-gb", str(CALIBRATION_RAM_MIN_AVAILABLE_GB),
                ),
            )

        _run_step(
            "02_artifact_smoke",
            _python_module(
                "analysis.runners.h1_performance_gap",
                "--symbols", "AAPL", "MSFT",
                "--start", EVAL_START,
                "--end", EVAL_START,
                "--artifacts", str(FILL_MODEL_DIR),
                "--out", str(ARTIFACT_ROOT / "artifact_smoke"),
                "--workers", "1",
                "--fill-spec", "tape_replay_queue",
            ),
        )
        _run_step(
            "03_worker_benchmark",
            _python_module(
                "analysis.runners.benchmark_hypothesis_workers",
                "--artifacts", str(FILL_MODEL_DIR),
                "--out", str(ARTIFACT_ROOT / "benchmark"),
                "--universe", "sp500",
                "--start", EVAL_START,
                "--days", "2",
                "--symbols", "50",
            ),
        )
        selection = json.loads(
            (ARTIFACT_ROOT / "benchmark" / "worker_selection.json").read_text(
                encoding="utf-8",
            )
        )
        selected_workers = int(selection["selected_workers"])
        os.environ["THESIS_SELECTED_POOL_BACKEND"] = selection["selected_backend"]

        for name, module in [
            ("04_h1_dry_run", "analysis.runners.h1_performance_gap"),
            ("05_h2_dry_run", "analysis.runners.h2_signal_efficiency"),
            ("06_h3_dry_run", "analysis.runners.h3_te_tradeoff"),
        ]:
            _run_step(
                name,
                _python_module(
                    module,
                    "--dry-run",
                    "--universe", "sp500",
                    "--start", EVAL_START,
                    "--end", EVAL_END,
                    "--artifacts", str(FILL_MODEL_DIR),
                    "--fill-spec", "tape_replay_queue",
                    "--workers", str(selected_workers),
                ),
            )
        _run_step(
            "07_hypotheses",
            _python_module(
                "analysis.runners.run_all_hypotheses",
                "--run-id", RUN_ID,
                "--run-root", str(RUN_ROOT),
                "--universe", "sp500",
                "--start", EVAL_START,
                "--end", EVAL_END,
                "--artifacts", str(FILL_MODEL_DIR),
                "--workers", str(selected_workers),
                "--fill-spec", "tape_replay_queue",
            ),
        )
        _write_executive_summary("complete")
        _write_status("complete", summary=str(ARTIFACT_ROOT / "EXECUTIVE_SUMMARY.md"))
    except Exception as exc:
        _write_executive_summary("failed", error=str(exc))
        _write_status("failed", error=str(exc), summary=str(ARTIFACT_ROOT / "EXECUTIVE_SUMMARY.md"))
        raise
    finally:
        stop.set()
        monitor.join(timeout=5.0)


if __name__ == "__main__":
    main()
