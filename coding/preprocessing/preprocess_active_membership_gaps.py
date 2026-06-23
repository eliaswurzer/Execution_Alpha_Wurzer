#!/usr/bin/env python3
"""Supplement processed TAQ Parquets for active S&P membership gaps.

The unified data audit distinguishes conservative-union gaps from active
point-in-time S&P 500 membership gaps. This runner processes only raw-present
active membership gaps and leaves raw-missing gaps as external data blockers.
It never overwrites existing Parquets by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
STREAMING_SCRIPT = SCRIPT_DIR / "preprocess_taq_streaming.py"
DEFAULT_AUDIT_DIR = Path(os.environ.get("THESIS_DATA_AUDIT_DIR", "artifacts/data_audit/final_2018_2019"))
SIDE_DIR = {"trade": "Trade", "nbbo": "NBBO"}
SIDE_PREFIX = {"trade": "EQY_US_ALL_TRADE", "nbbo": "EQY_US_ALL_NBBO"}
SIDE_PARQUET_DIR = {"trade": "trades", "nbbo": "nbbo"}
SUPPLEMENT_QUEUE = "raw_present_but_active_symbol_supplement_needed.csv"
RAW_CHECKLIST = "required_raw_files_for_refinitiv_completion.csv"


@dataclass(frozen=True)
class SupplementJob:
    date: str
    side: str
    raw_root: Path
    raw_path: Path
    processed_root: Path
    symbols: tuple[str, ...]
    status_before: str
    trade_done_before: str
    nbbo_done_before: str
    qc_status: str


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace(" ", "_").replace("/", "_")


def _logical_symbol_from_path(path: Path) -> str:
    return path.stem.replace("_", " ")


def _parse_symbols(value: str | None) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return sorted({str(item).strip() for item in parsed if str(item).strip()})
    except json.JSONDecodeError:
        pass
    return sorted({part.strip() for part in re.split(r"[,;]", text) if part.strip()})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_audit_rows(audit_dir: Path) -> list[dict[str, str]]:
    path = audit_dir / "date_side_audit.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing audit CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _path_values(raw_field: str) -> list[Path]:
    """Parse one or more Windows paths from an audit path field.

    The audit field can contain paths with spaces. We therefore look for drive
    rooted `.gz` paths instead of splitting on whitespace.
    """
    text = str(raw_field or "").strip()
    if not text:
        return []
    matches = re.findall(r"[A-Za-z]:\\.*?\.gz(?=\s+[A-Za-z]:\\|$)", text)
    if matches:
        return [Path(match) for match in matches]
    return [Path(text)]


def raw_root_from_raw_path(raw_path: Path, side: str) -> Path:
    side_dir = SIDE_DIR[side].lower()
    parent = raw_path.parent
    if parent.name.lower() == side_dir.lower():
        return parent.parent
    return parent


def _raw_candidates(raw_root: Path, date: str, side: str) -> Iterable[Path]:
    prefix = SIDE_PREFIX[side]
    side_dir = SIDE_DIR[side]
    yield raw_root / side_dir / f"{prefix}_{date}.gz"
    yield raw_root / f"{prefix}_{date}.gz"
    yield from sorted((raw_root / side_dir).glob(f"{prefix}_{date}*.gz")) if (raw_root / side_dir).exists() else []
    yield from sorted(raw_root.glob(f"{prefix}_{date}*.gz")) if raw_root.exists() else []


def resolve_raw(raw_roots: list[Path], row: dict[str, str], date: str, side: str) -> tuple[Path | None, Path | None]:
    field = row.get(f"raw_{side}_paths", "")
    for raw_path in _path_values(field):
        if raw_path.exists():
            return raw_root_from_raw_path(raw_path, side), raw_path
    for raw_root in raw_roots:
        for candidate in _raw_candidates(raw_root, date, side):
            if candidate.exists():
                return raw_root, candidate
    return None, None


def existing_side_symbols(processed_root: Path, date: str, side: str) -> set[str]:
    side_dir = processed_root / date / SIDE_PARQUET_DIR[side]
    if not side_dir.exists():
        return set()
    return {_logical_symbol_from_path(path) for path in side_dir.glob("*.parquet")}


def _side_enabled(side: str, side_filter: str) -> bool:
    return side_filter == "both" or side_filter == side


def build_jobs(
    audit_dir: Path,
    *,
    raw_roots: list[Path] | None = None,
    start: str | None = None,
    end: str | None = None,
    side_filter: str = "both",
    limit: int | None = None,
) -> tuple[list[SupplementJob], list[dict[str, Any]]]:
    """Build raw-present supplement jobs and raw-missing checklist rows."""
    audit_dir = Path(audit_dir)
    summary = _read_json(audit_dir / "date_side_audit_summary.json")
    roots = raw_roots or [Path(value) for value in summary.get("raw_roots", [])]
    rows = sorted(_read_audit_rows(audit_dir), key=lambda item: str(item.get("date", "")))
    jobs: list[SupplementJob] = []
    raw_missing: list[dict[str, Any]] = []

    for row in rows:
        date = str(row.get("date", "")).strip()
        if not date:
            continue
        if start and date < start:
            continue
        if end and date > end:
            continue
        processed_root_text = str(row.get("processed_root", "")).strip()
        if not processed_root_text:
            continue
        processed_root = Path(processed_root_text)
        for side in ("trade", "nbbo"):
            if not _side_enabled(side, side_filter):
                continue
            symbols = _parse_symbols(row.get(f"active_missing_{side}_symbols"))
            if not symbols:
                continue
            raw_available = _truthy(row.get(f"raw_{side}_gz_available"))
            raw_root, raw_path = resolve_raw(roots, row, date, side) if raw_available else (None, None)
            if raw_root is None or raw_path is None:
                raw_missing.append({
                    "date": date,
                    "side": side,
                    "expected_raw_filename": f"{SIDE_PREFIX[side]}_{date}.gz",
                    "candidate_roots": json.dumps([str(root) for root in roots]),
                    "active_missing_count": len(symbols),
                    "active_missing_symbols": json.dumps(symbols),
                    "status": row.get("status", ""),
                })
                continue
            existing = existing_side_symbols(processed_root, date, side)
            pending = tuple(symbol for symbol in symbols if symbol not in existing)
            if not pending:
                continue
            if limit is not None and len(jobs) >= limit:
                continue
            jobs.append(SupplementJob(
                date=date,
                side=side,
                raw_root=raw_root,
                raw_path=raw_path,
                processed_root=processed_root,
                symbols=pending,
                status_before=row.get("status", ""),
                trade_done_before=row.get("trade_side_done", ""),
                nbbo_done_before=row.get("nbbo_side_done", ""),
                qc_status=row.get("trade_qc_status", ""),
            ))
    return jobs, raw_missing


def write_queue_files(audit_dir: Path, jobs: list[SupplementJob], raw_missing: list[dict[str, Any]]) -> tuple[Path, Path]:
    queue_rows = [
        {
            "date": job.date,
            "side": job.side,
            "processed_root": str(job.processed_root),
            "raw_root": str(job.raw_root),
            "raw_path": str(job.raw_path),
            "symbol_count": len(job.symbols),
            "symbols": json.dumps(list(job.symbols)),
            "status_before": job.status_before,
        }
        for job in jobs
    ]
    queue_path = audit_dir / SUPPLEMENT_QUEUE
    raw_path = audit_dir / RAW_CHECKLIST
    _write_csv(
        queue_path,
        queue_rows,
        ["date", "side", "processed_root", "raw_root", "raw_path", "symbol_count", "symbols", "status_before"],
    )
    _write_csv(
        raw_path,
        raw_missing,
        ["date", "side", "expected_raw_filename", "candidate_roots", "active_missing_count", "active_missing_symbols", "status"],
    )
    return queue_path, raw_path


def _write_symbol_file(out_dir: Path, job: SupplementJob) -> Path:
    symbol_dir = out_dir / "symbol_overrides"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    path = symbol_dir / f"{job.date}_{job.side}_active_refinitiv.txt"
    path.write_text("\n".join(job.symbols) + "\n", encoding="utf-8")
    return path


def build_command(
    job: SupplementJob,
    *,
    out_dir: Path,
    chunksize: int,
    schema: str,
    trade_filter_policy: str,
) -> list[str]:
    symbol_file = _write_symbol_file(out_dir, job)
    command = [
        sys.executable,
        "-B",
        str(STREAMING_SCRIPT),
        "--date",
        job.date,
        "--raw-root",
        str(job.raw_root),
        "--output-dir",
        str(job.processed_root),
        "--chunksize",
        str(chunksize),
        "--schema",
        schema,
        "--trade-filter-policy",
        trade_filter_policy,
        "--supplement-missing",
    ]
    if job.side == "trade":
        command.extend(["--trade-symbols-file", str(symbol_file), "--skip-nbbo"])
    else:
        command.extend(["--nbbo-symbols-file", str(symbol_file), "--skip-trades"])
    return command


def run_job(
    job: SupplementJob,
    *,
    out_dir: Path,
    chunksize: int,
    schema: str,
    trade_filter_policy: str,
    dry_run: bool,
) -> dict[str, Any]:
    before_existing = existing_side_symbols(job.processed_root, job.date, job.side)
    command = build_command(
        job,
        out_dir=out_dir,
        chunksize=chunksize,
        schema=schema,
        trade_filter_policy=trade_filter_policy,
    )
    row: dict[str, Any] = {
        "date": job.date,
        "processed_side": job.side,
        "raw_root": str(job.raw_root),
        "raw_path": str(job.raw_path),
        "processed_root": str(job.processed_root),
        "symbol_count": len(job.symbols),
        "symbols": json.dumps(list(job.symbols)),
        "status_before": job.status_before,
        "trade_done_before": job.trade_done_before,
        "nbbo_done_before": job.nbbo_done_before,
        "qc_status": job.qc_status,
        "command": subprocess.list2cmdline(command),
    }
    if dry_run:
        row.update({
            "exit_code": 0,
            "status_after": "dry_run",
            "present_after_count": sum(1 for symbol in job.symbols if symbol in before_existing),
            "missing_after_count": len(job.symbols),
            "error_detail": "",
            "runtime_seconds": "0.000",
        })
        return row

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.date}_{job.side}.log"
    t0 = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        runtime = time.time() - t0
        log_path.write_text(
            "Command:\n"
            + subprocess.list2cmdline(command)
            + "\n\nSTDOUT:\n"
            + (completed.stdout or "")
            + "\n\nSTDERR:\n"
            + (completed.stderr or ""),
            encoding="utf-8",
        )
        after_existing = existing_side_symbols(job.processed_root, job.date, job.side)
        present_after = sum(1 for symbol in job.symbols if symbol in after_existing)
        missing_after = len(job.symbols) - present_after
        if completed.returncode != 0:
            status_after = "failed"
        elif missing_after:
            status_after = "partial_no_rows_or_symbol_absent"
        else:
            status_after = "complete"
        row.update({
            "exit_code": completed.returncode,
            "status_after": status_after,
            "present_after_count": present_after,
            "missing_after_count": missing_after,
            "error_detail": (completed.stderr or "").strip().splitlines()[-1] if completed.stderr else "",
            "runtime_seconds": f"{runtime:.3f}",
            "log_path": str(log_path),
        })
        return row
    except Exception as exc:  # pragma: no cover - defensive process boundary
        runtime = time.time() - t0
        row.update({
            "exit_code": -1,
            "status_after": "runner_error",
            "present_after_count": 0,
            "missing_after_count": len(job.symbols),
            "error_detail": repr(exc),
            "runtime_seconds": f"{runtime:.3f}",
            "log_path": str(log_path),
        })
        return row


def run_jobs(
    jobs: list[SupplementJob],
    *,
    out_dir: Path,
    workers: int,
    chunksize: int,
    schema: str,
    trade_filter_policy: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if dry_run or workers <= 1:
        return [
            run_job(
                job,
                out_dir=out_dir,
                chunksize=chunksize,
                schema=schema,
                trade_filter_policy=trade_filter_policy,
                dry_run=dry_run,
            )
            for job in jobs
        ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_job,
                job,
                out_dir=out_dir,
                chunksize=chunksize,
                schema=schema,
                trade_filter_policy=trade_filter_policy,
                dry_run=False,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"[supplement] completed {len(rows)}/{len(jobs)}", flush=True)
    return sorted(rows, key=lambda row: (str(row.get("date")), str(row.get("processed_side"))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--raw-root", type=Path, action="append", dest="raw_roots")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--schema", choices=["slim", "rich"], default="slim")
    parser.add_argument("--trade-filter-policy", default="preprocessing")
    parser.add_argument("--side", choices=["trade", "nbbo", "both"], default="both")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (audit_dir / f"active_membership_supplement_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, raw_missing = build_jobs(
        audit_dir,
        raw_roots=args.raw_roots,
        start=args.start,
        end=args.end,
        side_filter=args.side,
        limit=args.limit,
    )
    queue_path, raw_path = write_queue_files(audit_dir, jobs, raw_missing)
    print(f"[supplement] raw-present queue: {queue_path} ({len(jobs)} jobs)")
    print(f"[supplement] raw-missing checklist: {raw_path} ({len(raw_missing)} rows)")
    if not jobs:
        return

    planned_path = out_dir / "planned_jobs.csv"
    _write_csv(
        planned_path,
        [
            {
                "date": job.date,
                "side": job.side,
                "raw_root": str(job.raw_root),
                "raw_path": str(job.raw_path),
                "processed_root": str(job.processed_root),
                "symbol_count": len(job.symbols),
                "symbols": json.dumps(list(job.symbols)),
            }
            for job in jobs
        ],
        ["date", "side", "raw_root", "raw_path", "processed_root", "symbol_count", "symbols"],
    )
    rows = run_jobs(
        jobs,
        out_dir=out_dir,
        workers=max(1, args.workers),
        chunksize=args.chunksize,
        schema=args.schema,
        trade_filter_policy=args.trade_filter_policy,
        dry_run=args.dry_run,
    )
    summary_path = out_dir / "active_membership_supplement_summary.csv"
    _write_csv(
        summary_path,
        rows,
        [
            "date",
            "processed_side",
            "raw_root",
            "raw_path",
            "processed_root",
            "symbol_count",
            "symbols",
            "status_before",
            "trade_done_before",
            "nbbo_done_before",
            "qc_status",
            "exit_code",
            "status_after",
            "present_after_count",
            "missing_after_count",
            "runtime_seconds",
            "error_detail",
            "log_path",
            "command",
        ],
    )
    print(f"[supplement] summary: {summary_path}")


if __name__ == "__main__":
    main()
