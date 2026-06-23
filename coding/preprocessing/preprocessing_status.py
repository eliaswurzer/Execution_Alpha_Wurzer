"""Shared raw/processed TAQ availability and resume-status helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

STATUS_RAW_MISSING = "raw_missing"
STATUS_UNPROCESSED = "unprocessed"
STATUS_TRADE_ONLY = "trade_only_processed"
STATUS_NBBO_ONLY = "nbbo_only_processed"
STATUS_COMPLETE = "complete"
STATUS_COMPLETE_WITH_MISSING = "complete_with_missing_symbols"
STATUS_MANIFEST_INCONSISTENT = "manifest_inconsistent"
STATUS_QC_PROBLEM = "qc_missing_or_failed"


SIDE_TO_MANIFEST_COL = {
    "trade": "trade_rows",
    "nbbo": "nbbo_rows",
}


def safe_symbol(symbol: str) -> str:
    return str(symbol).strip().replace(" ", "_").replace("/", "_")


def load_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _raw_date_from_path(path: Path, prefix: str) -> str | None:
    stem = path.stem
    match = re.match(rf"^{re.escape(prefix)}(\d{{8}})(?:\D.*)?$", stem)
    if match:
        return match.group(1)
    return None


def discover_raw_date_sides(raw_roots: Iterable[Path]) -> dict[str, dict]:
    """Discover Trade/NBBO gz availability across one or more raw roots."""
    out: dict[str, dict] = {}
    for raw_root in sorted({Path(p) for p in raw_roots}, key=lambda p: str(p).lower()):
        for side, subdir, prefix in (
            ("trade", "Trade", "EQY_US_ALL_TRADE_"),
            ("nbbo", "NBBO", "EQY_US_ALL_NBBO_"),
        ):
            candidate_dirs = []
            for candidate in (raw_root / subdir, raw_root):
                if candidate.is_dir() and candidate not in candidate_dirs:
                    candidate_dirs.append(candidate)
            for side_dir in candidate_dirs:
                for path in sorted(side_dir.glob(f"{prefix}*.gz"), key=lambda p: str(p).lower()):
                    date = _raw_date_from_path(path, prefix)
                    if date is None:
                        continue
                    row = out.setdefault(
                        date,
                        {
                            "trade": False,
                            "nbbo": False,
                            "trade_paths": [],
                            "nbbo_paths": [],
                            "raw_roots": [],
                        },
                    )
                    row[side] = True
                    if str(path) not in row[f"{side}_paths"]:
                        row[f"{side}_paths"].append(str(path))
                    if str(raw_root) not in row["raw_roots"]:
                        row["raw_roots"].append(str(raw_root))
    return out


def discover_processed_dates(processed_root: Path) -> set[str]:
    root = Path(processed_root)
    if not root.is_dir():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit()
    }


def manifest_side_symbols(path: Path, side: str) -> set[str]:
    if side not in SIDE_TO_MANIFEST_COL:
        raise ValueError(f"Unknown side: {side}")
    path = Path(path)
    if not path.exists():
        return set()
    col = SIDE_TO_MANIFEST_COL[side]
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "symbol" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            symbol = row.get("symbol")
            if not symbol:
                continue
            try:
                n_rows = int(float(row.get(col, 0) or 0))
            except ValueError:
                n_rows = 0
            if n_rows > 0:
                out.add(safe_symbol(symbol))
    return out


def manifest_symbols(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "symbol" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            if row.get("symbol"):
                out.add(safe_symbol(row["symbol"]))
    return out


def parquet_symbols(path: Path) -> set[str]:
    path = Path(path)
    if not path.is_dir():
        return set()
    return {p.stem for p in path.glob("*.parquet")}


def trade_qc_status(date_dir: Path) -> tuple[str, str]:
    qc_path = Path(date_dir) / "qc" / "trade_qc_summary.json"
    if not qc_path.exists():
        return "missing", ""
    try:
        payload = json.loads(qc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "invalid_json", f"{type(exc).__name__}: {exc}"
    status = str(payload.get("status", "ok")).strip().lower()
    if status in {"failed", "error"}:
        return "failed", json.dumps(payload, sort_keys=True)
    return "ok", ""


@dataclass
class DateSideStatus:
    date: str
    processed_root: str
    expected_symbols: int = 0
    trade_symbols: set[str] = field(default_factory=set)
    nbbo_symbols: set[str] = field(default_factory=set)
    trade_manifest_symbols: set[str] = field(default_factory=set)
    nbbo_manifest_symbols: set[str] = field(default_factory=set)
    missing_trade_symbols: list[str] = field(default_factory=list)
    missing_nbbo_symbols: list[str] = field(default_factory=list)
    manifest_exists: bool = False
    coverage_exists: bool = False
    trade_qc_status: str = "missing"
    trade_qc_detail: str = ""
    manifest_consistent: bool = True
    trade_done: bool = False
    nbbo_done: bool = False
    complete_done: bool = False
    status: str = STATUS_UNPROCESSED

    @property
    def complete_symbols(self) -> set[str]:
        return self.trade_symbols & self.nbbo_symbols

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "processed_root": self.processed_root,
            "expected_symbols": self.expected_symbols,
            "trade_symbols": self.trade_symbols,
            "nbbo_symbols": self.nbbo_symbols,
            "trade_manifest_symbols": self.trade_manifest_symbols,
            "nbbo_manifest_symbols": self.nbbo_manifest_symbols,
            "trade_count": len(self.trade_symbols),
            "nbbo_count": len(self.nbbo_symbols),
            "complete_count": len(self.complete_symbols),
            "missing_trade_symbols": self.missing_trade_symbols,
            "missing_nbbo_symbols": self.missing_nbbo_symbols,
            "missing_trade_count": len(self.missing_trade_symbols),
            "missing_nbbo_count": len(self.missing_nbbo_symbols),
            "manifest_exists": self.manifest_exists,
            "coverage_exists": self.coverage_exists,
            "trade_qc_status": self.trade_qc_status,
            "trade_qc_detail": self.trade_qc_detail,
            "manifest_consistent": self.manifest_consistent,
            "trade_done": self.trade_done,
            "nbbo_done": self.nbbo_done,
            "complete_done": self.complete_done,
            "status": self.status,
        }


def date_side_status(
    out_root: Path,
    date: str,
    symbols: Iterable[str] | None = None,
    *,
    raw_trade: bool = False,
    raw_nbbo: bool = False,
) -> DateSideStatus:
    out_root = Path(out_root)
    date_dir = out_root / str(date)
    manifest = date_dir / "manifest.csv"
    coverage_summary = out_root / "coverage" / f"{date}_summary.json"
    expected = {safe_symbol(s) for s in (symbols or [])}
    trade = parquet_symbols(date_dir / "trades")
    nbbo = parquet_symbols(date_dir / "nbbo")
    trade_manifest = manifest_side_symbols(manifest, "trade")
    nbbo_manifest = manifest_side_symbols(manifest, "nbbo")
    qc_status, qc_detail = trade_qc_status(date_dir)

    manifest_consistent = True
    if trade or trade_manifest:
        manifest_consistent = manifest_consistent and trade == trade_manifest
    if nbbo or nbbo_manifest:
        manifest_consistent = manifest_consistent and nbbo == nbbo_manifest

    trade_done = bool(trade) and manifest.exists() and qc_status == "ok" and trade == trade_manifest
    nbbo_done = bool(nbbo) and manifest.exists() and nbbo == nbbo_manifest
    # Date-level completeness means both sides were processed consistently
    # with their manifests and the coverage summary exists. Requiring equal
    # (or nested) symbol sets across sides was a single-run proxy; after the
    # 2018/2019 consolidation a date legitimately carries side-specific
    # extras because trade and NBBO supplement runs used different symbol
    # lists. Whether every ACTIVE index member has both sides is the
    # audit-level question and is enforced there through the
    # active_missing_trade/nbbo hard blockers.
    complete_done = trade_done and nbbo_done and coverage_summary.exists()

    missing_trade = sorted(expected - trade) if expected else []
    missing_nbbo = sorted(expected - nbbo) if expected else []

    if not raw_trade and not raw_nbbo and not trade and not nbbo:
        status = STATUS_RAW_MISSING
    elif not manifest_consistent:
        status = STATUS_MANIFEST_INCONSISTENT
    elif trade and qc_status != "ok":
        status = STATUS_QC_PROBLEM
    elif complete_done:
        missing_complete = sorted(expected - (trade & nbbo)) if expected else []
        status = STATUS_COMPLETE_WITH_MISSING if missing_complete else STATUS_COMPLETE
    elif trade_done and not nbbo_done:
        status = STATUS_TRADE_ONLY
    elif nbbo_done and not trade_done:
        status = STATUS_NBBO_ONLY
    else:
        status = STATUS_UNPROCESSED

    return DateSideStatus(
        date=str(date),
        processed_root=str(out_root),
        expected_symbols=len(expected),
        trade_symbols=trade,
        nbbo_symbols=nbbo,
        trade_manifest_symbols=trade_manifest,
        nbbo_manifest_symbols=nbbo_manifest,
        missing_trade_symbols=missing_trade,
        missing_nbbo_symbols=missing_nbbo,
        manifest_exists=manifest.exists(),
        coverage_exists=coverage_summary.exists(),
        trade_qc_status=qc_status,
        trade_qc_detail=qc_detail,
        manifest_consistent=manifest_consistent,
        trade_done=trade_done,
        nbbo_done=nbbo_done,
        complete_done=complete_done,
        status=status,
    )


def choose_best_status(
    statuses: Iterable[DateSideStatus],
) -> DateSideStatus | None:
    ranking = {
        STATUS_COMPLETE: 80,
        STATUS_COMPLETE_WITH_MISSING: 75,
        STATUS_TRADE_ONLY: 60,
        STATUS_NBBO_ONLY: 55,
        STATUS_UNPROCESSED: 20,
        STATUS_QC_PROBLEM: 10,
        STATUS_MANIFEST_INCONSISTENT: 5,
        STATUS_RAW_MISSING: 0,
    }
    items = list(statuses)
    if not items:
        return None
    return sorted(
        items,
        key=lambda s: (
            ranking.get(s.status, -1),
            s.complete_done,
            s.trade_done + s.nbbo_done,
            len(s.complete_symbols),
            s.processed_root.lower(),
        ),
        reverse=True,
    )[0]


def done_status(out_root: Path, date: str, symbols: Iterable[str]) -> tuple[bool, list[str]]:
    status = date_side_status(out_root, date, symbols)
    expected = {safe_symbol(s) for s in symbols}
    missing = sorted(expected - status.complete_symbols)
    done = status.complete_done and not missing and status.expected_symbols == len(expected)
    return done, missing
