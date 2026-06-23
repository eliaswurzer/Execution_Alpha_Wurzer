#!/usr/bin/env python3
"""Streaming TAQ preprocessing for large symbol universes.

Unlike ``preprocess_taq.py``, this writer does not buffer all rows for all
symbols in memory. It scans each raw daily file once and appends chunk-level row
groups to one Parquet file per symbol.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_CODING_ROOT = _HERE.parents[0]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_CODING_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODING_ROOT))

import preprocess_taq as base  # noqa: E402
from analysis.data import trade_conditions as tc  # noqa: E402


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace(" ", "_").replace("/", "_")


def _load_symbols(path: Path | None, inline: list[str] | None) -> set[str] | None:
    symbols: set[str] = set()
    if inline:
        symbols.update(s.strip() for s in inline if s.strip())
    if path:
        symbols.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return symbols or None


def _symbol_override(path: Path | None, inline: list[str] | None, fallback: set[str] | None) -> set[str] | None:
    override = _load_symbols(path, inline)
    return override if override is not None else fallback


def _prepare_date_dir(
    output_dir: Path,
    date: str,
    overwrite: bool,
    *,
    process_trades: bool,
    process_nbbo: bool,
    supplement_missing: bool = False,
) -> Path:
    date_dir = output_dir / date
    if date_dir.exists() and any(date_dir.iterdir()):
        if supplement_missing and not overwrite:
            date_dir.mkdir(parents=True, exist_ok=True)
            return date_dir
        if process_trades and process_nbbo:
            if not overwrite:
                raise FileExistsError(f"Output date directory is not empty: {date_dir}")
            shutil.rmtree(date_dir)
        else:
            targets = []
            if process_trades:
                targets.append(date_dir / "trades")
            if process_nbbo:
                targets.append(date_dir / "nbbo")
            for target in targets:
                if target.exists() and any(target.iterdir()):
                    if not overwrite:
                        raise FileExistsError(
                            f"Output side directory is not empty: {target}. "
                            "Use --overwrite to rebuild this side."
                        )
                    shutil.rmtree(target)
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


class SymbolParquetWriters:
    """Keep one ParquetWriter per symbol for append-style row-group writes."""

    def __init__(self, out_dir: Path, *, fail_existing: bool = False):
        self.out_dir = out_dir
        self.fail_existing = fail_existing
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._schemas: dict[str, pa.Schema] = {}
        self.stats: dict[str, int] = {}

    def write(self, symbol: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        safe = _safe_symbol(symbol)
        if safe not in self._writers:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            path = self.out_dir / f"{safe}.parquet"
            if self.fail_existing and path.exists():
                raise FileExistsError(f"Supplement target already exists: {path}")
            writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            self._writers[safe] = writer
            self._schemas[safe] = table.schema
        else:
            table = pa.Table.from_pandas(
                frame, schema=self._schemas[safe], preserve_index=False,
            )
            writer = self._writers[safe]
        writer.write_table(table)
        self.stats[symbol] = self.stats.get(symbol, 0) + int(len(frame))

    def close(self) -> None:
        errors = []
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - defensive close path
                errors.append(exc)
        self._writers.clear()
        if errors:
            raise errors[0]


def _read_manifest_stats(output_dir: Path, date: str) -> tuple[dict[str, int], dict[str, int]]:
    path = output_dir / date / "manifest.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        return {}, {}
    trade_stats: dict[str, int] = {}
    nbbo_stats: dict[str, int] = {}
    for _, row in df.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        trade_rows = int(row.get("trade_rows", 0) or 0)
        nbbo_rows = int(row.get("nbbo_rows", 0) or 0)
        if trade_rows > 0:
            trade_stats[sym] = trade_rows
        if nbbo_rows > 0:
            nbbo_stats[sym] = nbbo_rows
    return trade_stats, nbbo_stats


def _write_manifest(
    output_dir: Path,
    date: str,
    trade_stats: dict,
    nbbo_stats: dict,
    *,
    merge_existing: bool = True,
) -> None:
    if merge_existing:
        existing_trade, existing_nbbo = _read_manifest_stats(output_dir, date)
        existing_trade.update(trade_stats)
        existing_nbbo.update(nbbo_stats)
        trade_stats = existing_trade
        nbbo_stats = existing_nbbo
    all_symbols = sorted(set(trade_stats) | set(nbbo_stats))
    rows = [
        {
            "symbol": sym,
            "trade_rows": trade_stats.get(sym, 0),
            "nbbo_rows": nbbo_stats.get(sym, 0),
        }
        for sym in all_symbols
    ]
    pd.DataFrame(rows).to_csv(output_dir / date / "manifest.csv", index=False)


def _write_trade_qc(output_dir: Path, date: str, summary: dict) -> None:
    qc_dir = output_dir / date / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "trade_qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )


def process_trades_streaming(
    raw_root: Path,
    output_dir: Path,
    date: str,
    symbols: set[str] | None,
    chunksize: int,
    schema: str,
    trade_filter_policy: str,
    *,
    fail_existing: bool = False,
) -> dict[str, int]:
    schema = base._validate_schema(schema)
    trade_filter_policy = tc.normalize_policy(trade_filter_policy)
    keep_cols = base.TRADE_RICH_KEEP_COLS if schema == "rich" else base.TRADE_KEEP_COLS
    out_cols = base.TRADE_SLIM_OUT_COLS + (
        base.TRADE_RICH_EXTRA_OUT_COLS if schema == "rich" else []
    )
    path = base._resolve_path(str(raw_root), "EQY_US_ALL_TRADE", date)
    writer = SymbolParquetWriters(output_dir / date / "trades", fail_existing=fail_existing)
    total_read = 0
    total_symbol_rows = 0
    total_kept = 0
    qc_counts: dict[str, int] = {}
    truncated_gzip = False
    t0 = time.time()

    reader = pd.read_csv(
        path, sep="|", dtype=str, header=0, names=base.TRADE_ALL_COLS,
        usecols=keep_cols, chunksize=chunksize, on_bad_lines="skip",
    )
    try:
        for chunk_i, chunk in enumerate(reader, 1):
            total_read += len(chunk)
            df = chunk.copy()
            df["Symbol"] = df["Symbol"].str.strip()
            if symbols:
                df = df[df["Symbol"].isin(symbols)]
            total_symbol_rows += len(df)
            if df.empty:
                continue
            df, chunk_qc = base.filter_trades_c1(
                df, policy=trade_filter_policy, return_qc=True,
            )
            base._add_qc_counts(qc_counts, chunk_qc)
            if df.empty:
                continue
            df["Trade Price"] = pd.to_numeric(df["Trade Price"], errors="coerce")
            df["Trade Volume"] = pd.to_numeric(
                df["Trade Volume"], errors="coerce",
            ).astype("int32")
            if schema == "rich":
                for col in ["Sequence Number", "Trade Id"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df["time"] = base.parse_taq_time(df["Time"], date)
            df = df.rename(columns=base.TRADE_RENAME).drop(columns=["Time"])
            df = df[base._ordered_existing(df, out_cols)]
            total_kept += len(df)
            for sym, grp in df.groupby("symbol", sort=False):
                writer.write(sym, grp.reset_index(drop=True))
            if chunk_i % 20 == 0:
                print(
                    f"[TRADE {date}] {total_read:,} read, {total_kept:,} kept, "
                    f"{len(writer.stats)} symbols, {time.time() - t0:.0f}s",
                    flush=True,
                )
    except EOFError:
        truncated_gzip = True
        print(
            f"[TRADE {date}] WARNING: truncated gzip after {total_read:,} rows; "
            f"writing {total_kept:,} kept rows from {len(writer.stats)} symbols",
            flush=True,
        )
    finally:
        writer.close()

    _write_trade_qc(output_dir, date, {
        "date": date,
        "kind": "trades",
        "schema": schema,
        "trade_filter_policy": trade_filter_policy,
        "trade_condition_policy_version": tc.POLICY_VERSION,
        "input_path": path,
        "chunksize": chunksize,
        "symbols_filter_count": len(symbols) if symbols else None,
        "total_read_rows": total_read,
        "total_symbol_filter_rows": total_symbol_rows,
        "total_kept_rows": total_kept,
        "symbols_written": len(writer.stats),
        "truncated_gzip": truncated_gzip,
        "qc_counts": qc_counts,
    })
    print(
        f"[TRADE {date}] done: {total_read:,} read -> {total_kept:,} kept "
        f"across {len(writer.stats)} symbols in {time.time() - t0:.0f}s",
        flush=True,
    )
    return writer.stats


def process_nbbo_streaming(
    raw_root: Path,
    output_dir: Path,
    date: str,
    symbols: set[str] | None,
    chunksize: int,
    schema: str,
    *,
    fail_existing: bool = False,
) -> dict[str, int]:
    schema = base._validate_schema(schema)
    keep_cols = base.NBBO_RICH_KEEP_COLS if schema == "rich" else base.NBBO_KEEP_COLS
    out_cols = base.NBBO_SLIM_OUT_COLS + (
        base.NBBO_RICH_EXTRA_OUT_COLS if schema == "rich" else []
    )
    path = base._resolve_path(str(raw_root), "EQY_US_ALL_NBBO", date)
    writer = SymbolParquetWriters(output_dir / date / "nbbo", fail_existing=fail_existing)
    total_read = 0
    total_kept = 0
    truncated_gzip = False
    t0 = time.time()

    reader = pd.read_csv(
        path, sep="|", dtype=str, header=0, usecols=keep_cols,
        chunksize=chunksize, on_bad_lines="skip",
    )
    try:
        for chunk_i, chunk in enumerate(reader, 1):
            total_read += len(chunk)
            df = base.filter_nbbo_quality(chunk.copy())
            df["Symbol"] = df["Symbol"].str.strip()
            if symbols:
                df = df[df["Symbol"].isin(symbols)]
            if df.empty:
                continue
            for col in ["Best_Bid_Price", "Best_Offer_Price"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            for col in ["Best_Bid_Size", "Best_Offer_Size"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("int32")
            if schema == "rich":
                for col in ["Bid_Price", "Offer_Price"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                for col in ["Bid_Size", "Offer_Size", "Sequence_Number"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df["time"] = base.parse_taq_time(df["Time"], date)
            df = df.rename(columns=base.NBBO_RENAME).drop(columns=["Time"])
            df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
            df = df[base._ordered_existing(df, out_cols)]
            total_kept += len(df)
            for sym, grp in df.groupby("symbol", sort=False):
                writer.write(sym, grp.reset_index(drop=True))
            if chunk_i % 40 == 0:
                print(
                    f"[NBBO  {date}] {total_read:,} read, {total_kept:,} kept, "
                    f"{len(writer.stats)} symbols, {time.time() - t0:.0f}s",
                    flush=True,
                )
    except EOFError:
        truncated_gzip = True
        print(
            f"[NBBO  {date}] WARNING: truncated gzip after {total_read:,} rows; "
            f"writing {total_kept:,} kept rows from {len(writer.stats)} symbols",
            flush=True,
        )
    finally:
        writer.close()

    print(
        f"[NBBO  {date}] done: {total_read:,} read -> {total_kept:,} kept "
        f"across {len(writer.stats)} symbols in {time.time() - t0:.0f}s",
        flush=True,
    )
    if truncated_gzip:
        qc_dir = output_dir / date / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        (qc_dir / "nbbo_qc_summary.json").write_text(
            json.dumps({
                "date": date,
                "kind": "nbbo",
                "schema": schema,
                "input_path": path,
                "chunksize": chunksize,
                "symbols_filter_count": len(symbols) if symbols else None,
                "total_read_rows": total_read,
                "total_kept_rows": total_kept,
                "symbols_written": len(writer.stats),
                "truncated_gzip": True,
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return writer.stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("THESIS_RAW_TAQ_ROOT", "data/raw")),
        help="Root containing licensed raw DTAQ files. Defaults to THESIS_RAW_TAQ_ROOT or data/raw.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--trade-symbols-file", type=Path)
    parser.add_argument("--trade-symbols", nargs="*")
    parser.add_argument("--nbbo-symbols-file", type=Path)
    parser.add_argument("--nbbo-symbols", nargs="*")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--schema", choices=sorted(base.SCHEMAS), default="slim")
    parser.add_argument(
        "--trade-filter-policy",
        choices=sorted(tc.TRADE_FILTER_POLICIES),
        default="preprocessing",
    )
    parser.add_argument("--skip-trades", action="store_true")
    parser.add_argument("--skip-nbbo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--supplement-missing", action="store_true",
        help="Allow writing missing symbols into an existing date/side directory without deleting existing Parquets.",
    )
    args = parser.parse_args()

    symbols = _load_symbols(args.symbols_file, args.symbols)
    trade_symbols = _symbol_override(args.trade_symbols_file, args.trade_symbols, symbols)
    nbbo_symbols = _symbol_override(args.nbbo_symbols_file, args.nbbo_symbols, symbols)
    process_trades = not args.skip_trades
    process_nbbo = not args.skip_nbbo
    if not process_trades and not process_nbbo:
        raise SystemExit("Nothing to process: both --skip-trades and --skip-nbbo were set")
    _prepare_date_dir(
        args.output_dir,
        args.date,
        args.overwrite,
        process_trades=process_trades,
        process_nbbo=process_nbbo,
        supplement_missing=args.supplement_missing,
    )

    trade_stats: dict[str, int] = {}
    nbbo_stats: dict[str, int] = {}
    if process_trades:
        trade_stats = process_trades_streaming(
            args.raw_root, args.output_dir, args.date, trade_symbols,
            args.chunksize, args.schema, args.trade_filter_policy,
            fail_existing=args.supplement_missing,
        )
    if process_nbbo:
        nbbo_stats = process_nbbo_streaming(
            args.raw_root, args.output_dir, args.date, nbbo_symbols,
            args.chunksize, args.schema,
            fail_existing=args.supplement_missing,
        )
    _write_manifest(args.output_dir, args.date, trade_stats, nbbo_stats)
    print(f"[MANIFEST] {args.output_dir / args.date / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
