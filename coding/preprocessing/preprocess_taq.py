#!/usr/bin/env python3
"""
preprocess_taq.py -- Slim down raw NYSE TAQ daily files (TRADE + NBBO)
into per-symbol Parquet files, applying Ait-Sahalia et al. (2025) C.1
filters.

Two output schemas are supported:
    slim   Minimal analysis schema used by H1/H2/H3 (default).
    rich   Slim schema plus TAQ metadata for audit/publication checks.

Input
-----
Raw pipe-delimited TAQ files (plain or .gz):
    EQY_US_ALL_TRADE_YYYYMMDD[.gz]
    EQY_US_ALL_NBBO_YYYYMMDD[.gz]

Output
------
Per-symbol Parquet files under ``<output_dir>/<YYYYMMDD>/``:
    trades/<SYMBOL>.parquet
    nbbo/<SYMBOL>.parquet

Usage
-----
    python preprocess_taq.py --date 20180501 [--data-dir ../data/raw] [--output-dir ../data/processed]
    python preprocess_taq.py --date 20180501 --symbols AAPL MSFT GOOG
    python preprocess_taq.py --date 20180501 --symbols-file sp500.txt
    python preprocess_taq.py --date 20180501 --schema rich --symbols AAPL

Reads gzipped files automatically if the uncompressed version is absent.
"""

import argparse
import glob
import gzip
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODING_ROOT = Path(__file__).resolve().parents[1]
if str(_CODING_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODING_ROOT))

from analysis.data import trade_conditions as tc  # noqa: E402

# ======================================================================
# Column specifications
# ======================================================================

SCHEMAS = {"slim", "rich"}

TRADE_ALL_COLS = [
    "Time", "Exchange", "Symbol", "Sale Condition", "Trade Volume",
    "Trade Price", "Trade Stop Stock Indicator",
    "Trade Correction Indicator", "Sequence Number", "Trade Id",
    "Source of Trade", "Trade Reporting Facility",
    "Participant Timestamp",
    "Trade Reporting Facility TRF Timestamp",
    "Trade Through Exempt Indicator",
]

TRADE_KEEP_COLS = [
    "Time", "Exchange", "Symbol", "Sale Condition",
    "Trade Volume", "Trade Price", "Trade Correction Indicator",
]

TRADE_RICH_KEEP_COLS = [
    "Time", "Exchange", "Symbol", "Sale Condition",
    "Trade Volume", "Trade Price", "Trade Stop Stock Indicator",
    "Trade Correction Indicator", "Sequence Number", "Trade Id",
    "Source of Trade", "Trade Reporting Facility",
    "Participant Timestamp",
    "Trade Reporting Facility TRF Timestamp",
    "Trade Through Exempt Indicator",
]

NBBO_ALL_COLS = [
    "Time", "Exchange", "Symbol", "Bid_Price", "Bid_Size",
    "Offer_Price", "Offer_Size", "Quote_Condition", "Sequence_Number",
    "National_BBO_Ind", "FINRA_BBO_Indicator",
    "FINRA_ADF_MPID_Indicator", "Quote_Cancel_Correction",
    "Source_Of_Quote", "Best Bid Quote Condition", "Best_Bid_Exchange",
    "Best_Bid_Price", "Best_Bid_Size",
    "Best_Bid_FINRA_Market_Maker_ID", "Best_Offer_Quote_Condition",
    "Best_Offer_Exchange", "Best_Offer_Price", "Best_Offer_Size",
    "Best_Offer_FINRA_Market_Maker_ID", "LULD_Indicator",
    "LULD_NBBO_Indicator", "SIP_Generated_Message_Identifier",
    "Participant_Timestamp", "FINRA_ADF_Timestamp",
    "Security_Status_Indicator",
]

NBBO_KEEP_COLS = [
    "Time", "Symbol",
    "Best_Bid_Price", "Best_Bid_Size",
    "Best_Offer_Price", "Best_Offer_Size",
]

NBBO_RICH_KEEP_COLS = [
    "Time", "Exchange", "Symbol", "Bid_Price", "Bid_Size",
    "Offer_Price", "Offer_Size", "Quote_Condition", "Sequence_Number",
    "National_BBO_Ind", "Quote_Cancel_Correction", "Source_Of_Quote",
    "Best Bid Quote Condition", "Best_Bid_Exchange",
    "Best_Bid_Price", "Best_Bid_Size",
    "Best_Offer_Quote_Condition", "Best_Offer_Exchange",
    "Best_Offer_Price", "Best_Offer_Size",
    "LULD_Indicator", "LULD_NBBO_Indicator",
    "SIP_Generated_Message_Identifier", "Participant_Timestamp",
    "Security_Status_Indicator",
]

TRADE_RENAME = {
    "Symbol": "symbol",
    "Exchange": "exchange",
    "Sale Condition": "sale_condition",
    "Trade Volume": "volume",
    "Trade Price": "price",
    "Trade Stop Stock Indicator": "stop_stock_indicator",
    "Trade Correction Indicator": "correction",
    "Sequence Number": "sequence_number",
    "Trade Id": "trade_id",
    "Source of Trade": "source_of_trade",
    "Trade Reporting Facility": "trade_reporting_facility",
    "Participant Timestamp": "participant_timestamp",
    "Trade Reporting Facility TRF Timestamp": "trf_timestamp",
    "Trade Through Exempt Indicator": "trade_through_exempt",
}

TRADE_SLIM_OUT_COLS = [
    "time", "symbol", "exchange", "sale_condition",
    "volume", "price", "correction",
]

TRADE_RICH_EXTRA_OUT_COLS = [
    "stop_stock_indicator", "sequence_number", "trade_id",
    "source_of_trade", "trade_reporting_facility",
    "participant_timestamp", "trf_timestamp", "trade_through_exempt",
]

NBBO_RENAME = {
    "Symbol": "symbol",
    "Exchange": "exchange",
    "Bid_Price": "bid_price",
    "Bid_Size": "bid_size",
    "Offer_Price": "offer_price",
    "Offer_Size": "offer_size",
    "Quote_Condition": "quote_condition",
    "Sequence_Number": "sequence_number",
    "National_BBO_Ind": "national_bbo_ind",
    "Quote_Cancel_Correction": "quote_cancel_correction",
    "Source_Of_Quote": "source_of_quote",
    "Best Bid Quote Condition": "best_bid_quote_condition",
    "Best_Bid_Exchange": "best_bid_exchange",
    "Best_Bid_Price": "best_bid",
    "Best_Bid_Size": "best_bid_size",
    "Best_Offer_Quote_Condition": "best_offer_quote_condition",
    "Best_Offer_Exchange": "best_offer_exchange",
    "Best_Offer_Price": "best_offer",
    "Best_Offer_Size": "best_offer_size",
    "LULD_Indicator": "luld_indicator",
    "LULD_NBBO_Indicator": "luld_nbbo_indicator",
    "SIP_Generated_Message_Identifier": "sip_generated_message_id",
    "Participant_Timestamp": "participant_timestamp",
    "Security_Status_Indicator": "security_status_indicator",
}

NBBO_SLIM_OUT_COLS = [
    "time", "symbol", "best_bid", "best_bid_size",
    "best_offer", "best_offer_size", "mid",
]

NBBO_RICH_EXTRA_OUT_COLS = [
    "exchange", "bid_price", "bid_size", "offer_price", "offer_size",
    "quote_condition", "sequence_number", "national_bbo_ind",
    "quote_cancel_correction", "source_of_quote",
    "best_bid_quote_condition", "best_bid_exchange",
    "best_offer_quote_condition", "best_offer_exchange",
    "luld_indicator", "luld_nbbo_indicator",
    "sip_generated_message_id", "participant_timestamp",
    "security_status_indicator",
]

# ======================================================================
# C.1 trade-quality filters (following Ait-Sahalia, Jacod, Xiu 2025)
# ======================================================================

def filter_trades_c1(
    df: pd.DataFrame,
    policy: str = "preprocessing",
    *,
    return_qc: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Apply C.1-style trade quality filters.

    1. Keep correction indicator 00 or 01 only.
    2. Remove trades with sale conditions excluded by the selected policy.
    3. Remove trades with price <= 0 or volume <= 0.
    """
    policy = tc.normalize_policy(policy)
    qc: dict[str, int] = {
        "policy_input_rows": int(len(df)),
        "dropped_bad_correction": 0,
        "dropped_preprocess_bad_condition": 0,
        "dropped_eval_bad_condition_if_applied": 0,
        "dropped_bad_price_volume": 0,
        "kept_closing_auction_condition": 0,
        "kept_opening_auction_condition": 0,
    }

    corr = df["Trade Correction Indicator"]
    keep_corr = tc.valid_correction_mask(corr)
    qc["dropped_bad_correction"] = int((~keep_corr).sum())
    work = df.loc[keep_corr].copy()

    sc = work["Sale Condition"].fillna("")
    bad_sc = tc.bad_sale_condition_mask(sc, policy)
    if policy == "preprocessing":
        qc["dropped_preprocess_bad_condition"] = int(bad_sc.sum())
    else:
        qc["dropped_eval_bad_condition_if_applied"] = int(bad_sc.sum())
    work = work.loc[~bad_sc].copy()

    if policy == "preprocessing":
        eval_bad = tc.bad_sale_condition_mask(work["Sale Condition"].fillna(""), "evaluation")
        qc["dropped_eval_bad_condition_if_applied"] = int(eval_bad.sum())

    price = pd.to_numeric(work["Trade Price"], errors="coerce")
    vol = pd.to_numeric(work["Trade Volume"], errors="coerce")
    keep_pv = (price > 0) & (vol > 0)
    qc["dropped_bad_price_volume"] = int((~keep_pv).sum())
    out = work.loc[keep_pv].copy()

    out_sc = out["Sale Condition"].fillna("")
    qc["kept_closing_auction_condition"] = int(tc.auction_condition_mask(out_sc).sum())
    qc["kept_opening_auction_condition"] = int(tc.opening_auction_condition_mask(out_sc).sum())

    if return_qc:
        return out, qc
    return out


# ======================================================================
# Timestamp parsing
# ======================================================================

def parse_taq_time(ts_series: pd.Series, date_str: str) -> pd.Series:
    """Convert TAQ HHMMSSsssssssss timestamps to datetime64[ns]."""
    s = ts_series.astype(str).str.zfill(15)
    iso = (date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8]
           + " " + s.str[:2] + ":" + s.str[2:4] + ":" + s.str[4:6]
           + "." + s.str[6:])
    return pd.to_datetime(iso, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")


# ======================================================================
# Chunked file reader
# ======================================================================

def _resolve_path(data_dir: str, prefix: str, date_str: str) -> str:
    """Find uncompressed or gzipped TAQ file in flat or side-subdir layouts."""
    fname = f"{prefix}_{date_str}"
    subdir = "Trade" if prefix.endswith("_TRADE") else "NBBO"
    search_dirs = []
    for directory in (data_dir, os.path.join(data_dir, subdir)):
        if directory not in search_dirs:
            search_dirs.append(directory)
    candidates = [os.path.join(directory, fname) for directory in search_dirs]
    for base in candidates:
        if os.path.isfile(base):
            return base
        gz = base + ".gz"
        if os.path.isfile(gz):
            return gz
    for directory in search_dirs:
        matches = sorted(glob.glob(os.path.join(directory, fname + "*.gz")))
        if matches:
            return matches[0]
    checked = ", ".join(candidates + [c + ".gz" for c in candidates])
    raise FileNotFoundError(f"No raw TAQ file found. Checked: {checked}")


def _open_file(path: str):
    """Open plain or gzipped file transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _validate_schema(schema: str) -> str:
    """Normalize and validate output schema name."""
    schema = str(schema).lower().strip()
    if schema not in SCHEMAS:
        raise ValueError(f"schema must be one of {sorted(SCHEMAS)}, got {schema!r}")
    return schema


def _ordered_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return requested columns that exist in df, preserving order."""
    return [c for c in cols if c in df.columns]


def _add_qc_counts(total: dict[str, int], part: dict[str, int]) -> None:
    """Accumulate integer QC counters in place."""
    for k, v in part.items():
        total[k] = int(total.get(k, 0)) + int(v)


def _write_trade_qc_summary(
    output_dir: str,
    date_str: str,
    summary: dict,
) -> None:
    """Persist a small machine-readable trade preprocessing QC summary."""
    qc_dir = os.path.join(output_dir, date_str, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    path = os.path.join(qc_dir, "trade_qc_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)


# ======================================================================
# Trade processing
# ======================================================================

def process_trades(
    data_dir: str,
    date_str: str,
    output_dir: str,
    symbols: Optional[set] = None,
    chunksize: int = 500_000,
    schema: str = "slim",
    trade_filter_policy: str = "preprocessing",
) -> dict:
    """Read, filter, and write per-symbol trade Parquet files.

    Returns dict of {symbol: n_rows_written}.
    """
    schema = _validate_schema(schema)
    trade_filter_policy = tc.normalize_policy(trade_filter_policy)
    keep_cols = TRADE_RICH_KEEP_COLS if schema == "rich" else TRADE_KEEP_COLS
    out_cols = TRADE_SLIM_OUT_COLS + (
        TRADE_RICH_EXTRA_OUT_COLS if schema == "rich" else []
    )

    path = _resolve_path(data_dir, "EQY_US_ALL_TRADE", date_str)
    out_base = os.path.join(output_dir, date_str, "trades")
    os.makedirs(out_base, exist_ok=True)

    print(f"[TRADE] Reading {path}")
    print(f"[TRADE] Schema: {schema}")
    print(f"[TRADE] Filter policy: {trade_filter_policy} ({tc.POLICY_VERSION})")
    print(f"[TRADE] Keeping columns: {keep_cols}")

    # Accumulate per symbol
    buffers: dict[str, list[pd.DataFrame]] = {}
    total_read = 0
    total_symbol_rows = 0
    total_kept = 0
    qc_counts: dict[str, int] = {}
    t0 = time.time()

    reader = pd.read_csv(
        path, sep="|", dtype=str, header=0,
        names=TRADE_ALL_COLS, usecols=keep_cols, chunksize=chunksize,
        on_bad_lines="skip",
    )

    try:
        for chunk_i, chunk in enumerate(reader):
            total_read += len(chunk)

            # --- Column selection ---
            df = chunk.copy()

            # --- Symbol filter ---
            df["Symbol"] = df["Symbol"].str.strip()
            if symbols:
                df = df[df["Symbol"].isin(symbols)]
            total_symbol_rows += len(df)

            # --- C.1 filtering ---
            df, chunk_qc = filter_trades_c1(
                df, policy=trade_filter_policy, return_qc=True,
            )
            _add_qc_counts(qc_counts, chunk_qc)

            if df.empty:
                continue

            # --- Parse numerics ---
            df["Trade Price"] = pd.to_numeric(df["Trade Price"], errors="coerce")
            df["Trade Volume"] = pd.to_numeric(df["Trade Volume"], errors="coerce").astype("int32")
            if schema == "rich":
                for c in ["Sequence Number", "Trade Id"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            # --- Parse timestamp ---
            df["time"] = parse_taq_time(df["Time"], date_str)

            # --- Rename to clean schema ---
            df = df.rename(columns=TRADE_RENAME).drop(columns=["Time"])

            # Reorder
            df = df[_ordered_existing(df, out_cols)]

            total_kept += len(df)

            # Buffer by symbol
            for sym, grp in df.groupby("symbol", sort=False):
                buffers.setdefault(sym, []).append(grp)

            if (chunk_i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  ... {total_read:>12,} rows read, "
                      f"{total_kept:>10,} kept, "
                      f"{len(buffers):>5,} symbols, "
                      f"{elapsed:.0f}s")
    except EOFError:
        print(f"[TRADE] WARNING: truncated gzip after {total_read:,} rows — "
              f"writing {total_kept:,} kept rows from {len(buffers)} symbols")

    # --- Write per-symbol Parquet ---
    stats = {}
    for sym, parts in buffers.items():
        df_sym = pd.concat(parts, ignore_index=True)
        safe_name = sym.replace(" ", "_").replace("/", "_")
        outpath = os.path.join(out_base, f"{safe_name}.parquet")
        df_sym.to_parquet(outpath, index=False, engine="pyarrow")
        stats[sym] = len(df_sym)

    elapsed = time.time() - t0
    print(f"[TRADE] Done: {total_read:,} read -> {total_kept:,} kept "
          f"across {len(stats)} symbols in {elapsed:.0f}s")
    _write_trade_qc_summary(output_dir, date_str, {
        "date": date_str,
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
        "symbols_written": len(stats),
        "qc_counts": qc_counts,
    })
    return stats


# ======================================================================
# NBBO processing
# ======================================================================

def filter_nbbo_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Basic NBBO quality filter: drop rows with non-positive best prices."""
    bid = pd.to_numeric(df["Best_Bid_Price"], errors="coerce")
    ask = pd.to_numeric(df["Best_Offer_Price"], errors="coerce")
    return df[(bid > 0) & (ask > 0) & (ask >= bid)].copy()


def process_nbbo(
    data_dir: str,
    date_str: str,
    output_dir: str,
    symbols: Optional[set] = None,
    chunksize: int = 500_000,
    schema: str = "slim",
) -> dict:
    """Read, filter, and write per-symbol NBBO Parquet files.

    Returns dict of {symbol: n_rows_written}.
    """
    schema = _validate_schema(schema)
    keep_cols = NBBO_RICH_KEEP_COLS if schema == "rich" else NBBO_KEEP_COLS
    out_cols = NBBO_SLIM_OUT_COLS + (
        NBBO_RICH_EXTRA_OUT_COLS if schema == "rich" else []
    )

    path = _resolve_path(data_dir, "EQY_US_ALL_NBBO", date_str)
    out_base = os.path.join(output_dir, date_str, "nbbo")
    os.makedirs(out_base, exist_ok=True)

    print(f"[NBBO]  Reading {path}")
    print(f"[NBBO]  Schema: {schema}")
    print(f"[NBBO]  Keeping columns: {keep_cols}")

    buffers: dict[str, list[pd.DataFrame]] = {}
    total_read = 0
    total_kept = 0
    t0 = time.time()

    reader = pd.read_csv(
        path, sep="|", dtype=str, header=0,
        usecols=keep_cols,
        chunksize=chunksize,
        on_bad_lines="skip",
    )

    try:
        for chunk_i, chunk in enumerate(reader):
            total_read += len(chunk)
            df = chunk.copy()

            # --- Quality filter ---
            df = filter_nbbo_quality(df)

            # --- Symbol filter ---
            df["Symbol"] = df["Symbol"].str.strip()
            if symbols:
                df = df[df["Symbol"].isin(symbols)]

            if df.empty:
                continue

            # --- Parse numerics ---
            for c in ["Best_Bid_Price", "Best_Offer_Price"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            for c in ["Best_Bid_Size", "Best_Offer_Size"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")
            if schema == "rich":
                for c in ["Bid_Price", "Offer_Price"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                for c in ["Bid_Size", "Offer_Size", "Sequence_Number"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            # --- Parse timestamp ---
            df["time"] = parse_taq_time(df["Time"], date_str)

            # --- Rename to clean schema ---
            df = df.rename(columns=NBBO_RENAME).drop(columns=["Time"])

            # Add derived midpoint
            df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0

            # Reorder
            df = df[_ordered_existing(df, out_cols)]

            total_kept += len(df)

            for sym, grp in df.groupby("symbol", sort=False):
                buffers.setdefault(sym, []).append(grp)

            if (chunk_i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  ... {total_read:>12,} rows read, "
                      f"{total_kept:>10,} kept, "
                      f"{len(buffers):>5,} symbols, "
                      f"{elapsed:.0f}s")
    except EOFError:
        print(f"[NBBO]  WARNING: truncated gzip after {total_read:,} rows — "
              f"writing {total_kept:,} kept rows from {len(buffers)} symbols")

    # --- Write per-symbol Parquet ---
    stats = {}
    for sym, parts in buffers.items():
        df_sym = pd.concat(parts, ignore_index=True)
        safe_name = sym.replace(" ", "_").replace("/", "_")
        outpath = os.path.join(out_base, f"{safe_name}.parquet")
        df_sym.to_parquet(outpath, index=False, engine="pyarrow")
        stats[sym] = len(df_sym)

    elapsed = time.time() - t0
    print(f"[NBBO]  Done: {total_read:,} read -> {total_kept:,} kept "
          f"across {len(stats)} symbols in {elapsed:.0f}s")
    return stats


# ======================================================================
# Summary report
# ======================================================================

def write_manifest(
    output_dir: str,
    date_str: str,
    trade_stats: dict,
    nbbo_stats: dict,
) -> None:
    """Write a manifest CSV listing all symbols and their row counts."""
    all_syms = sorted(set(trade_stats) | set(nbbo_stats))
    rows = []
    for sym in all_syms:
        rows.append({
            "symbol": sym,
            "trade_rows": trade_stats.get(sym, 0),
            "nbbo_rows": nbbo_stats.get(sym, 0),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, date_str, "manifest.csv")
    df.to_csv(path, index=False)
    print(f"\n[MANIFEST] {path}")
    print(f"  {len(all_syms)} symbols total")
    print(f"  Trades: {df['trade_rows'].sum():,} rows")
    print(f"  NBBO:   {df['nbbo_rows'].sum():,} rows")

    # Top 10 by trade count
    top = df.nlargest(10, "trade_rows")
    print(f"\n  Top 10 symbols by trade count:")
    for _, r in top.iterrows():
        print(f"    {r['symbol']:>8s}  trades={r['trade_rows']:>10,}  nbbo={r['nbbo_rows']:>10,}")


# ======================================================================
# CLI
# ======================================================================

def parse_symbols_arg(args) -> Optional[set]:
    """Resolve --symbols and --symbols-file into a set, or None for all."""
    syms = set()
    if args.symbols:
        syms.update(s.strip() for s in args.symbols)
    if args.symbols_file:
        with open(args.symbols_file) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    syms.add(s)
    return syms if syms else None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw NYSE TAQ files into per-symbol Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", required=True,
                        help="Trading date YYYYMMDD (e.g. 20180501)")
    parser.add_argument("--data-dir", default="../data/raw",
                        help="Directory containing raw TAQ files (default: ../data/raw)")
    parser.add_argument("--output-dir", default="../data/processed",
                        help="Output directory for Parquet files (default: ../data/processed)")
    parser.add_argument("--symbols", nargs="*",
                        help="Specific symbols to keep (e.g. AAPL MSFT)")
    parser.add_argument("--symbols-file",
                        help="File with one symbol per line")
    parser.add_argument("--chunksize", type=int, default=500_000,
                        help="Rows per chunk for reading (default: 500000)")
    parser.add_argument("--schema", choices=sorted(SCHEMAS), default="slim",
                        help="Output schema: slim for analysis, rich for audit metadata (default: slim)")
    parser.add_argument("--trade-filter-policy",
                        choices=sorted(tc.TRADE_FILTER_POLICIES),
                        default="preprocessing",
                        help="Trade sale-condition policy for preprocessing (default: preprocessing)")
    parser.add_argument("--skip-trades", action="store_true",
                        help="Skip trade file processing")
    parser.add_argument("--skip-nbbo", action="store_true",
                        help="Skip NBBO file processing")

    args = parser.parse_args()
    symbols = parse_symbols_arg(args)

    if symbols:
        print(f"Filtering to {len(symbols)} symbols: "
              f"{', '.join(sorted(symbols)[:10])}"
              f"{'...' if len(symbols) > 10 else ''}")
    else:
        print("Processing ALL symbols.")

    print(f"Date: {args.date}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Schema: {args.schema}")
    print(f"Trade filter policy: {args.trade_filter_policy}")
    print()

    trade_stats = {}
    nbbo_stats = {}

    if not args.skip_trades:
        trade_stats = process_trades(
            args.data_dir, args.date, args.output_dir,
            symbols=symbols, chunksize=args.chunksize, schema=args.schema,
            trade_filter_policy=args.trade_filter_policy,
        )
        print()

    if not args.skip_nbbo:
        nbbo_stats = process_nbbo(
            args.data_dir, args.date, args.output_dir,
            symbols=symbols, chunksize=args.chunksize, schema=args.schema,
        )
        print()

    write_manifest(args.output_dir, args.date, trade_stats, nbbo_stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
