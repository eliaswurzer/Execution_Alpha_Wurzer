#!/usr/bin/env python3
"""Build public approximate 2018-01-02 S&P500/Nasdaq100 start lists.

The preferred final thesis source is Bloomberg/CRSP point-in-time membership.
This script is only for a public-source technical pilot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd


WIKI_API = "https://en.wikipedia.org/w/api.php"
SP500_TITLE = "List of S&P 500 companies"
NASDAQ100_TITLE = "Nasdaq-100"


def normalize_taq_symbol(symbol: str, date: dt.date = dt.date(2018, 1, 2)) -> str:
    """Normalize public-source symbols to observed 2018 TAQ conventions."""
    s = str(symbol).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", " ").replace(".", " ")
    if date < dt.date(2022, 6, 9) and s == "META":
        return "FB"
    if s == "GOOGL":
        return "GOOG L"
    return s


def _revision_url(title: str, date: dt.date) -> tuple[str, str]:
    """Return a Wikipedia old-revision URL at or before date end UTC."""
    import requests

    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvprop": "ids|timestamp",
        "rvstart": f"{date.isoformat()}T23:59:59Z",
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    rev = page["revisions"][0]
    oldid = rev["revid"]
    ts = rev["timestamp"]
    url_title = title.replace(" ", "_")
    return f"https://en.wikipedia.org/w/index.php?title={url_title}&oldid={oldid}", ts


def _find_symbol_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        label = " ".join(str(x) for x in (c if isinstance(c, tuple) else [c])).lower()
        if "symbol" in label or "ticker" in label:
            return c
    return None


def _extract_symbols_from_tables(url: str, min_count: int) -> pd.DataFrame:
    tables = pd.read_html(url)
    candidates = []
    for table in tables:
        col = _find_symbol_column(table)
        if col is None:
            continue
        syms = (
            table[col].astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
        )
        syms = [s for s in syms if s and s.lower() not in {"nan", "symbol", "ticker"}]
        if len(syms) >= min_count:
            candidates.append((len(syms), table, col))
    if not candidates:
        raise RuntimeError(f"No table with >= {min_count} symbols found at {url}")
    _, table, col = max(candidates, key=lambda x: x[0])
    out = pd.DataFrame({"source_symbol": table[col].astype(str)})
    name_col = None
    for c in table.columns:
        label = " ".join(str(x) for x in (c if isinstance(c, tuple) else [c])).lower()
        if "company" in label or "security" in label or "name" in label:
            name_col = c
            break
    out["company_name"] = table[name_col].astype(str) if name_col is not None else ""
    return out


def build_lists(date: dt.date, out_dir: Path) -> dict[str, pd.DataFrame]:
    sp_url, sp_ts = _revision_url(SP500_TITLE, date)
    ndx_url, ndx_ts = _revision_url(NASDAQ100_TITLE, date)

    sp = _extract_symbols_from_tables(sp_url, min_count=450)
    ndx = _extract_symbols_from_tables(ndx_url, min_count=90)
    outputs = {}
    for index_id, frame, url, ts, min_count in (
        ("sp500", sp, sp_url, sp_ts, 450),
        ("nasdaq100", ndx, ndx_url, ndx_ts, 90),
    ):
        out = frame.copy()
        out["symbol"] = [normalize_taq_symbol(s, date) for s in out["source_symbol"]]
        out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
        if len(out) < min_count:
            raise RuntimeError(f"{index_id} list too small after normalization: {len(out)}")
        out["index_id"] = index_id
        out["asof_date"] = date.isoformat()
        out["source_url"] = url
        out["source_revision_timestamp"] = ts
        outputs[index_id] = out[
            ["index_id", "asof_date", "symbol", "source_symbol", "company_name",
             "source_url", "source_revision_timestamp"]
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs["sp500"].to_csv(out_dir / "public_sp500_20180102_symbols.csv", index=False)
    outputs["nasdaq100"].to_csv(out_dir / "public_nasdaq100_20180102_symbols.csv", index=False)
    union = sorted(set(outputs["sp500"]["symbol"]) | set(outputs["nasdaq100"]["symbol"]))
    (out_dir / "public_index_union_20180102_symbols.txt").write_text(
        "\n".join(union) + "\n", encoding="utf-8",
    )
    meta = {
        "date": date.isoformat(),
        "sp500_count": int(len(outputs["sp500"])),
        "nasdaq100_count": int(len(outputs["nasdaq100"])),
        "union_count": int(len(union)),
        "sources": {
            "sp500": sp_url,
            "nasdaq100": ndx_url,
        },
    }
    (out_dir / "public_index_union_20180102_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date(2018, 1, 2))
    parser.add_argument("--out-dir", type=Path, default=Path("reference/index_membership"))
    args = parser.parse_args()
    outputs = build_lists(args.date, args.out_dir)
    union = sorted(set(outputs["sp500"]["symbol"]) | set(outputs["nasdaq100"]["symbol"]))
    print(f"sp500 symbols: {len(outputs['sp500'])}")
    print(f"nasdaq100 symbols: {len(outputs['nasdaq100'])}")
    print(f"union symbols: {len(union)}")
    print(args.out_dir / "public_index_union_20180102_symbols.txt")


if __name__ == "__main__":
    main()
