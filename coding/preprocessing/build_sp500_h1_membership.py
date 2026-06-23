#!/usr/bin/env python3
"""Build public-approximation S&P 500 2018-2019 membership intervals.

This is a reproducible technical input, not a publication-grade Bloomberg,
CRSP, FactSet, S&P, or another licensed point-in-time constituent file. It starts from the user-provided
2018-01-02 S&P 500 anchor and applies local official-source additions and
removals from SP500/official_sources_2018_2019.csv through 2019-12-31.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE_DIR = Path("reference/index_membership")
DEFAULT_START = DEFAULT_REFERENCE_DIR / "public_sp500_20180102_symbols.csv"
DEFAULT_CHANGES_JSON = Path("SP500/changes.json")
DEFAULT_OFFICIAL_SOURCES = Path("SP500/official_sources_2018_2019.csv")
DEFAULT_TICKER_EVENTS = DEFAULT_REFERENCE_DIR / "sp500_ticker_events_2018_2019.csv"
DEFAULT_OUT = DEFAULT_REFERENCE_DIR / "sp500_membership_intervals.csv"
DEFAULT_AUDIT = DEFAULT_REFERENCE_DIR / "sp500_membership_intervals_audit.json"

START_DATE = dt.date(2018, 1, 2)
END_DATE = dt.date(2019, 12, 31)


# Used only to resolve source/current symbols in the 2018-01-02 anchor and
# removal tickers back to the active historical TAQ ticker. Additions use the
# event ticker as listed in the local official-source file.
HISTORICAL_TAQ_SYMBOLS = {
    "BKNG": "PCLN",
    "BRK.B": "BRK B",
    "BRK/B": "BRK B",
    "CBRE": "CBG",
    "CMCSA": "CMCS A",
    "CPRI": "KORS",
    "DISCA": "DISC A",
    "DISCK": "DISC K",
    "GOOGL": "GOOG L",
    "JEF": "LUK",
    "KDP": "DPS",
    "META": "FB",
    "WELL": "HCN",
    "WYND": "WYN",
}


FIELDNAMES = [
    "index_id",
    "symbol",
    "effective_from",
    "effective_to",
    "company_name",
    "sector",
    "listing_exchange",
    "source",
    "source_note",
]


def _normalize_symbol(symbol: str) -> str:
    out = str(symbol or "").strip().upper().replace("/", " ").replace(".", " ")
    return " ".join(out.split())


def _normalize_anchor_symbol(symbol: str) -> str:
    raw = _normalize_symbol(symbol)
    return HISTORICAL_TAQ_SYMBOLS.get(raw, raw)


def _normalize_event_add_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol)


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def _load_start_rows(path: Path, *, start_date: dt.date) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    active: dict[str, dict[str, str]] = {}
    source_to_taq: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = _normalize_anchor_symbol(row.get("symbol", ""))
            source_symbol = _normalize_symbol(row.get("source_symbol", symbol))
            if not symbol:
                continue
            row = dict(row)
            row["symbol"] = symbol
            row["source_symbol"] = source_symbol
            row["_effective_from"] = start_date.isoformat()
            active.setdefault(symbol, row)
            source_to_taq[source_symbol] = symbol
            source_to_taq[_normalize_anchor_symbol(source_symbol)] = symbol
            source_to_taq[symbol] = symbol
    return active, source_to_taq


def _close_interval(
    intervals: list[dict[str, str]],
    row: dict[str, str],
    symbol: str,
    effective: dt.date,
    *,
    source: str,
    source_note: str,
) -> None:
    intervals.append({
        "index_id": "sp500",
        "symbol": symbol,
        "effective_from": row.get("_effective_from", START_DATE.isoformat()),
        "effective_to": (effective - dt.timedelta(days=1)).isoformat(),
        "company_name": row.get("company_name", ""),
        "sector": row.get("sector", ""),
        "listing_exchange": row.get("listing_exchange", ""),
        "source": source,
        "source_note": source_note,
    })


def _active_removal_symbol(raw_symbol: str, source_to_taq: dict[str, str]) -> str:
    raw = _normalize_symbol(raw_symbol)
    return source_to_taq.get(raw, source_to_taq.get(_normalize_anchor_symbol(raw), _normalize_anchor_symbol(raw)))


def _load_official_source_changes(path: Path) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("effective_date"):
                continue
            changes.append({
                "effective": _parse_date(row["effective_date"]),
                "addition_ticker": _normalize_event_add_symbol(row.get("add_ticker", "")),
                "addition_name": str(row.get("add_name", "") or ""),
                "removal_ticker": _normalize_symbol(row.get("remove_ticker", "")),
                "removal_name": str(row.get("remove_name", "") or ""),
                "event_type": str(row.get("event_type", "") or ""),
                "source_url": str(row.get("official_source_url", "") or ""),
                "source_status": str(row.get("source_status", "") or ""),
                "notes": str(row.get("notes", "") or ""),
            })
    return sorted(changes, key=lambda item: (item["effective"], item["addition_ticker"], item["removal_ticker"]))


def _load_ticker_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("effective_date"):
                continue
            events.append({
                "effective": _parse_date(row["effective_date"]),
                "old_symbol": _normalize_anchor_symbol(row.get("old_symbol", "")),
                "new_symbol": _normalize_event_add_symbol(row.get("new_symbol", "")),
                "event_type": str(row.get("event_type", "") or ""),
                "source": str(row.get("source", "") or ""),
                "source_note": str(row.get("source_note", "") or ""),
            })
    return sorted(events, key=lambda item: (item["effective"], item["old_symbol"], item["new_symbol"]))


def _load_changes_json_backstop(path: Path, *, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for item in raw.get("changes", []):
        if not item.get("effectiveDate"):
            continue
        effective = _parse_date(item["effectiveDate"])
        if start_date < effective <= end_date:
            add = item.get("addition") or {}
            rem = item.get("removal") or {}
            out.append({
                "effective": effective.isoformat(),
                "addition_ticker": _normalize_symbol(add.get("ticker", "")),
                "removal_ticker": _normalize_symbol(rem.get("ticker", "")),
                "source_url": item.get("sourceUrl", ""),
            })
    return sorted(out, key=lambda item: (item["effective"], item["addition_ticker"], item["removal_ticker"]))


def build_intervals(
    start_file: Path,
    official_sources_file: Path,
    out_file: Path,
    audit_file: Path,
    *,
    ticker_events_file: Path = DEFAULT_TICKER_EVENTS,
    changes_json_file: Path = DEFAULT_CHANGES_JSON,
    start_date: dt.date = START_DATE,
    end_date: dt.date = END_DATE,
) -> dict[str, Any]:
    active, source_to_taq = _load_start_rows(start_file, start_date=start_date)
    intervals: list[dict[str, str]] = []
    applied_changes: list[dict[str, str]] = []
    applied_ticker_events: list[dict[str, str]] = []
    skipped_ticker_events: list[dict[str, str]] = []
    warnings: list[str] = []

    official_changes = [
        c for c in _load_official_source_changes(official_sources_file)
        if start_date < c["effective"] <= end_date
    ]
    ticker_events = [
        e for e in _load_ticker_events(ticker_events_file)
        if start_date < e["effective"] <= end_date
    ]
    dates = sorted({c["effective"] for c in official_changes} | {e["effective"] for e in ticker_events})

    for effective in dates:
        day_official = [c for c in official_changes if c["effective"] == effective]
        day_ticker = [e for e in ticker_events if e["effective"] == effective]

        for change in day_official:
            add_symbol = change["addition_ticker"]
            rem_source = change["removal_ticker"]
            rem_symbol = _active_removal_symbol(rem_source, source_to_taq) if rem_source else ""
            source_status = change.get("source_status", "")
            source_url = change.get("source_url", "")
            event_type = change.get("event_type", "")
            source = "sp500_official_sources_2018_2019_csv"

            if rem_symbol:
                existing = active.pop(rem_symbol, None)
                if existing is None:
                    warnings.append(
                        f"{effective.isoformat()}: removal {rem_source}->{rem_symbol} was not active"
                    )
                else:
                    _close_interval(
                        intervals,
                        existing,
                        rem_symbol,
                        effective,
                        source=source,
                        source_note=(
                            f"Removed by SP500/official_sources_2018_2019.csv on "
                            f"{effective.isoformat()} ({rem_source}); event={event_type}; "
                            f"source_status={source_status}; url={source_url}"
                        ),
                    )

            if add_symbol:
                active[add_symbol] = {
                    "symbol": add_symbol,
                    "source_symbol": add_symbol,
                    "company_name": change.get("addition_name", ""),
                    "sector": "",
                    "listing_exchange": "",
                    "_effective_from": effective.isoformat(),
                }
                source_to_taq[add_symbol] = add_symbol
                source_to_taq[_normalize_anchor_symbol(add_symbol)] = add_symbol

            applied_changes.append({
                "effective_date": effective.isoformat(),
                "addition_symbol": add_symbol,
                "removal_source": rem_source,
                "removal_symbol": rem_symbol,
                "event_type": event_type,
                "source_status": source_status,
                "source_url": source_url,
            })

        for event in day_ticker:
            old_symbol = _active_removal_symbol(event["old_symbol"], source_to_taq)
            new_symbol = event["new_symbol"]
            if not old_symbol or not new_symbol:
                continue
            existing = active.pop(old_symbol, None)
            if existing is None:
                skipped_ticker_events.append({
                    "effective_date": effective.isoformat(),
                    "old_symbol": old_symbol,
                    "new_symbol": new_symbol,
                    "reason": "old_symbol_not_active_or_already_handled_by_official_change",
                    "event_type": event.get("event_type", ""),
                })
                source_to_taq[event["old_symbol"]] = new_symbol
                continue

            _close_interval(
                intervals,
                existing,
                old_symbol,
                effective,
                source="sp500_ticker_events_2018_2019_csv",
                source_note=(
                    f"Ticker event {old_symbol}->{new_symbol} on {effective.isoformat()}; "
                    f"event={event.get('event_type', '')}; source={event.get('source', '')}; "
                    f"note={event.get('source_note', '')}"
                ),
            )
            if new_symbol in active:
                skipped_ticker_events.append({
                    "effective_date": effective.isoformat(),
                    "old_symbol": old_symbol,
                    "new_symbol": new_symbol,
                    "reason": "new_symbol_already_active_after_official_change_old_closed_only",
                    "event_type": event.get("event_type", ""),
                })
            else:
                active[new_symbol] = {
                    "symbol": new_symbol,
                    "source_symbol": new_symbol,
                    "company_name": existing.get("company_name", ""),
                    "sector": existing.get("sector", ""),
                    "listing_exchange": existing.get("listing_exchange", ""),
                    "_effective_from": effective.isoformat(),
                }
            source_to_taq[event["old_symbol"]] = new_symbol
            source_to_taq[old_symbol] = new_symbol
            source_to_taq[new_symbol] = new_symbol
            applied_ticker_events.append({
                "effective_date": effective.isoformat(),
                "old_symbol": old_symbol,
                "new_symbol": new_symbol,
                "event_type": event.get("event_type", ""),
                "source": event.get("source", ""),
            })

    for symbol, row in sorted(active.items()):
        intervals.append({
            "index_id": "sp500",
            "symbol": symbol,
            "effective_from": row.get("_effective_from", start_date.isoformat()),
            "effective_to": end_date.isoformat(),
            "company_name": row.get("company_name", ""),
            "sector": row.get("sector", ""),
            "listing_exchange": row.get("listing_exchange", ""),
            "source": "public_sp500_20180102_anchor_plus_official_sources_2018_2019_csv",
            "source_note": (
                "Public technical approximation from user-provided 2018-01-02 anchor plus "
                "SP500/official_sources_2018_2019.csv and local ticker events; validate with "
                "an independently licensed point-in-time constituent source before publication-grade claims."
            ),
        })

    intervals = sorted(intervals, key=lambda r: (r["effective_from"], r["symbol"], r["effective_to"]))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(intervals)

    backstop = _load_changes_json_backstop(changes_json_file, start_date=start_date, end_date=end_date)

    audit = {
        "index_id": "sp500",
        "not_publication_grade": True,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "start_file": str(start_file),
        "official_sources_file": str(official_sources_file),
        "ticker_events_file": str(ticker_events_file),
        "changes_json_file": str(changes_json_file),
        "out_file": str(out_file),
        "interval_rows": len(intervals),
        "active_on_end": len(active),
        "applied_changes": applied_changes,
        "applied_change_count": len(applied_changes),
        "applied_ticker_events": applied_ticker_events,
        "applied_ticker_event_count": len(applied_ticker_events),
        "skipped_ticker_events": skipped_ticker_events,
        "warnings": warnings,
        "changes_json_backstop_rows": len(backstop),
        "note": (
            "Technical 2018-2019 public approximation. It is suitable for preprocessing, "
            "pipeline validation, and exploratory evaluation, but final thesis claims still "
            "need validated point-in-time S&P 500 membership from an independently licensed point-in-time constituent source."
        ),
    }
    audit_file.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-file", type=Path, default=DEFAULT_START)
    parser.add_argument("--official-sources-file", type=Path, default=DEFAULT_OFFICIAL_SOURCES)
    parser.add_argument("--ticker-events-file", type=Path, default=DEFAULT_TICKER_EVENTS)
    parser.add_argument("--changes-json-file", type=Path, default=DEFAULT_CHANGES_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--start-date", default=START_DATE.isoformat())
    parser.add_argument("--end-date", default=END_DATE.isoformat())
    args = parser.parse_args()

    audit = build_intervals(
        args.start_file,
        args.official_sources_file,
        args.out,
        args.audit,
        ticker_events_file=args.ticker_events_file,
        changes_json_file=args.changes_json_file,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
    )
    print(json.dumps({
        "out_file": audit["out_file"],
        "interval_rows": audit["interval_rows"],
        "active_on_end": audit["active_on_end"],
        "applied_changes": audit["applied_change_count"],
        "applied_ticker_events": audit["applied_ticker_event_count"],
        "warnings": len(audit["warnings"]),
    }, indent=2))


if __name__ == "__main__":
    main()
