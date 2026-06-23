"""Build the derived per-symbol listing-exchange reference map.

The licensed membership snapshot leaves ``listing_exchange`` empty, so H1
previously re-derived the venue from the trade tape on every run. The primary
source here is the membership RIC root recorded in the membership
``source_note`` column (suffix ``.N`` = NYSE, ``.OQ`` = Nasdaq, ``.Z`` =
Cboe BZX): this is authoritative listing information and matches the known
NYSE/Nasdaq split of the index. Only symbols without a usable RIC root fall
back to the trade-tape heuristic (``listing_exchange_from_trades``) sampled
on three days of the membership interval with a majority vote; the tape
heuristic alone misclassifies a substantial share of NYSE names because most
tape volume executes off the listing venue. Output:
``reference/index_membership/<index>_listing_exchange.csv``;
``load_index_membership`` merges the map into blank ``listing_exchange``
values, so the venue then flows into every simulated panel row.

Usage::

    python -m analysis.runners.build_listing_exchange_map --universe sp500
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ..data.index_universe import (
    listing_exchange_map_path,
    load_index_membership,
    membership_path,
    normalize_index_id,
)
from ..data.listing_exchange import listing_exchange_from_trades
from ..data.taq_loader import list_dates, load_trades
from ._common import _eval_dates

log = logging.getLogger(__name__)

DEFAULT_SAMPLES_PER_SYMBOL = 3

_RIC_SUFFIX_TO_VENUE = {
    "N": "NYSE",
    "A": "NYSE",      # NYSE American
    "OQ": "NASDAQ",
    "O": "NASDAQ",
    "Z": "CBOE",      # Cboe BZX listings
}
_RIC_PATTERN = re.compile(r"\b[A-Za-z0-9]+\.([A-Z]+)\b")


def _ric_listing(source_notes: pd.Series) -> str | None:
    """Unanimous listing venue from RIC root suffixes, else None."""
    venues: set[str] = set()
    for note in source_notes.dropna().astype(str):
        for suffix in _RIC_PATTERN.findall(note):
            venue = _RIC_SUFFIX_TO_VENUE.get(suffix)
            if venue:
                venues.add(venue)
    if len(venues) == 1:
        return next(iter(venues))
    return None


def _sample_dates(
    available: list[_dt.date],
    start: _dt.date,
    end: _dt.date,
    n_samples: int,
) -> list[_dt.date]:
    window = [d for d in available if start <= d <= end]
    if not window:
        return []
    if len(window) <= n_samples:
        return window
    idx = [round(i * (len(window) - 1) / (n_samples - 1)) for i in range(n_samples)]
    return [window[i] for i in sorted(set(idx))]


def build_map(
    universe: str,
    *,
    root: Path | None = None,
    n_samples: int = DEFAULT_SAMPLES_PER_SYMBOL,
    out_path: Path | None = None,
) -> pd.DataFrame:
    idx = normalize_index_id(universe)
    membership = load_index_membership(idx, root)
    intervals = (
        membership.groupby("symbol")
        .agg(
            start=("effective_from", "min"),
            end=("effective_to", "max"),
            ric_listing=("source_note", _ric_listing),
        )
        .reset_index()
    )
    years = sorted({intervals["start"].min().year, intervals["end"].max().year})
    available = _eval_dates(
        _dt.date(min(years), 1, 1), _dt.date(max(years), 12, 31),
    )
    if not available:
        raise RuntimeError("No preprocessed TAQ dates available for sampling")

    rows: list[dict] = []
    for item in intervals.itertuples(index=False):
        if item.ric_listing:
            rows.append({
                "symbol": item.symbol,
                "listing_exchange": item.ric_listing,
                "votes_nyse": 0,
                "votes_nasdaq": 0,
                "sample_dates": "",
                "method": "membership_ric_root_v1",
            })
            continue
        samples = _sample_dates(available, item.start, item.end, n_samples)
        votes: Counter[str] = Counter()
        used: list[str] = []
        for day in samples:
            try:
                trades = load_trades(day, item.symbol, rth_only=False)
            except FileNotFoundError:
                continue
            votes[listing_exchange_from_trades(trades)] += 1
            used.append(day.isoformat())
        listing = votes.most_common(1)[0][0] if votes else "NASDAQ"
        rows.append({
            "symbol": item.symbol,
            "listing_exchange": listing,
            "votes_nyse": votes.get("NYSE", 0),
            "votes_nasdaq": votes.get("NASDAQ", 0),
            "sample_dates": " ".join(used),
            "method": "tape_heuristic_majority_v1",
        })

    frame = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    target = out_path or listing_exchange_map_path(idx, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    log.info(
        "Wrote %s: %d symbols (%s) — methods: %s — from %s",
        target, len(frame),
        frame["listing_exchange"].value_counts().to_dict(),
        frame["method"].value_counts().to_dict(),
        membership_path(idx, root),
    )
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", default="sp500")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_SYMBOL)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    build_map(args.universe, n_samples=args.samples, out_path=args.out)


if __name__ == "__main__":
    main()
