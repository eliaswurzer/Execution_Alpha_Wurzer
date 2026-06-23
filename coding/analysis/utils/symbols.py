"""Symbol alias helpers for TAQ file-safe tickers and artifact tickers."""

from __future__ import annotations

from collections.abc import Mapping


def canonical_symbol(symbol: str) -> str:
    """Return the logical TAQ symbol used in analysis tables.

    File-safe underscores and vendor share-class dots are aliases, not
    distinct securities.
    """
    return str(symbol).strip().upper().replace("_", " ").replace(".", " ")


def symbol_aliases(symbol: str) -> set[str]:
    """Return common aliases for symbols such as ``BRK B`` / ``BRK_B``."""
    s = canonical_symbol(symbol)
    aliases = {s}
    for old, new in ((" ", "_"),):
        if old in s:
            aliases.add(s.replace(old, new))
    return aliases


def expand_symbol_to_tier(mapping: Mapping[str, int]) -> dict[str, int]:
    """Add non-conflicting ticker aliases to a symbol-to-tier mapping."""
    out: dict[str, int] = {}
    for symbol, tier in mapping.items():
        tier_int = int(tier)
        for alias in symbol_aliases(str(symbol)):
            out.setdefault(alias, tier_int)
    return out
