"""
adverse_selection.py -- Glosten-1988 Regression fuer Adverse-Selection-Kosten
(Thesis §4.2.4 Eq. 4.3/4.4).

Modell::

    Delta m_tau = alpha + beta * 1_{fill} + epsilon

mit ``Delta m_tau`` = Midquote-Change ueber Horizont Delta (Default 5 Minuten)
nach einem Kandidaten-Zeitpunkt, und ``1_{fill}`` = 1 wenn ein passiver
Limit-Order in diesem Fenster gefuellt worden waere.

Der Koeffizient ``beta`` ist der differentielle Preis-Move bei Fill,
interpretiert als Adverse-Selection-Kosten (signiert je Side).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg

log = logging.getLogger(__name__)


@dataclass
class GlostenASModel:
    """Haelt die geschaetzten (alpha, beta) und Standardfehler.

    Side-Specific: BUY-Limits filllen wenn Seller kommen (Delta m < 0), SELL-
    Limits wenn Buyer kommen (Delta m > 0). Der Koeffizient ist fuer beide
    Seiten getrennt gespeichert.
    """

    alpha_buy: float = float("nan")
    beta_buy: float = float("nan")
    alpha_sell: float = float("nan")
    beta_sell: float = float("nan")
    n_buy: int = 0
    n_sell: int = 0

    def as_bps(self, side: str) -> float:
        """Adverse-Selection-Kosten in *Absolutwerten* Basispunkten.

        Das Vorzeichen ist in Thesis Eq. 4.4 negativ fuer Buy-Orders (Preis
        faellt nach Fill). Da in ``net_alpha`` ``-phi * |AS|`` steht,
        liefern wir hier ``|beta|`` als nicht-negativen Wert.
        """
        b = self.beta_buy if side.upper() == "BUY" else self.beta_sell
        if np.isnan(b):
            return float("nan")
        return abs(float(b))


# ---------------------------------------------------------------------------
# Panel construction: fills vs non-fills
# ---------------------------------------------------------------------------

def build_as_panel(
    nbbo: pd.DataFrame,
    trades: pd.DataFrame,
    horizon_seconds: int = cfg.AS_HORIZON_SECONDS,
    sample_every_seconds: int = 60,
    max_rows: int | None = None,
    sample_seed: int | None = None,
) -> pd.DataFrame:
    """Panel mit Delta m_tau und Fill-Indikator pro Zeitpunkt und Seite.

    Liefert eine long-format-Tabelle mit Spalten ``t0``, ``side``, ``dm_bps``,
    ``fill``.
    """
    if nbbo.empty or trades.empty:
        return pd.DataFrame(columns=["t0", "side", "dm_bps", "fill"])

    nb = nbbo.sort_values("time").reset_index(drop=True)
    tr = trades.sort_values("time")[["time", "price"]].reset_index(drop=True)

    grid = pd.date_range(nb["time"].iloc[0], nb["time"].iloc[-1], freq=f"{sample_every_seconds}s")
    nb_times = nb["time"].values.astype("datetime64[ns]").astype("int64")
    tr_times = tr["time"].values.astype("datetime64[ns]").astype("int64")
    tr_prices = tr["price"].to_numpy(dtype=float)
    bids = nb["best_bid"].to_numpy(dtype=float)
    asks = nb["best_offer"].to_numpy(dtype=float)
    mids = nb["mid"].to_numpy(dtype=float)
    grid_ns = grid.values.astype("datetime64[ns]").astype("int64")
    horizon_ns = int(horizon_seconds * 1_000_000_000)

    start_idx = np.searchsorted(nb_times, grid_ns, side="right") - 1
    end_idx = np.searchsorted(nb_times, grid_ns + horizon_ns, side="right") - 1
    safe_start = np.maximum(start_idx, 0)
    safe_end = np.maximum(end_idx, 0)
    valid = (start_idx >= 0) & (end_idx >= 0)
    valid &= np.isfinite(bids[safe_start]) & np.isfinite(asks[safe_start])
    valid &= np.isfinite(mids[safe_end])
    valid &= (bids[safe_start] > 0) & (asks[safe_start] > 0)
    valid &= asks[safe_start] > bids[safe_start]
    valid &= mids[safe_end] > 0

    grid = grid[valid]
    grid_ns = grid_ns[valid]
    start_idx = start_idx[valid]
    end_idx = end_idx[valid]
    if len(grid) == 0:
        return pd.DataFrame(columns=["t0", "side", "dm_bps", "fill"])

    total_candidates = len(grid) * 2
    selected_ordinals: set[int] | None = None
    if max_rows is not None and max_rows > 0 and total_candidates > max_rows:
        rng = np.random.default_rng(sample_seed if sample_seed is not None else cfg.DEFAULT_SEED)
        selected_ordinals = set(
            int(x) for x in rng.choice(total_candidates, size=int(max_rows), replace=False)
        )

    rows = []
    row_ordinal = 0
    for t0, t0_ns, q_idx, q_end_idx in zip(grid, grid_ns, start_idx, end_idx):
        bid = float(bids[q_idx])
        ask = float(asks[q_idx])
        mid0 = 0.5 * (bid + ask)
        mid_end = float(mids[q_end_idx])
        if mid0 <= 0 or mid_end <= 0:
            row_ordinal += 2
            continue
        dm_bps = (mid_end - mid0) / mid0 * 1e4
        t_end_ns = int(t0_ns + horizon_ns)
        tr_start = int(np.searchsorted(tr_times, t0_ns, side="right"))
        tr_end = int(np.searchsorted(tr_times, t_end_ns, side="right"))
        win_prices = tr_prices[tr_start:tr_end]
        buy_fill = int(win_prices.size > 0 and bool((win_prices <= bid).any()))
        sell_fill = int(win_prices.size > 0 and bool((win_prices >= ask).any()))

        for side, fill in (("BUY", buy_fill), ("SELL", sell_fill)):
            include = selected_ordinals is None or row_ordinal in selected_ordinals
            row_ordinal += 1
            if include:
                rows.append({"t0": t0, "side": side, "dm_bps": float(dm_bps), "fill": int(fill)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_glosten_as(panel: pd.DataFrame) -> GlostenASModel:
    """Schaetzt side-specific ``alpha, beta`` per OLS."""
    if panel.empty:
        return GlostenASModel()

    result = GlostenASModel()
    for side in ("BUY", "SELL"):
        sub = panel[panel["side"] == side].dropna(subset=["dm_bps", "fill"])
        if len(sub) < 30 or sub["fill"].sum() < 5:
            continue
        x = sub["fill"].to_numpy(dtype=float)
        y = sub["dm_bps"].to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            log.warning("fit_glosten_as: non-finite values in %s regression — skipping", side)
            continue
        X = np.column_stack([np.ones_like(x), x])
        # Standard OLS
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        if side == "BUY":
            result.alpha_buy, result.beta_buy, result.n_buy = alpha, beta, len(sub)
        else:
            result.alpha_sell, result.beta_sell, result.n_sell = alpha, beta, len(sub)
    return result


# ---------------------------------------------------------------------------
# Multi-Horizont Schaetzung und Persistierung
# ---------------------------------------------------------------------------

def fit_glosten_as_horizons(
    nbbo: pd.DataFrame,
    trades: pd.DataFrame,
    horizons_seconds: tuple[int, ...] = cfg.AS_HORIZON_GRID_SECONDS,
    sample_every_seconds: int = 60,
) -> dict[int, GlostenASModel]:
    """Schaetzt das Glosten-Modell fuer jede der angegebenen Horizonte.

    Liefert einen dict ``horizon_seconds -> GlostenASModel``. Headline-Run
    nutzt ``cfg.AS_HEADLINE_HORIZON_SECONDS`` (30s), Robustness ueber alle.
    """
    results: dict[int, GlostenASModel] = {}
    for h in horizons_seconds:
        panel = build_as_panel(
            nbbo, trades, horizon_seconds=h,
            sample_every_seconds=sample_every_seconds,
        )
        results[int(h)] = fit_glosten_as(panel)
    return results


def save_glosten_grid(
    grid: dict[int, GlostenASModel], out_dir: Path,
) -> None:
    """Persistiert alle Horizont-Modelle als CSV (panel-friendly).

    Ein File ``glosten_as_horizons.csv`` mit Spalten ``horizon_seconds``,
    ``alpha_buy``, ``beta_buy``, ``alpha_sell``, ``beta_sell``,
    ``n_buy``, ``n_sell``. Zusaetzlich ein JSON-Pickle des Headline-Modells
    unter ``glosten_as.csv`` (kompatibel zu altem Loader).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, m in sorted(grid.items()):
        d = asdict(m)
        d["horizon_seconds"] = int(h)
        rows.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "glosten_as_horizons.csv", index=False)

    headline = grid.get(int(cfg.AS_HEADLINE_HORIZON_SECONDS))
    if headline is None and grid:
        headline = grid[sorted(grid.keys())[len(grid) // 2]]
    if headline is not None:
        pd.DataFrame([asdict(headline)]).to_csv(out_dir / "glosten_as.csv", index=False)


def load_glosten_grid(in_dir: Path) -> dict[int, GlostenASModel]:
    """Laedt das Horizont-Gitter, falls vorhanden. Sonst leeres Dict."""
    p = in_dir / "glosten_as_horizons.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[int, GlostenASModel] = {}
    for _, row in df.iterrows():
        h = int(row["horizon_seconds"])
        out[h] = GlostenASModel(
            alpha_buy=float(row.get("alpha_buy", float("nan"))),
            beta_buy=float(row.get("beta_buy", float("nan"))),
            alpha_sell=float(row.get("alpha_sell", float("nan"))),
            beta_sell=float(row.get("beta_sell", float("nan"))),
            n_buy=int(row.get("n_buy", 0) or 0),
            n_sell=int(row.get("n_sell", 0) or 0),
        )
    return out


def select_horizon(
    grid: dict[int, GlostenASModel],
    horizon_seconds: int = cfg.AS_HEADLINE_HORIZON_SECONDS,
) -> GlostenASModel:
    """Liefert das Modell zum gewaehlten Horizont; faellt bei Bedarf
    auf den naechstgelegenen vorhandenen Horizont zurueck."""
    if not grid:
        return GlostenASModel()
    if horizon_seconds in grid:
        return grid[horizon_seconds]
    # naechstgelegener Horizont
    closest = min(grid.keys(), key=lambda k: abs(k - horizon_seconds))
    log.warning("AS horizon %ds nicht gefittet -- nutze %ds", horizon_seconds, closest)
    return grid[closest]
