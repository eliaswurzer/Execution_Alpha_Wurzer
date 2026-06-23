"""End-to-end S5 value-model smoke workflow.

Default synthetic mode is fast and independent of external TAQ data. Optional
real-data mode is a tiny path-wiring smoke and must not be read as empirical
evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data.taq_loader import (
    extract_closing_auction_details,
    filter_trades_near_quotes,
    filter_valid_quotes,
    filter_valid_trades,
    load_symbol_day,
)
from ..fill_model.rolling import assign_monthly_anchor, build_monthly_training_schedule
from ..fill_model.state_vector import build_event_panel
from ..fill_model.value_model import SideTieredXGBValueModel, attach_value_labels
from ..reporting.preliminary_templates import write_preliminary_templates
from ..reporting.static_posting_curve import posting_curve_summary, save_posting_curve_figure
from ..strategies.base import MarketState
from ..strategies.value_aware import ValueAwareXGBStrategy

REQUIRED_CANDIDATE_COLUMNS = [
    "symbol", "date", "side", "tier", "sector", "listing_exchange",
    "limit_offset_bps", "event", "close_price", "limit_price",
    "adverse_selection_bps", "size_frac", "target_net_alpha_vs_moc_bps",
]


class _ConstantValueModel:
    def __init__(self, values: list[float]):
        self.values = list(values)

    def predict_candidates(self, candidates: pd.DataFrame) -> np.ndarray:
        vals = self.values[: len(candidates)]
        if len(vals) < len(candidates):
            vals.extend([vals[-1] if vals else float("-inf")] * (len(candidates) - len(vals)))
        return np.asarray(vals, dtype=float)


def _business_dates(start: dt.date, periods: int) -> list[dt.date]:
    return [pd.Timestamp(x).date() for x in pd.bdate_range(start, periods=periods)]


def build_synthetic_candidate_panel(
    *,
    n_dates: int = 90,
    symbols: tuple[str, ...] = ("AAPL", "MSFT"),
    offset_grid_bps: tuple[float, ...] = cfg.VALUE_MODEL_OFFSET_GRID_BPS,
) -> pd.DataFrame:
    """Create a deterministic candidate panel with side/tier/sector variation."""
    rows: list[dict] = []
    dates = _business_dates(dt.date(2018, 1, 2), n_dates)
    sectors = {"AAPL": "Information Technology", "MSFT": "Information Technology", "JNJ": "Health Care"}
    exchanges = {"AAPL": "NASDAQ", "MSFT": "NASDAQ", "JNJ": "NYSE"}
    for dpos, date in enumerate(dates):
        for spos, symbol in enumerate(symbols):
            tier = 1 if spos % 2 == 0 else 2
            base_mid = 100.0 + 0.02 * dpos + 0.5 * spos
            close_shift = ((dpos % 7) - 3) * 0.015
            close_price = base_mid + close_shift
            for side in ("BUY", "SELL"):
                side_sign = 1.0 if side == "BUY" else -1.0
                for offset in offset_grid_bps:
                    offset = float(offset)
                    event = int(offset <= (2.0 + 0.5 * (tier == 1)) and (dpos + int(offset * 10) + spos) % 4 != 0)
                    if side == "BUY":
                        limit_price = base_mid - 0.01 - offset / 1e4 * base_mid
                    else:
                        limit_price = base_mid + 0.01 + offset / 1e4 * base_mid
                    rows.append({
                        "symbol": symbol,
                        "date": date,
                        "side": side,
                        "tier": tier,
                        "sector": sectors.get(symbol, "Unknown"),
                        "listing_exchange": exchanges.get(symbol, "Unknown"),
                        "q0": 900 + 40 * tier + 3 * dpos,
                        "D0": 1800 + 80 * tier + 5 * dpos,
                        "ofi_z": side_sign * np.sin(dpos / 5.0 + offset / 3.0),
                        "sigma": 0.0008 + 0.00001 * (dpos % 20) + 0.0001 * tier,
                        "limit_offset_bps": offset,
                        "half_spread_bps": 0.5 + 0.1 * tier,
                        "size_frac": cfg.PARENT_ORDER_PRIMARY_FRACTION,
                        "time_to_cutoff_seconds": 1200 - 10 * (dpos % 60),
                        "close_price": close_price,
                        "limit_price": limit_price,
                        "event": event,
                        "adverse_selection_bps": (0.15 * tier + 0.02 * offset) * (1 if event else 0),
                        **{f"tod_{h}": float(h == 15) for h in cfg.TOD_HOUR_BINS},
                    })
    return attach_value_labels(pd.DataFrame(rows))


def build_realdata_candidate_panel(
    *,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    max_rows_per_symbol_day: int = 96,
    offset_grid_bps: tuple[float, ...] = cfg.VALUE_MODEL_OFFSET_GRID_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tiny real-data candidate panel for path-wiring checks."""
    rows: list[pd.DataFrame] = []
    skips: list[dict] = []
    dates = [pd.Timestamp(x).date() for x in pd.bdate_range(start, end)]
    for date in dates:
        for spos, symbol in enumerate(symbols):
            try:
                trades, nbbo = load_symbol_day(date, symbol, rth_only=True)
                nbbo = filter_valid_quotes(nbbo)
                trades = filter_trades_near_quotes(filter_valid_trades(trades), nbbo)
                auction = extract_closing_auction_details(trades)
                if auction.volume <= 0 or not np.isfinite(auction.price):
                    raise ValueError("missing usable closing auction")
                panel = build_event_panel(
                    nbbo,
                    trades,
                    symbol,
                    date,
                    offset_grid_bps=offset_grid_bps,
                    max_rows=max_rows_per_symbol_day,
                    sample_seed=cfg.DEFAULT_SEED + int(date.strftime("%j")) + spos,
                )
                if panel.empty:
                    raise ValueError("empty candidate panel")
                panel["close_price"] = float(auction.price)
                panel["tier"] = 1 if spos % 2 == 0 else 2
                panel["sector"] = "Unknown"
                panel["listing_exchange"] = "Unknown"
                panel["size_frac"] = cfg.PARENT_ORDER_PRIMARY_FRACTION
                panel["adverse_selection_bps"] = 0.0
                rows.append(attach_value_labels(panel))
            except Exception as exc:
                skips.append({"date": date, "symbol": symbol, "reason": type(exc).__name__, "detail": str(exc)})
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=REQUIRED_CANDIDATE_COLUMNS)
    return frame, pd.DataFrame(skips)


def _synthetic_market_state() -> MarketState:
    times = pd.to_datetime(["2018-04-02 15:30:00", "2018-04-02 15:30:30", "2018-04-02 15:31:00"])
    nbbo = pd.DataFrame({
        "time": times,
        "best_bid": [100.00, 100.01, 100.02],
        "best_offer": [100.10, 100.11, 100.12],
        "best_bid_size": [1000, 1020, 1010],
        "best_offer_size": [1200, 1180, 1190],
        "mid": [100.05, 100.06, 100.07],
    })
    return MarketState(
        symbol="AAPL",
        date=dt.date(2018, 4, 2),
        nbbo=nbbo,
        trades=pd.DataFrame(),
        close_price=100.0,
        close_volume=1_000_000,
        ofi=pd.DataFrame(),
        rv=pd.Series(dtype=float),
        imbalance=pd.DataFrame(),
        nbbo_times=nbbo["time"].values.astype("datetime64[ns]").astype("int64"),
        nbbo_mid=nbbo[["time", "mid"]],
    )


def s5_dry_run_decisions(value_model, *, offset_grid_bps: tuple[float, ...]) -> pd.DataFrame:
    state = _synthetic_market_state()
    t = pd.Timestamp("2018-04-02 15:30:00")
    cutoff = pd.Timestamp("2018-04-02 15:50:00")
    cases = [
        ("trained_model", value_model, "BUY"),
        ("positive_stub", _ConstantValueModel([-1.0, 0.2, 0.9, 0.4, 0.1, -0.1]), "BUY"),
        ("nonpositive_stub", _ConstantValueModel([-0.5, -0.2, -0.1, -0.05, -0.01, -0.001]), "SELL"),
    ]
    rows = []
    for name, model, side in cases:
        strategy = ValueAwareXGBStrategy(
            value_model=model,
            tier=1,
            sector="Information Technology",
            listing_exchange="NASDAQ",
            offset_grid_bps=offset_grid_bps,
        )
        offset = strategy.limit_offset_bps(t, side, state, 0.0, 10.0)
        slice_qty = strategy.slice_size(t, cutoff, 1000, side, state)
        rows.append({
            "case": name,
            "side": side,
            "chosen_offset_bps": float(offset),
            "slice_qty": int(slice_qty),
            "posted_passively": int(slice_qty > 0),
            "last_predicted_value_bps": float(strategy._last_value_bps),
        })
    return pd.DataFrame(rows)


def _candidate_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [
        ("rows", len(panel)),
        ("symbols", panel["symbol"].nunique()),
        ("dates", pd.to_datetime(panel["date"]).dt.date.nunique()),
        ("sides", panel["side"].nunique()),
        ("tiers", panel["tier"].nunique()),
        ("sectors", panel["sector"].nunique()),
        ("offsets", panel["limit_offset_bps"].nunique()),
        ("fill_probability", float(pd.to_numeric(panel["event"], errors="coerce").mean())),
        ("mean_target_net_alpha_bps", float(pd.to_numeric(panel["target_net_alpha_vs_moc_bps"], errors="coerce").mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def run_smoke(
    *,
    mode: str = "synthetic",
    out_dir: Path,
    symbols: list[str] | None = None,
    start: dt.date = dt.date(2018, 2, 1),
    end: dt.date = dt.date(2018, 2, 2),
    n_estimators: int = 20,
    xgb_device: str = "cpu",
    min_rows_global: int = 30,
    min_rows_side: int = 20,
    min_rows_side_tier: int = 10,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = symbols or ["AAPL", "MSFT"]
    if mode == "synthetic":
        panel = build_synthetic_candidate_panel(symbols=tuple(symbols))
        skips = pd.DataFrame(columns=["date", "symbol", "reason", "detail"])
    elif mode == "realdata":
        panel, skips = build_realdata_candidate_panel(symbols=symbols, start=start, end=end)
    else:
        raise ValueError("mode must be 'synthetic' or 'realdata'")
    if panel.empty:
        raise RuntimeError("Candidate panel is empty")
    missing = sorted(set(REQUIRED_CANDIDATE_COLUMNS) - set(panel.columns))
    if missing:
        raise RuntimeError(f"Candidate panel missing required columns: {missing}")

    panel.to_parquet(out_dir / "candidate_panel.parquet", index=False)
    _candidate_summary(panel).to_csv(out_dir / "candidate_panel_summary.csv", index=False)
    skips.to_csv(out_dir / "candidate_panel_skips.csv", index=False)

    candidate_dates = sorted(pd.to_datetime(panel["date"]).dt.date.unique())
    schedule = build_monthly_training_schedule(candidate_dates)
    mapping = assign_monthly_anchor(candidate_dates, schedule)
    schedule.to_csv(out_dir / "rolling_schedule.csv", index=False)
    mapping.to_csv(out_dir / "rolling_anchor_map.csv", index=False)

    model = SideTieredXGBValueModel().fit_panel(
        panel,
        min_rows_global=min_rows_global,
        min_rows_side=min_rows_side,
        min_rows_side_tier=min_rows_side_tier,
        n_estimators=n_estimators,
        xgb_device=xgb_device,
    )
    model.save(out_dir)

    dry_run = s5_dry_run_decisions(model, offset_grid_bps=cfg.VALUE_MODEL_OFFSET_GRID_BPS)
    dry_run.to_csv(out_dir / "s5_dry_run_orders.csv", index=False)
    if not (dry_run.loc[dry_run["case"] == "positive_stub", "posted_passively"].iloc[0] == 1):
        raise RuntimeError("S5 positive branch did not post passively")
    if not (dry_run.loc[dry_run["case"] == "nonpositive_stub", "posted_passively"].iloc[0] == 0):
        raise RuntimeError("S5 non-positive branch did not route to MOC")

    posting_summary = posting_curve_summary(panel)
    posting_summary.to_csv(out_dir / "posting_curve_summary.csv", index=False)
    save_posting_curve_figure(posting_summary, out_dir / "posting_curve.png")
    template_outputs = write_preliminary_templates(
        out_dir / "reporting_templates",
        posting_summary=posting_summary,
        title="S5 Value-Model Smoke Reporting Template",
    )

    manifest = {
        "status": "complete",
        "mode": mode,
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(panel)),
        "policy": cfg.VALUE_MODEL_POLICY_VERSION,
        "xgb_device": xgb_device,
        "n_estimators": int(n_estimators),
        "value_model_keys": sorted(model.models),
        "trainable_rolling_anchors": int((schedule.get("status", pd.Series(dtype=str)) == "trainable").sum()) if not schedule.empty else 0,
        "template_outputs": template_outputs,
    }
    (out_dir / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "realdata"], default="synthetic")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=["AAPL", "MSFT"])
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2018, 2, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2018, 2, 2))
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (cfg.ARTIFACTS_DIR / f"value_model_smoke_{stamp}")
    manifest = run_smoke(
        mode=args.mode,
        out_dir=out_dir,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        n_estimators=args.n_estimators,
        xgb_device=args.xgb_device,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
