"""Tests for the parent-size grid rework, side matching, the impact-coefficient
sweep, and the rolling-window stability utility."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from analysis import config as cfg
from analysis.runners.parent_size_grid import (
    DEFAULT_SIZE_GRID,
    impact_sweep,
    summarize_grid_panel,
)
from analysis.runners._common import rolling_window_panel
from analysis.runners.master_panel import run_master_panel
from analysis.simulation.parent_orders import (
    _side_start_offset,
    build_parent_orders,
)


# ---------------------------------------------------------------------------
# Side matching (G2)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_single_size_orders_match_legacy_running_index_rule() -> None:
    """Headline/bounds invariance: with one size the window-keyed side rule
    must replicate the legacy running-order-index rule bit for bit."""
    symbol, date = "AAPL", dt.date(2018, 7, 2)
    orders = build_parent_orders(symbol, date, 100_000, size_fractions=(0.01,))
    offset = _side_start_offset(symbol, date, cfg.DEFAULT_SEED)
    legacy_sides = [
        "BUY" if (i + offset) % 2 == 0 else "SELL"
        for i in range(len(cfg.EXECUTION_WINDOWS))
    ]
    assert orders["side"].tolist() == legacy_sides
    for row in orders.itertuples(index=False):
        assert row.order_id == (
            f"{date.isoformat()}_{symbol}_{row.window}_0100bp_{row.side}"
        )


@pytest.mark.unit
def test_parent_order_size_tags_are_decimal_safe() -> None:
    orders = build_parent_orders(
        "MSFT", dt.date(2018, 7, 2), 1_000_000,
        size_fractions=(0.005, 0.01, 0.02),
        windows={"B": cfg.EXECUTION_WINDOWS["B"]},
    )

    tags = orders["order_id"].str.extract(r"_B_([^_]+)_")[0].tolist()
    assert tags == ["0050bp", "0100bp", "0200bp"]
    assert orders["order_id"].is_unique


@pytest.mark.unit
def test_grid_sizes_share_side_within_window() -> None:
    orders = build_parent_orders(
        "MSFT", dt.date(2018, 7, 2), 1_000_000,
        size_fractions=DEFAULT_SIZE_GRID,
    )
    per_window_sides = orders.groupby("window")["side"].nunique()
    assert (per_window_sides == 1).all()
    # Windows still alternate sides.
    window_sides = orders.drop_duplicates("window").sort_values("window")
    assert window_sides["side"].nunique() == 2


@pytest.mark.unit
def test_windows_subset_restricts_orders() -> None:
    orders = build_parent_orders(
        "MSFT", dt.date(2018, 7, 2), 1_000_000,
        size_fractions=DEFAULT_SIZE_GRID,
        windows={"B": cfg.EXECUTION_WINDOWS["B"]},
    )
    assert set(orders["window"]) == {"B"}
    assert len(orders) == len(DEFAULT_SIZE_GRID)


@pytest.mark.unit
def test_run_master_panel_rejects_unknown_window(artifact_dir) -> None:
    with pytest.raises(ValueError, match="Unknown execution windows"):
        run_master_panel(
            strategies=["S0_MOC"],
            start=dt.date(2018, 7, 2),
            end=dt.date(2018, 7, 3),
            artifacts_dir=artifact_dir,
            run_root=artifact_dir / "run",
            windows=("B", "X"),
        )


# ---------------------------------------------------------------------------
# Grid summaries and impact sweep (G1/G3)
# ---------------------------------------------------------------------------

def _grid_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for d in range(8):
        date = dt.date(2018, 7, 2) + dt.timedelta(days=d)
        for sym in ("AAPL", "MSFT", "NVDA"):
            for size in (0.01, 0.05):
                for strat in ("S0_MOC", "S3_FULL"):
                    impact = (
                        cfg.IMPACT_COEF_BPS * np.sqrt(size)
                        if (strat != "S0_MOC" and size > cfg.IMPACT_ACTIVATION_THRESHOLD)
                        else 0.0
                    )
                    rows.append({
                        "order_id": f"{date}_{sym}_B_{size}_{strat}",
                        "symbol": sym,
                        "date": date,
                        "strategy": strat,
                        "size_frac": size,
                        "net_alpha_bps": float(rng.normal(-1.0, 0.3)),
                        "net_alpha_vs_moc_bps": float(rng.normal(-1.0, 0.3)),
                        "alpha_bps": float(rng.normal(0.0, 0.3)),
                        "fill_rate": float(rng.uniform(0.5, 1.0)),
                        "adverse_selection_bps": float(rng.normal(-1.0, 0.5)),
                        "adverse_selection_cost_bps": float(rng.uniform(0, 2)),
                        "impact_bps": impact,
                        "window": "B",
                    })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_summarize_grid_panel_writes_clustered_and_table_csv(artifact_dir) -> None:
    panel = _grid_panel()
    summary = summarize_grid_panel(panel, artifact_dir)
    assert set(summary["metric"]) == {"net_alpha_bps", "net_alpha_vs_moc_bps"}
    assert (summary["n"] > 0).all()
    table = pd.read_csv(artifact_dir / "size_table_summary.csv")
    assert sorted(table["size_bucket"].tolist()) == [0.01, 0.05]
    assert (table["strategy"] == "S3_FULL").all()
    assert {"mean_net_alpha_bps", "se_twoway", "tev", "mean_fill_rate",
            "mean_as_markout_bps", "n"}.issubset(table.columns)


@pytest.mark.unit
def test_impact_sweep_shifts_net_alpha_by_coefficient_delta(artifact_dir) -> None:
    panel = _grid_panel()
    sweep = impact_sweep(panel, artifact_dir)
    base = sweep[
        np.isclose(sweep["impact_coef_bps"], cfg.IMPACT_COEF_BPS)
    ].set_index(["size_bucket", "strategy"])["mean"]
    doubled = sweep[
        np.isclose(sweep["impact_coef_bps"], 2 * cfg.IMPACT_COEF_BPS)
    ].set_index(["size_bucket", "strategy"])["mean"]
    # At the headline coefficient the sweep reproduces the panel net alpha.
    raw = panel[
        (panel["strategy"] == "S3_FULL") & np.isclose(panel["size_frac"], 0.05)
    ]["net_alpha_bps"].mean()
    assert base[(0.05, "S3_FULL")] == pytest.approx(raw, abs=1e-9)
    # Doubling kappa lowers net alpha by exactly the stored impact for active
    # rows and leaves the threshold bucket (1 percent) untouched.
    expected_shift = cfg.IMPACT_COEF_BPS * np.sqrt(0.05)
    assert doubled[(0.05, "S3_FULL")] == pytest.approx(
        base[(0.05, "S3_FULL")] - expected_shift, abs=1e-9,
    )
    assert doubled[(0.01, "S3_FULL")] == pytest.approx(
        base[(0.01, "S3_FULL")], abs=1e-9,
    )


# ---------------------------------------------------------------------------
# Rolling-window stability (G4)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rolling_window_panel_window_arithmetic_and_se() -> None:
    dates = pd.bdate_range("2018-07-02", "2019-12-31")
    rows = []
    rng = np.random.default_rng(3)
    for d in dates:
        for sym in ("AAPL", "MSFT"):
            rows.append({
                "symbol": sym,
                "date": d.date(),
                "strategy": "S3_FULL",
                "net_alpha_vs_moc_bps": float(rng.normal(-1.0, 1.0)),
            })
    panel = pd.DataFrame(rows)
    out = rolling_window_panel(panel, alpha_col="net_alpha_vs_moc_bps")
    # Span 2018-07-02..2019-12-31, 6-month windows stepped monthly: the last
    # admissible window start is 2019-06-02 -> 12 overlapping windows.
    assert len(out) == 12
    assert (out["window_end"] > out["window_start"]).all()
    # Thin two-way panels can clamp a variance diagonal to zero; SEs must be
    # finite and non-negative, and informative in most windows.
    assert out["clustered_se"].ge(0).all()
    assert out["clustered_se"].gt(0).sum() >= len(out) - 2
    assert out["n"].gt(0).all()
    # Windows are exactly six months long.
    spans = pd.to_datetime(out["window_end"]) - pd.to_datetime(out["window_start"])
    assert spans.min() >= pd.Timedelta(days=180)


@pytest.mark.unit
def test_rolling_window_panel_empty_input() -> None:
    assert rolling_window_panel(pd.DataFrame()).empty
