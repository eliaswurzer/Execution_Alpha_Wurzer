from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from analysis.data.adv_spread_buckets import (
    assign_adv_spread_buckets,
    merge_adv_spread_buckets,
    summarize_adv_spread_buckets,
)
from analysis.runners import h1_performance_gap, h2_signal_efficiency, h3_te_tradeoff


@pytest.mark.unit
def test_adv_spread_buckets_assign_expected_terciles() -> None:
    rows = []
    specs = [
        ("A", 900.0, 1.0), ("B", 800.0, 2.0), ("C", 700.0, 3.0),
        ("D", 600.0, 4.0), ("E", 500.0, 5.0), ("F", 400.0, 6.0),
        ("G", 300.0, 7.0), ("H", 200.0, 8.0), ("I", 100.0, 9.0),
    ]
    for symbol, adv, quoted_spread in specs:
        rows.append({
            "symbol": symbol,
            "date": dt.date(2018, 1, 2),
            "adv_dollar": adv,
            "adv_shares": adv / 10.0,
            "avg_half_spread_bps": quoted_spread / 2.0,
        })
    out = assign_adv_spread_buckets(pd.DataFrame(rows), min_days=1)
    lookup = out.set_index("symbol")
    assert int(lookup.loc["A", "adv_bucket"]) == 1
    assert int(lookup.loc["I", "adv_bucket"]) == 3
    assert int(lookup.loc["A", "spread_bucket"]) == 1
    assert int(lookup.loc["I", "spread_bucket"]) == 3
    assert lookup.loc["A", "adv_spread_bucket"] == "ADV1_SPREAD1"
    summary = summarize_adv_spread_buckets(out)
    assert set(summary["adv_spread_bucket"]) == {
        "ADV1_SPREAD1", "ADV2_SPREAD2", "ADV3_SPREAD3",
    }


@pytest.mark.unit
def test_adv_spread_merge_preserves_existing_tier_and_marks_unassigned() -> None:
    panel = pd.DataFrame({
        "symbol": ["A", "Z"],
        "tier": [2, 1],
        "net_alpha_bps": [1.0, 2.0],
    })
    bucket_map = pd.DataFrame({
        "symbol": ["A"],
        "adv_bucket": [1],
        "spread_bucket": [3],
        "adv_spread_bucket": ["ADV1_SPREAD3"],
        "mean_adv_dollar": [1000.0],
        "mean_adv_shares": [100.0],
        "avg_quoted_spread_bps": [4.0],
        "n_calibration_days": [10],
    })
    out = merge_adv_spread_buckets(panel, bucket_map)
    assert out["tier"].tolist() == [2, 1]
    assert out.loc[out["symbol"] == "A", "adv_spread_bucket"].iloc[0] == "ADV1_SPREAD3"
    assert out.loc[out["symbol"] == "Z", "adv_spread_bucket"].iloc[0] == "unassigned"


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    strategies = [
        "S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
        "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD",
    ]
    base_date = dt.date(2018, 2, 1)
    for day in range(2):
        date = base_date + dt.timedelta(days=day)
        for idx, symbol in enumerate(["AAA", "BBB", "CCC"]):
            order_id = f"{date}_{symbol}_BUY"
            for strategy in strategies:
                diff = {
                    "S0_MOC": 0.0,
                    "S1_STATIC": -0.2,
                    "S2_TIME_ADAPTIVE": -0.1,
                    "S3_OFI": 0.1,
                    "S3_IMB": 0.05,
                    "S3_FULL": 0.2,
                    "S4_TOD": -0.3,
                }[strategy]
                rows.append({
                    "order_id": order_id,
                    "symbol": symbol,
                    "date": date,
                    "strategy": strategy,
                    "size_frac": 0.01,
                    "window": "B",
                    "side": "BUY",
                    "tier": idx + 1,
                    "adv_bucket": (idx % 3) + 1,
                    "spread_bucket": (idx % 3) + 1,
                    "adv_spread_bucket": f"ADV{(idx % 3) + 1}_SPREAD{(idx % 3) + 1}",
                    "net_alpha_bps": 1.0 + diff,
                    "alpha_bps": 1.2 + diff,
                    "net_alpha_vs_moc_bps": diff,
                    "fill_rate": 0.0 if strategy == "S0_MOC" else 0.4 + 0.1 * idx,
                    "adverse_selection_cost_bps": 0.0 if strategy == "S0_MOC" else 0.1 * (idx + 1),
                    "impact_bps": 0.0,
                    "arrival_time": pd.Timestamp(f"{date} 15:30:00"),
                    "listing_exchange": "NYSE" if idx % 2 == 0 else "NASDAQ",
                })
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_h1_h2_h3_write_adv_spread_bucket_outputs(artifact_dir) -> None:
    panel = _synthetic_panel()
    h1_dir = artifact_dir / "h1"
    h2_dir = artifact_dir / "h2"
    h3_dir = artifact_dir / "h3"

    h1_performance_gap.analyze_panel(panel, h1_dir)
    assert (h1_dir / "h1_subgroup_adv_bucket.csv").exists()
    assert (h1_dir / "h1_subgroup_spread_bucket.csv").exists()
    assert (h1_dir / "h1_subgroup_adv_spread_bucket.csv").exists()

    h2_panel = panel[panel["strategy"].isin(["S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL"])]
    h2_signal_efficiency.analyze_panel(h2_panel, h2_dir, n_bins=3)
    assert (h2_dir / "h2_pooled_by_adv_bucket.csv").exists()
    assert (h2_dir / "h2_pooled_by_spread_bucket.csv").exists()
    assert (h2_dir / "h2_pooled_by_adv_spread_bucket.csv").exists()

    h3_panel = panel[panel["strategy"].isin(["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE", "S3_FULL", "S4_TOD"])]
    h3_te_tradeoff.analyze_panel(h3_panel, h3_dir, etas=[0.01])
    assert (h3_dir / "h3_tev_by_adv_bucket.csv").exists()
    assert (h3_dir / "h3_raear_by_spread_bucket.csv").exists()
    assert (h3_dir / "h3_raear_by_adv_spread_bucket.csv").exists()


@pytest.mark.unit
def test_h3_uses_moc_relative_alpha_for_tev_and_raear(artifact_dir) -> None:
    rows = []
    date = dt.date(2018, 1, 2)
    for idx, moc_diff in enumerate([1.0, 3.0, 5.0], start=1):
        order_id = f"o{idx}"
        rows.append({
            "order_id": order_id,
            "symbol": f"SYM{idx}",
            "date": date,
            "strategy": "S0_MOC",
            "size_frac": 0.01,
            "net_alpha_bps": -0.1 * idx,
            "net_alpha_vs_moc_bps": 0.0,
            "adv_bucket": "G1",
            "spread_bucket": "G1",
            "adv_spread_bucket": "G1_G1",
        })
        rows.append({
            "order_id": order_id,
            "symbol": f"SYM{idx}",
            "date": date,
            "strategy": "S3_FULL",
            "size_frac": 0.01,
            "net_alpha_bps": 100.0 * idx,
            "net_alpha_vs_moc_bps": moc_diff,
            "adv_bucket": "G1",
            "spread_bucket": "G1",
            "adv_spread_bucket": "G1_G1",
        })

    out_dir = artifact_dir / "h3_moc_relative"
    h3_te_tradeoff.analyze_panel(pd.DataFrame(rows), out_dir, etas=[0.25])

    tev = pd.read_csv(out_dir / "h3_tev.csv").set_index("strategy")
    raear = pd.read_csv(out_dir / "h3_raear.csv").set_index("strategy")
    grouped = pd.read_csv(out_dir / "h3_tev_by_adv_bucket.csv").set_index("strategy")
    manifest = json.loads((out_dir / "h3_metric_manifest.json").read_text(encoding="utf-8"))

    assert tev.loc["S0_MOC", "mean_alpha"] == pytest.approx(0.0)
    assert tev.loc["S0_MOC", "tev"] == pytest.approx(0.0)
    assert raear.loc["S0_MOC", "raear_eta_0.25"] == pytest.approx(0.0)
    assert pd.isna(raear.loc["S0_MOC", "ir"])

    assert tev.loc["S3_FULL", "mean_alpha"] == pytest.approx(3.0)
    assert tev.loc["S3_FULL", "tev"] == pytest.approx(4.0)
    assert raear.loc["S3_FULL", "tes"] == pytest.approx(2.0)
    assert raear.loc["S3_FULL", "ir"] == pytest.approx(1.5)
    assert raear.loc["S3_FULL", "raear_eta_0.25"] == pytest.approx(2.0)
    assert grouped.loc["S3_FULL", "tev"] == pytest.approx(4.0)
    assert manifest["alpha_col"] == h3_te_tradeoff.H3_ALPHA_COL
    assert manifest["metric_policy_version"] == h3_te_tradeoff.H3_METRIC_POLICY_VERSION


@pytest.mark.unit
def test_h3_filters_to_primary_window_for_raear(artifact_dir) -> None:
    rows = []
    date = dt.date(2018, 1, 2)
    by_window = {
        "A": [100.0, 100.0, 100.0],
        "B": [1.0, 3.0, 5.0],
        "C": [-100.0, -100.0, -100.0],
    }
    for window, diffs in by_window.items():
        for idx, moc_diff in enumerate(diffs, start=1):
            order_id = f"{window}{idx}"
            common = {
                "symbol": f"SYM{idx}",
                "date": date,
                "window": window,
                "size_frac": 0.01,
                "adv_bucket": "G1",
                "spread_bucket": "G1",
                "adv_spread_bucket": "G1_G1",
            }
            rows.append({
                **common,
                "order_id": order_id,
                "strategy": "S0_MOC",
                "net_alpha_vs_moc_bps": 0.0,
            })
            rows.append({
                **common,
                "order_id": order_id,
                "strategy": "S3_FULL",
                "net_alpha_vs_moc_bps": moc_diff,
            })

    out_dir = artifact_dir / "h3_primary_window"
    h3_te_tradeoff.analyze_panel(pd.DataFrame(rows), out_dir, etas=[0.25])

    tev = pd.read_csv(out_dir / "h3_tev.csv").set_index("strategy")
    raear = pd.read_csv(out_dir / "h3_raear.csv").set_index("strategy")
    grouped = pd.read_csv(out_dir / "h3_tev_by_adv_bucket.csv").set_index("strategy")
    manifest = json.loads((out_dir / "h3_metric_manifest.json").read_text(encoding="utf-8"))

    assert tev.loc["S3_FULL", "mean_alpha"] == pytest.approx(3.0)
    assert tev.loc["S3_FULL", "tev"] == pytest.approx(4.0)
    assert raear.loc["S3_FULL", "ir"] == pytest.approx(1.5)
    assert grouped.loc["S3_FULL", "mean_alpha"] == pytest.approx(3.0)
    assert manifest["n_rows"] == 6
    assert manifest["metric_policy_version"] == h3_te_tradeoff.H3_METRIC_POLICY_VERSION


@pytest.mark.unit
def test_h1_subgroup_filters_to_primary_window(artifact_dir) -> None:
    """H1 subgroup tables must use only the primary Window-B surface.

    Regression guard: if ``_primary_surface`` stopped filtering on ``window``,
    each symbol-day would contribute three arrival-window rows (A, B, C), so the
    per-group ``n`` would triple and the subgroup mean would average across
    windows instead of reporting the Window-B differential.
    """
    rows = []
    sym_tier = {"AAA": 1, "BBB": 1, "CCC": 2, "DDD": 2}
    vs_moc_by_window = {"A": 9.0, "B": 0.5, "C": -9.0}
    for day in range(2):
        date = dt.date(2018, 1, 2) + dt.timedelta(days=day)
        for sym, tier in sym_tier.items():
            for window, vs_moc in vs_moc_by_window.items():
                common = {
                    "order_id": f"{date}_{sym}_{window}",
                    "symbol": sym,
                    "date": date,
                    "window": window,
                    "size_frac": 0.01,
                    "side": "BUY",
                    "tier": tier,
                    "arrival_time": pd.Timestamp(f"{date} 15:30:00"),
                    "listing_exchange": "NYSE" if tier == 1 else "NASDAQ",
                    "impact_bps": 0.0,
                }
                rows.append({
                    **common, "strategy": "S0_MOC", "net_alpha_bps": 0.0,
                    "alpha_bps": 0.0, "net_alpha_vs_moc_bps": 0.0,
                    "fill_rate": 0.0, "adverse_selection_cost_bps": 0.0,
                })
                rows.append({
                    **common, "strategy": "S3_FULL", "net_alpha_bps": vs_moc,
                    "alpha_bps": vs_moc, "net_alpha_vs_moc_bps": vs_moc,
                    "fill_rate": 0.5, "adverse_selection_cost_bps": 0.1,
                })

    out_dir = artifact_dir / "h1_primary_window"
    h1_performance_gap.analyze_panel(pd.DataFrame(rows), out_dir)
    sub = pd.read_csv(out_dir / "h1_subgroup_tier.csv")

    # Two dates x two symbols per tier on Window B only -> 4 cells per tier, 8
    # total; never the 24 that pooling windows A/B/C would produce.
    assert sorted(sub["n"].tolist()) == [4, 4]
    assert int(sub["n"].sum()) == 8
    assert sub["n"].max() <= 8
    # Means report the Window-B differential (0.5), not the A/B/C average (~0.17).
    for mean in sub["mean"]:
        assert mean == pytest.approx(0.5)
