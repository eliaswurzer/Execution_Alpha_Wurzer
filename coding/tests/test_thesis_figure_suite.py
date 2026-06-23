from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from analysis.reporting.thesis_figures.data import (
    FigureBuildContext,
    close_volume_share,
    close_volume_share_trend_appendix,
)
from analysis.reporting.thesis_figures.io import write_yaml
from analysis.reporting.thesis_figures.spec import FigureSpec, load_specs
from analysis.reporting.thesis_figures.style import THEME, apply_thesis_style, color_for, sample_sequential
from analysis.reporting.thesis_figures.renderers import (
    _assert_layout_quality,
    _cluster_label_defaults,
    _render_line,
    _render_line_panels,
    _render_scatter,
    _resolve_cmap,
)
from analysis.reporting.thesis_figures.suite import curated_input_path, render_figures


def _make_panel(root: Path) -> None:
    rows = []
    strategies = ["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD"]
    dates = [dt.date(2018, 7, 2) + dt.timedelta(days=30 * i) for i in range(6)]
    for d_i, date in enumerate(dates):
        for sym_i, symbol in enumerate(("AAPL", "MSFT", "JPM")):
            tier = sym_i + 1
            for strategy in strategies:
                fill_rate = 0.0 if strategy == "S0_MOC" else 0.45 + 0.05 * sym_i + 0.01 * d_i
                net_alpha = -0.10 + 0.03 * sym_i - 0.01 * d_i
                rows.append({
                    "order_id": f"{date}_{symbol}_{strategy}",
                    "symbol": symbol,
                    "date": date,
                    "side": "BUY",
                    "strategy": strategy,
                    "window": "B",
                    "alpha_bps": net_alpha + 0.05,
                    "net_alpha_bps": net_alpha,
                    "net_alpha_vs_moc_bps": net_alpha + 0.10,
                    "fill_rate": fill_rate,
                    "adverse_selection_bps": -0.5 - 0.1 * sym_i,
                    "impact_bps": 0.0,
                    "size_frac": 0.01,
                    "tier": tier,
                    "close_trade_volume": 1000 + 100 * sym_i,
                    "expected_vc": 9000 + 200 * sym_i,
                })
    h1 = root / "hypotheses" / "h1"
    h1.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(h1 / "h1_panel.parquet", index=False)


def _make_run(root: Path) -> None:
    _make_panel(root)
    h2 = root / "hypotheses" / "h2"
    h3 = root / "hypotheses" / "h3"
    h2.mkdir(parents=True, exist_ok=True)
    h3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"label": label, "bin": b, "mean": 0.1 * (b - 2)}
        for label in ("OFI_marginal", "IMB_marginal", "FULL_vs_S2", "interaction")
        for b in range(5)
    ]).to_csv(h2 / "h2_per_bin_differentials.csv", index=False)
    pd.DataFrame([
        {"strategy": s, "mean_alpha": 0.2 - i * 0.05, "tev": 1.0 + i,
         "raear_eta_0.01": 0.1, "raear_eta_0.05": 0.0}
        for i, s in enumerate(("S0_MOC", "S2_TIME_ADAPTIVE", "S3_FULL"))
    ]).to_csv(h3 / "h3_raear.csv", index=False)


def _make_volume_sources(base: Path) -> tuple[Path, Path, Path]:
    membership_root = base / "membership"
    membership_root.mkdir(parents=True, exist_ok=True)
    (membership_root / "sp500_membership_intervals.csv").write_text(
        "index_id,symbol,effective_from,effective_to\n"
        "sp500,AAPL,2018-01-01,2019-12-31\n"
        "sp500,MSFT,2018-01-01,2019-12-31\n"
        "sp500,JPM,2018-01-01,2019-12-31\n"
        "sp500,OLD,2018-01-01,2018-06-29\n",
        encoding="utf-8",
    )

    tier_map = base / "completed_symbol_tier_map.csv"
    tier_map.write_text(
        "symbol,tier,tier_source\n"
        "AAPL,1,calibrated\n"
        "MSFT,2,calibrated\n"
        "JPM,3,calibrated\n",
        encoding="utf-8",
    )

    volume_db = base / "volume.duckdb"
    con = duckdb.connect(str(volume_db))
    con.execute("""
        CREATE TABLE daily_volume (
            Ticker VARCHAR,
            Date DATE,
            Total_Daily_Val DOUBLE,
            Close_Auction_Val DOUBLE,
            Official_Close_Marker_Val DOUBLE,
            Official_Close_Marker_Rows BIGINT
        )
    """)
    rows = [
        ("AAPL", dt.date(2018, 1, 2), 100.0, 10.0, 1.0, 1),
        ("MSFT", dt.date(2018, 1, 2), 300.0, 30.0, 2.0, 1),
        ("JPM", dt.date(2018, 1, 2), 600.0, 60.0, 3.0, 1),
        ("OLD", dt.date(2018, 1, 2), 100.0, 20.0, 0.0, 0),
        ("AAPL", dt.date(2018, 7, 2), 100.0, 5.0, 0.0, 0),
        ("MSFT", dt.date(2018, 7, 2), 100.0, 15.0, 0.0, 0),
        ("JPM", dt.date(2018, 7, 2), 100.0, 25.0, 0.0, 0),
        # Documented early close; deliberately impossible if not excluded.
        ("AAPL", dt.date(2018, 7, 3), 100.0, 500.0, 0.0, 0),
        ("AAPL", dt.date(2019, 1, 2), 200.0, 20.0, 0.0, 0),
        ("MSFT", dt.date(2019, 1, 2), 200.0, 30.0, 0.0, 0),
        ("JPM", dt.date(2019, 1, 2), 200.0, 40.0, 0.0, 0),
    ]
    con.executemany("INSERT INTO daily_volume VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.close()
    return volume_db, tier_map, membership_root


def _make_external_sources(base: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    stats = base / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": m, "tier": t, "auc": 0.75 + 0.01 * t, "absolute_calibration_error": 0.01 * t}
        for m in ("cox", "km", "xgb")
        for t in (1, 2, 3)
    ]).to_csv(stats / "fill_model_oos_calibration.csv", index=False)
    pd.DataFrame([
        {"spec": "tape_replay_queue", "label": "Queue-aware replay (headline)",
         "mean_fill_rate": 0.65, "mean_net_alpha_vs_moc_bps": -0.1},
        {"spec": "tape_replay", "label": "At-or-through replay (upper bound)",
         "mean_fill_rate": 0.75, "mean_net_alpha_vs_moc_bps": 0.2},
        {"spec": "xgb", "label": "XGBoost survival",
         "mean_fill_rate": 0.80, "mean_net_alpha_vs_moc_bps": 2.5},
    ]).to_csv(stats / "fill_model_economic_tests.csv", index=False)
    pd.DataFrame([
        {"spec": spec, "label": label, "metric": metric, "mean_diff": value}
        for spec, label in (("tape_replay", "At-or-through replay"), ("xgb", "XGBoost survival"))
        for metric, value in (("net_alpha_vs_moc_bps", 0.2), ("fill_rate", 0.1), ("as_markout_bps", -0.2))
    ]).to_csv(stats / "fill_model_vs_queue_tests.csv", index=False)

    as_horizon = base / "as_horizon_summary.csv"
    pd.DataFrame([
        {"horizon_seconds": h, "as_markout_bps": 1.0 + i * 0.1,
         "as_component_bps": -0.6 - i * 0.05, "mean_net_alpha_vs_moc_bps": -0.1}
        for i, h in enumerate((5, 15, 30, 60, 300))
    ]).to_csv(as_horizon, index=False)

    size_grid = base / "size_grid"
    size_grid.mkdir(exist_ok=True)
    pd.DataFrame([
        {"size_bucket": s, "mean_net_alpha_bps": -s * 10, "se_twoway": 0.1,
         "mean_fill_rate": 0.7 - s, "mean_as_markout_bps": 1.0 + s, "tev": 10.0, "n": 100}
        for s in (0.005, 0.01, 0.02, 0.05)
    ]).to_csv(size_grid / "size_table_summary.csv", index=False)
    volume_db, tier_map, membership_root = _make_volume_sources(base)
    return stats, as_horizon, size_grid, volume_db, tier_map, membership_root


@pytest.mark.unit
def test_thesis_style_sets_pdf_fonttype() -> None:
    apply_thesis_style()
    assert plt.rcParams["font.size"] == 11
    assert plt.rcParams["xtick.labelsize"] == 10
    assert plt.rcParams["legend.fontsize"] == 10
    assert not plt.rcParams["axes.grid"]
    assert plt.rcParams["pdf.fonttype"] == 42
    assert "STIX Two Text" in plt.rcParams["font.serif"]
    assert "Times New Roman" not in plt.rcParams["font.serif"]


@pytest.mark.unit
def test_line_renderer_uses_uniform_width_and_no_default_x_grid() -> None:
    df = pd.DataFrame({
        "eta": [0.0, 0.5, 0.0, 0.5],
        "raear": [0.0, 0.0, 0.0, -1.0],
        "label": ["S0 MOC", "S0 MOC", "S3 Full", "S3 Full"],
    })
    spec = SimpleNamespace(
        aesthetics={"x": "eta", "y": "raear", "hue": "label", "color_role": "strategy", "emphasize": "S0 MOC"},
        layout={"xlabel": "", "ylabel": "", "legend_title": "Strategy", "line_width": 1.35},
    )
    fig = _render_line(df, spec)
    ax = fig.axes[0]
    assert {line.get_linewidth() for line in ax.lines} == {1.35}
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)


@pytest.mark.unit
def test_scatter_renderer_can_use_legend_without_point_labels() -> None:
    df = pd.DataFrame({
        "fill_rate": [0.0, 0.65],
        "net_alpha": [-0.1, -0.2],
        "label": ["S0 MOC", "S3 Full"],
    })
    spec = SimpleNamespace(
        aesthetics={
            "x": "fill_rate",
            "y": "net_alpha",
            "label": "label",
            "color_role": "strategy",
            "legend_by_label": True,
            "show_labels": False,
        },
        layout={"xlabel": "", "ylabel": "", "legend_title": "Strategy", "legend_ncols": 2},
    )
    fig = _render_scatter(df, spec)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["S0 MOC", "S3 Full"]
    assert len(ax.texts) == 0
    plt.close(fig)


@pytest.mark.unit
def test_line_panels_support_parent_size_one_percent_x_grid() -> None:
    df = pd.DataFrame({
        "panel": ["Net alpha"] * 3,
        "parent_size_pct": [0.5, 1.0, 2.0],
        "series": ["Net alpha vs. MOC"] * 3,
        "value": [-0.05, -0.1, -1.0],
    })
    spec = SimpleNamespace(
        aesthetics={"panel": "panel", "x": "parent_size_pct", "y": "value", "hue": "series", "color_role": "metric"},
        layout={"xlabel": "", "ylabel": "", "x_major_step": 1, "x_grid": True, "line_width": 1.35},
    )
    fig = _render_line_panels(df, spec)
    ax = fig.axes[0]
    assert 1.0 in [round(float(tick), 6) for tick in ax.get_xticks()]
    assert any(line.get_visible() for line in ax.get_xgridlines())
    plt.close(fig)


@pytest.mark.unit
def test_semantic_colors_are_stable_across_ordering() -> None:
    assert color_for("strategy", "S3 Full", 0) == THEME["purple"]
    assert color_for("strategy", "S3 Full", 6) == THEME["purple"]
    assert color_for("strategy", "S0 MOC", 4) == THEME["black"]
    assert color_for("tier", "All", 2) == THEME["black"]
    assert color_for(None, "unmapped", 0) != color_for(None, "unmapped", 1)
    assert len(set(sample_sequential(3))) == 3


@pytest.mark.unit
def test_heatmap_cmap_aliases_resolve_to_theme_maps() -> None:
    assert _resolve_cmap("sequential").name == "thesis_purple_orange_sequential"
    assert _resolve_cmap("diverging").name == "PuOr"


@pytest.mark.unit
def test_layout_qa_detects_overlapping_annotations() -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(0.5, 0.5, "Alpha")
    ax.text(0.5, 0.5, "Beta")
    fig.tight_layout()
    with pytest.raises(RuntimeError, match="overlapping annotations"):
        _assert_layout_quality(fig, SimpleNamespace(id="synthetic_overlap"))
    plt.close(fig)


@pytest.mark.unit
def test_layout_qa_detects_clipped_text_and_tiny_effective_font() -> None:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(-0.25, 0.5, "Outside")
    ax.text(0.5, 0.5, "Too small", fontsize=7)
    fig.tight_layout()
    spec = SimpleNamespace(
        id="synthetic_clipped",
        layout={"min_effective_font_pt": 7.0},
        latex={"include_width": "0.50\\textwidth"},
    )
    with pytest.raises(RuntimeError, match="outside figure bounds|effective font below"):
        _assert_layout_quality(fig, spec)
    plt.close(fig)


@pytest.mark.unit
def test_cluster_label_defaults_separate_dense_scatter_labels() -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    offsets = _cluster_label_defaults(
        ax,
        [("A", 0.50, 0.50), ("B", 0.51, 0.50), ("C", 0.52, 0.50)],
    )
    assert set(offsets) == {"A", "B", "C"}
    assert len({tuple(value["xytext"]) for value in offsets.values()}) == 3
    plt.close(fig)


@pytest.mark.unit
def test_spec_validation_rejects_bad_plot_type() -> None:
    spec = load_specs()["fig_raear_curve"]
    payload = {
        "id": spec.id,
        "plot_type": "banana",
        "source": spec.source,
        "transform": spec.transform,
        "table": spec.table,
        "aesthetics": spec.aesthetics,
        "layout": spec.layout,
        "latex": spec.latex,
        "required_columns": list(spec.required_columns),
    }
    with pytest.raises(ValueError, match="Invalid plot_type"):
        FigureSpec.from_dict(payload)


@pytest.mark.unit
def test_render_thesis_figures_all_outputs_manifest(artifact_dir: Path) -> None:
    run = artifact_dir / "run"
    out = artifact_dir / "figures"
    _make_run(run)
    stats, as_horizon, size_grid, volume_db, tier_map, membership_root = _make_external_sources(artifact_dir)
    ctx = FigureBuildContext(
        run_root=run,
        out_dir=out,
        stats_dir=stats,
        as_horizon_csv=as_horizon,
        size_grid_root=size_grid,
        volume_db=volume_db,
        tier_map_csv=tier_map,
        membership_root=membership_root,
    )

    manifest = render_figures(ctx, figure="all", mode="artifact", refresh_data=True, fail_fast=True)

    assert not manifest["skipped"]
    assert "fig_raear_curve" in manifest["outputs"]
    assert (out / "fig_raear_curve.pdf").exists()
    assert (out / "fig_raear_curve.png").exists()
    assert (out / "fig_raear_curve.tex").exists()
    assert (out / "figure_inputs" / "fig_raear_curve.csv").exists()
    assert "fig_alpha_decomposition:run:h1_panel.parquet" in manifest["inputs_sha256"]


@pytest.mark.unit
def test_curated_table_and_override_are_used(artifact_dir: Path) -> None:
    run = artifact_dir / "run"
    out = artifact_dir / "figures"
    _make_run(run)
    stats, as_horizon, size_grid, volume_db, tier_map, membership_root = _make_external_sources(artifact_dir)
    ctx = FigureBuildContext(
        run_root=run,
        out_dir=out,
        stats_dir=stats,
        as_horizon_csv=as_horizon,
        size_grid_root=size_grid,
        volume_db=volume_db,
        tier_map_csv=tier_map,
        membership_root=membership_root,
    )
    render_figures(ctx, figure="fig_alpha_fill_frontier", mode="artifact", refresh_data=True, fail_fast=True)
    spec = load_specs()["fig_alpha_fill_frontier"]
    curated = pd.read_csv(out / "figure_inputs" / "fig_alpha_fill_frontier.csv")
    curated.loc[0, "net_alpha"] = 9.99
    curated_path = curated_input_path(out, spec)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(curated_path, index=False)
    write_yaml(out / "overrides" / "fig_alpha_fill_frontier.yaml", {
        "layout": {"xlabel": "Edited fill rate"},
    })

    manifest = render_figures(ctx, figure="fig_alpha_fill_frontier", mode="curated", refresh_data=False, fail_fast=True)

    keys = manifest["inputs_sha256"]
    assert "fig_alpha_fill_frontier:table:fig_alpha_fill_frontier:curated" in keys
    assert "fig_alpha_fill_frontier:override:fig_alpha_fill_frontier" in keys
    assert (out / "fig_alpha_fill_frontier.pdf").exists()


@pytest.mark.unit
def test_close_volume_share_uses_total_daily_volume_and_exports_audit_workbook(artifact_dir: Path) -> None:
    run = artifact_dir / "run"
    out = artifact_dir / "figures"
    _make_run(run)
    volume_db, tier_map, membership_root = _make_volume_sources(artifact_dir)
    workbook = out / "closing_auction_share_daily_values.xlsx"
    ctx = FigureBuildContext(
        run_root=run,
        out_dir=out,
        volume_db=volume_db,
        tier_map_csv=tier_map,
        membership_root=membership_root,
        close_share_xlsx_path=workbook,
    )

    data = close_volume_share(ctx)
    frame = data.frame.copy()

    assert "expected_vc" not in frame.columns
    assert pd.to_datetime(frame["date"]).min().date() == dt.date(2018, 1, 2)
    assert dt.date(2018, 7, 3) not in set(pd.to_datetime(frame["date"]).dt.date)
    assert set(frame["series"]) == {"Tier 1", "Tier 2", "Tier 3", "All"}
    assert frame["close_share_pct"].max() <= 100.0

    jan_all = frame[(frame["date"] == dt.date(2018, 1, 2)) & (frame["series"] == "All")].iloc[0]
    assert jan_all["close_share_pct"] == pytest.approx(100.0 * 120.0 / 1100.0)

    assert workbook.exists()
    unclassified = pd.read_excel(workbook, sheet_name="UnclassifiedH1")
    assert unclassified["symbol"].tolist() == ["OLD"]
    daily = pd.read_excel(workbook, sheet_name="DailyByTier")
    assert "Unclassified" in set(daily["series"])


@pytest.mark.unit
def test_close_volume_share_trend_appendix_adds_tau_and_fitted_values(artifact_dir: Path) -> None:
    run = artifact_dir / "run"
    out = artifact_dir / "figures"
    _make_run(run)
    volume_db, tier_map, membership_root = _make_volume_sources(artifact_dir)
    ctx = FigureBuildContext(
        run_root=run,
        out_dir=out,
        volume_db=volume_db,
        tier_map_csv=tier_map,
        membership_root=membership_root,
    )

    trend = close_volume_share_trend_appendix(ctx).frame

    assert {"date", "series", "tau", "close_share_pct", "fitted_close_share_pct"} <= set(trend.columns)
    assert trend["fitted_close_share_pct"].notna().all()
    tau_by_date = trend.groupby("date")["tau"].nunique()
    assert tau_by_date.max() == 1
    assert trend["tau"].min() == 0
    assert set(trend["series"]) == {"All", "Tier 1", "Tier 2", "Tier 3"}
