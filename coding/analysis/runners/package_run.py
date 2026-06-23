"""Package hypothesis outputs, volume tables, and thesis figures for one run."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from .run_all_hypotheses import make_run_layout


STRATEGY_ORDER = [
    "S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
    "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD",
]


def _fmt_strategy(name: str) -> str:
    return {
        "S0_MOC": "S0 MOC",
        "S1_STATIC": "S1 Static",
        "S2_TIME_ADAPTIVE": "S2 Time",
        "S3_OFI": "S3 OFI",
        "S3_IMB": "S3 IMB",
        "S3_FULL": "S3 Full",
        "S4_TOD": "S4 TOD",
    }.get(name, name)


def _ordered(df: pd.DataFrame, col: str = "strategy") -> pd.DataFrame:
    out = df.copy()
    out["_order"] = out[col].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
    return out.sort_values(["_order", col]).drop(columns="_order")




def _has_adv_spread_grid(df: pd.DataFrame) -> bool:
    return {"adv_bucket", "spread_bucket", "adv_spread_bucket"}.issubset(df.columns)


def _save_table(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    df.to_markdown(path.with_suffix(".md"), index=index)
    df.to_latex(path.with_suffix(".tex"), index=index, float_format="%.4f")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _load_panels(run_root: Path) -> dict[str, pd.DataFrame]:
    hyp = run_root / "hypotheses"
    return {
        "h1": pd.read_parquet(hyp / "h1" / "h1_panel.parquet"),
        "h2": pd.read_parquet(hyp / "h2" / "h2_panel.parquet"),
        "h3": pd.read_parquet(hyp / "h3" / "h3_panel.parquet"),
    }


def _panel_checks(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in panels.items():
        rows.append({
            "panel": name.upper(),
            "rows": len(p),
            "orders": p["order_id"].nunique() if "order_id" in p else np.nan,
            "symbol_days": p[["symbol", "date"]].drop_duplicates().shape[0],
            "symbols": p["symbol"].nunique(),
            "dates": p["date"].nunique(),
            "size_fracs": ", ".join(f"{x:g}" for x in sorted(p["size_frac"].astype(float).unique())),
            "has_moc": bool((p["strategy"] == "S0_MOC").any()),
            "has_s4": bool((p["strategy"] == "S4_TOD").any()),
            "has_moc_diff": "net_alpha_vs_moc_bps" in p.columns,
            "has_arrival_time": "arrival_time" in p.columns,
        })
    return pd.DataFrame(rows)


def _attach_as_markout(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-row signed AS markout as a cost figure, NaN where no passive fill.

    Negated mean signed post-fill drift conditional on fills; positive values
    denote adverse drift. Primary reporting diagnostic, replacing the
    one-sided ``adverse_selection_cost_bps`` in summary tables (the one-sided
    column remains available in the panel as a downside measure).
    """
    out = panel.copy()
    if "adverse_selection_bps" in out.columns and "fill_rate" in out.columns:
        out["as_markout_bps"] = np.where(
            pd.to_numeric(out["fill_rate"], errors="coerce") > 0,
            -pd.to_numeric(out["adverse_selection_bps"], errors="coerce"),
            np.nan,
        )
    else:
        out["as_markout_bps"] = np.nan
    return out


def _write_result_tables(run_root: Path, panels: dict[str, pd.DataFrame]) -> None:
    tables = run_root / "tables"
    h1 = _attach_as_markout(panels["h1"])
    h2 = _attach_as_markout(panels["h2"])
    h3 = panels["h3"]

    h1_summary = h1.groupby("strategy").agg(
        n=("net_alpha_bps", "count"),
        net_alpha_bps=("net_alpha_bps", "mean"),
        gross_alpha_bps=("alpha_bps", "mean"),
        moc_diff_bps=("net_alpha_vs_moc_bps", "mean"),
        fill_rate=("fill_rate", "mean"),
        as_markout_bps=("as_markout_bps", "mean"),
        impact_bps=("impact_bps", "mean"),
    ).reset_index()
    _save_table(_ordered(h1_summary), tables / "thesis" / "h1_strategy_summary.csv")

    h1_window = h1.groupby(["strategy", "window"]).agg(
        n=("net_alpha_bps", "count"),
        net_alpha_bps=("net_alpha_bps", "mean"),
        moc_diff_bps=("net_alpha_vs_moc_bps", "mean"),
        fill_rate=("fill_rate", "mean"),
    ).reset_index()
    _save_table(_ordered(h1_window), tables / "analysis" / "h1_by_window.csv")

    h2_summary = h2.groupby("strategy").agg(
        n=("net_alpha_bps", "count"),
        net_alpha_bps=("net_alpha_bps", "mean"),
        fill_rate=("fill_rate", "mean"),
        as_markout_bps=("as_markout_bps", "mean"),
    ).reset_index()
    _save_table(_ordered(h2_summary), tables / "thesis" / "h2_strategy_summary.csv")

    h2_pooled = pd.read_csv(run_root / "hypotheses" / "h2" / "h2_pooled.csv")
    _save_table(h2_pooled, tables / "thesis" / "h2_signal_decomposition.csv")

    h3_raear = pd.read_csv(run_root / "hypotheses" / "h3" / "h3_raear.csv")
    _save_table(_ordered(h3_raear), tables / "thesis" / "h3_raear.csv")
    h3_tev = pd.read_csv(run_root / "hypotheses" / "h3" / "h3_tev.csv")
    _save_table(_ordered(h3_tev), tables / "analysis" / "h3_tev_portfolio_bounds.csv")

    checks = _panel_checks(panels)
    _save_table(checks, tables / "checks" / "panel_acceptance_checks.csv")

    if _has_adv_spread_grid(h1):
        grid_summary = (
            h1.groupby(["adv_bucket", "spread_bucket", "adv_spread_bucket", "strategy"], dropna=False)
            .agg(
                n=("net_alpha_bps", "count"),
                net_alpha_bps=("net_alpha_bps", "mean"),
                moc_diff_bps=("net_alpha_vs_moc_bps", "mean"),
                fill_rate=("fill_rate", "mean"),
                as_markout_bps=("as_markout_bps", "mean"),
            )
            .reset_index()
        )
        _save_table(_ordered(grid_summary), tables / "analysis" / "h1_strategy_by_adv_spread_bucket.csv")

    for source, target in (
        ("h1_subgroup_adv_bucket.csv", "h1_subgroup_adv_bucket.csv"),
        ("h1_subgroup_spread_bucket.csv", "h1_subgroup_spread_bucket.csv"),
        ("h1_subgroup_adv_spread_bucket.csv", "h1_subgroup_adv_spread_bucket.csv"),
    ):
        source_path = run_root / "hypotheses" / "h1" / source
        if source_path.exists():
            _save_table(pd.read_csv(source_path), tables / "analysis" / target)

    primary = pd.read_csv(run_root / "hypotheses" / "h1" / "h1_primary_ttest.csv")
    _save_table(primary, tables / "thesis" / "h1_primary_test.csv")


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")




def _adv_spread_heatmap(h1: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path, *, cmap: str = "vlag") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not _has_adv_spread_grid(h1) or value_col not in h1.columns:
        return
    subset = h1[h1["strategy"] == "S3_FULL"].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index="spread_bucket", columns="adv_bucket", values=value_col, aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, center=0 if cmap == "vlag" else None, ax=ax)
    ax.set_xlabel("ADV bucket (1 = highest dollar ADV)")
    ax.set_ylabel("Spread bucket (1 = tightest spread)")
    ax.set_title(title)
    _save_fig(fig, path)
    plt.close(fig)


def _write_figures(run_root: Path, panels: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures = run_root / "figures"
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = sns.color_palette("colorblind")

    h1_summary = pd.read_csv(run_root / "tables" / "thesis" / "h1_strategy_summary.csv")
    h1_summary = _ordered(h1_summary)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(h1_summary["strategy"].map(_fmt_strategy), h1_summary["net_alpha_bps"], color=palette[:len(h1_summary)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net alpha (bps)")
    ax.set_title("Headline Net Alpha by Strategy")
    ax.tick_params(axis="x", rotation=30)
    _save_fig(fig, figures / "h1_net_alpha_by_strategy")
    plt.close(fig)

    h1 = _attach_as_markout(panels["h1"])
    _adv_spread_heatmap(h1, "net_alpha_vs_moc_bps", "S3 Full MOC Differential by ADV x Spread Bucket", "S3 Full - MOC (bps)", figures / "h1_s3_full_moc_diff_adv_spread_heatmap")
    _adv_spread_heatmap(h1, "fill_rate", "S3 Full Fill Rate by ADV x Spread Bucket", "Fill rate", figures / "h1_s3_full_fill_rate_adv_spread_heatmap", cmap="viridis")
    _adv_spread_heatmap(h1, "as_markout_bps", "S3 Full Adverse-Selection Markout by ADV x Spread Bucket", "AS markout (bps)", figures / "h1_s3_full_as_markout_adv_spread_heatmap", cmap="mako")

    s3_window = h1[h1["strategy"] == "S3_FULL"].groupby("window", as_index=False)["net_alpha_vs_moc_bps"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(s3_window["window"], s3_window["net_alpha_vs_moc_bps"], color=palette[2])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Execution window")
    ax.set_ylabel("S3 Full - MOC (bps)")
    ax.set_title("S3 Full MOC Differential by Window")
    _save_fig(fig, figures / "h1_s3_full_moc_diff_by_window")
    plt.close(fig)

    passive = h1[h1["strategy"] != "S0_MOC"].copy()
    passive["strategy_label"] = passive["strategy"].map(_fmt_strategy)
    passive_order = [_fmt_strategy(s) for s in STRATEGY_ORDER if s in set(passive["strategy"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.boxplot(data=passive, x="strategy_label", y="fill_rate", order=passive_order,
                ax=ax, color=palette[0], showfliers=False)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Passive fill rate")
    ax.set_xlabel("")
    ax.set_title("Fill-Rate Distribution")
    _save_fig(fig, figures / "fill_rate_distribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.boxplot(data=passive, x="strategy_label", y="as_markout_bps",
                order=passive_order,
                ax=ax, color=palette[1], showfliers=False)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("AS markout (bps)")
    ax.set_xlabel("")
    ax.set_title("Adverse-Selection Markout Distribution (conditional on fill)")
    _save_fig(fig, figures / "adverse_selection_markout_distribution")
    plt.close(fig)

    h2 = pd.read_csv(run_root / "tables" / "thesis" / "h2_signal_decomposition.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(h2["label"], h2["mean"], yerr=1.96 * h2["se_twoway"], color=palette[3], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Differential net alpha (bps)")
    ax.set_title("H2 Signal Marginal Contributions")
    ax.tick_params(axis="x", rotation=25)
    _save_fig(fig, figures / "h2_signal_marginals")
    plt.close(fig)

    h3 = pd.read_csv(run_root / "tables" / "thesis" / "h3_raear.csv")
    eta = np.linspace(0, 0.1, 101)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, row in _ordered(h3).iterrows():
        ax.plot(eta, row["mean_alpha"] - eta * row["tev"], label=_fmt_strategy(row["strategy"]),
                color=palette[i % len(palette)], linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Risk aversion $\eta$")
    ax.set_ylabel("RAEAR (bps)")
    ax.set_title("Risk-Adjusted Execution Alpha")
    ax.legend(fontsize=8)
    _save_fig(fig, figures / "h3_raear_curves")
    plt.close(fig)


def _volume_queries(volume_db: Path, start: _dt.date, end: _dt.date) -> dict[str, pd.DataFrame]:
    import duckdb

    con = duckdb.connect(str(volume_db), read_only=True)
    try:
        daily_cols = set(con.execute("PRAGMA table_info('daily_volume')").df()["name"])
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        marker_val = (
            "sum(Official_Close_Marker_Val)"
            if "Official_Close_Marker_Val" in daily_cols else "0.0"
        )
        marker_rows = (
            "sum(Official_Close_Marker_Rows)"
            if "Official_Close_Marker_Rows" in daily_cols else "0"
        )
        date_filter = "where Date between ? and ?"
        params = [start, end]
        overview = con.execute(f"""
            select count(*) as n_rows,
                   count(distinct Date) as n_dates,
                   count(distinct Ticker) as n_tickers,
                   min(Date) as first_date,
                   max(Date) as last_date,
                   sum(case when Close_Auction_Val > 0 then 1 else 0 end) as close_nonzero_rows,
                   round(100 * sum(Close_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as pooled_close_share_pct,
                   round(100 * avg(Close_Auction_Val / nullif(Total_Daily_Val, 0)), 4) as mean_row_close_share_pct,
                   round(100 * median(Close_Auction_Val / nullif(Total_Daily_Val, 0)), 4) as median_row_close_share_pct,
                   {marker_rows} as official_close_marker_rows,
                   round({marker_val}, 2) as official_close_marker_value,
                   round(100 * {marker_val} / nullif(sum(Total_Daily_Val), 0), 4) as official_close_marker_share_pct
              from daily_volume
              {date_filter}
        """, params).df()
        by_symbol = con.execute(f"""
            select Ticker,
                   count(*) as n_days,
                   round(avg(Total_Daily_Val), 2) as mean_daily_dollar_volume,
                   round(avg(Close_Auction_Val), 2) as mean_close_auction_dollar_volume,
                   round(100 * sum(Close_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as pooled_close_share_pct,
                   round(100 * avg(Close_Auction_Val / nullif(Total_Daily_Val, 0)), 4) as mean_row_close_share_pct,
                   sum(case when Close_Auction_Val > 0 then 1 else 0 end) as close_nonzero_days,
                   {marker_rows} as official_close_marker_rows,
                   round({marker_val}, 2) as official_close_marker_value,
                   round(100 * {marker_val} / nullif(sum(Total_Daily_Val), 0), 4) as official_close_marker_share_pct
              from daily_volume
              {date_filter}
             group by Ticker
             order by pooled_close_share_pct desc
        """, params).df()
        by_date = con.execute(f"""
            select Date,
                   count(*) as n_symbols,
                   round(sum(Total_Daily_Val), 2) as total_dollar_volume,
                   round(sum(Close_Auction_Val), 2) as close_auction_dollar_volume,
                   round(100 * sum(Close_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as pooled_close_share_pct,
                   {marker_rows} as official_close_marker_rows,
                   round({marker_val}, 2) as official_close_marker_value,
                   round(100 * {marker_val} / nullif(sum(Total_Daily_Val), 0), 4) as official_close_marker_share_pct
              from daily_volume
              {date_filter}
             group by Date
             order by Date
        """, params).df()
        buckets = con.execute(f"""
            select round(100 * sum(Pre_Market_Val) / nullif(sum(Total_Daily_Val), 0), 4) as pre_market_pct,
                   round(100 * sum(Open_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as open_auction_pct,
                   round(100 * sum(Morning_30m_Val) / nullif(sum(Total_Daily_Val), 0), 4) as morning_30m_pct,
                   round(100 * sum(Mid_Day_Val) / nullif(sum(Total_Daily_Val), 0), 4) as midday_pct,
                   round(100 * sum(Afternoon_30m_Val) / nullif(sum(Total_Daily_Val), 0), 4) as afternoon_30m_pct,
                   round(100 * sum(Close_Auction_Val) / nullif(sum(Total_Daily_Val), 0), 4) as close_auction_pct,
                   round(100 * sum(Post_Market_Val) / nullif(sum(Total_Daily_Val), 0), 4) as post_market_pct,
                   round(100 * {marker_val} / nullif(sum(Total_Daily_Val), 0), 4) as official_close_marker_pct
              from daily_volume
              {date_filter}
        """, params).df()
        zero_close = con.execute(f"""
            select Ticker, Date, Total_Daily_Val, Close_Auction_Val
              from daily_volume
              {date_filter}
               and Close_Auction_Val <= 0
             order by Ticker, Date
        """, params).df()
        if "daily_volume_skipped" in tables:
            skipped_summary = con.execute("""
                select Reason,
                       count(*) as n_rows,
                       count(distinct Date) as n_dates,
                       count(distinct Ticker) as n_tickers
                  from daily_volume_skipped
                 where Date between ? and ?
                 group by Reason
                 order by n_rows desc, Reason
            """, params).df()
            skipped_detail = con.execute("""
                select Ticker, Date, Reason, Detail, Source_Path
                  from daily_volume_skipped
                 where Date between ? and ?
                 order by Date, Ticker, Reason
            """, params).df()
        else:
            skipped_summary = pd.DataFrame(columns=["Reason", "n_rows", "n_dates", "n_tickers"])
            skipped_detail = pd.DataFrame(columns=["Ticker", "Date", "Reason", "Detail", "Source_Path"])
    finally:
        con.close()
    return {
        "volume_overview": overview,
        "volume_by_symbol": by_symbol,
        "volume_by_date": by_date,
        "volume_bucket_share": buckets,
        "volume_zero_close_days": zero_close,
        "volume_skipped_summary": skipped_summary,
        "volume_skipped_detail": skipped_detail,
    }


def _write_volume_report(run_root: Path, volume_db: Path, start: _dt.date, end: _dt.date) -> None:
    tables = _volume_queries(volume_db, start, end)
    vol_dir = run_root / "volume"
    for name, df in tables.items():
        _save_table(df, vol_dir / f"{name}.csv")

    overview = tables["volume_overview"].iloc[0].to_dict()
    md = [
        "# Volume Analysis Template",
        "",
        f"Sample: {start.isoformat()} to {end.isoformat()}",
        "",
        "## Key Checks",
        "",
        f"- Rows: {int(overview['n_rows'])}",
        f"- Dates: {int(overview['n_dates'])}",
        f"- Tickers: {int(overview['n_tickers'])}",
        f"- Positive close-auction rows: {int(overview['close_nonzero_rows'])}",
        f"- Pooled close-auction share: {overview['pooled_close_share_pct']:.4f}%",
        f"- Mean row close-auction share: {overview['mean_row_close_share_pct']:.4f}%",
        f"- Median row close-auction share: {overview['median_row_close_share_pct']:.4f}%",
        f"- Official-close marker rows excluded from buckets: {int(overview['official_close_marker_rows'])}",
        f"- Official-close marker value share: {overview['official_close_marker_share_pct']:.4f}%",
        "",
        "## Files",
        "",
        "- `volume_overview.csv`: one-row headline checks.",
        "- `volume_by_symbol.csv`: ticker-level close-auction shares and coverage.",
        "- `volume_by_date.csv`: date-level close-auction shares.",
        "- `volume_bucket_share.csv`: pooled intraday bucket distribution.",
        "- `volume_zero_close_days.csv`: symbol-days with zero close-auction volume.",
        "- `volume_skipped_summary.csv`: skipped symbol-day counts by reason.",
        "- `volume_skipped_detail.csv`: skipped symbol-day audit trail.",
    ]
    (vol_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _mirror_volume_template(run_root: Path, volume_db: Path) -> None:
    """Copy run-specific volume CSV/Markdown tables to volume/analysis_templates."""
    target = volume_db.parent / "analysis_templates" / run_root.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_root / "volume", target)


def _write_volume_figures(run_root: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    vol_dir = run_root / "volume"
    fig_dir = run_root / "figures" / "volume"
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    by_symbol = pd.read_csv(vol_dir / "volume_by_symbol.csv")
    by_date = pd.read_csv(vol_dir / "volume_by_date.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    top = by_symbol.sort_values("pooled_close_share_pct", ascending=False)
    sns.barplot(data=top, x="Ticker", y="pooled_close_share_pct", color="#4C78A8", ax=ax)
    ax.set_ylabel("Pooled close-auction share (%)")
    ax.set_xlabel("")
    ax.set_title("Closing-Auction Share by Symbol")
    ax.tick_params(axis="x", rotation=75)
    _save_fig(fig, fig_dir / "volume_close_share_by_symbol")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_date["Date"] = pd.to_datetime(by_date["Date"])
    ax.plot(by_date["Date"], by_date["pooled_close_share_pct"], color="#F58518", linewidth=1.5)
    ax.set_ylabel("Pooled close-auction share (%)")
    ax.set_xlabel("")
    ax.set_title("Closing-Auction Share by Date")
    fig.autofmt_xdate()
    _save_fig(fig, fig_dir / "volume_close_share_by_date")
    plt.close(fig)


def _write_reporting_evaluation(run_root: Path) -> None:
    text = """# Reporting Setup Evaluation

## Assessment

The old flat artifact layout was useful while debugging but is weak for thesis work:
H1/H2/H3 folders and logs were spread across the top-level artifact directory, old
versions were easy to confuse with repaired outputs, and the volume DuckDB required
direct database access to verify key numbers.

The new run-bundle layout is closer to academic replication practice. Each run now has
one folder containing raw hypothesis outputs, derived thesis tables, acceptance checks,
volume diagnostics, figures, and metadata. This makes it clear which tables belong
to which sample, fill mechanism, and parent-size convention.

## Added Tables

- H1 strategy summary with net alpha, gross alpha, fill rate, adverse-selection cost,
  impact, and MOC differential.
- H1 primary paired test table.
- H1 by-window table for the A/B/C timing comparison.
- H2 signal-decomposition table with matched realized-fill-rate semantics.
- H3 RAEAR and portfolio tracking-error tables.
- Panel acceptance checks for row counts, symbols, headline-size filtering, MOC rows,
  S4 rows, and metadata columns.
- Volume overview, by-symbol, by-date, bucket-share, and zero-close-day tables.

## Added Figures

- Net alpha by strategy.
- S3 Full minus MOC by execution window.
- Fill-rate distribution by strategy.
- Adverse-selection cost distribution by strategy.
- H2 signal marginal effects with confidence intervals.
- H3 RAEAR curves.
- Closing-auction share by symbol and by date.

## Remaining Academic Enhancements

- For the full sample, add confidence intervals based on the same two-way clustered
  estimator used in the hypothesis tests, not only descriptive standard errors.
- Add robustness-table panels for parent size, Cox/XGB/KM fill specifications,
  adverse-selection horizon, and trade-signing variants.
- Add a compact LaTeX table builder for final thesis tables once the full 2018--2019
  run is available.
"""
    (run_root / "REPORTING_EVALUATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", type=Path, default=cfg.ARTIFACTS_DIR / "runs")
    p.add_argument("--source-artifacts", type=Path, default=cfg.ARTIFACTS_DIR)
    p.add_argument("--volume-db", type=Path, default=cfg.VOLUME_DB_PATH)
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.EVAL_END)
    p.add_argument("--source-h1", default="h1_v7")
    p.add_argument("--source-h2", default="h2_v7")
    p.add_argument("--source-h3", default="h3_v7")
    args = p.parse_args()

    run_root = args.run_root / args.run_id
    paths = make_run_layout(run_root)
    _copy_tree(args.source_artifacts / args.source_h1, paths["h1"])
    _copy_tree(args.source_artifacts / args.source_h2, paths["h2"])
    _copy_tree(args.source_artifacts / args.source_h3, paths["h3"])

    meta = {
        "run_id": args.run_id,
        "source_artifacts": str(args.source_artifacts),
        "source_h1": args.source_h1,
        "source_h2": args.source_h2,
        "source_h3": args.source_h3,
        "volume_db": str(args.volume_db),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "headline_size_frac": cfg.PARENT_ORDER_PRIMARY_FRACTION,
    }
    (paths["metadata"] / "package_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    panels = _load_panels(run_root)
    _write_result_tables(run_root, panels)
    _write_volume_report(run_root, args.volume_db, args.start, args.end)
    _mirror_volume_template(run_root, args.volume_db)
    _write_figures(run_root, panels)
    _write_volume_figures(run_root)
    _write_reporting_evaluation(run_root)

    readme = (
        f"# {args.run_id}\n\n"
        f"Packaged thesis run for {args.start} to {args.end}.\n\n"
        "## Structure\n\n"
        "- `hypotheses/h1`, `hypotheses/h2`, `hypotheses/h3`: raw runner outputs.\n"
        "- `tables/thesis`: compact tables suitable for thesis drafting.\n"
        "- `tables/analysis`: supporting diagnostics and decomposition tables.\n"
        "- `tables/checks`: panel acceptance checks.\n"
        "- `volume`: CSV/Markdown volume analysis template.\n"
        "- `figures`: PNG/PDF plots for reporting.\n"
        "- `metadata`: run configuration and packaging manifest.\n"
    )
    (run_root / "README.md").write_text(readme, encoding="utf-8")
    print(f"Packaged run: {run_root}")


if __name__ == "__main__":
    main()
