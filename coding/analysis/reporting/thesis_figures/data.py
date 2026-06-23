"""Artifact-to-plot-table transforms for standardized thesis figures."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from ... import config as cfg
from ...data.index_universe import load_index_membership
from ...runners import render_thesis_results as rtr
from ...runners._common import rolling_window_panel
from ...utils.symbols import canonical_symbol
from .io import hash_file


@dataclass
class FigureBuildContext:
    run_root: Path
    out_dir: Path
    compare_runs: dict[str, Path] = field(default_factory=dict)
    stats_dir: Path | None = None
    as_horizon_csv: Path | None = None
    size_grid_root: Path | None = None
    volume_bucket_csv: Path | None = None
    volume_by_date_csv: Path | None = None
    volume_db: Path | None = None
    tier_map_csv: Path | None = None
    membership_root: Path | None = None
    close_share_xlsx_path: Path | None = None


@dataclass
class FigureData:
    frame: pd.DataFrame
    inputs: dict[str, str]


def _hash(path: Path, label: str | None = None) -> dict[str, str]:
    path = Path(path)
    return {label or str(path): hash_file(path)} if path.exists() else {}


def _first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def _default_stats_dir(ctx: FigureBuildContext) -> Path | None:
    return _first_existing(
        ctx.stats_dir,
        ctx.run_root / "statistical_tests",
        cfg.RUN_ROOT / "final_submission_20260618" / "evidence" / "statistical_tests_20260618_final_v4_xgb",
    )


def _default_as_horizon_csv(ctx: FigureBuildContext) -> Path | None:
    return _first_existing(
        ctx.as_horizon_csv,
        cfg.ARTIFACTS_DIR / "as_horizon_robustness_20260619" / "as_horizon_summary.csv",
    )


def _default_size_grid_root(ctx: FigureBuildContext) -> Path | None:
    return _first_existing(
        ctx.size_grid_root,
        cfg.ARTIFACTS_DIR / "runs" / "final_v4_20260619_size_grid_w6",
        cfg.ARTIFACTS_DIR / "runs" / "size_grid_20260613",
    )


def _read_csv(path: Path, label: str) -> FigureData:
    return FigureData(pd.read_csv(path), _hash(path, label))


def _read_h1_panel(ctx: FigureBuildContext) -> FigureData:
    path = ctx.run_root / "hypotheses" / "h1" / "h1_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return FigureData(pd.read_parquet(path), _hash(path, "run:h1_panel.parquet"))


def _strategy_labels(values: pd.Series) -> pd.Series:
    return values.map(lambda s: rtr.STRATEGY_NAMES.get(str(s), str(s)).replace("$-$", "-"))


def alpha_decomposition(ctx: FigureBuildContext) -> FigureData:
    panel_data = _read_h1_panel(ctx)
    panel = rtr._primary_h1_surface(panel_data.frame)
    diag = rtr._panel_strategy_diagnostics(panel)
    frame = pd.DataFrame({
        "strategy": diag.index,
        "label": _strategy_labels(pd.Series(diag.index, index=diag.index)).to_numpy(),
        "gross_alpha": diag["gross_alpha"].to_numpy(dtype=float),
        "maker_rebate": diag["fill_rate"].to_numpy(dtype=float) * cfg.MAKER_REBATE_BPS,
        "commission": -cfg.COMMISSION_BPS,
        "self_impact": 0.0,
        "net_alpha": diag["net_alpha"].to_numpy(dtype=float),
        "as_component": diag["as_component"].to_numpy(dtype=float),
    })
    if "impact_bps" in panel.columns:
        impact = panel.groupby("strategy")["impact_bps"].mean()
        frame["self_impact"] = -frame["strategy"].map(impact).fillna(0.0)
    return FigureData(frame, panel_data.inputs)


def alpha_fill_frontier(ctx: FigureBuildContext) -> FigureData:
    panel_data = _read_h1_panel(ctx)
    diag = rtr._panel_strategy_diagnostics(rtr._primary_h1_surface(panel_data.frame))
    frame = pd.DataFrame({
        "strategy": diag.index,
        "label": _strategy_labels(pd.Series(diag.index, index=diag.index)).to_numpy(),
        "fill_rate": diag["fill_rate"].to_numpy(dtype=float),
        "net_alpha": diag["net_alpha"].to_numpy(dtype=float),
    })
    return FigureData(frame, panel_data.inputs)


def h2_heatmap(ctx: FigureBuildContext) -> FigureData:
    path = ctx.run_root / "hypotheses" / "h2" / "h2_per_bin_differentials.csv"
    data = _read_csv(path, "run:h2_per_bin_differentials.csv")
    frame = data.frame.copy()
    frame["label"] = frame["label"].map(lambda x: rtr.H2_LABELS.get(str(x), str(x)).replace("$-$", "-"))
    frame["bin_label"] = frame["bin"].map(lambda x: f"B{int(x) + 1}")
    return FigureData(frame[["label", "bin_label", "mean"]], data.inputs)


def raear_curve(ctx: FigureBuildContext) -> FigureData:
    path = ctx.run_root / "hypotheses" / "h3" / "h3_raear.csv"
    data = _read_csv(path, "run:h3_raear.csv")
    raear = data.frame.set_index("strategy")
    eta_cols = [c for c in raear.columns if c.startswith("raear_eta_")]
    eta_max = max((float(c.replace("raear_eta_", "")) for c in eta_cols), default=0.5)
    grid = np.linspace(0.0, eta_max, 120)
    rows = []
    for strategy in [s for s in rtr.STRATEGY_ORDER if s in raear.index]:
        row = raear.loc[strategy]
        label = rtr.STRATEGY_NAMES.get(strategy, strategy).replace("$-$", "-")
        values = float(row["mean_alpha"]) - grid * float(row["tev"])
        rows.extend({"strategy": strategy, "label": label, "eta": eta, "raear": value}
                    for eta, value in zip(grid, values))
    return FigureData(pd.DataFrame(rows), data.inputs)


def _stats_csv(ctx: FigureBuildContext, name: str) -> Path:
    stats_dir = _default_stats_dir(ctx)
    if stats_dir is None:
        raise FileNotFoundError(f"stats dir not found for {name}")
    path = stats_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fill_model_oos_calibration(ctx: FigureBuildContext) -> FigureData:
    data = _read_csv(_stats_csv(ctx, "fill_model_oos_calibration.csv"), "stats:fill_model_oos_calibration.csv")
    labels = {"cox": "Cox", "km": "Kaplan-Meier", "xgb": "XGBoost"}
    rows = []
    for row in data.frame.itertuples(index=False):
        model = str(row.model)
        tier = int(row.tier)
        rows.append({
            "metric": "AUC",
            "model": model,
            "model_label": labels.get(model, model),
            "tier_label": f"Tier {tier}",
            "value": float(row.auc),
        })
        rows.append({
            "metric": "Abs. calibration error (pp)",
            "model": model,
            "model_label": labels.get(model, model),
            "tier_label": f"Tier {tier}",
            "value": float(row.absolute_calibration_error) * 100.0,
        })
    return FigureData(pd.DataFrame(rows), data.inputs)


def fill_spec_frontier(ctx: FigureBuildContext) -> FigureData:
    inputs: dict[str, str] = {}
    stats_path = _default_stats_dir(ctx)
    econ = stats_path / "fill_model_economic_tests.csv" if stats_path is not None else None
    if econ is not None and econ.exists():
        data = _read_csv(econ, "stats:fill_model_economic_tests.csv")
        frame = data.frame.copy()
        inputs.update(data.inputs)
        out = pd.DataFrame({
            "spec": frame["spec"],
            "spec_label": frame["label"].astype(str).str.replace(r" \(.*\)$", "", regex=True),
            "spec_group": np.where(frame["spec"].isin(rtr.MODEL_SPECS), "Model-based", "Tape replay"),
            "fill_rate": pd.to_numeric(frame["mean_fill_rate"], errors="coerce"),
            "net_alpha": pd.to_numeric(frame["mean_net_alpha_vs_moc_bps"], errors="coerce"),
        })
        return FigureData(out.dropna(subset=["fill_rate", "net_alpha"]), inputs)

    rows = []
    headline = rtr._spec_diagnostics(_read_h1_panel(ctx).frame)
    if headline:
        rows.append(("tape_replay_queue", headline))
    for spec, root in ctx.compare_runs.items():
        path = Path(root) / "hypotheses" / "h1" / "h1_panel.parquet"
        if not path.exists():
            continue
        panel = pd.read_parquet(path)
        inputs.update(_hash(path, f"compare:{spec}:h1_panel.parquet"))
        rows.append((spec, rtr._spec_diagnostics(panel)))
    out = pd.DataFrame([{
        "spec": spec,
        "spec_label": rtr.FILL_SPEC_NAMES.get(spec, spec).split(" (")[0],
        "spec_group": "Model-based" if spec in rtr.MODEL_SPECS else "Tape replay",
        "fill_rate": d.get("fill_rate"),
        "net_alpha": d.get("net_alpha"),
    } for spec, d in rows if d])
    return FigureData(out, inputs)


def alt_vs_queue_diagnostics(ctx: FigureBuildContext) -> FigureData:
    inputs: dict[str, str] = {}
    stats_path = _default_stats_dir(ctx)
    paired = stats_path / "fill_model_vs_queue_tests.csv" if stats_path is not None else None
    if paired is not None and paired.exists():
        data = _read_csv(paired, "stats:fill_model_vs_queue_tests.csv")
        frame = data.frame.copy()
        metric_labels = {
            "net_alpha_vs_moc_bps": "Net alpha vs. MOC (bps)",
            "fill_rate": "Fill rate",
            "as_markout_bps": "AS markout (bps)",
        }
        out = pd.DataFrame({
            "metric": frame["metric"],
            "metric_label": frame["metric"].map(metric_labels).fillna(frame["metric"]),
            "spec": frame["spec"],
            "spec_label": frame["label"].astype(str).str.replace(r" \(.*\)$", "", regex=True),
            "spec_group": np.where(frame["spec"].isin(rtr.MODEL_SPECS), "Model-based", "Tape replay"),
            "mean_diff": pd.to_numeric(frame["mean_diff"], errors="coerce"),
        })
        return FigureData(out.dropna(subset=["mean_diff"]), data.inputs)

    base = rtr._spec_diagnostics(_read_h1_panel(ctx).frame)
    rows = []
    for spec, root in ctx.compare_runs.items():
        path = Path(root) / "hypotheses" / "h1" / "h1_panel.parquet"
        if not path.exists():
            continue
        d = rtr._spec_diagnostics(pd.read_parquet(path))
        inputs.update(_hash(path, f"compare:{spec}:h1_panel.parquet"))
        for key, label in (
            ("net_alpha", "Net alpha vs. MOC (bps)"),
            ("fill_rate", "Fill rate"),
            ("as_markout", "AS markout (bps)"),
        ):
            rows.append({
                "metric": key,
                "metric_label": label,
                "spec": spec,
                "spec_label": rtr.FILL_SPEC_NAMES.get(spec, spec).split(" (")[0],
                "spec_group": "Model-based" if spec in rtr.MODEL_SPECS else "Tape replay",
                "mean_diff": d.get(key, np.nan) - base.get(key, np.nan),
            })
    return FigureData(pd.DataFrame(rows), inputs)


def as_horizon_robustness(ctx: FigureBuildContext) -> FigureData:
    path = _default_as_horizon_csv(ctx)
    if path is None:
        raise FileNotFoundError("as_horizon_summary.csv not found")
    data = _read_csv(path, "as_horizon:as_horizon_summary.csv")
    labels = {
        "as_markout_bps": "Adverse-selection markout",
        "as_component_bps": "Adverse-selection component",
        "mean_net_alpha_vs_moc_bps": "Net alpha vs. MOC",
    }
    rows = []
    for _, row in data.frame.sort_values("horizon_seconds").iterrows():
        for col, label in labels.items():
            rows.append({
                "horizon_seconds": float(row["horizon_seconds"]),
                "series": label,
                "value": float(row[col]),
            })
    return FigureData(pd.DataFrame(rows), data.inputs)


def parent_size_robustness(ctx: FigureBuildContext) -> FigureData:
    root = _default_size_grid_root(ctx)
    if root is None:
        raise FileNotFoundError("size grid root not found")
    path = root / "size_table_summary.csv"
    data = _read_csv(path, "size_grid:size_table_summary.csv")
    rows = []
    for _, row in data.frame.sort_values("size_bucket").iterrows():
        pct = float(row["size_bucket"]) * 100.0
        mean = float(row["mean_net_alpha_bps"])
        se = float(row.get("se_twoway", np.nan))
        rows.append({
            "panel": "Net alpha",
            "parent_size_pct": pct,
            "series": "Net alpha vs. MOC",
            "value": mean,
            "ci_low": mean - 1.96 * se if np.isfinite(se) else np.nan,
            "ci_high": mean + 1.96 * se if np.isfinite(se) else np.nan,
        })
        rows.append({
            "panel": "Fill / AS diagnostics",
            "parent_size_pct": pct,
            "series": "Fill rate",
            "value": float(row["mean_fill_rate"]),
            "ci_low": np.nan,
            "ci_high": np.nan,
        })
        as_col = "mean_as_markout_bps" if "mean_as_markout_bps" in row else "mean_as_cost_bps"
        rows.append({
            "panel": "Fill / AS diagnostics",
            "parent_size_pct": pct,
            "series": "AS markout (bps)",
            "value": float(row[as_col]),
            "ci_low": np.nan,
            "ci_high": np.nan,
        })
    return FigureData(pd.DataFrame(rows), data.inputs)


def rolling_stability(ctx: FigureBuildContext) -> FigureData:
    panel_data = _read_h1_panel(ctx)
    panel = panel_data.frame
    sub = panel[panel["strategy"] == "S3_FULL"].copy()
    if "net_alpha_vs_moc_bps" not in sub.columns:
        sub["net_alpha_vs_moc_bps"] = sub["net_alpha_bps"]
    dates = pd.to_datetime(sub["date"], errors="coerce")
    span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
    if span_days >= 360:
        rolling = rolling_window_panel(sub, alpha_col="net_alpha_vs_moc_bps")
        if not rolling.empty:
            rolling["ci_low"] = rolling["mean_alpha"] - 1.96 * rolling["clustered_se"]
            rolling["ci_high"] = rolling["mean_alpha"] + 1.96 * rolling["clustered_se"]
            return FigureData(
                rolling[["window_end", "mean_alpha", "ci_low", "ci_high"]],
                panel_data.inputs,
            )
    grouped = (
        sub.assign(date=pd.to_datetime(sub["date"], errors="coerce"))
        .groupby("date", as_index=False)["net_alpha_vs_moc_bps"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"date": "window_end", "mean": "mean_alpha"})
    )
    grouped["sem"] = grouped["sem"].fillna(0.0)
    grouped["ci_low"] = grouped["mean_alpha"] - 1.96 * grouped["sem"]
    grouped["ci_high"] = grouped["mean_alpha"] + 1.96 * grouped["sem"]
    return FigureData(grouped[["window_end", "mean_alpha", "ci_low", "ci_high"]], panel_data.inputs)


def data_coverage(ctx: FigureBuildContext) -> FigureData:
    panel_data = _read_h1_panel(ctx)
    panel = panel_data.frame.copy()
    tier_map_path = ctx.run_root / "metadata" / "completed_symbol_tier_map.csv"
    inputs = dict(panel_data.inputs)
    tier_map = pd.DataFrame(columns=["symbol", "tier"])
    if tier_map_path.exists():
        tier_map = pd.read_csv(tier_map_path, usecols=["symbol", "tier"])
        inputs.update(_hash(tier_map_path, "run:completed_symbol_tier_map.csv"))

    success_cols = ["date", "symbol"] + (["tier"] if "tier" in panel.columns else [])
    successes = panel[success_cols].drop_duplicates().copy()
    if "tier" not in successes.columns and not tier_map.empty:
        successes = successes.merge(tier_map, on="symbol", how="left")
    successes["date"] = pd.to_datetime(successes["date"], errors="coerce")
    successes = successes.dropna(subset=["date", "symbol", "tier"])
    successes["month"] = successes["date"].dt.strftime("%Y-%m")
    successes["tier_label"] = successes["tier"].astype(int).map(lambda x: f"Tier {x}")

    failure_path = ctx.run_root / "metadata" / "simulation_failures.csv"
    failures = pd.DataFrame(columns=["date", "symbol", "month", "tier_label"])
    if failure_path.exists():
        raw_failures = pd.read_csv(failure_path)
        inputs.update(_hash(failure_path, "run:simulation_failures.csv"))
        if {"date", "symbol"}.issubset(raw_failures.columns):
            failures = raw_failures[["date", "symbol"]].drop_duplicates().copy()
            if not tier_map.empty:
                failures = failures.merge(tier_map, on="symbol", how="left")
            failures["date"] = pd.to_datetime(failures["date"], errors="coerce")
            failures = failures.dropna(subset=["date", "symbol", "tier"])
            failures = failures.merge(
                successes[["date", "symbol"]],
                on=["date", "symbol"],
                how="left",
                indicator=True,
            )
            failures = failures[failures["_merge"] == "left_only"].drop(columns="_merge")
            failures["month"] = failures["date"].dt.strftime("%Y-%m")
            failures["tier_label"] = failures["tier"].astype(int).map(lambda x: f"Tier {x}")

    by = ["tier_label", "month"]
    simulated = successes.groupby(by, as_index=False).size().rename(columns={"size": "simulated"})
    failed = failures.groupby(by, as_index=False).size().rename(columns={"size": "failed"})
    counts = simulated.merge(failed, on=by, how="outer").fillna(0.0)
    counts["simulated"] = counts["simulated"].astype(int)
    counts["failed"] = counts["failed"].astype(int)
    counts["total"] = counts["simulated"] + counts["failed"]
    counts["coverage_pct"] = np.where(
        counts["total"] > 0,
        100.0 * counts["simulated"] / counts["total"],
        np.nan,
    )
    counts["tier_order"] = counts["tier_label"].str.extract(r"(\d+)").astype(int)
    counts = counts.sort_values(["tier_order", "month"])
    return FigureData(counts[["tier_label", "month", "coverage_pct", "simulated", "total"]], inputs)


_CLOSE_SHARE_START = _dt.date(2018, 1, 2)
_CLOSE_SHARE_END = _dt.date(2019, 12, 31)
_H1_2018_END = _dt.date(2018, 6, 29)
_EXCLUDED_DATE_REASONS = {
    _dt.date(2018, 7, 3): "early close 13:00",
    _dt.date(2018, 11, 23): "early close 13:00",
    _dt.date(2018, 12, 24): "early close 13:00",
    _dt.date(2019, 5, 13): "raw TAQ trade file unobtainable",
    _dt.date(2019, 7, 3): "early close 13:00",
    _dt.date(2019, 11, 29): "early close 13:00",
    _dt.date(2019, 12, 24): "early close 13:00",
}


def _default_volume_db(ctx: FigureBuildContext) -> Path:
    path = _first_existing(
        ctx.volume_db,
        cfg.RUN_ROOT / "volume" / "dollar_volume_sp500_2018_2019.duckdb",
        cfg.VOLUME_DB_PATH,
    )
    if path is None:
        raise FileNotFoundError("No dollar-volume DuckDB found for close-volume share")
    return path


def _default_tier_map(ctx: FigureBuildContext) -> Path:
    path = _first_existing(
        ctx.tier_map_csv,
        cfg.ARTIFACTS_DIR / "runs" / "final_v4_20260618_queue" / "metadata" / "completed_symbol_tier_map.csv",
        cfg.ARTIFACTS_DIR / "runs" / "final_v4_20260618_queue" / "metadata" / "liquidity_tier_audit.csv",
        cfg.ARTIFACTS_DIR / "fill_model_v4_winsor_cutoff_20260617" / "symbol_tier_map.csv",
    )
    if path is None:
        raise FileNotFoundError("No symbol-tier map found for close-volume share")
    return path


def _load_close_volume_source(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = set(con.execute("PRAGMA table_info('daily_volume')").df()["name"])
        marker_val = (
            "Official_Close_Marker_Val"
            if "Official_Close_Marker_Val" in cols else "0.0 AS Official_Close_Marker_Val"
        )
        marker_rows = (
            "Official_Close_Marker_Rows"
            if "Official_Close_Marker_Rows" in cols else "0 AS Official_Close_Marker_Rows"
        )
        frame = con.execute(f"""
            SELECT
                Ticker,
                Date,
                Total_Daily_Val,
                Close_Auction_Val,
                {marker_val},
                {marker_rows}
            FROM daily_volume
            WHERE Total_Daily_Val > 0
        """).df()
    finally:
        con.close()
    frame["date"] = pd.to_datetime(frame["Date"]).dt.date
    frame["symbol"] = frame["Ticker"].map(canonical_symbol)
    return frame.drop(columns=["Date"])


def _active_close_volume_symbol_days(ctx: FigureBuildContext) -> tuple[pd.DataFrame, dict[str, str]]:
    db_path = _default_volume_db(ctx)
    tier_path = _default_tier_map(ctx)
    membership_root = ctx.membership_root or cfg.INDEX_MEMBERSHIP_DIR

    volume = _load_close_volume_source(db_path)
    eligible_dates = {
        d for d in volume["date"].dropna().unique()
        if _CLOSE_SHARE_START <= d <= _CLOSE_SHARE_END and d not in cfg.EXCLUDED_EVAL_DATES
    }
    volume = volume[volume["date"].isin(eligible_dates)].copy()

    membership = load_index_membership("sp500", membership_root)
    active = volume.merge(
        membership[["symbol", "effective_from", "effective_to"]],
        on="symbol",
        how="inner",
    )
    active = active[
        (active["effective_from"] <= active["date"])
        & (active["effective_to"] >= active["date"])
    ].drop_duplicates(["symbol", "date"])

    tiers = pd.read_csv(tier_path)
    if "symbol" not in tiers.columns or "tier" not in tiers.columns:
        raise ValueError(f"Tier map must contain symbol and tier columns: {tier_path}")
    tiers = tiers.copy()
    tiers["symbol"] = tiers["symbol"].map(canonical_symbol)
    keep_cols = ["symbol", "tier"]
    if "tier_source" in tiers.columns:
        keep_cols.append("tier_source")
    tiers = tiers[keep_cols].drop_duplicates("symbol")
    active = active.merge(tiers, on="symbol", how="left")
    active["tier"] = pd.to_numeric(active["tier"], errors="coerce")
    active["tier_group"] = active["tier"].map(
        lambda x: f"Tier {int(x)}" if pd.notna(x) else "Unclassified"
    )
    if "tier_source" not in active.columns:
        active["tier_source"] = np.where(active["tier"].notna(), "tier_map", np.nan)
    active["tier_source"] = active["tier_source"].fillna("unclassified_missing_tier_map")
    active["year"] = pd.to_datetime(active["date"]).dt.year
    active = active.rename(columns={
        "Ticker": "ticker",
        "Total_Daily_Val": "total_daily_dollar_volume",
        "Close_Auction_Val": "close_auction_dollar_volume",
        "Official_Close_Marker_Val": "official_close_marker_dollar_volume",
        "Official_Close_Marker_Rows": "official_close_marker_rows",
    })
    inputs = {
        **_hash(db_path, "volume:dollar_volume_db"),
        **_hash(tier_path, "volume:completed_symbol_tier_map"),
    }
    membership_path = Path(membership_root) / "sp500_membership_intervals.csv"
    inputs.update(_hash(membership_path, "volume:sp500_membership_intervals.csv"))
    return active, inputs


def _aggregate_close_share(panel: pd.DataFrame, group_col: str, series_name: str | None = None) -> pd.DataFrame:
    keys = ["date", "year"] if series_name is not None else ["date", "year", group_col]
    grouped = (
        panel.groupby(keys, as_index=False)
        .agg(
            n_symbols=("symbol", "nunique"),
            symbol_days=("symbol", "size"),
            total_daily_dollar_volume=("total_daily_dollar_volume", "sum"),
            close_auction_dollar_volume=("close_auction_dollar_volume", "sum"),
            official_close_marker_dollar_volume=("official_close_marker_dollar_volume", "sum"),
            official_close_marker_rows=("official_close_marker_rows", "sum"),
        )
    )
    if series_name is None:
        grouped = grouped.rename(columns={group_col: "series"})
    else:
        grouped["series"] = series_name
    grouped["close_share_pct"] = (
        100.0
        * grouped["close_auction_dollar_volume"]
        / grouped["total_daily_dollar_volume"].replace({0.0: np.nan})
    )
    grouped["official_close_marker_share_pct"] = (
        100.0
        * grouped["official_close_marker_dollar_volume"]
        / grouped["total_daily_dollar_volume"].replace({0.0: np.nan})
    )
    return grouped


def _close_share_tables(ctx: FigureBuildContext) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    symbol_days, inputs = _active_close_volume_symbol_days(ctx)

    classified = symbol_days[symbol_days["tier"].notna()].copy()
    by_tier = _aggregate_close_share(classified, "tier_group") if not classified.empty else pd.DataFrame()
    unclassified = symbol_days[symbol_days["tier"].isna()].copy()
    by_unclassified = (
        _aggregate_close_share(unclassified, "tier_group") if not unclassified.empty else pd.DataFrame()
    )
    all_rows = _aggregate_close_share(symbol_days, "tier_group", "All")
    daily_by_tier = pd.concat([by_tier, by_unclassified, all_rows], ignore_index=True)
    order = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Unclassified": 4, "All": 5}
    daily_by_tier["series_order"] = daily_by_tier["series"].map(order).fillna(99).astype(int)
    daily_by_tier = daily_by_tier.sort_values(["year", "date", "series_order"]).drop(columns="series_order")

    plot = daily_by_tier[daily_by_tier["series"].isin(["Tier 1", "Tier 2", "Tier 3", "All"])].copy()

    h1_unclassified = unclassified[unclassified["date"] <= _H1_2018_END]
    unclassified_h1 = (
        h1_unclassified.groupby("symbol", as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            symbol_days=("date", "nunique"),
            total_daily_dollar_volume=("total_daily_dollar_volume", "sum"),
            close_auction_dollar_volume=("close_auction_dollar_volume", "sum"),
        )
        .sort_values("symbol")
    )
    if not unclassified_h1.empty:
        unclassified_h1["close_share_pct"] = (
            100.0
            * unclassified_h1["close_auction_dollar_volume"]
            / unclassified_h1["total_daily_dollar_volume"].replace({0.0: np.nan})
        )

    db_dates = set(symbol_days["date"].dropna().unique())
    excluded = pd.DataFrame([
        {
            "date": date,
            "reason": _EXCLUDED_DATE_REASONS.get(date, "documented evaluation exclusion"),
            "present_in_active_volume_panel": date in db_dates,
        }
        for date in sorted(d for d in cfg.EXCLUDED_EVAL_DATES if _CLOSE_SHARE_START <= d <= _CLOSE_SHARE_END)
    ])

    checks = _close_share_checks(symbol_days, daily_by_tier, plot, unclassified_h1, excluded)
    source_cols = [
        "date", "year", "symbol", "ticker", "tier", "tier_group", "tier_source",
        "total_daily_dollar_volume", "close_auction_dollar_volume",
        "official_close_marker_dollar_volume", "official_close_marker_rows",
    ]
    return {
        "plot": plot,
        "daily_by_tier": daily_by_tier,
        "symbol_day_source": symbol_days[source_cols].sort_values(["date", "symbol"]),
        "unclassified_h1": unclassified_h1,
        "excluded_dates": excluded,
        "checks": checks,
    }, inputs


def _close_share_checks(
    symbol_days: pd.DataFrame,
    daily_by_tier: pd.DataFrame,
    plot: pd.DataFrame,
    unclassified_h1: pd.DataFrame,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"check": "source_symbol_days", "value": float(len(symbol_days))},
        {"check": "source_symbols", "value": float(symbol_days["symbol"].nunique())},
        {"check": "source_dates", "value": float(symbol_days["date"].nunique())},
        {"check": "plot_rows", "value": float(len(plot))},
        {"check": "daily_by_tier_rows", "value": float(len(daily_by_tier))},
        {"check": "max_plotted_close_share_pct", "value": float(plot["close_share_pct"].max())},
        {"check": "min_plotted_close_share_pct", "value": float(plot["close_share_pct"].min())},
        {"check": "max_daily_by_tier_close_share_pct", "value": float(daily_by_tier["close_share_pct"].max())},
        {"check": "unclassified_h1_symbols", "value": float(unclassified_h1["symbol"].nunique())},
        {"check": "unclassified_h1_symbol_days", "value": float(unclassified_h1["symbol_days"].sum())},
        {"check": "excluded_dates_documented", "value": float(len(excluded))},
    ]
    for year, group in symbol_days.groupby("year"):
        rows.append({
            "check": f"{int(year)}_pooled_close_share_pct",
            "value": float(
                100.0 * group["close_auction_dollar_volume"].sum()
                / group["total_daily_dollar_volume"].sum()
            ),
        })
    return pd.DataFrame(rows)


def _write_close_share_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_map = {
        "DailyByTier": tables["daily_by_tier"],
        "SymbolDaySource": tables["symbol_day_source"],
        "UnclassifiedH1": tables["unclassified_h1"],
        "ExcludedDates": tables["excluded_dates"],
        "Checks": tables["checks"],
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in sheet_map.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.book[sheet]
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                header = str(column_cells[0].value or "")
                width = min(max(len(header) + 2, 12), 34)
                ws.column_dimensions[column_cells[0].column_letter].width = width


def close_volume_share(ctx: FigureBuildContext) -> FigureData:
    tables, inputs = _close_share_tables(ctx)
    if ctx.close_share_xlsx_path is not None:
        _write_close_share_workbook(ctx.close_share_xlsx_path, tables)
        inputs.update(_hash(ctx.close_share_xlsx_path, "volume:closing_auction_share_daily_values.xlsx"))
    columns = [
        "date", "year", "series", "n_symbols", "symbol_days",
        "total_daily_dollar_volume", "close_auction_dollar_volume",
        "close_share_pct", "official_close_marker_dollar_volume",
        "official_close_marker_rows", "official_close_marker_share_pct",
    ]
    return FigureData(tables["plot"][columns], inputs)


def close_volume_share_trend_appendix(ctx: FigureBuildContext) -> FigureData:
    close_share_csv = ctx.out_dir / "figure_inputs" / "fig_close_volume_share.csv"
    if close_share_csv.exists():
        source = _read_csv(close_share_csv, "table:fig_close_volume_share.csv")
    else:
        source = close_volume_share(ctx)

    frame = source.frame.copy()
    required = {"date", "series", "close_share_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"close-share trend input missing columns {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "series", "close_share_pct"]).copy()
    frame["date_key"] = frame["date"].dt.strftime("%Y-%m-%d")
    tau_map = {date: i for i, date in enumerate(sorted(frame["date_key"].unique()))}
    frame["tau"] = frame["date_key"].map(tau_map).astype(int)
    frame["close_share_pct"] = pd.to_numeric(frame["close_share_pct"], errors="coerce")
    frame = frame.dropna(subset=["close_share_pct"]).copy()

    rows: list[pd.DataFrame] = []
    preferred = ["All", "Tier 1", "Tier 2", "Tier 3"]
    series_order = [s for s in preferred if s in set(frame["series"].astype(str))]
    series_order += [s for s in sorted(set(frame["series"].astype(str))) if s not in series_order]
    for series in series_order:
        sub = frame[frame["series"].astype(str) == series].sort_values("tau").copy()
        if len(sub) < 2:
            continue
        x = sub["tau"].to_numpy(dtype=float)
        y = sub["close_share_pct"].to_numpy(dtype=float)
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
        sub["fitted_close_share_pct"] = intercept + slope * x
        sub["intercept_pct"] = intercept
        sub["slope_pp_per_day"] = slope
        sub["trend_pp_per_year"] = slope * 252.0
        rows.append(sub)

    if not rows:
        raise ValueError("close-share trend input has no series with at least two observations")
    out = pd.concat(rows, ignore_index=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    columns = [
        "date", "series", "tau", "close_share_pct", "fitted_close_share_pct",
        "intercept_pct", "slope_pp_per_day", "trend_pp_per_year",
    ]
    return FigureData(out[columns], source.inputs)


TRANSFORMS: dict[str, Callable[[FigureBuildContext], FigureData]] = {
    "alpha_decomposition": alpha_decomposition,
    "alpha_fill_frontier": alpha_fill_frontier,
    "alt_vs_queue_diagnostics": alt_vs_queue_diagnostics,
    "as_horizon_robustness": as_horizon_robustness,
    "close_volume_share": close_volume_share,
    "close_volume_share_trend_appendix": close_volume_share_trend_appendix,
    "data_coverage": data_coverage,
    "fill_model_oos_calibration": fill_model_oos_calibration,
    "fill_spec_frontier": fill_spec_frontier,
    "h2_heatmap": h2_heatmap,
    "parent_size_robustness": parent_size_robustness,
    "raear_curve": raear_curve,
    "rolling_stability": rolling_stability,
}


def build_figure_data(transform: str, ctx: FigureBuildContext) -> FigureData:
    try:
        builder = TRANSFORMS[transform]
    except KeyError as exc:
        raise ValueError(f"Unknown figure transform: {transform}") from exc
    return builder(ctx)
