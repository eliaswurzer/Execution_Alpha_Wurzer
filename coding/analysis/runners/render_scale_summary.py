"""Render a read-only scale-of-analysis summary for a thesis run.

The runner intentionally does not scan raw TAQ data.  It aggregates existing
manifests, run shards, the calibrated fill-model manifest, and the volume
DuckDB so the summary can be refreshed while a long run is still in progress.

Examples
--------
python -m analysis.runners.render_scale_summary \
  --run-root "<artifact-root>/runs/final_20260611_queue" \
  --allow-incomplete
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config as cfg


class ScaleSummaryError(RuntimeError):
    """Raised when the summary cannot be rendered safely."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if not isinstance(value, (list, tuple, dict, str, bytes)):
        try:
            missing = pd.isna(value)
        except TypeError:
            missing = False
        if isinstance(missing, bool) and missing:
            return None
    return value


def _sql_literal(path: Path | str) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _metric(
    rows: list[dict[str, Any]],
    section: str,
    metric: str,
    value: Any,
    *,
    unit: str = "",
    source: str = "",
    note: str = "",
) -> None:
    rows.append({
        "section": section,
        "metric": metric,
        "value": _jsonable(value),
        "unit": unit,
        "source": source,
        "note": note,
    })


def _default_volume_db() -> Path:
    preferred = cfg.RUN_ROOT / "volume" / "dollar_volume_sp500_2018_2019.duckdb"
    return preferred if preferred.exists() else cfg.VOLUME_DB_PATH


def _latest_audit_root() -> Path | None:
    audit_parent = cfg.ARTIFACTS_DIR / "data_audit"
    if not audit_parent.exists():
        return None
    candidates = [p for p in audit_parent.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_fill_artifacts(run_root: Path, fill_artifacts: Path | None) -> Path | None:
    if fill_artifacts is not None:
        return fill_artifacts
    run_config = run_root / "metadata" / "run_config.json"
    if run_config.exists():
        artifacts = _read_json(run_config).get("artifacts")
        if artifacts:
            return Path(artifacts)
    preferred = cfg.ARTIFACTS_DIR / "fill_model_v2"
    if preferred.exists():
        return preferred
    fallback = cfg.ARTIFACTS_DIR / "fill_model"
    return fallback if fallback.exists() else None


def _collect_preprocessing(rows: list[dict[str, Any]], processed_root: Path | None) -> None:
    if processed_root is None or not processed_root.exists():
        _metric(rows, "Preprocessing", "trade_qc_manifests", 0, source=str(processed_root or ""))
        return

    manifests = sorted(processed_root.glob("*/qc/trade_qc_summary.json"))
    dates: list[str] = []
    totals = {
        "total_read_rows": 0,
        "total_symbol_filter_rows": 0,
        "total_kept_rows": 0,
        "policy_input_rows": 0,
        "kept_opening_auction_condition": 0,
        "kept_closing_auction_condition": 0,
        "dropped_bad_correction": 0,
        "dropped_bad_price_volume": 0,
        "dropped_preprocess_bad_condition": 0,
        "dropped_eval_bad_condition_if_applied": 0,
        "symbols_written_sum": 0,
    }
    policies: set[str] = set()

    for path in manifests:
        payload = _read_json(path)
        date = str(payload.get("date") or path.parents[1].name)
        dates.append(date)
        qc_counts = payload.get("qc_counts") or {}
        totals["total_read_rows"] += int(payload.get("total_read_rows") or 0)
        totals["total_symbol_filter_rows"] += int(payload.get("total_symbol_filter_rows") or 0)
        totals["total_kept_rows"] += int(payload.get("total_kept_rows") or 0)
        totals["symbols_written_sum"] += int(payload.get("symbols_written") or 0)
        for key in (
            "policy_input_rows",
            "kept_opening_auction_condition",
            "kept_closing_auction_condition",
            "dropped_bad_correction",
            "dropped_bad_price_volume",
            "dropped_preprocess_bad_condition",
            "dropped_eval_bad_condition_if_applied",
        ):
            totals[key] += int(qc_counts.get(key) or 0)
        if payload.get("trade_condition_policy_version"):
            policies.add(str(payload["trade_condition_policy_version"]))

    _metric(rows, "Preprocessing", "trade_qc_manifests", len(manifests), unit="days", source=str(processed_root))
    if dates:
        _metric(rows, "Preprocessing", "first_trade_qc_date", min(dates), source=str(processed_root))
        _metric(rows, "Preprocessing", "last_trade_qc_date", max(dates), source=str(processed_root))
    for key, value in totals.items():
        note = "sum over daily QC manifests" if key == "symbols_written_sum" else ""
        _metric(rows, "Preprocessing", key, value, unit="rows" if key.endswith("rows") else "", source=str(processed_root), note=note)
    if policies:
        _metric(rows, "Preprocessing", "trade_condition_policy_versions", ", ".join(sorted(policies)), source=str(processed_root))


def _collect_audit(rows: list[dict[str, Any]], audit_root: Path | None) -> None:
    if audit_root is None:
        _metric(rows, "Data audit", "audit_summary_found", False)
        return
    summary = audit_root / "date_side_audit_summary.json"
    if not summary.exists():
        _metric(rows, "Data audit", "audit_summary_found", False, source=str(audit_root))
        return

    payload = _read_json(summary)
    keys = (
        "dates",
        "complete_dates",
        "partial_dates",
        "active_membership_symbol_days",
        "active_membership_complete_symbol_days",
        "active_membership_coverage",
        "active_membership_missing_trade_symbol_days",
        "active_membership_missing_nbbo_symbol_days",
        "union_symbol_days",
        "union_complete_symbol_days",
        "union_coverage",
    )
    for key in keys:
        if key in payload:
            unit = "symbol-days" if "symbol_days" in key else ("days" if key.endswith("dates") or key == "dates" else "")
            _metric(rows, "Data audit", key, payload[key], unit=unit, source=str(summary))
    excluded = payload.get("excluded_dates") or []
    _metric(rows, "Data audit", "excluded_dates", ", ".join(excluded), source=str(summary))


def _collect_volume(rows: list[dict[str, Any]], volume_db: Path | None) -> None:
    if volume_db is None or not volume_db.exists():
        _metric(rows, "Volume DB", "volume_db_found", False, source=str(volume_db or ""))
        return

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - project dependency in normal runs
        _metric(rows, "Volume DB", "volume_db_error", f"duckdb import failed: {exc}", source=str(volume_db))
        return

    try:
        con = duckdb.connect(str(volume_db), read_only=True)
        summary = con.execute(
            """
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT Date) AS dates,
              CAST(MIN(Date) AS VARCHAR) AS min_date,
              CAST(MAX(Date) AS VARCHAR) AS max_date,
              SUM(Total_Daily_Val) AS total_daily_dollar_volume,
              SUM(Close_Auction_Val) AS close_auction_dollar_volume,
              SUM(Official_Close_Marker_Val) AS official_close_marker_dollar_volume,
              SUM(Official_Close_Marker_Rows) AS official_close_marker_rows
            FROM daily_volume
            """
        ).fetchone()
        names = [
            "rows",
            "dates",
            "min_date",
            "max_date",
            "total_daily_dollar_volume",
            "close_auction_dollar_volume",
            "official_close_marker_dollar_volume",
            "official_close_marker_rows",
        ]
        data = dict(zip(names, summary))
        for key, value in data.items():
            unit = "USD" if key.endswith("dollar_volume") else ("rows" if key.endswith("rows") or key == "rows" else "days" if key == "dates" else "")
            _metric(rows, "Volume DB", key, value, unit=unit, source=str(volume_db))
        total = float(data.get("total_daily_dollar_volume") or 0.0)
        close = float(data.get("close_auction_dollar_volume") or 0.0)
        if total:
            _metric(rows, "Volume DB", "close_auction_share_of_total", close / total, unit="share", source=str(volume_db))

        skipped = con.execute(
            """
            SELECT Reason, COUNT(*) AS rows
            FROM daily_volume_skipped
            GROUP BY Reason
            ORDER BY rows DESC, Reason
            """
        ).fetchall()
        for reason, count in skipped:
            _metric(rows, "Volume DB", f"skipped_{reason}", count, unit="rows", source=str(volume_db))
        con.close()
    except Exception as exc:  # pragma: no cover - defensive around locked/corrupt DBs
        _metric(rows, "Volume DB", "volume_db_error", str(exc), source=str(volume_db))


def _collect_fill_artifacts(rows: list[dict[str, Any]], fill_artifacts: Path | None) -> None:
    if fill_artifacts is None:
        _metric(rows, "Calibration", "fill_artifacts_found", False)
        return
    manifest = fill_artifacts / "calibration_manifest.json"
    if not manifest.exists():
        _metric(rows, "Calibration", "calibration_manifest_found", False, source=str(fill_artifacts))
        return
    payload = _read_json(manifest)
    for key in (
        "status",
        "feature_policy",
        "coverage",
        "n_critical_failures",
        "n_event_rows",
        "n_daily_feature_rows",
        "n_as_rows",
        "xgb_survival_status",
        "km_status",
    ):
        if key in payload:
            unit = "rows" if key.startswith("n_") else ""
            _metric(rows, "Calibration", key, payload[key], unit=unit, source=str(manifest))

    validation = fill_artifacts / "validation.csv"
    if validation.exists():
        try:
            n_rows = len(pd.read_csv(validation))
            _metric(rows, "Calibration", "validation_rows", n_rows, unit="rows", source=str(validation))
        except Exception as exc:  # pragma: no cover - malformed optional file
            _metric(rows, "Calibration", "validation_rows_error", str(exc), source=str(validation))


def _collect_expected_vc(rows: list[dict[str, Any]], run_root: Path) -> None:
    cache = run_root / "cache"
    manifest = cache / "expected_vc_manifest.json"
    if manifest.exists():
        payload = _read_json(manifest)
        for key in ("history_dates", "history_rows", "expected_vc_rows", "fingerprint"):
            if key in payload:
                unit = "rows" if key.endswith("rows") else ("days" if key.endswith("dates") else "")
                _metric(rows, "Expected VC cache", key, payload[key], unit=unit, source=str(manifest))

    vc_history = cache / "vc_history.parquet"
    if not vc_history.exists():
        return
    try:
        import duckdb

        con = duckdb.connect()
        path = _sql_literal(vc_history)
        data = con.execute(
            f"""
            SELECT
              SUM(vc_shares) AS vc_shares,
              SUM(close_trade_volume) AS close_trade_volume,
              SUM(close_trade_rows) AS close_trade_rows,
              SUM(official_close_marker_volume) AS official_close_marker_volume,
              SUM(official_close_marker_rows) AS official_close_marker_rows
            FROM read_parquet({path})
            """
        ).fetchone()
        con.close()
        keys = [
            "vc_shares",
            "close_trade_volume",
            "close_trade_rows",
            "official_close_marker_volume",
            "official_close_marker_rows",
        ]
        for key, value in zip(keys, data):
            unit = "shares" if key.endswith("volume") or key.endswith("shares") else "rows"
            _metric(rows, "Expected VC cache", key, value, unit=unit, source=str(vc_history))
    except Exception as exc:  # pragma: no cover - defensive around concurrent files
        _metric(rows, "Expected VC cache", "vc_history_error", str(exc), source=str(vc_history))


def _collect_simulation_manifest(rows: list[dict[str, Any]], run_root: Path) -> None:
    manifest = run_root / "metadata" / "simulation_manifest.csv"
    failures = run_root / "metadata" / "simulation_failures.csv"
    if not manifest.exists():
        _metric(rows, "Simulation", "simulation_manifest_found", False, source=str(manifest))
        return

    frame = pd.read_csv(manifest)
    _metric(rows, "Simulation", "manifest_dates", int(frame["date"].nunique()), unit="days", source=str(manifest))
    if not frame.empty:
        _metric(rows, "Simulation", "first_manifest_date", str(frame["date"].min()), source=str(manifest))
        _metric(rows, "Simulation", "last_manifest_date", str(frame["date"].max()), source=str(manifest))
    for key in ("eligible_symbol_days", "successful_symbol_days", "failed_symbol_days", "rows", "runtime_seconds"):
        if key in frame.columns:
            unit = "seconds" if key == "runtime_seconds" else ("rows" if key == "rows" else "symbol-days")
            _metric(rows, "Simulation", f"manifest_{key}_sum", frame[key].sum(), unit=unit, source=str(manifest))
    if "status" in frame.columns:
        for status, count in frame["status"].value_counts(dropna=False).items():
            _metric(rows, "Simulation", f"manifest_status_{status}", int(count), unit="days", source=str(manifest))

    if failures.exists():
        fail = pd.read_csv(failures)
        _metric(rows, "Simulation", "failure_rows", len(fail), unit="rows", source=str(failures))
        if "reason" in fail.columns:
            for reason, count in fail["reason"].value_counts(dropna=False).items():
                _metric(rows, "Simulation", f"failure_reason_{reason}", int(count), unit="rows", source=str(failures))


def _collect_panel_shards(rows: list[dict[str, Any]], run_root: Path) -> None:
    shard_files = sorted((run_root / "panel_shards").glob("date=*/panel.parquet"))
    _metric(rows, "Simulation panel", "panel_shard_files", len(shard_files), unit="files", source=str(run_root / "panel_shards"))
    if not shard_files:
        return

    try:
        import duckdb

        con = duckdb.connect()
        glob = _sql_literal(run_root / "panel_shards" / "date=*" / "panel.parquet")
        panel = f"read_parquet({glob}, hive_partitioning=false)"
        summary = con.execute(
            f"""
            WITH p AS (SELECT * FROM {panel}),
            parent_orders AS (
              SELECT DISTINCT symbol, date, order_id, qty_intended
              FROM p
            ),
            symbol_days AS (
              SELECT DISTINCT symbol, date, close_trade_volume, close_trade_rows,
                     official_close_marker_volume, official_close_marker_rows,
                     official_close_marker_fallback_volume
              FROM p
            )
            SELECT
              (SELECT COUNT(*) FROM p) AS panel_rows,
              (SELECT COUNT(DISTINCT symbol) FROM p) AS symbols,
              (SELECT COUNT(DISTINCT date) FROM p) AS dates,
              (SELECT COUNT(DISTINCT strategy) FROM p) AS strategies,
              (SELECT COUNT(*) FROM parent_orders) AS parent_orders,
              (SELECT COUNT(*) FROM symbol_days) AS symbol_days,
              (SELECT SUM(qty_intended) FROM p) AS strategy_order_intended_shares,
              (SELECT SUM(qty_filled_passive) FROM p) AS passive_filled_shares,
              (SELECT SUM(qty_filled_moc) FROM p) AS moc_filled_shares,
              (SELECT SUM(qty_intended) FROM parent_orders) AS parent_order_intended_shares,
              (SELECT SUM(close_trade_volume) FROM symbol_days) AS distinct_symbol_day_close_trade_volume,
              (SELECT SUM(close_trade_rows) FROM symbol_days) AS distinct_symbol_day_close_trade_rows,
              (SELECT SUM(official_close_marker_volume) FROM symbol_days) AS distinct_symbol_day_official_marker_volume,
              (SELECT SUM(official_close_marker_rows) FROM symbol_days) AS distinct_symbol_day_official_marker_rows,
              (SELECT SUM(official_close_marker_fallback_volume) FROM symbol_days) AS distinct_symbol_day_official_marker_fallback_volume
            """
        ).fetchdf().iloc[0].to_dict()
        for key, value in summary.items():
            if key.endswith("shares") or key.endswith("volume"):
                unit = "shares"
            elif key.endswith("rows") or key == "panel_rows":
                unit = "rows"
            elif key.endswith("orders"):
                unit = "orders"
            elif key.endswith("days") or key == "dates":
                unit = "days" if key == "dates" else "symbol-days"
            else:
                unit = ""
            note = ""
            if key == "strategy_order_intended_shares":
                note = "counts one intended quantity per strategy-row"
            if key == "parent_order_intended_shares":
                note = "deduplicated by symbol/date/order_id"
            _metric(rows, "Simulation panel", key, value, unit=unit, source=str(run_root / "panel_shards"), note=note)

        source_counts = con.execute(
            f"""
            WITH p AS (SELECT * FROM {panel}),
            source_days AS (
              SELECT DISTINCT symbol, date, close_price_source, close_volume_source
              FROM p
            )
            SELECT close_price_source, close_volume_source, COUNT(*) AS symbol_days
            FROM source_days
            GROUP BY close_price_source, close_volume_source
            ORDER BY symbol_days DESC, close_price_source, close_volume_source
            """
        ).fetchall()
        for price_source, volume_source, count in source_counts:
            _metric(
                rows,
                "Simulation panel",
                f"close_source_symbol_days:{price_source}/{volume_source}",
                count,
                unit="symbol-days",
                source=str(run_root / "panel_shards"),
            )
        con.close()
    except Exception as exc:  # pragma: no cover - defensive around concurrent shards
        _metric(rows, "Simulation panel", "panel_shard_error", str(exc), source=str(run_root / "panel_shards"))


def _collect_run_status(rows: list[dict[str, Any]], run_root: Path) -> str:
    status_path = run_root / "run_status.json"
    status = "missing"
    if status_path.exists():
        payload = _read_json(status_path)
        status = str(payload.get("status", "unknown"))
        _metric(rows, "Run", "status", status, source=str(status_path))
        if payload.get("current_step"):
            _metric(rows, "Run", "current_step", payload["current_step"], source=str(status_path))
        if payload.get("updated_at"):
            _metric(rows, "Run", "updated_at", payload["updated_at"], source=str(status_path))
    else:
        _metric(rows, "Run", "status", "missing", source=str(status_path))

    run_config = run_root / "metadata" / "run_config.json"
    if run_config.exists():
        payload = _read_json(run_config)
        for key in ("run_id", "start", "end", "fill_specification", "universe", "workers", "tier_policy", "artifacts"):
            if key in payload:
                _metric(rows, "Run", key, payload[key], source=str(run_config))
    sim_config = run_root / "metadata" / "simulation_config.json"
    if sim_config.exists():
        payload = _read_json(sim_config)
        for key in ("fingerprint", "schema_version", "simulation_source_sha256", "feature_policy", "trade_condition_policy", "pool_backend"):
            if key in payload:
                _metric(rows, "Run", key, payload[key], source=str(sim_config))
    return status


def _write_tables(rows: list[dict[str, Any]], run_root: Path) -> dict[str, str]:
    df = pd.DataFrame(rows)
    tables_dir = run_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "scale_summary.csv"
    md_path = tables_dir / "scale_summary.md"
    tex_path = tables_dir / "scale_summary.tex"
    df.to_csv(csv_path, index=False)
    try:
        df.to_markdown(md_path, index=False)
    except ImportError:  # pragma: no cover - tabulate is installed in normal runs
        md_path.write_text(df.to_csv(index=False), encoding="utf-8")
    df.to_latex(tex_path, index=False, escape=True)
    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }


def render(
    run_root: Path,
    *,
    processed_root: Path | None = None,
    audit_root: Path | None = None,
    volume_db: Path | None = None,
    fill_artifacts: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    run_root = Path(run_root)
    if not run_root.exists():
        raise ScaleSummaryError(f"Run root not found: {run_root}")

    processed_root = processed_root or cfg.CONSOLIDATED_TAQ_ROOT
    audit_root = audit_root if audit_root is not None else _latest_audit_root()
    volume_db = volume_db or _default_volume_db()
    fill_artifacts = _resolve_fill_artifacts(run_root, fill_artifacts)

    rows: list[dict[str, Any]] = []
    status = _collect_run_status(rows, run_root)
    if status != "complete" and not allow_incomplete:
        raise ScaleSummaryError(
            f"Run status is {status!r}; pass --allow-incomplete for a draft scale summary."
        )

    _collect_preprocessing(rows, processed_root)
    _collect_audit(rows, audit_root)
    _collect_volume(rows, volume_db)
    _collect_fill_artifacts(rows, fill_artifacts)
    _collect_expected_vc(rows, run_root)
    _collect_simulation_manifest(rows, run_root)
    _collect_panel_shards(rows, run_root)

    outputs = _write_tables(rows, run_root)
    manifest_path = run_root / "metadata" / "scale_summary.json"
    outputs["json"] = str(manifest_path)
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "draft": status != "complete",
        "run_status": status,
        "processed_root": str(processed_root) if processed_root else None,
        "audit_root": str(audit_root) if audit_root else None,
        "volume_db": str(volume_db) if volume_db else None,
        "fill_artifacts": str(fill_artifacts) if fill_artifacts else None,
        "outputs": outputs,
        "metrics": rows,
    }
    meta = run_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--volume-db", type=Path, default=None)
    parser.add_argument("--fill-artifacts", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        manifest = render(
            args.run_root,
            processed_root=args.processed_root,
            audit_root=args.audit_root,
            volume_db=args.volume_db,
            fill_artifacts=args.fill_artifacts,
            allow_incomplete=args.allow_incomplete,
        )
    except ScaleSummaryError as exc:
        raise SystemExit(str(exc))
    print(json.dumps({
        "run_root": manifest["run_root"],
        "draft": manifest["draft"],
        "outputs": manifest["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
