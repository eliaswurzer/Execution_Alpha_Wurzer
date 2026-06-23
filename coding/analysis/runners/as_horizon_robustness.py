"""Targeted adverse-selection horizon robustness runner.

This runner replays the queue-aware simulation for one configured
``AS_HORIZON_SECONDS`` value, materializes only the H1 panel, and then
aggregates the completed horizon runs with the existing 30-second headline
anchor. It deliberately does not rerun H2/H3 because the horizon affects the
reported markout diagnostic, not the strategy's fill rule or net-alpha formula.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from . import h1_performance_gap
from .master_panel import (
    TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    TIER_POLICY_CHOICES,
    materialize_panel,
    run_master_panel,
    _sha256,
)
from .run_all_hypotheses import (
    HYPOTHESIS_REQUIRED,
    MASTER_STRATEGIES,
    make_run_layout,
    run_dir,
    _hypothesis_complete,
    _write_status,
)

log = logging.getLogger(__name__)

DEFAULT_NEW_HORIZONS = (5, 15, 60, 300)
SUMMARY_HORIZONS = (5, 15, 30, 60, 300)
DEFAULT_RUN_ID_PREFIX = "final_v4_20260619_as_horizon_queue"
DEFAULT_BASELINE_RUN = (
    cfg.ARTIFACTS_DIR / "runs" / "final_v4_20260618_queue"
)
DEFAULT_SUMMARY_DIR = (
    cfg.ARTIFACTS_DIR / "as_horizon_robustness_20260619"
)
PRIMARY_PANEL_COLUMNS = [
    "order_id",
    "strategy",
    "window",
    "size_frac",
    "net_alpha_vs_moc_bps",
    "fill_rate",
    "adverse_selection_bps",
    "adverse_selection_cost_bps",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _horizon_run_id(prefix: str, horizon_seconds: int) -> str:
    return f"{prefix}_h{int(horizon_seconds):03d}"


def _write_run_config(paths: dict[str, Path], args: argparse.Namespace) -> None:
    payload = {
        "run_id": args.run_id,
        "analysis_scope": "h1_as_horizon_robustness",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "fill_specification": args.fill_spec,
        "workers": args.workers,
        "symbols": args.symbols,
        "universe": args.universe,
        "artifacts": str(args.artifacts),
        "headline_size_frac": cfg.PARENT_ORDER_PRIMARY_FRACTION,
        "as_horizon_seconds": int(cfg.AS_HORIZON_SECONDS),
        "resume": not args.no_resume,
        "tier_policy": args.tier_policy,
        "preprocessing_symbols_file": (
            str(args.preprocessing_symbols_file)
            if args.preprocessing_symbols_file else None
        ),
        "adv_spread_bucket_map": (
            str(args.adv_spread_bucket_map)
            if args.adv_spread_bucket_map else None
        ),
    }
    _write_json(paths["metadata"] / "run_config.json", payload)


def _run_h1_only(paths: dict[str, Path], simulation_fingerprint: str, *, resume: bool) -> None:
    h1_dir = paths["h1"]
    status_path = h1_dir / "status.json"
    if resume and _hypothesis_complete("h1", h1_dir, simulation_fingerprint):
        log.info("H1 already complete for fingerprint %s; skipping.", simulation_fingerprint)
        return

    _write_status(status_path, "running")
    panel_path = h1_dir / "h1_panel.parquet"
    try:
        materialize_panel(paths["panel_shards"], MASTER_STRATEGIES, panel_path)
        panel = pd.read_parquet(panel_path)
        h1_performance_gap.analyze_panel(panel, h1_dir, panel_path=panel_path)
        missing = [
            filename for filename in HYPOTHESIS_REQUIRED["h1"]
            if not (h1_dir / filename).exists()
            or (h1_dir / filename).stat().st_size <= 0
        ]
        if missing:
            raise RuntimeError(f"H1 output validation failed: {missing}")
        _write_status(
            status_path,
            "complete",
            panel_rows=len(panel),
            strategies=MASTER_STRATEGIES,
            simulation_fingerprint=simulation_fingerprint,
            as_horizon_seconds=int(cfg.AS_HORIZON_SECONDS),
        )
    except Exception as exc:
        _write_status(status_path, "failed", error=str(exc))
        raise


def run_one_horizon(args: argparse.Namespace) -> Path:
    base = run_dir(args.run_id, args.run_root)
    paths = make_run_layout(base)
    _write_run_config(paths, args)

    log.info(
        "AS-horizon run bundle: %s  horizon=%ss  workers=%s",
        base, cfg.AS_HORIZON_SECONDS, args.workers,
    )
    top_status = base / "run_status.json"
    _write_status(
        top_status,
        "running",
        current_step="simulation",
        as_horizon_seconds=int(cfg.AS_HORIZON_SECONDS),
    )
    try:
        simulation = run_master_panel(
            strategies=MASTER_STRATEGIES,
            start=args.start,
            end=args.end,
            artifacts_dir=args.artifacts,
            run_root=base,
            symbols=args.symbols,
            universe=args.universe,
            max_dates=args.max_dates,
            workers=args.workers,
            fill_specification=args.fill_spec,
            resume=not args.no_resume,
            min_eligible_coverage=args.min_eligible_coverage,
            min_index_coverage=args.min_index_coverage,
            tier_policy=args.tier_policy,
            preprocessing_symbols_file=args.preprocessing_symbols_file,
            adv_spread_bucket_map=args.adv_spread_bucket_map,
        )
        _write_status(
            top_status,
            "running",
            current_step="h1",
            simulation=simulation,
            as_horizon_seconds=int(cfg.AS_HORIZON_SECONDS),
        )
        _run_h1_only(paths, simulation["fingerprint"], resume=not args.no_resume)
        _write_status(
            top_status,
            "complete",
            simulation=simulation,
            as_horizon_seconds=int(cfg.AS_HORIZON_SECONDS),
            analysis_scope="h1_as_horizon_robustness",
        )
    except Exception as exc:
        _write_status(
            top_status,
            "failed",
            error=str(exc),
            as_horizon_seconds=int(cfg.AS_HORIZON_SECONDS),
        )
        raise

    readme = (
        f"# {args.run_id}\n\n"
        f"- Analysis scope: H1 adverse-selection horizon robustness\n"
        f"- AS horizon: `{int(cfg.AS_HORIZON_SECONDS)}` seconds\n"
        f"- Sample: {args.start} to {args.end}\n"
        f"- Fill specification: `{args.fill_spec}`\n"
        f"- Workers: `{args.workers}`\n"
        f"- H1 outputs: `hypotheses/h1`\n"
        f"- Run config: `metadata/run_config.json`\n"
    )
    (base / "README.md").write_text(readme, encoding="utf-8")
    return base


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _primary_s3(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[panel["strategy"] == "S3_FULL"].copy()
    if "window" in out.columns:
        out = out[out["window"] == cfg.PRIMARY_WINDOW]
    if "size_frac" in out.columns:
        size = pd.to_numeric(out["size_frac"], errors="coerce")
        out = out[(size - cfg.PARENT_ORDER_PRIMARY_FRACTION).abs() <= 1e-12]
    return out.copy()


def _as_markout(sub: pd.DataFrame) -> float:
    filled = sub[pd.to_numeric(sub["fill_rate"], errors="coerce") > 0]
    if filled.empty:
        return 0.0
    return float(-pd.to_numeric(filled["adverse_selection_bps"], errors="coerce").mean())


def _as_component(sub: pd.DataFrame) -> float:
    return float((
        pd.to_numeric(sub["fill_rate"], errors="coerce")
        * pd.to_numeric(sub["adverse_selection_bps"], errors="coerce")
    ).mean())


def _horizon_from_run(run_root: Path, default: int | None = None) -> int:
    sim_config = run_root / "metadata" / "simulation_config.json"
    run_config = run_root / "metadata" / "run_config.json"
    for path in (sim_config, run_config):
        if path.exists():
            payload = _load_json(path)
            if "as_horizon_seconds" in payload:
                return int(payload["as_horizon_seconds"])
    if default is None:
        raise RuntimeError(f"Run {run_root} has no as_horizon_seconds metadata")
    return int(default)


def _validate_run_root(run_root: Path, expected_horizon: int, *, baseline: bool) -> list[str]:
    problems: list[str] = []
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return [f"{run_root}: missing run_status.json"]
    status = _load_json(status_path)
    if status.get("status") != "complete":
        problems.append(f"{run_root}: status is {status.get('status')!r}")
    simulation = status.get("simulation") or {}
    if simulation.get("status") != "complete":
        problems.append(f"{run_root}: simulation is {simulation.get('status')!r}")
    if int(simulation.get("dates_with_valid_shards", 0) or 0) != 371:
        problems.append(f"{run_root}: expected 371 valid shards")
    if int(simulation.get("critical_failures", 0) or 0) != 0:
        problems.append(f"{run_root}: critical failures present")
    if float(simulation.get("eligible_coverage", 0.0) or 0.0) < 0.995:
        problems.append(f"{run_root}: eligible coverage below 0.995")
    if not baseline:
        actual = _horizon_from_run(run_root)
        if actual != expected_horizon:
            problems.append(
                f"{run_root}: horizon metadata {actual}, expected {expected_horizon}"
            )
    h1_status_path = run_root / "hypotheses" / "h1" / "status.json"
    if h1_status_path.exists():
        h1_status = _load_json(h1_status_path)
        if h1_status.get("status") != "complete":
            problems.append(f"{run_root}: H1 status is {h1_status.get('status')!r}")
        sim_fp = simulation.get("fingerprint")
        h1_fp = h1_status.get("simulation_fingerprint")
        if sim_fp and h1_fp and sim_fp != h1_fp:
            problems.append(f"{run_root}: H1 fingerprint does not match simulation")
        h1_horizon = h1_status.get("as_horizon_seconds")
        if not baseline and h1_horizon is not None and int(h1_horizon) != expected_horizon:
            problems.append(
                f"{run_root}: H1 horizon metadata {h1_horizon}, expected {expected_horizon}"
            )
    else:
        problems.append(f"{run_root}: missing H1 status.json")
    for rel in HYPOTHESIS_REQUIRED["h1"]:
        path = run_root / "hypotheses" / "h1" / rel
        if not path.exists() or path.stat().st_size <= 0:
            problems.append(f"{run_root}: missing/nonempty H1 output {rel}")
    return problems


def _failure_reason_counts(run_root: Path) -> dict[str, int]:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return {}
    simulation = (_load_json(status_path).get("simulation") or {})
    counts = simulation.get("failure_reason_counts") or {}
    return {str(k): int(v) for k, v in counts.items()}


def _worker_count(run_root: Path) -> int | None:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return None
    simulation = (_load_json(status_path).get("simulation") or {})
    workers = simulation.get("workers")
    return int(workers) if workers is not None else None


def _fingerprint_audit(run_root: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "run_status": None,
        "h1_status": None,
        "simulation_config": None,
        "simulation_summary": None,
    }
    status_path = run_root / "run_status.json"
    if status_path.exists():
        out["run_status"] = (
            (_load_json(status_path).get("simulation") or {}).get("fingerprint")
        )
    h1_status_path = run_root / "hypotheses" / "h1" / "status.json"
    if h1_status_path.exists():
        out["h1_status"] = _load_json(h1_status_path).get("simulation_fingerprint")
    sim_config_path = run_root / "metadata" / "simulation_config.json"
    if sim_config_path.exists():
        out["simulation_config"] = _load_json(sim_config_path).get("fingerprint")
    sim_summary_path = run_root / "metadata" / "simulation_summary.json"
    if sim_summary_path.exists():
        out["simulation_summary"] = _load_json(sim_summary_path).get("fingerprint")
    return out


def _primary_s3_from_run(run_root: Path) -> pd.DataFrame:
    panel_path = run_root / "hypotheses" / "h1" / "h1_panel.parquet"
    panel = pd.read_parquet(panel_path, columns=PRIMARY_PANEL_COLUMNS)
    return _primary_s3(panel)


def _validate_supplement_root(run_root: Path, expected_horizon: int) -> list[str]:
    problems: list[str] = []
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        return [f"{run_root}: missing supplement run_status.json"]
    status = _load_json(status_path)
    if status.get("status") != "complete":
        problems.append(f"{run_root}: supplement status is {status.get('status')!r}")
    simulation = status.get("simulation") or {}
    if simulation.get("status") != "complete":
        problems.append(
            f"{run_root}: supplement simulation is {simulation.get('status')!r}"
        )
    if int(simulation.get("dates_with_valid_shards", 0) or 0) < 1:
        problems.append(f"{run_root}: supplement has no valid shards")
    if int(simulation.get("critical_failures", 0) or 0) != 0:
        problems.append(f"{run_root}: supplement critical failures present")
    actual = _horizon_from_run(run_root, default=expected_horizon)
    if actual != expected_horizon:
        problems.append(
            f"{run_root}: supplement horizon metadata {actual}, expected {expected_horizon}"
        )
    h1_status_path = run_root / "hypotheses" / "h1" / "status.json"
    if h1_status_path.exists():
        h1_status = _load_json(h1_status_path)
        if h1_status.get("status") != "complete":
            problems.append(
                f"{run_root}: supplement H1 status is {h1_status.get('status')!r}"
            )
        sim_fp = simulation.get("fingerprint")
        h1_fp = h1_status.get("simulation_fingerprint")
        if sim_fp and h1_fp and sim_fp != h1_fp:
            problems.append(f"{run_root}: supplement H1 fingerprint mismatch")
    else:
        problems.append(f"{run_root}: missing supplement H1 status.json")
    for rel in HYPOTHESIS_REQUIRED["h1"]:
        path = run_root / "hypotheses" / "h1" / rel
        if not path.exists() or path.stat().st_size <= 0:
            problems.append(f"{run_root}: missing/nonempty supplement H1 output {rel}")
    return problems


def _supplement_roots(supplement_root: Path | None, horizon: int) -> list[Path]:
    if supplement_root is None:
        return []
    root = Path(supplement_root)
    if not root.exists():
        return []
    return sorted(
        path for path in root.glob(f"as_horizon_supplement_h{int(horizon):03d}_*")
        if path.is_dir()
    )


def _augment_primary_with_supplements(
    primary_s3: pd.DataFrame,
    supplement_root: Path | None,
    horizon: int,
) -> tuple[pd.DataFrame, list[dict], dict[str, str], list[str]]:
    """Append validated supplement rows that fill missing primary order IDs."""
    audit: list[dict] = []
    input_sha256: dict[str, str] = {}
    problems: list[str] = []
    roots = _supplement_roots(supplement_root, horizon)
    if not roots:
        return primary_s3, audit, input_sha256, problems

    existing_ids = set(primary_s3["order_id"].astype(str))
    frames = [primary_s3]
    for root in roots:
        root_problems = _validate_supplement_root(root, horizon)
        problems.extend(root_problems)
        try:
            supplement = _primary_s3_from_run(root)
        except Exception as exc:
            problems.append(f"{root}: could not read supplement primary rows: {exc}")
            supplement = pd.DataFrame(columns=primary_s3.columns)
        if not supplement.empty:
            supplement = supplement.copy()
            ids = supplement["order_id"].astype(str)
            to_add = supplement.loc[~ids.isin(existing_ids)].copy()
        else:
            to_add = supplement
        if not to_add.empty:
            frames.append(to_add)
            existing_ids.update(to_add["order_id"].astype(str))
        audit.append({
            "run_id": root.name,
            "run_root": str(root),
            "workers": _worker_count(root),
            "candidate_rows": int(len(supplement)),
            "added_rows": int(len(to_add)),
            "validation_problems": root_problems,
            "added_order_ids": sorted(to_add["order_id"].astype(str).tolist())
            if not to_add.empty else [],
        })
        for rel in (
            "run_status.json",
            "metadata/simulation_config.json",
            "metadata/simulation_summary.json",
            "hypotheses/h1/status.json",
            "hypotheses/h1/h1_panel.parquet",
            "hypotheses/h1/h1_primary_ttest.csv",
        ):
            path = root / rel
            if path.exists():
                input_sha256[f"supplement:{horizon}:{root.name}:{rel}"] = _sha256(path)

    augmented = pd.concat(frames, ignore_index=True)
    return augmented, audit, input_sha256, problems


def _row_for_run(run_root: Path, horizon: int, *, source: str, primary_s3: pd.DataFrame) -> dict:
    primary_path = run_root / "hypotheses" / "h1" / "h1_primary_ttest.csv"
    primary = pd.read_csv(primary_path).iloc[0].to_dict()
    return {
        "horizon_seconds": int(horizon),
        "source": source,
        "run_id": run_root.name,
        "run_root": str(run_root),
        "n": int(len(primary_s3)),
        "mean_net_alpha_vs_moc_bps": float(
            pd.to_numeric(primary_s3["net_alpha_vs_moc_bps"], errors="coerce").mean()
        ),
        "mean_fill_rate": float(
            pd.to_numeric(primary_s3["fill_rate"], errors="coerce").mean()
        ),
        "as_markout_bps": _as_markout(primary_s3),
        "as_component_bps": _as_component(primary_s3),
        "mean_as_cost_bps": float(
            pd.to_numeric(primary_s3["adverse_selection_cost_bps"], errors="coerce").mean()
        ),
        "primary_t": float(primary.get("t", np.nan)),
        "primary_p_value": float(primary.get("p_value", np.nan)),
        "primary_p_one_sided": float(primary.get("p_one_sided", np.nan)),
    }


def _common_sample_summary(primary_panels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    common_ids: set[str] | None = None
    for panel in primary_panels.values():
        ids = set(panel["order_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        return pd.DataFrame()
    rows = []
    for horizon, panel in primary_panels.items():
        sub = panel[panel["order_id"].astype(str).isin(common_ids)].copy()
        rows.append({
            "horizon_seconds": int(horizon),
            "n": int(len(sub)),
            "mean_net_alpha_vs_moc_bps": float(
                pd.to_numeric(sub["net_alpha_vs_moc_bps"], errors="coerce").mean()
            ),
            "mean_fill_rate": float(
                pd.to_numeric(sub["fill_rate"], errors="coerce").mean()
            ),
            "as_markout_bps": _as_markout(sub),
            "as_component_bps": _as_component(sub),
        })
    out = pd.DataFrame(rows).sort_values("horizon_seconds").reset_index(drop=True)
    baseline = out[out["horizon_seconds"] == 30].iloc[0]
    out["net_alpha_delta_from_30"] = (
        out["mean_net_alpha_vs_moc_bps"] - float(baseline["mean_net_alpha_vs_moc_bps"])
    )
    out["fill_rate_delta_from_30"] = (
        out["mean_fill_rate"] - float(baseline["mean_fill_rate"])
    )
    return out


def _write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    rows = [
        "# Adverse-Selection Horizon Robustness",
        "",
        "This table reports the S3-full, Window-B, one-percent parent-order cell. "
        "The 30-second row is the existing validated headline run.",
        "",
        "| Horizon | Source | Net alpha vs MOC | Fill rate | AS markout | AS component | n |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.horizon_seconds}s | {row.source} | "
            f"{row.mean_net_alpha_vs_moc_bps:.4f} | "
            f"{row.mean_fill_rate:.4f} | {row.as_markout_bps:.4f} | "
            f"{row.as_component_bps:.4f} | {int(row.n):,} |"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_latex(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Adverse-selection horizon robustness}",
        r"\label{tab:as-horizon-robustness}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Horizon & Net alpha & Fill rate & AS markout & AS component & $N$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{int(row.horizon_seconds)}s & "
            f"{row.mean_net_alpha_vs_moc_bps:.2f} & "
            f"{row.mean_fill_rate:.3f} & "
            f"{row.as_markout_bps:.2f} & "
            f"{row.as_component_bps:.2f} & "
            f"{int(row.n):,} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_note(
    summary: pd.DataFrame,
    manifest: dict,
    out_path: Path,
    *,
    common_summary: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# AS-Horizon Robustness Audit Note",
        "",
        f"Manifest status: `{manifest['status']}`.",
        "",
        "This artifact aggregates the targeted H1 adverse-selection horizon "
        "robustness grid. The 30-second row is sourced from the existing "
        "headline queue-aware run, while the other rows are new horizon runs.",
        "",
    ]
    if manifest.get("allow_noncritical_sample_drift"):
        lines.extend([
            "Non-critical sample drift mode was enabled for aggregation. "
            "The drift is retained in the table and manifest instead of being "
            "silently rounded away.",
            "",
            "The strict exact-sample gate remains available by omitting "
            "`--allow-noncritical-sample-drift`.",
            "",
        ])
    if manifest.get("validation_warnings"):
        lines.extend(["## Validation Warnings", ""])
        lines.extend(f"- {item}" for item in manifest["validation_warnings"])
        lines.append("")
    if manifest.get("validation_problems"):
        lines.extend(["## Validation Problems", ""])
        lines.extend(f"- {item}" for item in manifest["validation_problems"])
        lines.append("")
    if manifest.get("supplement_root"):
        lines.extend([
            "## Supplemented Rows",
            "",
            "The aggregate includes validated one-symbol-day supplement runs "
            "only where an original primary S3-full order ID was absent from "
            "the horizon run.",
            "",
            "| Horizon | Supplement runs | Added primary rows |",
            "| ---: | ---: | ---: |",
        ])
        for horizon in SUMMARY_HORIZONS:
            audit_rows = manifest.get("supplement_audit", {}).get(str(horizon), [])
            if not audit_rows:
                continue
            lines.append(
                f"| {int(horizon)}s | {len(audit_rows)} | "
                f"{sum(int(item.get('added_rows', 0)) for item in audit_rows):,} |"
            )
        lines.append("")
    lines.extend([
        "## Drift Audit",
        "",
        "| Horizon | n | Delta n | Delta net alpha | Delta fill rate |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in manifest.get("drift_audit", []):
        lines.append(
            f"| {int(row['horizon_seconds'])}s | {int(row['n']):,} | "
            f"{int(row['n_delta_from_30']):+d} | "
            f"{float(row['net_alpha_delta_from_30']):+.6f} | "
            f"{float(row['fill_rate_delta_from_30']):+.8f} |"
        )
    if common_summary is not None and not common_summary.empty:
        lines.extend([
            "",
            "## Common-Sample Diagnostic",
            "",
            "This diagnostic restricts every horizon to the intersection of "
            "primary S3-full order IDs available in all five rows.",
            "",
            "| Horizon | n | Delta net alpha | Delta fill rate | AS markout | AS component |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in common_summary.itertuples(index=False):
            lines.append(
                f"| {int(row.horizon_seconds)}s | {int(row.n):,} | "
                f"{float(row.net_alpha_delta_from_30):+.6f} | "
                f"{float(row.fill_rate_delta_from_30):+.8f} | "
                f"{float(row.as_markout_bps):.4f} | "
                f"{float(row.as_component_bps):.4f} |"
            )
    lines.extend([
        "",
        "## Report Table",
        "",
        "| Horizon | Source | Net alpha vs MOC | Fill rate | AS markout | AS component | n |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon_seconds}s | {row.source} | "
            f"{row.mean_net_alpha_vs_moc_bps:.4f} | "
            f"{row.mean_fill_rate:.4f} | {row.as_markout_bps:.4f} | "
            f"{row.as_component_bps:.4f} | {int(row.n):,} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_horizons(args: argparse.Namespace) -> pd.DataFrame:
    baseline = Path(args.baseline_run)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    problems: list[str] = []
    warnings: list[str] = []
    drift_audit: list[dict] = []
    run_roots: dict[str, str] = {}
    input_sha256: dict[str, str] = {}
    workers_by_horizon: dict[str, int | None] = {}
    failure_reason_counts: dict[str, dict[str, int]] = {}
    fingerprint_audit: dict[str, dict[str, str | None]] = {}
    supplement_audit: dict[str, list[dict]] = {}
    primary_panels: dict[int, pd.DataFrame] = {}
    supplement_root = getattr(args, "supplement_root", None)
    supplement_root = Path(supplement_root) if supplement_root else None
    allow_drift = bool(getattr(args, "allow_noncritical_sample_drift", False))
    max_n_drift = int(getattr(args, "max_row_count_drift", 0) or 0)
    max_net_drift = float(getattr(args, "max_net_alpha_drift_bps", 0.0) or 0.0)
    max_fill_drift = float(getattr(args, "max_fill_rate_drift", 0.0) or 0.0)

    for horizon in SUMMARY_HORIZONS:
        if horizon == 30:
            root = baseline
            source = "existing_30s_headline"
            is_baseline = True
        else:
            root = run_root / _horizon_run_id(args.run_id_prefix, horizon)
            source = "new_horizon_run"
            is_baseline = False
        run_roots[str(horizon)] = str(root)
        workers_by_horizon[str(horizon)] = _worker_count(root)
        failure_reason_counts[str(horizon)] = _failure_reason_counts(root)
        fp_audit = _fingerprint_audit(root)
        fingerprint_audit[str(horizon)] = fp_audit
        status_fp = fp_audit.get("run_status")
        h1_fp = fp_audit.get("h1_status")
        config_fp = fp_audit.get("simulation_config")
        if status_fp and h1_fp and status_fp == h1_fp and config_fp and config_fp != status_fp:
            msg = (
                f"{horizon}s metadata/simulation_config fingerprint differs from "
                "the matching run_status/H1 fingerprint"
            )
            if allow_drift:
                warnings.append(msg)
            else:
                problems.append(msg)
        problems.extend(_validate_run_root(root, horizon, baseline=is_baseline))
        primary_s3 = _primary_s3_from_run(root)
        if supplement_root is not None and horizon != 30:
            primary_s3, supp_audit, supp_hashes, supp_problems = (
                _augment_primary_with_supplements(primary_s3, supplement_root, horizon)
            )
            supplement_audit[str(horizon)] = supp_audit
            input_sha256.update(supp_hashes)
            problems.extend(supp_problems)
        else:
            supplement_audit[str(horizon)] = []
        primary_panels[int(horizon)] = primary_s3
        rows.append(_row_for_run(root, horizon, source=source, primary_s3=primary_s3))
        for rel in (
            "run_status.json",
            "metadata/simulation_config.json",
            "metadata/simulation_summary.json",
            "hypotheses/h1/status.json",
            "hypotheses/h1/h1_panel.parquet",
            "hypotheses/h1/h1_primary_ttest.csv",
        ):
            path = root / rel
            if path.exists():
                input_sha256[f"{horizon}:{rel}"] = _sha256(path)

    summary = pd.DataFrame(rows).sort_values("horizon_seconds").reset_index(drop=True)
    if summary["horizon_seconds"].tolist() != list(SUMMARY_HORIZONS):
        problems.append("summary does not contain exactly the expected five horizons")

    baseline_row = summary[summary["horizon_seconds"] == 30].iloc[0]
    expected_n = args.expected_n
    if expected_n is not None and int(baseline_row["n"]) != int(expected_n):
        problems.append(f"30s baseline n {int(baseline_row['n'])} != expected {expected_n}")
    for row in summary.itertuples(index=False):
        n_delta = int(row.n) - int(baseline_row.n)
        net_delta = float(row.mean_net_alpha_vs_moc_bps) - float(baseline_row.mean_net_alpha_vs_moc_bps)
        fill_delta = float(row.mean_fill_rate) - float(baseline_row.mean_fill_rate)
        drift_audit.append({
            "horizon_seconds": int(row.horizon_seconds),
            "n": int(row.n),
            "n_delta_from_30": int(n_delta),
            "net_alpha_delta_from_30": float(net_delta),
            "fill_rate_delta_from_30": float(fill_delta),
            "failure_reason_counts": failure_reason_counts.get(str(row.horizon_seconds), {}),
        })
        if int(row.horizon_seconds) == 30:
            continue
        if int(row.n) != int(baseline_row.n):
            msg = f"{row.horizon_seconds}s n {int(row.n)} != baseline {int(baseline_row.n)}"
            if allow_drift and abs(n_delta) <= max_n_drift:
                warnings.append(msg)
            else:
                problems.append(msg)
        if not np.isclose(
            row.mean_net_alpha_vs_moc_bps,
            baseline_row.mean_net_alpha_vs_moc_bps,
            atol=args.metric_tolerance,
        ):
            msg = f"{row.horizon_seconds}s net alpha differs from 30s beyond tolerance"
            if allow_drift and abs(net_delta) <= max_net_drift:
                warnings.append(msg)
            else:
                problems.append(msg)
        if not np.isclose(
            row.mean_fill_rate,
            baseline_row.mean_fill_rate,
            atol=args.metric_tolerance,
        ):
            msg = f"{row.horizon_seconds}s fill rate differs from 30s beyond tolerance"
            if allow_drift and abs(fill_delta) <= max_fill_drift:
                warnings.append(msg)
            else:
                problems.append(msg)

    summary["net_alpha_delta_from_30"] = (
        summary["mean_net_alpha_vs_moc_bps"] - float(baseline_row["mean_net_alpha_vs_moc_bps"])
    )
    summary["fill_rate_delta_from_30"] = (
        summary["mean_fill_rate"] - float(baseline_row["mean_fill_rate"])
    )
    summary.to_csv(out_dir / "as_horizon_summary.csv", index=False)
    _write_markdown(summary, out_dir / "as_horizon_summary.md")
    _write_latex(summary, out_dir / "tab_as_horizon_robustness.tex")
    common_summary = _common_sample_summary(primary_panels)
    common_sample_n = None
    if not common_summary.empty:
        common_summary.to_csv(out_dir / "as_horizon_common_sample_summary.csv", index=False)
        common_sample_n = int(common_summary["n"].iloc[0])

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": (
            "failed_validation" if problems
            else "complete_with_warnings" if warnings
            else "complete"
        ),
        "horizons_seconds": list(SUMMARY_HORIZONS),
        "new_horizons_seconds": list(DEFAULT_NEW_HORIZONS),
        "baseline_horizon_seconds": 30,
        "baseline_run": str(baseline),
        "run_id_prefix": args.run_id_prefix,
        "run_roots": run_roots,
        "workers_by_horizon": workers_by_horizon,
        "input_sha256": input_sha256,
        "failure_reason_counts": failure_reason_counts,
        "fingerprint_audit": fingerprint_audit,
        "supplement_root": str(supplement_root) if supplement_root else None,
        "supplement_audit": supplement_audit,
        "common_sample_n": common_sample_n,
        "expected_n": expected_n,
        "metric_tolerance": args.metric_tolerance,
        "allow_noncritical_sample_drift": allow_drift,
        "drift_tolerances": {
            "max_row_count_drift": max_n_drift,
            "max_net_alpha_drift_bps": max_net_drift,
            "max_fill_rate_drift": max_fill_drift,
        },
        "drift_audit": drift_audit,
        "validation_problems": problems,
        "validation_warnings": warnings,
        "outputs": [
            "as_horizon_summary.csv",
            "as_horizon_summary.md",
            "as_horizon_audit_note.md",
            "as_horizon_common_sample_summary.csv",
            "tab_as_horizon_robustness.tex",
            "manifest.json",
        ],
    }
    _write_audit_note(
        summary, manifest, out_dir / "as_horizon_audit_note.md",
        common_summary=common_summary,
    )
    _write_json(out_dir / "manifest.json", manifest)
    if problems and not args.allow_validation_problems:
        raise RuntimeError("AS-horizon aggregation failed validation: " + "; ".join(problems))
    return summary


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", type=Path, default=cfg.ARTIFACTS_DIR / "runs")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500")
    p.add_argument("--start", type=dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--end", type=dt.date.fromisoformat, default=cfg.EVAL_END)
    p.add_argument("--artifacts", type=Path, default=cfg.ARTIFACTS_DIR / "fill_model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fill-spec", default="tape_replay_queue",
                   choices=["tape_replay", "tape_replay_haircut",
                            "tape_replay_volume", "tape_replay_volume_haircut",
                            "tape_replay_strict", "tape_replay_queue",
                            "cox", "km", "infinite_depth",
                            "infinite_depth_haircut", "xgb"])
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--min-eligible-coverage", type=float, default=0.995)
    p.add_argument("--min-index-coverage", type=float, default=0.95)
    p.add_argument("--preprocessing-symbols-file", type=Path, default=None)
    p.add_argument("--adv-spread-bucket-map", type=Path, default=None)
    p.add_argument(
        "--tier-policy",
        choices=sorted(TIER_POLICY_CHOICES),
        default=TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one AS-horizon H1-only bundle")
    _add_common_run_args(run_p)

    agg_p = sub.add_parser("aggregate", help="Aggregate completed AS-horizon bundles")
    agg_p.add_argument("--run-root", type=Path, default=cfg.ARTIFACTS_DIR / "runs")
    agg_p.add_argument("--run-id-prefix", default=DEFAULT_RUN_ID_PREFIX)
    agg_p.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    agg_p.add_argument("--supplement-root", type=Path, default=None)
    agg_p.add_argument("--out-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    agg_p.add_argument("--expected-n", type=int, default=187_309)
    agg_p.add_argument("--metric-tolerance", type=float, default=1e-8)
    agg_p.add_argument("--allow-noncritical-sample-drift", action="store_true")
    agg_p.add_argument("--max-row-count-drift", type=int, default=10)
    agg_p.add_argument("--max-net-alpha-drift-bps", type=float, default=0.002)
    agg_p.add_argument("--max-fill-rate-drift", type=float, default=5e-5)
    agg_p.add_argument("--allow-validation-problems", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_one_horizon(args)
    elif args.command == "aggregate":
        aggregate_horizons(args)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
