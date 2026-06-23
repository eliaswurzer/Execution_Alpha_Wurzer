"""Run all supplementary statistical tests and render thesis snippets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis import config as analysis_cfg

from . import config as st_cfg
from .economic_tests import (
    adjusted_h2_pooled,
    fill_spec_summary,
    paired_vs_queue_tests,
    sha256_file,
)
from .h3_inference import h3_bootstrap, read_h3_panel
from .oos_calibration import build_oos_event_panel, score_event_panel
from .render_outputs import write_outputs
from .test_registry import build_test_registry


HYPOTHESIS_REQUIRED = {
    "h1": ["h1_panel.parquet", "h1_tev.csv", "h1_primary_ttest.csv"],
    "h2": ["h2_panel.parquet", "h2_per_bin.csv", "h2_pooled.csv"],
    "h3": ["h3_panel.parquet", "h3_tev.csv", "h3_raear.csv"],
}


def _hash_if_exists(path: Path, key: str, hashes: dict[str, str]) -> None:
    path = Path(path)
    if path.exists() and path.is_file():
        hashes[key] = sha256_file(path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_gate_errors(spec: str, root: Path) -> list[str]:
    """Return validation problems for one canonical statistical input run."""
    root = Path(root)
    errors: list[str] = []
    if not root.exists():
        return [f"{spec}: run root missing: {root}"]

    status_path = root / "run_status.json"
    status: dict = {}
    if not status_path.exists():
        errors.append(f"{spec}: run_status.json missing")
    else:
        try:
            status = _read_json(status_path)
        except Exception as exc:
            errors.append(f"{spec}: run_status.json unreadable: {exc}")
        if status and status.get("status") != "complete":
            errors.append(
                f"{spec}: run_status status is {status.get('status')!r}, expected 'complete'"
            )

    sim = status.get("simulation") or {}
    summary_path = root / "metadata" / "simulation_summary.json"
    if summary_path.exists():
        try:
            summary = _read_json(summary_path)
            sim = {**summary, **sim}
        except Exception as exc:
            errors.append(f"{spec}: simulation_summary.json unreadable: {exc}")
    if sim:
        expected = sim.get("dates_expected")
        valid = sim.get("dates_with_valid_shards")
        if expected != st_cfg.EXPECTED_EVAL_DATES or valid != st_cfg.EXPECTED_EVAL_DATES:
            errors.append(
                f"{spec}: expected {st_cfg.EXPECTED_EVAL_DATES} valid shards, "
                f"got {valid}/{expected}"
            )
        if int(sim.get("critical_failures", -1)) != 0:
            errors.append(f"{spec}: critical_failures is {sim.get('critical_failures')!r}")
    else:
        errors.append(f"{spec}: simulation summary missing from status/metadata")

    sim_config_path = root / "metadata" / "simulation_config.json"
    if not sim_config_path.exists():
        errors.append(f"{spec}: metadata/simulation_config.json missing")
    else:
        try:
            sim_config = _read_json(sim_config_path)
            if sim_config.get("feature_policy") != analysis_cfg.FEATURE_POLICY_VERSION:
                errors.append(
                    f"{spec}: feature_policy {sim_config.get('feature_policy')!r} "
                    f"!= {analysis_cfg.FEATURE_POLICY_VERSION!r}"
                )
            if sim_config.get("trade_policy") != analysis_cfg.TRADE_CONDITION_POLICY_VERSION:
                errors.append(
                    f"{spec}: trade_policy {sim_config.get('trade_policy')!r} "
                    f"!= {analysis_cfg.TRADE_CONDITION_POLICY_VERSION!r}"
                )
        except Exception as exc:
            errors.append(f"{spec}: simulation_config.json unreadable: {exc}")

    run_config_path = root / "metadata" / "run_config.json"
    if not run_config_path.exists():
        errors.append(f"{spec}: metadata/run_config.json missing")
    else:
        try:
            run_config = _read_json(run_config_path)
            artifact_root = Path(str(run_config.get("artifacts", "")))
            cal_manifest = artifact_root / "calibration_manifest.json"
            if not cal_manifest.exists():
                errors.append(f"{spec}: calibration manifest missing via run_config")
            else:
                cal = _read_json(cal_manifest)
                if cal.get("status") != "complete":
                    errors.append(
                        f"{spec}: calibration status is {cal.get('status')!r}"
                    )
                if cal.get("feature_policy") != analysis_cfg.FEATURE_POLICY_VERSION:
                    errors.append(
                        f"{spec}: calibration feature_policy {cal.get('feature_policy')!r} "
                        f"!= {analysis_cfg.FEATURE_POLICY_VERSION!r}"
                    )
        except Exception as exc:
            errors.append(f"{spec}: run_config/calibration gate unreadable: {exc}")

    for hypothesis, filenames in HYPOTHESIS_REQUIRED.items():
        hdir = root / "hypotheses" / hypothesis
        hstatus_path = hdir / "status.json"
        if not hstatus_path.exists():
            errors.append(f"{spec}: {hypothesis} status.json missing")
        else:
            try:
                hstatus = _read_json(hstatus_path)
                if hstatus.get("status") != "complete":
                    errors.append(
                        f"{spec}: {hypothesis} status is {hstatus.get('status')!r}"
                    )
            except Exception as exc:
                errors.append(f"{spec}: {hypothesis} status unreadable: {exc}")
        for filename in filenames:
            path = hdir / filename
            if not path.exists() or path.stat().st_size <= 0:
                errors.append(f"{spec}: required {hypothesis} artifact missing/empty: {filename}")
    return errors


def validate_canonical_inputs(runs: dict[str, Path] | None = None) -> None:
    """Hard gate for final statistical outputs.

    The suite must not silently mix old deterministic runs, incomplete v4
    reruns, or calibration artifacts produced under a different policy.
    """
    runs = runs or st_cfg.FILL_SPEC_RUNS
    errors: list[str] = []
    for spec in st_cfg.FILL_SPEC_ORDER:
        root = runs.get(spec)
        if root is None:
            errors.append(f"{spec}: missing from FILL_SPEC_RUNS")
            continue
        errors.extend(_run_gate_errors(spec, Path(root)))

    cal_manifest = st_cfg.FILL_MODEL_DIR / "calibration_manifest.json"
    if not cal_manifest.exists():
        errors.append(f"fill_model: calibration_manifest.json missing in {st_cfg.FILL_MODEL_DIR}")
    else:
        try:
            cal = _read_json(cal_manifest)
            if cal.get("status") != "complete":
                errors.append(f"fill_model: status is {cal.get('status')!r}")
            if cal.get("feature_policy") != analysis_cfg.FEATURE_POLICY_VERSION:
                errors.append(
                    f"fill_model: feature_policy {cal.get('feature_policy')!r} "
                    f"!= {analysis_cfg.FEATURE_POLICY_VERSION!r}"
                )
        except Exception as exc:
            errors.append(f"fill_model: calibration manifest unreadable: {exc}")

    if errors:
        raise RuntimeError(
            "Statistical-suite input gate failed:\n  - " + "\n  - ".join(errors)
        )


def _manifest_base(args, selected_dates: list | None = None) -> dict:
    run_roots = {k: str(v) for k, v in st_cfg.FILL_SPEC_RUNS.items()}
    input_hashes: dict[str, str] = {}
    for spec, root in st_cfg.FILL_SPEC_RUNS.items():
        root = Path(root)
        _hash_if_exists(root / "run_status.json", f"{spec}:run_status", input_hashes)
        _hash_if_exists(root / "metadata" / "run_config.json", f"{spec}:run_config", input_hashes)
        _hash_if_exists(root / "metadata" / "simulation_config.json", f"{spec}:simulation_config", input_hashes)
        _hash_if_exists(root / "metadata" / "simulation_summary.json", f"{spec}:simulation_summary", input_hashes)
        _hash_if_exists(root / "metadata" / "simulation_failures.csv", f"{spec}:simulation_failures", input_hashes)
        for hypothesis, filenames in HYPOTHESIS_REQUIRED.items():
            hdir = root / "hypotheses" / hypothesis
            _hash_if_exists(hdir / "status.json", f"{spec}:{hypothesis}:status", input_hashes)
            for filename in filenames:
                _hash_if_exists(hdir / filename, f"{spec}:{hypothesis}:{filename}", input_hashes)
    h2 = st_cfg.HEADLINE_RUN / "hypotheses" / "h2" / "h2_pooled.csv"
    if h2.exists():
        input_hashes["headline:h2_pooled"] = sha256_file(h2)
    cal_manifest = st_cfg.FILL_MODEL_DIR / "calibration_manifest.json"
    feature_policy = "unknown"
    if cal_manifest.exists():
        payload = json.loads(cal_manifest.read_text(encoding="utf-8"))
        feature_policy = str(payload.get("feature_policy", "unknown"))
        input_hashes["fill_model:calibration_manifest"] = sha256_file(cal_manifest)
    for name in ("validation.csv", "km_validation.csv", "xgb_validation.csv"):
        path = st_cfg.FILL_MODEL_DIR / name
        if path.exists():
            input_hashes[f"fill_model:{name}"] = sha256_file(path)
    out_dir = Path(getattr(args, "out_dir", st_cfg.OUTPUT_DIR))
    _hash_if_exists(out_dir / "oos_event_status.csv", "oos:event_status", input_hashes)
    _hash_if_exists(out_dir / "oos_event_manifest.json", "oos:event_manifest", input_hashes)
    return {
        "run_roots": run_roots,
        "fill_spec_runs": run_roots,
        "fill_spec_order": list(st_cfg.FILL_SPEC_ORDER),
        "h2_confirmatory_surface": st_cfg.H2_CONFIRMATORY_SURFACE,
        "h3_inference_role": st_cfg.H3_INFERENCE_ROLE,
        "fill_model_dir": str(st_cfg.FILL_MODEL_DIR),
        "model_specs": list(st_cfg.MODEL_SPECS),
        "output_policy": "diagnostic_skip_oos" if args.skip_oos else "final",
        "feature_policy": feature_policy,
        "input_sha256": input_hashes,
        "oos_dates": [str(d) for d in selected_dates] if selected_dates else [],
        "bootstrap": {
            "skip_bootstrap": bool(getattr(args, "skip_bootstrap", False)),
            "b_primary": int(getattr(args, "bootstrap_b", st_cfg.BOOTSTRAP_B)),
            "b_h3": int(st_cfg.BOOTSTRAP_B_H3),
            "b_union": int(st_cfg.BOOTSTRAP_B_UNION),
            "weights": st_cfg.BOOTSTRAP_WEIGHTS,
            "two_way": bool(st_cfg.BOOTSTRAP_TWO_WAY),
            "seed": int(st_cfg.BOOTSTRAP_SEED),
            "alternative": st_cfg.PRIMARY_ALTERNATIVE,
            "mde_alpha": float(st_cfg.MDE_ALPHA),
            "mde_power": float(st_cfg.MDE_POWER),
            "ci_alpha": float(st_cfg.CI_ALPHA),
        },
        "settings": {
            "skip_oos": bool(args.skip_oos),
            "diagnostic_skip_oos": bool(args.diagnostic_skip_oos),
            "force_oos": bool(args.force_oos),
            "days_per_quarter": int(args.days_per_quarter),
            "event_sample_per_symbol_day": None if args.event_sample_per_symbol_day is None else int(args.event_sample_per_symbol_day),
            "workers": int(args.workers),
        },
    }


def _h2_union_from_headline():
    """Per-bin H2 union test on the headline H2 panel, if available."""
    from analysis.runners.h2_signal_efficiency import per_bin_union_test

    path = Path(st_cfg.HEADLINE_RUN) / "hypotheses" / "h2" / "h2_panel.parquet"
    if not path.exists():
        return None
    panel = pd.read_parquet(path)
    if panel.empty:
        return None
    return per_bin_union_test(panel, n_boot=st_cfg.BOOTSTRAP_B_UNION, seed=st_cfg.BOOTSTRAP_SEED)


def _h3_inference_from_headline() -> dict:
    """Block-bootstrap H3 risk-ranking inference on the headline H3 panel."""
    panel = read_h3_panel(st_cfg.HEADLINE_RUN)
    if panel.empty:
        return {}
    return h3_bootstrap(panel, n_boot=st_cfg.BOOTSTRAP_B_H3, seed=st_cfg.BOOTSTRAP_SEED)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=st_cfg.OUTPUT_DIR)
    parser.add_argument("--skip-oos", action="store_true", help="Skip expensive OOS event construction and write an empty calibration table.")
    parser.add_argument("--diagnostic-skip-oos", action="store_true", help="Allow --skip-oos only for explicitly diagnostic output, never final thesis tables.")
    parser.add_argument("--force-oos", action="store_true", help="Rebuild OOS event shards even if a status file exists.")
    parser.add_argument("--days-per-quarter", type=int, default=st_cfg.OOS_DAYS_PER_QUARTER)
    parser.add_argument("--event-sample-per-symbol-day", type=int, default=st_cfg.OOS_EVENT_SAMPLE_PER_SYMBOL_DAY)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip the wild cluster / block bootstraps (keep analytic, one-sided, and MDE columns only).")
    parser.add_argument("--bootstrap-b", type=int, default=st_cfg.BOOTSTRAP_B,
                        help="Replications for the primary fill-spec wild cluster bootstrap.")
    args = parser.parse_args(argv)

    if args.skip_oos and not args.diagnostic_skip_oos:
        parser.error("--skip-oos is diagnostic only; pass --diagnostic-skip-oos to write draft diagnostics.")

    validate_canonical_inputs()

    run_bootstrap = not args.skip_bootstrap
    economic = fill_spec_summary(run_bootstrap=run_bootstrap, n_boot=args.bootstrap_b)
    paired = paired_vs_queue_tests()
    h2 = adjusted_h2_pooled()

    # H2 per-bin union test and H3 risk-ranking bootstrap (from existing panels).
    h2_union = _h2_union_from_headline() if run_bootstrap else None
    h3_tables = _h3_inference_from_headline() if run_bootstrap else {}
    registry = build_test_registry(
        economic=economic, paired=paired, h2=h2, h2_union=h2_union,
        h3_rank_stability=h3_tables.get("rank_stability"),
        headline_run=st_cfg.HEADLINE_RUN,
    )

    selected_dates = []
    if args.skip_oos:
        calibration = pd.DataFrame(columns=[
            "model", "tier", "n", "observed_fill_rate",
            "mean_predicted_probability", "absolute_calibration_error",
            "brier", "reliability", "resolution", "uncertainty", "auc",
        ])
    else:
        event_panel, selected_dates, status = build_oos_event_panel(
            out_dir=args.out_dir,
            days_per_quarter=args.days_per_quarter,
            workers=args.workers,
            event_sample_per_symbol_day=args.event_sample_per_symbol_day,
            force=args.force_oos,
        )
        status.to_csv(args.out_dir / "oos_event_status.csv", index=False)
        calibration = score_event_panel(event_panel)
        if calibration.empty:
            raise RuntimeError(
                "OOS calibration produced no rows; refusing to write final statistical outputs."
            )

    manifest = _manifest_base(args, selected_dates)
    write_outputs(
        args.out_dir,
        calibration=calibration,
        economic=economic,
        paired=paired,
        h2=h2,
        manifest=manifest,
        h3_strategy_ci=h3_tables.get("strategy_ci"),
        h3_rank_stability=h3_tables.get("rank_stability"),
        h3_pairwise=h3_tables.get("pairwise"),
        h2_union=h2_union,
        registry=registry,
    )
    print(f"Wrote statistical validation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
