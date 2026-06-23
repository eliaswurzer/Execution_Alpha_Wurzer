"""Run H1/H2/H3 into one structured run directory.

Example:
    python -m analysis.runners.run_all_hypotheses --run-id run_20260523_v8 \
        --symbols AAPL MSFT ... --workers 8 --fill-spec tape_replay
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import pandas as pd

from .. import config as cfg
from . import h1_performance_gap, h2_signal_efficiency, h3_te_tradeoff
from .master_panel import (
    TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
    TIER_POLICY_CHOICES,
    materialize_panel,
    run_master_panel,
)

log = logging.getLogger(__name__)

MASTER_STRATEGIES = [
    "S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE",
    "S3_OFI", "S3_IMB", "S3_FULL", "S4_TOD",
]
HYPOTHESIS_STRATEGIES = {
    "h1": MASTER_STRATEGIES,
    "h2": ["S2_TIME_ADAPTIVE", "S3_OFI", "S3_IMB", "S3_FULL"],
    "h3": ["S0_MOC", "S1_STATIC", "S2_TIME_ADAPTIVE", "S3_FULL", "S4_TOD"],
}
HYPOTHESIS_REQUIRED = {
    "h1": ["h1_panel.parquet", "h1_tev.csv", "h1_primary_ttest.csv"],
    "h2": ["h2_panel.parquet", "h2_per_bin.csv", "h2_pooled.csv"],
    "h3": ["h3_panel.parquet", "h3_tev.csv", "h3_raear.csv"],
}


def run_dir(run_id: str, root: Path | None = None) -> Path:
    return (root or cfg.ARTIFACTS_DIR / "runs") / run_id


def make_run_layout(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "hypotheses": base / "hypotheses",
        "h1": base / "hypotheses" / "h1",
        "h2": base / "hypotheses" / "h2",
        "h3": base / "hypotheses" / "h3",
        "tables": base / "tables",
        "figures": base / "figures",
        "volume": base / "volume",
        "logs": base / "logs",
        "metadata": base / "metadata",
        "cache": base / "cache",
        "panel_shards": base / "panel_shards",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _write_status(path: Path, status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _hypothesis_complete(
    name: str,
    out_dir: Path,
    simulation_fingerprint: str,
) -> bool:
    status_path = out_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if status.get("status") != "complete":
        return False
    if status.get("simulation_fingerprint") != simulation_fingerprint:
        return False
    for filename in HYPOTHESIS_REQUIRED[name]:
        path = out_dir / filename
        if not path.exists() or path.stat().st_size <= 0:
            return False
    try:
        panel = pd.read_parquet(
            out_dir / f"{name}_panel.parquet",
            columns=["strategy", "order_id"],
        )
    except Exception:
        return False
    return (
        not panel.empty
        and set(panel["strategy"].unique()) == set(HYPOTHESIS_STRATEGIES[name])
    )


def _load_master_frame(paths: dict[str, Path]) -> pd.DataFrame:
    """Materialize the full master panel once per invocation.

    ``HYPOTHESIS_STRATEGIES['h1']`` equals the master strategy set, so the
    H1 panel file doubles as the master materialization and H2/H3 are pure
    strategy slices of it; this replaces three independent DuckDB scans of
    the daily shards with one. On resume the existing (possibly enriched)
    H1 panel is reused when it still carries every master strategy.
    """
    h1_panel_path = paths["h1"] / "h1_panel.parquet"
    if h1_panel_path.exists():
        try:
            frame = pd.read_parquet(h1_panel_path)
            if set(MASTER_STRATEGIES).issubset(set(frame["strategy"].unique())):
                return frame
        except Exception as exc:
            log.warning("existing h1 panel unreadable (%s); rematerializing", exc)
    materialize_panel(paths["panel_shards"], MASTER_STRATEGIES, h1_panel_path)
    return pd.read_parquet(h1_panel_path)


def _run_hypothesis(
    name: str,
    paths: dict[str, Path],
    master_frame: pd.DataFrame,
    *,
    simulation_fingerprint: str,
) -> None:
    out_dir = paths[name]
    status_path = out_dir / "status.json"
    _write_status(status_path, "running")
    panel_path = out_dir / f"{name}_panel.parquet"
    try:
        if name == "h1":
            panel = master_frame
            h1_performance_gap.analyze_panel(
                panel, out_dir, panel_path=panel_path,
            )
        else:
            panel = master_frame[
                master_frame["strategy"].isin(HYPOTHESIS_STRATEGIES[name])
            ].reset_index(drop=True)
            missing = set(HYPOTHESIS_STRATEGIES[name]) - set(panel["strategy"].unique())
            if missing:
                raise RuntimeError(
                    f"{name.upper()} slice missing strategies: {sorted(missing)}"
                )
            if name == "h3":
                panel = h3_te_tradeoff.primary_surface(panel)
                missing = set(HYPOTHESIS_STRATEGIES[name]) - set(panel["strategy"].unique())
                if missing:
                    raise RuntimeError(
                        f"{name.upper()} primary slice missing strategies: {sorted(missing)}"
                    )
            panel.to_parquet(panel_path, index=False)
            if name == "h2":
                h2_signal_efficiency.analyze_panel(panel, out_dir, n_bins=10)
            else:
                h3_te_tradeoff.analyze_panel(panel, out_dir)
        if not all(
            (out_dir / filename).exists()
            and (out_dir / filename).stat().st_size > 0
            for filename in HYPOTHESIS_REQUIRED[name]
        ):
            raise RuntimeError(f"{name.upper()} output validation failed")
        _write_status(
            status_path,
            "complete",
            panel_rows=len(panel),
            strategies=HYPOTHESIS_STRATEGIES[name],
            simulation_fingerprint=simulation_fingerprint,
        )
    except Exception as exc:
        _write_status(status_path, "failed", error=str(exc))
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", type=Path, default=cfg.ARTIFACTS_DIR / "runs")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--universe", choices=["sp500", "nasdaq100"], default="sp500",
                   help="Point-in-time index universe; ignored when --symbols is set")
    p.add_argument("--start", type=_dt.date.fromisoformat, default=cfg.EVAL_START)
    p.add_argument("--end", type=_dt.date.fromisoformat, default=cfg.EVAL_END)
    p.add_argument("--artifacts", type=Path, default=cfg.ARTIFACTS_DIR / "fill_model")
    p.add_argument("--workers", type=int, default=1)
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
    p.add_argument(
        "--include-value-aware", action="store_true",
        help="Include optional S5_VALUE_AWARE_XGB comparison strategy",
    )
    p.add_argument(
        "--preprocessing-symbols-file", type=Path, default=None,
        help="Conservative preprocessing union file used only for guardrail auditing",
    )
    p.add_argument(
        "--adv-spread-bucket-map", type=Path, default=None,
        help="Optional fixed H1-2018 ADV x spread bucket map merged into master panels",
    )
    p.add_argument(
        "--no-render-thesis", action="store_true",
        help="Skip rendering copy-ready thesis LaTeX snippets after H3",
    )
    p.add_argument(
        "--tier-policy",
        choices=sorted(TIER_POLICY_CHOICES),
        default=TIER_POLICY_CALIBRATED_PLUS_FALLBACK,
        help=(
            "Liquidity-tier completion policy. Use calibrated_only for "
            "strict publication checks; calibrated_plus_fallback assigns "
            "data-complete symbols missing from calibration to tier 3."
        ),
    )
    args = p.parse_args()

    if args.include_value_aware and "S5_VALUE_AWARE_XGB" not in MASTER_STRATEGIES:
        MASTER_STRATEGIES.append("S5_VALUE_AWARE_XGB")
        HYPOTHESIS_STRATEGIES["h1"] = [*HYPOTHESIS_STRATEGIES["h1"], "S5_VALUE_AWARE_XGB"]
        HYPOTHESIS_STRATEGIES["h3"] = [*HYPOTHESIS_STRATEGIES["h3"], "S5_VALUE_AWARE_XGB"]

    base = run_dir(args.run_id, args.run_root)
    paths = make_run_layout(base)
    meta = {
        "run_id": args.run_id,
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
        "include_value_aware": bool(args.include_value_aware),
        "preprocessing_symbols_file": str(args.preprocessing_symbols_file) if args.preprocessing_symbols_file else None,
        "adv_spread_bucket_map": str(args.adv_spread_bucket_map) if args.adv_spread_bucket_map else None,
    }
    (paths["metadata"] / "run_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    log.info("Run bundle: %s", base)
    top_status = base / "run_status.json"
    _write_status(top_status, "running", current_step="simulation")
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
        master_frame: pd.DataFrame | None = None
        for name in ("h1", "h2", "h3"):
            _write_status(top_status, "running", current_step=name)
            if not args.no_resume and _hypothesis_complete(
                name, paths[name], simulation["fingerprint"],
            ):
                log.info("%s already complete; skipping.", name.upper())
                continue
            if master_frame is None:
                master_frame = _load_master_frame(paths)
            _run_hypothesis(
                name,
                paths,
                master_frame,
                simulation_fingerprint=simulation["fingerprint"],
            )
        _write_status(top_status, "complete", simulation=simulation)
    except Exception as exc:
        _write_status(top_status, "failed", error=str(exc))
        raise

    if not args.no_render_thesis:
        # Reporting is derived output; a rendering failure must not flip a
        # validated run to failed. The standalone CLI can re-render any time.
        try:
            from .render_thesis_results import render as render_thesis
            render_thesis(base, allow_incomplete=False)
        except Exception as exc:
            log.warning(
                "Thesis export rendering failed (%s). Re-run via "
                "python -m analysis.runners.render_thesis_results --run-root %s",
                exc, base,
            )

    readme = (
        f"# {args.run_id}\n\n"
        f"- Sample: {args.start} to {args.end}\n"
        f"- Fill specification: `{args.fill_spec}`\n"
        f"- Headline parent size: {cfg.PARENT_ORDER_PRIMARY_FRACTION:g}\n"
        f"- Hypothesis outputs: `hypotheses/h1`, `hypotheses/h2`, `hypotheses/h3`\n"
        f"- Run config: `metadata/run_config.json`\n"
        f"- Thesis exports: `thesis_exports/` (copy-ready Chapter-7 LaTeX snippets)\n"
    )
    (base / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

