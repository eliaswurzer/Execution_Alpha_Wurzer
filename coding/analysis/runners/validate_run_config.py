"""Validate locked final-run configuration files before heavy runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from analysis import config as cfg
from analysis.fill_model.value_model import VALUE_TARGET_COLUMN


ALLOWED_FILL_SPECS = {
    "tape_replay",
    "tape_replay_volume",
    "tape_replay_strict",
    "tape_replay_queue",
    "cox",
    "xgb",
    "infinite_depth_touch",
}
ALLOWED_TIER_POLICIES = {"calibrated_only", "calibrated_plus_fallback"}
ALLOWED_XGB_DEVICES = {"cpu", "cuda", "auto"}


def _date(value: str, label: str, errors: list[str]) -> date | None:
    try:
        return date.fromisoformat(value)
    except Exception:
        errors.append(f"{label} must be an ISO date: {value!r}")
        return None


def _require(data: dict[str, Any], key: str, errors: list[str]) -> Any:
    if key not in data:
        errors.append(f"Missing required key: {key}")
        return None
    return data[key]


def _path(value: Any, label: str, errors: list[str], *, must_exist: bool, allow_missing: bool) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty string path")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} must be absolute: {value}")
    if must_exist and not allow_missing and not path.exists():
        errors.append(f"{label} does not exist: {value}")
    return path


def validate_config(config: dict[str, Any], *, allow_missing_paths: bool = False) -> list[str]:
    errors: list[str] = []
    for key in [
        "run_id_prefix", "artifact_root", "data_roots", "volume_db",
        "membership_root", "universe", "symbols_file", "calibration",
        "evaluation", "value_model",
    ]:
        _require(config, key, errors)

    _path(config.get("artifact_root"), "artifact_root", errors, must_exist=False, allow_missing=allow_missing_paths)
    _path(config.get("volume_db"), "volume_db", errors, must_exist=True, allow_missing=allow_missing_paths)
    _path(config.get("membership_root"), "membership_root", errors, must_exist=True, allow_missing=allow_missing_paths)
    _path(config.get("symbols_file"), "symbols_file", errors, must_exist=True, allow_missing=allow_missing_paths)

    data_roots = config.get("data_roots")
    if not isinstance(data_roots, dict) or not data_roots:
        errors.append("data_roots must be a non-empty object")
    else:
        for year, roots in data_roots.items():
            if not isinstance(roots, list) or not roots:
                errors.append(f"data_roots.{year} must be a non-empty list")
                continue
            for i, root in enumerate(roots):
                _path(root, f"data_roots.{year}[{i}]", errors, must_exist=True, allow_missing=allow_missing_paths)

    calibration = config.get("calibration") or {}
    evaluation = config.get("evaluation") or {}
    if not isinstance(calibration, dict):
        errors.append("calibration must be an object")
        calibration = {}
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be an object")
        evaluation = {}

    warmup_start = _date(str(calibration.get("warmup_start", "")), "calibration.warmup_start", errors)
    eval_start = _date(str(evaluation.get("start", "")), "evaluation.start", errors)
    eval_end = _date(str(evaluation.get("end", "")), "evaluation.end", errors)
    if warmup_start and eval_start and warmup_start >= eval_start:
        errors.append("calibration.warmup_start must precede evaluation.start")
    if eval_start and eval_end and eval_start > eval_end:
        errors.append("evaluation.start must not be after evaluation.end")

    lookback = int(calibration.get("rolling_lookback_days", 0) or 0)
    min_train = int(calibration.get("rolling_min_train_days", 0) or 0)
    if lookback <= 0 or min_train <= 0:
        errors.append("rolling train windows must be positive")
    elif min_train > lookback:
        errors.append("rolling_min_train_days must be <= rolling_lookback_days")

    if int(calibration.get("workers", 0) or 0) < 1:
        errors.append("calibration.workers must be >= 1")
    if int(evaluation.get("workers", 0) or 0) < 1:
        errors.append("evaluation.workers must be >= 1")

    fill_spec = evaluation.get("fill_specification")
    if fill_spec not in ALLOWED_FILL_SPECS:
        errors.append(f"evaluation.fill_specification must be one of {sorted(ALLOWED_FILL_SPECS)}")
    tier_policy = evaluation.get("tier_policy")
    if tier_policy not in ALLOWED_TIER_POLICIES:
        errors.append(f"evaluation.tier_policy must be one of {sorted(ALLOWED_TIER_POLICIES)}")

    value_model = config.get("value_model") or {}
    if not isinstance(value_model, dict):
        errors.append("value_model must be an object")
        value_model = {}
    if value_model.get("enabled"):
        for key in ["strategy", "policy", "xgb_device", "target", "offset_grid_bps", "min_expected_alpha_bps"]:
            if key not in value_model:
                errors.append(f"value_model.{key} is required when value_model.enabled is true")
        if value_model.get("strategy") != "S5_VALUE_AWARE_XGB":
            errors.append("value_model.strategy must be S5_VALUE_AWARE_XGB when enabled")
        if value_model.get("policy") != cfg.VALUE_MODEL_POLICY_VERSION:
            errors.append(
                "value_model.policy must match current VALUE_MODEL_POLICY_VERSION "
                f"{cfg.VALUE_MODEL_POLICY_VERSION}"
            )
        if value_model.get("target") != VALUE_TARGET_COLUMN:
            errors.append(f"value_model.target must be {VALUE_TARGET_COLUMN}")
        if value_model.get("xgb_device") not in ALLOWED_XGB_DEVICES:
            errors.append(f"value_model.xgb_device must be one of {sorted(ALLOWED_XGB_DEVICES)}")
        grid = value_model.get("offset_grid_bps")
        if not isinstance(grid, list) or not grid:
            errors.append("value_model.offset_grid_bps must be a non-empty list")
        else:
            for item in grid:
                if not isinstance(item, (int, float)):
                    errors.append("value_model.offset_grid_bps must contain only numeric values")
                    break
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-missing-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8-sig"))
    errors = validate_config(payload, allow_missing_paths=args.allow_missing_paths)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    summary = {
        "config": str(args.config),
        "run_id_prefix": payload.get("run_id_prefix"),
        "universe": payload.get("universe"),
        "evaluation": payload.get("evaluation", {}),
        "value_model_enabled": bool(payload.get("value_model", {}).get("enabled")),
    }
    print("Config validation passed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

