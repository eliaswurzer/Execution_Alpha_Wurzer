"""Default paths for the supplementary statistical validation suite."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CODING_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = CODING_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent

RUNS_ROOT = WORKSPACE_ROOT / "artifacts" / "runs"
FILL_MODEL_DIR = WORKSPACE_ROOT / "artifacts" / "fill_model_v4_winsor_cutoff_20260617"
OUTPUT_DIR = CODING_ROOT / "artifacts" / "statistical_tests_20260618_final_v4_xgb"

HEADLINE_RUN = RUNS_ROOT / "final_v4_20260618_queue"
STRICT_RUN = RUNS_ROOT / "final_v4_20260618_strict"
AT_OR_THROUGH_RUN = RUNS_ROOT / "final_v4_20260618_at_or_through"
COX_RUN = RUNS_ROOT / "final_v4_20260617_cox"
KM_RUN = RUNS_ROOT / "final_v4_20260617_km"
# Included in the current model-based robustness family.
XGB_RUN = RUNS_ROOT / "final_v4_20260617_xgb"
SIZE_GRID_RUN = RUNS_ROOT / "size_grid_20260613"

FILL_SPEC_RUNS: dict[str, Path] = {
    "tape_replay_queue": HEADLINE_RUN,
    "tape_replay_strict": STRICT_RUN,
    "tape_replay": AT_OR_THROUGH_RUN,
    "cox": COX_RUN,
    "km": KM_RUN,
    "xgb": XGB_RUN,
}

FILL_SPEC_LABELS: dict[str, str] = {
    "tape_replay_queue": "Queue-aware replay (headline)",
    "tape_replay_strict": "Strictly-through replay (lower bound)",
    "tape_replay": "At-or-through replay (upper bound)",
    "cox": "Cox proportional hazards",
    "km": "Kaplan-Meier",
    "xgb": "XGBoost survival",
}

FILL_SPEC_ORDER = [
    "tape_replay_queue",
    "tape_replay_strict",
    "tape_replay",
    "cox",
    "km",
    "xgb",
]

MODEL_SPECS = ["cox", "km", "xgb"]
H2_FAMILY_LABELS = ["OFI_marginal", "IMB_marginal", "FULL_vs_S2", "interaction"]
EXPECTED_EVAL_DATES = 371
H2_CONFIRMATORY_SURFACE = "pooled_matched_differentials"
H3_INFERENCE_ROLE = "descriptive_risk_tradeoff"
OOS_CACHE_POLICY_VERSION = "oos_event_cache_v2"

PRIMARY_WINDOW = "B"
PRIMARY_SIZE_FRAC = 0.01
OOS_DAYS_PER_QUARTER = 10
OOS_EVENT_SAMPLE_PER_SYMBOL_DAY = 48

# Registered directional alternatives (Thesis Primary Hypothesis Designation):
# H1, H2a, H2b predict positive incremental alpha; H3 is two-sided.
PRIMARY_ALTERNATIVE = "greater"

# Bootstrap and power settings (recorded in the run manifest for reproducibility).
BOOTSTRAP_B = 9999
BOOTSTRAP_B_H3 = 2000
BOOTSTRAP_B_UNION = 2000
BOOTSTRAP_WEIGHTS = "webb"
BOOTSTRAP_TWO_WAY = True
BOOTSTRAP_SEED = 42
MDE_ALPHA = 0.05
MDE_POWER = 0.80
CI_ALPHA = 0.05
