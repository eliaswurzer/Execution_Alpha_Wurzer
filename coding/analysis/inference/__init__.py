"""Statistische Inferenz: Two-Way-Cluster-SE + Tests."""

from .bootstrap import (
    BootstrapResult,
    block_bootstrap_statistic,
    max_t_union_test,
    paired_diff_bootstrap,
    wild_cluster_bootstrap_mean,
)
from .clustering import (
    OLSResult,
    assert_size_stratification_ok,
    mean_with_twoway_se,
    two_way_cluster_ols,
)
from .power import minimum_detectable_effect, power_at_effect
from .tests import (
    TTestResult, holm_step_down, placebo_flag, primary_ttest, subgroup_ttests,
)

__all__ = [
    "BootstrapResult",
    "OLSResult",
    "TTestResult",
    "assert_size_stratification_ok",
    "block_bootstrap_statistic",
    "holm_step_down",
    "mean_with_twoway_se",
    "minimum_detectable_effect",
    "paired_diff_bootstrap",
    "placebo_flag",
    "power_at_effect",
    "primary_ttest",
    "subgroup_ttests",
    "two_way_cluster_ols",
    "wild_cluster_bootstrap_mean",
]
