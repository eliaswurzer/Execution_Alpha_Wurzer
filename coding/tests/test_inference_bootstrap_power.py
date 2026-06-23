"""Synthetic golden tests for the inference-hardening additions.

Covers the wild cluster bootstrap, the design-based MDE/power helpers, the
one-sided p-value reconciliation, the per-bin max-t union test, the H3 block
bootstrap, the FDR adjustment, and the consolidated test registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.inference.bootstrap import (
    block_bootstrap_statistic,
    max_t_union_test,
    wild_cluster_bootstrap_mean,
)
from analysis.inference.clustering import mean_with_twoway_se
from analysis.inference.power import minimum_detectable_effect, power_at_effect
from analysis.inference.tests import _one_sided_p, _two_sided_p
from statistical_tests import h3_inference
from statistical_tests.h3_inference import h3_bootstrap
from statistical_tests.multiple_testing import (
    benjamini_hochberg,
    one_sided_p_from_t,
    two_sided_p_from_t,
)
from statistical_tests.test_registry import build_test_registry


def _balanced_panel(mean: float, sd: float, n_sym: int, n_date: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    syms, dates, vals = [], [], []
    for s in range(n_sym):
        for d in range(n_date):
            syms.append(f"S{s}")
            dates.append(f"2019-01-{d + 1:02d}")
            vals.append(mean + sd * rng.standard_normal())
    return pd.Series(vals), pd.Series(syms), pd.Series(dates)


def test_bootstrap_se_matches_analytic_two_way_se() -> None:
    y, sym, date = _balanced_panel(0.3, 1.0, n_sym=20, n_date=20)
    _, se_analytic = mean_with_twoway_se(y, sym, date)

    boot = wild_cluster_bootstrap_mean(y, sym, date, n_boot=200, seed=1)

    # The bootstrap reports the same analytic two-way SE it studentizes against.
    assert np.isclose(boot.se_analytic, se_analytic, rtol=1e-10, atol=1e-12)


def test_bootstrap_p_agrees_with_analytic_in_large_cluster_limit() -> None:
    y, sym, date = _balanced_panel(0.4, 1.0, n_sym=25, n_date=25, seed=11)

    boot = wild_cluster_bootstrap_mean(
        y, sym, date, alternative="greater", n_boot=2000, seed=3,
    )

    # With many clusters the wild bootstrap one-sided p tracks the normal one.
    assert abs(boot.p_bootstrap_one_sided - boot.p_analytic_one_sided) < 0.05
    assert boot.p_bootstrap_one_sided < 0.05  # genuine positive effect


def test_bootstrap_p_is_large_under_pure_noise() -> None:
    y, sym, date = _balanced_panel(0.0, 1.0, n_sym=20, n_date=20, seed=99)

    boot = wild_cluster_bootstrap_mean(
        y, sym, date, alternative="greater", n_boot=2000, seed=5,
    )

    assert boot.p_bootstrap_two_sided > 0.10
    # Confidence interval brackets zero when the mean is null.
    assert boot.ci_lo < 0.0 < boot.ci_hi


def test_minimum_detectable_effect_matches_closed_form() -> None:
    # z_{0.95} + z_{0.80} = 1.644854 + 0.841621 = 2.486476
    mde = minimum_detectable_effect(1.0, alpha=0.05, power=0.80, one_sided=True)
    assert abs(mde - 2.486476) < 1e-3
    # Scales linearly in the standard error.
    assert abs(minimum_detectable_effect(2.5, alpha=0.05, power=0.80) - 2.5 * 2.486476) < 1e-2


def test_power_at_the_mde_is_the_target_power() -> None:
    se = 1.3
    mde = minimum_detectable_effect(se, alpha=0.05, power=0.80, one_sided=True)
    assert abs(power_at_effect(mde, se, alpha=0.05, one_sided=True) - 0.80) < 1e-2


def test_one_sided_p_is_half_two_sided_when_sign_matches() -> None:
    assert abs(2 * one_sided_p_from_t(2.0, greater=True) - two_sided_p_from_t(2.0)) < 1e-9
    assert abs(2 * _one_sided_p(2.0, greater=True) - _two_sided_p(2.0)) < 1e-9
    # Wrong-direction one-sided p is the complement.
    assert abs(one_sided_p_from_t(2.0, greater=False) - (1 - one_sided_p_from_t(2.0, greater=True))) < 1e-9


def test_benjamini_hochberg_known_values() -> None:
    adj = benjamini_hochberg([0.04, 0.01])
    assert np.isclose(adj[0], 0.04)
    assert np.isclose(adj[1], 0.02)
    # Equal step values collapse to a common adjusted level.
    flat = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(flat, 0.05)
    # NaNs are preserved.
    with_nan = benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(with_nan[1])


def _union_panel(effect_bin: int | None, n_bins: int, n_sym: int, n_date: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bins):
        for s in range(n_sym):
            for d in range(n_date):
                base = 5.0 if (effect_bin is not None and b == effect_bin) else 0.0
                rows.append({
                    "bin": b, "symbol": f"S{s}", "date": f"2019-02-{d + 1:02d}",
                    "diff": base + rng.standard_normal(),
                })
    return pd.DataFrame(rows)


def test_max_t_union_test_detects_a_single_active_bin() -> None:
    null = _union_panel(None, n_bins=5, n_sym=8, n_date=8, seed=21)
    active = _union_panel(2, n_bins=5, n_sym=8, n_date=8, seed=21)

    p_null = max_t_union_test(null, value_col="diff", group_col="bin", n_boot=500, seed=2)
    p_active = max_t_union_test(active, value_col="diff", group_col="bin", n_boot=500, seed=2)

    assert p_active["p_bootstrap"] < 0.05
    assert p_active["p_bootstrap"] < p_null["p_bootstrap"]
    assert p_active["n_groups"] == 5


def _h3_panel(seed: int = 13):
    rng = np.random.default_rng(seed)
    rows = []
    n_date, n_sym = 30, 6
    for d in range(n_date):
        for s in range(n_sym):
            common = {"symbol": f"S{s}", "date": f"2019-03-{d + 1:02d}"}
            rows.append({**common, "strategy": "S0_MOC", "net_alpha_vs_moc_bps": 0.0})
            rows.append({**common, "strategy": "S2_TIME_ADAPTIVE",
                         "net_alpha_vs_moc_bps": 0.5 + 2.0 * rng.standard_normal()})
            rows.append({**common, "strategy": "S3_FULL",
                         "net_alpha_vs_moc_bps": 2.0 + 2.0 * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_h3_bootstrap_ranks_and_brackets() -> None:
    panel = _h3_panel()
    out = h3_bootstrap(panel, n_boot=300, seed=4)

    ci = out["strategy_ci"].set_index("strategy")
    assert {"S2_TIME_ADAPTIVE", "S3_FULL"}.issubset(ci.index)
    # S3_FULL has the higher mean alpha and hence the higher information ratio.
    assert ci.loc["S3_FULL", "ir"] > ci.loc["S2_TIME_ADAPTIVE", "ir"]
    # Confidence intervals are ordered.
    assert ci.loc["S3_FULL", "ir_lo"] <= ci.loc["S3_FULL", "ir"] <= ci.loc["S3_FULL", "ir_hi"]
    rank = out["rank_stability"].iloc[0]
    assert 0.0 <= rank["p_ir_ranking_preserved"] <= 1.0
    assert rank["p_ir_ranking_preserved"] > 0.5


def test_h3_sufficient_stats_bootstrap_matches_row_resampling() -> None:
    panel = _h3_panel(seed=17)
    generic = block_bootstrap_statistic(
        panel, h3_inference._moments_stat, cluster_col="date", n_boot=25, seed=9,
    ).sort_index(axis=1)
    fast = h3_inference._bootstrap_moments_from_date_stats(
        panel, n_boot=25, seed=9,
    ).sort_index(axis=1)

    assert list(fast.columns) == list(generic.columns)
    assert np.allclose(fast.to_numpy(), generic.to_numpy(), equal_nan=True)


def test_registry_enumerates_every_test_with_a_family(artifact_dir) -> None:
    h1 = artifact_dir / "hypotheses" / "h1"
    h1.mkdir(parents=True)
    pd.DataFrame([{
        "mean": 0.3, "se": 0.1, "t": 3.0, "p_value": 0.0027,
        "n": 1000, "label": "primary", "alternative": "greater",
        "p_one_sided": 0.00135,
    }]).to_csv(h1 / "h1_primary_ttest.csv", index=False)
    pd.DataFrame([
        {"group": "tier", "level": 1, "mean": 0.1, "se": 0.2, "t": 0.5,
         "p_value": 0.6, "p_holm": 0.6, "n": 300},
        {"group": "tier", "level": 2, "mean": -0.1, "se": 0.2, "t": -0.5,
         "p_value": 0.6, "p_holm": 0.6, "n": 300},
    ]).to_csv(h1 / "h1_subgroup_tier.csv", index=False)

    economic = pd.DataFrame([
        {"spec": "tape_replay_queue", "label": "Queue", "is_headline": True,
         "mean_net_alpha_vs_moc_bps": -0.1, "se_twoway": 0.1, "t": -1.0,
         "p_value": 0.32, "p_one_sided": 0.84, "p_bootstrap_one_sided": 0.85,
         "p_holm": np.nan, "n": 1000},
        {"spec": "cox", "label": "Cox", "is_headline": False,
         "mean_net_alpha_vs_moc_bps": 0.2, "se_twoway": 0.1, "t": 2.0,
         "p_value": 0.05, "p_one_sided": 0.025, "p_bootstrap_one_sided": 0.03,
         "p_holm": 0.1, "n": 1000},
    ])
    h2 = pd.DataFrame([
        {"label": "OFI_marginal", "mean": 0.1, "se_twoway": 0.05, "t": 2.0,
         "p_value": 0.045, "p_one_sided": 0.0225, "p_holm": 0.09, "n": 800},
        {"label": "IMB_marginal", "mean": 0.2, "se_twoway": 0.05, "t": 4.0,
         "p_value": 0.0001, "p_one_sided": 0.00005, "p_holm": 0.0002, "n": 800},
        {"label": "FULL_vs_S2", "mean": 0.3, "se_twoway": 0.05, "t": 6.0,
         "p_value": 0.00001, "p_one_sided": 0.000005, "p_holm": 0.00002, "n": 800},
    ])
    h2_union = pd.DataFrame([
        {"label": "OFI_marginal", "max_abs_t": 2.5, "p_bootstrap": 0.04,
         "n": 800, "n_groups": 8},
        {"label": "IMB_marginal", "max_abs_t": 3.0, "p_bootstrap": 0.02,
         "n": 800, "n_groups": 8},
    ])

    registry = build_test_registry(
        economic=economic, paired=pd.DataFrame(), h2=h2, h2_union=h2_union,
        headline_run=artifact_dir,
    )

    assert not registry.empty
    assert (registry["family"].astype(str).str.len() > 0).all()
    assert registry["correction"].isin(
        {"none", "holm", "holm_one_sided", "fdr", "bootstrap",
         "bootstrap_one_sided", "bootstrap_max_t", "block_bootstrap"}
    ).all()
    assert "confirmatory" in set(registry["role"])
    assert (registry["test_id"] == "H1_primary").any()
    # The confirmatory primary decision uses the registered one-sided p-value.
    primary = registry[registry["test_id"] == "H1_primary"].iloc[0]
    assert np.isclose(primary["p_decision"], 0.00135)
    assert bool(primary["reject_5pct"]) is True

    h2_confirmatory = registry[
        (registry["hypothesis"] == "H2") & (registry["role"] == "confirmatory")
    ]["test_id"].tolist()
    assert h2_confirmatory == ["H2:OFI_marginal", "H2:IMB_marginal"]
    union_roles = registry[registry["test_id"].str.startswith("H2_union:")]["role"]
    assert set(union_roles) == {"exploratory"}
