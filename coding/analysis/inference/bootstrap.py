"""
bootstrap.py -- Cluster wild bootstrap and block bootstrap for the panel-mean
and paired-differential statistics used in the hypothesis tests.

The analytic inference layer (``clustering.py``) reports asymptotic two-way
clustered standard errors with a normal reference distribution. That reference
is accurate at the full-sample scale (several hundred date clusters) but can
overstate significance when a subgroup leaves few clusters in one dimension.
This module adds two distribution-free robustness devices:

* ``wild_cluster_bootstrap_mean`` -- a wild cluster bootstrap (Cameron, Gelbach
  & Miller 2008; Webb 2014; for the multiway construction Davezies,
  D'Haultfoeuille & Guyonvarch 2021) for the mean-against-zero / paired-diff
  model ``y = mu + eps``. The restricted (imposed-null) variant supplies the
  bootstrap p-value; the unrestricted variant supplies a studentized
  (percentile-t) confidence interval.

* ``block_bootstrap_statistic`` -- a generic by-cluster resampling helper that
  recomputes an arbitrary panel statistic on whole-cluster resamples. It is the
  engine behind the H3 risk-ranking confidence intervals.

The constant-only two-way cluster variance is computed in closed form here for
speed (so B can be large), and is verified against ``mean_with_twoway_se`` in
the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Callable

import numpy as np
import pandas as pd

from .. import config as cfg


# Webb (2014) six-point weights: robust when the number of clusters is small.
_WEBB_POINTS = np.array(
    [-sqrt(1.5), -1.0, -sqrt(0.5), sqrt(0.5), 1.0, sqrt(1.5)], dtype=float
)


def _two_sided_p_normal(t_value: float) -> float:
    if not np.isfinite(t_value):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(float(t_value)) / sqrt(2.0)))))


def _one_sided_p_normal(t_value: float, *, greater: bool) -> float:
    if not np.isfinite(t_value):
        return float("nan")
    upper = 1.0 - 0.5 * (1.0 + erf(float(t_value) / sqrt(2.0)))
    return float(upper if greater else 1.0 - upper)


def _draw_weights(rng: np.random.Generator, size: int, kind: str) -> np.ndarray:
    if kind == "rademacher":
        return rng.choice((-1.0, 1.0), size=size)
    if kind == "webb":
        return rng.choice(_WEBB_POINTS, size=size)
    raise ValueError(f"unknown weight kind: {kind!r}")


def _grouped_sq_sum(u: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    """Sum over clusters of the squared within-cluster residual sums."""
    s = np.bincount(codes, weights=u, minlength=n_groups)
    return float(np.dot(s, s))


def _const_twoway_se(
    u: np.ndarray,
    sym_codes: np.ndarray,
    n_sym: int,
    date_codes: np.ndarray,
    n_date: int,
    inter_codes: np.ndarray,
    n_inter: int,
) -> float:
    """Two-way clustered SE of the mean for the constant-only model.

    Equivalent to ``mean_with_twoway_se`` but specialized and vectorized for the
    intercept-only regression so the bootstrap can afford many replications.
    """
    n = u.shape[0]
    if n == 0:
        return float("nan")
    v_sym = _grouped_sq_sum(u, sym_codes, n_sym)
    v_date = _grouped_sq_sum(u, date_codes, n_date)
    v_inter = _grouped_sq_sum(u, inter_codes, n_inter)
    var = (v_sym + v_date - v_inter) / (n * n)
    return float(sqrt(var)) if var > 0 else 0.0


def _const_oneway_se(u: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    n = u.shape[0]
    if n == 0:
        return float("nan")
    var = _grouped_sq_sum(u, codes, n_groups) / (n * n)
    return float(sqrt(var)) if var > 0 else 0.0


@dataclass
class BootstrapResult:
    mean: float
    se_analytic: float
    t_analytic: float
    p_analytic_two_sided: float
    p_analytic_one_sided: float
    p_bootstrap_two_sided: float
    p_bootstrap_one_sided: float
    ci_lo: float
    ci_hi: float
    n: int
    n_boot: int
    weights: str
    two_way: bool
    n_clusters_sym: int
    n_clusters_date: int
    alternative: str


def wild_cluster_bootstrap_mean(
    values: pd.Series | np.ndarray,
    symbols: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
    *,
    alternative: str = "greater",
    n_boot: int = 9999,
    weights: str = "webb",
    two_way: bool = True,
    ci_alpha: float = 0.05,
    seed: int = cfg.DEFAULT_SEED,
) -> BootstrapResult:
    """Wild cluster bootstrap for the panel mean of ``values`` against zero.

    ``alternative`` is one of ``{"greater", "less", "two-sided"}`` and selects
    which one-sided bootstrap p-value is reported in
    ``p_bootstrap_one_sided``; the two-sided p-value is always reported as well.
    The restricted (imposed-null) resampling gives the p-values; an unrestricted
    studentized resampling gives the percentile-t confidence interval.
    """
    frame = pd.concat(
        {"y": pd.Series(values).reset_index(drop=True),
         "s": pd.Series(symbols).reset_index(drop=True),
         "d": pd.Series(dates).reset_index(drop=True)},
        axis=1,
    ).dropna(subset=["y", "s", "d"])
    y = frame["y"].to_numpy(dtype=float)
    n = y.shape[0]
    sym_codes, sym_uni = pd.factorize(frame["s"])
    date_codes, date_uni = pd.factorize(frame["d"])
    n_sym = len(sym_uni)
    n_date = len(date_uni)
    inter_codes = pd.factorize(
        pd.Series(list(zip(sym_codes.tolist(), date_codes.tolist())))
    )[0]
    n_inter = int(inter_codes.max()) + 1 if n else 0

    nan_result = BootstrapResult(
        mean=float("nan"), se_analytic=float("nan"), t_analytic=float("nan"),
        p_analytic_two_sided=float("nan"), p_analytic_one_sided=float("nan"),
        p_bootstrap_two_sided=float("nan"), p_bootstrap_one_sided=float("nan"),
        ci_lo=float("nan"), ci_hi=float("nan"), n=n, n_boot=n_boot,
        weights=weights, two_way=two_way, n_clusters_sym=n_sym,
        n_clusters_date=n_date, alternative=alternative,
    )
    if n < 2:
        return nan_result

    def se_of(u: np.ndarray) -> float:
        if two_way:
            return _const_twoway_se(
                u, sym_codes, n_sym, date_codes, n_date, inter_codes, n_inter
            )
        return _const_oneway_se(u, date_codes, n_date)

    mean = float(np.mean(y))
    se_obs = se_of(y - mean)
    if not (se_obs > 0):
        return nan_result
    t_obs = mean / se_obs
    greater = alternative != "less"

    rng = np.random.default_rng(seed)
    t_restr = np.full(n_boot, np.nan)
    tc_unrestr = np.full(n_boot, np.nan)
    resid_unrestr = y - mean  # unrestricted residuals (recentered at mu_hat)
    for b in range(n_boot):
        if two_way:
            w = (_draw_weights(rng, n_sym, weights)[sym_codes]
                 * _draw_weights(rng, n_date, weights)[date_codes])
        else:
            w = _draw_weights(rng, n_date, weights)[date_codes]
        # Restricted (H0: mu = 0): restricted residual equals y.
        y_r = y * w
        mu_r = float(np.mean(y_r))
        se_r = se_of(y_r - mu_r)
        if se_r > 0:
            t_restr[b] = mu_r / se_r
        # Unrestricted (centered at mu_hat) for the studentized CI.
        e_w = resid_unrestr * w
        mu_u = float(np.mean(e_w))
        se_u = se_of(e_w - mu_u)
        if se_u > 0:
            tc_unrestr[b] = mu_u / se_u

    valid_r = t_restr[np.isfinite(t_restr)]
    if valid_r.size == 0:
        return nan_result
    b_eff = valid_r.size
    p_two = (1.0 + np.sum(np.abs(valid_r) >= abs(t_obs))) / (b_eff + 1.0)
    if greater:
        p_one = (1.0 + np.sum(valid_r >= t_obs)) / (b_eff + 1.0)
    else:
        p_one = (1.0 + np.sum(valid_r <= t_obs)) / (b_eff + 1.0)

    valid_c = tc_unrestr[np.isfinite(tc_unrestr)]
    if valid_c.size:
        q_lo = np.quantile(valid_c, ci_alpha / 2.0)
        q_hi = np.quantile(valid_c, 1.0 - ci_alpha / 2.0)
        ci_lo = mean - q_hi * se_obs
        ci_hi = mean - q_lo * se_obs
    else:
        ci_lo = ci_hi = float("nan")

    return BootstrapResult(
        mean=mean, se_analytic=se_obs, t_analytic=float(t_obs),
        p_analytic_two_sided=_two_sided_p_normal(t_obs),
        p_analytic_one_sided=_one_sided_p_normal(t_obs, greater=greater),
        p_bootstrap_two_sided=float(p_two),
        p_bootstrap_one_sided=float(p_one),
        ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        n=n, n_boot=b_eff, weights=weights, two_way=two_way,
        n_clusters_sym=n_sym, n_clusters_date=n_date, alternative=alternative,
    )


def paired_diff_bootstrap(
    treatment: pd.Series | np.ndarray,
    control: pd.Series | np.ndarray,
    symbols: pd.Series | np.ndarray,
    dates: pd.Series | np.ndarray,
    **kwargs,
) -> BootstrapResult:
    """Wild cluster bootstrap of the paired differential ``treatment - control``."""
    diff = pd.Series(treatment).reset_index(drop=True) - pd.Series(control).reset_index(drop=True)
    return wild_cluster_bootstrap_mean(diff, symbols, dates, **kwargs)


def max_t_union_test(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    symbol_col: str = "symbol",
    date_col: str = "date",
    alternative: str = "two-sided",
    n_boot: int = 2000,
    weights: str = "webb",
    two_way: bool = True,
    seed: int = cfg.DEFAULT_SEED,
) -> dict:
    """Bootstrap max-t union test of H0: ``E[value] = 0`` in every group.

    This is the multiplicity-aware test for an "effect in at least one group"
    alternative (used for the per-bin H2a/H2b matched-fill comparison). With
    ``alternative="greater"`` the statistic is the maximum signed studentized
    group mean (the registered "positive in at least one bin" alternative); with
    ``alternative="two-sided"`` it is the maximum absolute studentized group
    mean. A single cluster-weight draw is shared across all groups in each
    replication, so the null distribution of the maximum respects cross-group
    dependence. Returns the observed max statistic, the bootstrap p-value, and
    bookkeeping.
    """
    cols = [value_col, group_col, symbol_col, date_col]
    f = frame[cols].dropna()
    n = len(f)
    signed = alternative in ("greater", "less")
    sign = -1.0 if alternative == "less" else 1.0
    out = {"max_abs_t": float("nan"), "p_bootstrap": float("nan"),
           "n_groups": 0, "n": n, "weights": weights, "two_way": two_way,
           "alternative": alternative}
    if n < 2:
        return out
    y = f[value_col].to_numpy(dtype=float)
    sym_codes, sym_uni = pd.factorize(f[symbol_col])
    date_codes, date_uni = pd.factorize(f[date_col])
    n_sym, n_date = len(sym_uni), len(date_uni)
    inter_codes = pd.factorize(
        pd.Series(list(zip(sym_codes.tolist(), date_codes.tolist())))
    )[0]
    n_inter = int(inter_codes.max()) + 1

    group_idx: list[np.ndarray] = []
    for _, g in f.groupby(group_col, sort=True):
        pos = f.index.get_indexer(g.index)
        if pos.size:
            group_idx.append(pos)
    if not group_idx:
        return out

    def se_of(u: np.ndarray, sc: np.ndarray, dc: np.ndarray, ic: np.ndarray) -> float:
        if two_way:
            return _const_twoway_se(u, sc, n_sym, dc, n_date, ic, n_inter)
        return _const_oneway_se(u, dc, n_date)

    def max_abs_t(vals: np.ndarray) -> float:
        best = -np.inf
        seen = False
        for idx in group_idx:
            yi = vals[idx]
            mi = float(np.mean(yi))
            se = se_of(yi - mi, sym_codes[idx], date_codes[idx], inter_codes[idx])
            if se > 0:
                seen = True
                t = sign * mi / se
                best = max(best, t if signed else abs(t))
        return best if seen else float("nan")

    t_obs = max_abs_t(y)
    if not np.isfinite(t_obs):
        return out

    rng = np.random.default_rng(seed)
    exceed = 0
    eff = 0
    for _ in range(n_boot):
        if two_way:
            w = (_draw_weights(rng, n_sym, weights)[sym_codes]
                 * _draw_weights(rng, n_date, weights)[date_codes])
        else:
            w = _draw_weights(rng, n_date, weights)[date_codes]
        t_star = max_abs_t(y * w)  # restricted null: residual equals y
        if np.isfinite(t_star):
            eff += 1
            if t_star >= t_obs:
                exceed += 1
    out["max_abs_t"] = float(t_obs)
    out["n_groups"] = len(group_idx)
    out["n_boot"] = eff
    out["p_bootstrap"] = (1.0 + exceed) / (eff + 1.0) if eff else float("nan")
    return out


def block_bootstrap_statistic(
    panel: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], pd.Series | dict | float],
    *,
    cluster_col: str = "date",
    n_boot: int = 2000,
    seed: int = cfg.DEFAULT_SEED,
) -> pd.DataFrame:
    """Resample whole clusters with replacement and recompute ``statistic``.

    Returns one row per bootstrap replication; columns are the keys of the
    statistic's return value. ``statistic`` receives the resampled panel and
    must return a mapping (or Series) of named scalar outputs.
    """
    if panel.empty or cluster_col not in panel.columns:
        return pd.DataFrame()
    groups = {k: g.index.to_numpy() for k, g in panel.groupby(cluster_col, sort=True)}
    cluster_keys = list(groups.keys())
    n_clusters = len(cluster_keys)
    if n_clusters < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[cluster_keys[j]] for j in drawn])
        resampled = panel.loc[idx]
        out = statistic(resampled)
        if isinstance(out, pd.Series):
            out = out.to_dict()
        if out:
            rows.append(dict(out))
    return pd.DataFrame(rows)
