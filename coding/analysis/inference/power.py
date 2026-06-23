"""
power.py -- Design-based power and minimum-detectable-effect helpers.

A non-rejection is only informative if the design could plausibly have detected
an economically meaningful effect. These helpers translate the achieved
clustered standard error into the smallest effect the test could reject at a
given significance level and power, so that null findings can be reported as
"the design rules out effects larger than X" rather than as bare
fail-to-reject statements. This is a design-based minimum detectable effect; it
deliberately avoids the post-hoc "observed power" computed from the realized
estimate, which adds no information beyond the p-value.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam rational approximation)."""
    if not (0.0 < p < 1.0):
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def minimum_detectable_effect(
    se: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
) -> float:
    """Smallest true effect detectable at ``alpha`` with probability ``power``.

    ``MDE = (z_{1-alpha} + z_{power}) * se`` for a one-sided test, with
    ``z_{1-alpha/2}`` substituted for a two-sided test. Returned in the same
    units as ``se`` (basis points for the alpha differentials).
    """
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    z_alpha = _norm_ppf(1.0 - alpha) if one_sided else _norm_ppf(1.0 - alpha / 2.0)
    z_power = _norm_ppf(power)
    if not (np.isfinite(z_alpha) and np.isfinite(z_power)):
        return float("nan")
    return float((z_alpha + z_power) * se)


def power_at_effect(
    effect: float,
    se: float,
    *,
    alpha: float = 0.05,
    one_sided: bool = True,
) -> float:
    """Probability of rejecting H0 when the true effect equals ``effect``."""
    if not np.isfinite(se) or se <= 0 or not np.isfinite(effect):
        return float("nan")
    z_alpha = _norm_ppf(1.0 - alpha) if one_sided else _norm_ppf(1.0 - alpha / 2.0)
    ncp = abs(effect) / se
    # Normal-approximation power of the (one- or two-sided) z-test.
    upper = 0.5 * (1.0 + erf((ncp - z_alpha) / sqrt(2.0)))
    if one_sided:
        return float(upper)
    lower = 0.5 * (1.0 + erf((-ncp - z_alpha) / sqrt(2.0)))
    return float(upper + lower)
