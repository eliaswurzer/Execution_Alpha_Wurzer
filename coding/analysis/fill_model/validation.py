"""
validation.py -- Out-of-sample Diagnostik fuer das Cox-PH Fill-Modell
(Thesis §4.2.6): Brier-Score Decomposition + AUC-ROC bei h = 5 min.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cox_ph import TieredFillModel


@dataclass
class ValidationReport:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    auc: float
    n: int
    # Calibration-level diagnostics consumed by the calibration status gate.
    # ``observed`` is the empirical fill rate at the horizon, ``mean_pred`` the
    # mean predicted fill probability, and ``base_fill_s0`` the implied baseline
    # fill ``1 - S0(h)`` for models with an explicit lifelines baseline (Cox);
    # NaN for models without one (KM/XGB). A collapsed Cox baseline shows up as
    # ``base_fill_s0`` ~ 0 and ``mean_pred`` far below ``observed``.
    observed: float = float("nan")
    mean_pred: float = float("nan")
    base_fill_s0: float = float("nan")


def _brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float, float, float]:
    """Murphy (1973) decomposition of the Brier score.

    ``BS = REL - RES + UNC`` -- reliability (calibration error, klein ist gut),
    resolution (discrimination, gross ist gut) und unvermeidbare Unsicherheit.
    """
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    bs = float(np.mean((y_prob - y_true) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)

    o = float(np.mean(y_true))
    unc = o * (1 - o)

    rel = 0.0
    res = 0.0
    for b in range(n_bins):
        m = ids == b
        if not m.any():
            continue
        fk = float(np.mean(y_prob[m]))
        ok = float(np.mean(y_true[m]))
        nk = int(m.sum()) / len(y_true)
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - o) ** 2

    return bs, rel, res, unc


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney-U-basierte AUC (ohne sklearn)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    # mittlere Raenge bei Ties
    pos = 0
    while pos < len(order):
        j = pos
        while j + 1 < len(order) and s[order[j + 1]] == s[order[pos]]:
            j += 1
        mean_rank = 0.5 * (pos + j) + 1
        for k in range(pos, j + 1):
            ranks[order[k]] = mean_rank
        pos = j + 1
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_tiered_model(
    model: TieredFillModel,
    eval_panel: pd.DataFrame,
    horizon_seconds: int,
) -> dict[int, ValidationReport]:
    """Ein Report pro Tier.

    ``eval_panel`` muss Spalten ``symbol``, ``duration``, ``event`` plus
    die Covariates enthalten.
    """
    panel = eval_panel.copy()
    panel["tier"] = panel["symbol"].map(model.symbol_to_tier)
    panel = panel.dropna(subset=["tier"])
    panel["tier"] = panel["tier"].astype(int)

    reports: dict[int, ValidationReport] = {}
    for tier, grp in panel.groupby("tier"):
        if tier not in model.models:
            continue
        m = model.models[int(tier)]
        y_true = (grp["event"].to_numpy(dtype=int) & (grp["duration"].to_numpy(dtype=float) <= horizon_seconds).astype(int))
        covariates = list(getattr(m, "covariates", []) or [])
        if covariates:
            # Cox/XGB consume numerical covariates only.
            xdf = grp[[c for c in covariates if c in grp.columns]].copy()
            for c in covariates:
                if c not in xdf.columns:
                    xdf[c] = 0.0
        else:
            # KM uses stratification variables such as t0 and limit_offset_bps.
            xdf = grp.copy()
        y_prob = np.asarray(m.fill_probability(horizon_seconds, xdf), dtype=float)
        bs, rel, res, unc = _brier_decomposition(y_true, y_prob)
        auc = _auc_roc(y_true, y_prob)
        observed = float(np.mean(y_true)) if len(y_true) else float("nan")
        mean_pred = float(np.nanmean(y_prob)) if len(y_prob) else float("nan")
        reports[int(tier)] = ValidationReport(
            brier=bs, reliability=rel, resolution=res, uncertainty=unc,
            auc=auc, n=len(grp),
            observed=observed, mean_pred=mean_pred,
            base_fill_s0=_baseline_fill(m, horizon_seconds),
        )
    return reports


def _baseline_fill(model, horizon_seconds: float) -> float:
    """Implied baseline fill ``1 - S0(h)`` for models with an explicit lifelines
    baseline (Cox). Returns NaN for models without one (KM/XGB)."""
    fitter = getattr(model, "fitter", None)
    bs = getattr(fitter, "baseline_survival_", None) if fitter is not None else None
    if bs is None:
        return float("nan")
    try:
        bst = bs.index.to_numpy(dtype=float)
        bsv = bs.iloc[:, 0].to_numpy(dtype=float)
        idx = int(np.searchsorted(bst, horizon_seconds, side="right")) - 1
        s0 = float(bsv[idx]) if idx >= 0 else 1.0
        return float(1.0 - s0)
    except Exception:  # pragma: no cover - diagnostic best-effort
        return float("nan")


# ---------------------------------------------------------------------------
# Calibration-Plot (Decile mit Wilson-Konfidenzintervall) -- P4.3
# ---------------------------------------------------------------------------

def _wilson_interval(
    successes: int, n: int, z: float = 1.96,
) -> tuple[float, float]:
    """Wilson-Score-Konfidenzintervall fuer eine Binomialproportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * float(np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)))
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_plot(
    model: TieredFillModel,
    eval_panel: pd.DataFrame,
    horizon_seconds: int,
    out_dir,
    n_bins: int = 10,
) -> dict:
    """Erzeugt einen Decile-level Predicted-vs-Realized Scatter mit
    Wilson-Konfidenzbaendern. Speichert PNG unter
    ``<out_dir>/diagnostics/calibration_tier_<tier>.png`` und liefert die
    Bin-Tabellen als dict zurueck.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    panel = eval_panel.copy()
    panel["tier"] = panel["symbol"].map(model.symbol_to_tier)
    panel = panel.dropna(subset=["tier"])
    panel["tier"] = panel["tier"].astype(int)

    tables: dict[int, pd.DataFrame] = {}
    for tier, grp in panel.groupby("tier"):
        if tier not in model.models:
            continue
        m = model.models[int(tier)]
        y_true = (
            grp["event"].to_numpy(dtype=int)
            & (grp["duration"].to_numpy(dtype=float) <= horizon_seconds).astype(int)
        )
        xdf = grp[[c for c in m.covariates if c in grp.columns]].copy()
        for c in m.covariates:
            if c not in xdf.columns:
                xdf[c] = 0.0
        y_prob = np.asarray(m.fill_probability(horizon_seconds, xdf), dtype=float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = ids == b
            if not mask.any():
                continue
            n = int(mask.sum())
            succ = int(y_true[mask].sum())
            mean_pred = float(np.mean(y_prob[mask]))
            obs = succ / n
            lo, hi = _wilson_interval(succ, n)
            rows.append({
                "bin": b, "n": n, "mean_pred": mean_pred,
                "observed": obs, "ci_low": lo, "ci_high": hi,
            })
        tbl = pd.DataFrame(rows)
        tables[int(tier)] = tbl
        if tbl.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            tbl["mean_pred"], tbl["observed"],
            yerr=[tbl["observed"] - tbl["ci_low"], tbl["ci_high"] - tbl["observed"]],
            fmt="o", capsize=3, color="#4472C4",
            label=f"tier {tier} (n={int(tbl['n'].sum())})",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.set_xlabel("Predicted fill probability (decile mean)")
        ax.set_ylabel("Observed fill rate (Wilson 95%)")
        ax.set_title(f"Cox-PH Calibration -- Tier {tier} (h={horizon_seconds}s)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(diag / f"calibration_tier_{tier}.png", dpi=120)
        plt.close(fig)
        tbl.to_csv(diag / f"calibration_tier_{tier}.csv", index=False)
    return tables
