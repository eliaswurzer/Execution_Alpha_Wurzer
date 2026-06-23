"""Consolidated test registry.

A single machine-generated enumeration of every statistical test the thesis
emits, with its confirmatory-or-exploratory role, its correction family, and the
p-value that drives the reject decision. The registry is the artifact that makes
the multiple-testing discipline auditable: a reader can see the full test
surface in one place and verify that each exploratory family is corrected and
that the confirmatory tests are the pre-registered ones.

The registry does not run any new test; it normalizes the outputs already
produced by the hypothesis runners and the supplementary economic tests into one
long table.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as st_cfg

REGISTRY_COLUMNS = [
    "test_id", "hypothesis", "role", "family", "label", "n",
    "estimate", "se", "p_analytic_two_sided", "p_one_sided", "p_bootstrap",
    "correction", "p_adjusted", "p_decision", "reject_5pct",
]


def _two_sided_p(t: float) -> float:
    if t is None or not np.isfinite(t):
        return float("nan")
    return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(float(t)) / sqrt(2.0)))))


def _one_sided_p(t: float, greater: bool = True) -> float:
    if t is None or not np.isfinite(t):
        return float("nan")
    upper = 1.0 - 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))
    return float(upper if greater else 1.0 - upper)


def _g(row, *keys, default=float("nan")):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def _row(
    *, test_id, hypothesis, role, family, label, n=np.nan, estimate=np.nan,
    se=np.nan, p_two=np.nan, p_one=np.nan, p_boot=np.nan, correction="none",
    p_adjusted=np.nan, p_decision=np.nan,
) -> dict:
    decision = bool(np.isfinite(p_decision) and p_decision < 0.05)
    return {
        "test_id": test_id, "hypothesis": hypothesis, "role": role,
        "family": family, "label": label, "n": n, "estimate": estimate,
        "se": se, "p_analytic_two_sided": p_two, "p_one_sided": p_one,
        "p_bootstrap": p_boot, "correction": correction,
        "p_adjusted": p_adjusted, "p_decision": p_decision,
        "reject_5pct": decision,
    }


def _h1_rows(headline_run: Path) -> list[dict]:
    rows: list[dict] = []
    h1 = Path(headline_run) / "hypotheses" / "h1"
    primary_path = h1 / "h1_primary_ttest.csv"
    if primary_path.exists():
        pr = pd.read_csv(primary_path)
        if not pr.empty:
            r = pr.iloc[0].to_dict()
            t = float(_g(r, "t"))
            p_one = float(_g(r, "p_one_sided", default=_one_sided_p(t, True)))
            rows.append(_row(
                test_id="H1_primary", hypothesis="H1", role="confirmatory",
                family="primary_H1", label=str(_g(r, "label", default="S3_FULL-S0_MOC:B")),
                n=_g(r, "n"), estimate=_g(r, "mean"), se=_g(r, "se"),
                p_two=_g(r, "p_value", default=_two_sided_p(t)), p_one=p_one,
                correction="none", p_decision=p_one,
            ))
    for sub_path in sorted(h1.glob("h1_subgroup_*.csv")):
        sub = pd.read_csv(sub_path)
        if sub.empty:
            continue
        fam = sub_path.stem.replace("h1_subgroup_", "subgroup_")
        for _, r in sub.iterrows():
            t = float(_g(r, "t"))
            p_holm = _g(r, "p_holm")
            rows.append(_row(
                test_id=f"{fam}:{_g(r, 'level', default='')}", hypothesis="H1",
                role="exploratory", family=fam,
                label=f"{_g(r, 'group', default='')}={_g(r, 'level', default='')}",
                n=_g(r, "n"), estimate=_g(r, "mean"), se=_g(r, "se"),
                p_two=_g(r, "p_value", default=_two_sided_p(t)),
                p_one=_g(r, "p_one_sided", default=_one_sided_p(t, True)),
                correction="holm", p_adjusted=p_holm, p_decision=p_holm,
            ))
    return rows


def _h2_rows(h2: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if h2 is None or h2.empty:
        return rows
    confirmatory = {"OFI_marginal", "IMB_marginal"}
    for _, r in h2.iterrows():
        label = str(_g(r, "label", default=""))
        role = "confirmatory" if label in confirmatory else "exploratory"
        # The registered H2a/H2b family is one-sided, so the decision uses the
        # one-sided Holm-adjusted p-value when available.
        p_holm_one = _g(r, "p_holm_one_sided")
        p_dec = p_holm_one if np.isfinite(p_holm_one) else _g(r, "p_holm")
        rows.append(_row(
            test_id=f"H2:{label}", hypothesis="H2", role=role,
            family="H2_pooled_signal", label=label, n=_g(r, "n"),
            estimate=_g(r, "mean"), se=_g(r, "se_twoway"),
            p_two=_g(r, "p_value"), p_one=_g(r, "p_one_sided"),
            correction="holm_one_sided", p_adjusted=p_dec, p_decision=p_dec,
        ))
    return rows


def _h2_union_rows(union: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if union is None or union.empty:
        return rows
    for _, r in union.iterrows():
        label = str(_g(r, "label", default=""))
        p_boot = _g(r, "p_bootstrap")
        rows.append(_row(
            test_id=f"H2_union:{label}", hypothesis="H2", role="exploratory",
            family="H2_per_bin_union", label=f"{label} (max-t over bins)",
            n=_g(r, "n"), estimate=_g(r, "max_abs_t"),
            p_boot=p_boot, correction="bootstrap_max_t", p_decision=p_boot,
        ))
    return rows


def _fill_spec_rows(economic: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if economic is None or economic.empty:
        return rows
    for _, r in economic.iterrows():
        is_head = bool(_g(r, "is_headline", default=False))
        role = "confirmatory" if is_head else "exploratory"
        p_holm = _g(r, "p_holm")
        # Headline decision uses the registered one-sided test, preferring the
        # wild cluster bootstrap p-value (the same statistic that drives the
        # headline stars in the renderer) and falling back to the analytic
        # one-sided p; robustness rows use the Holm-adjusted family p-value.
        if is_head:
            correction = "bootstrap_one_sided"
            p_dec = _g(r, "p_bootstrap_one_sided", "p_one_sided")
        else:
            correction = "holm"
            p_dec = p_holm
        rows.append(_row(
            test_id=f"fillspec:{_g(r, 'spec', default='')}", hypothesis="H1",
            role=role, family="fill_spec_robustness",
            label=str(_g(r, "label", default="")), n=_g(r, "n"),
            estimate=_g(r, "mean_net_alpha_vs_moc_bps"), se=_g(r, "se_twoway"),
            p_two=_g(r, "p_value"), p_one=_g(r, "p_one_sided"),
            p_boot=_g(r, "p_bootstrap_one_sided"), correction=correction,
            p_adjusted=p_holm, p_decision=p_dec,
        ))
    return rows


def _paired_rows(paired: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if paired is None or paired.empty:
        return rows
    for _, r in paired.iterrows():
        p_holm = _g(r, "p_holm")
        rows.append(_row(
            test_id=f"paired:{_g(r, 'spec', default='')}:{_g(r, 'metric', default='')}",
            hypothesis="H1", role="exploratory",
            family=str(_g(r, "family", default="alt_vs_queue")),
            label=f"{_g(r, 'label', default='')} ({_g(r, 'metric', default='')})",
            n=_g(r, "n"), estimate=_g(r, "mean_diff"), se=_g(r, "se_twoway"),
            p_two=_g(r, "p_value"), correction="holm",
            p_adjusted=p_holm, p_decision=p_holm,
        ))
    return rows


def _h3_rows(rank_stability: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if rank_stability is None or rank_stability.empty:
        return rows
    r = rank_stability.iloc[0]
    rows.append(_row(
        test_id="H3_rank_stability", hypothesis="H3", role="exploratory",
        family="H3_rank", label="P(RAEAR rank flip across eta)",
        n=_g(r, "n_boot"), estimate=_g(r, "p_raear_rank_flip_across_eta"),
        correction="block_bootstrap",
    ))
    rows.append(_row(
        test_id="H3_ir_ranking_preserved", hypothesis="H3", role="exploratory",
        family="H3_rank", label="P(IR ranking preserved)",
        n=_g(r, "n_boot"), estimate=_g(r, "p_ir_ranking_preserved"),
        correction="block_bootstrap",
    ))
    return rows


def build_test_registry(
    *,
    economic: pd.DataFrame | None = None,
    paired: pd.DataFrame | None = None,
    h2: pd.DataFrame | None = None,
    h2_union: pd.DataFrame | None = None,
    h3_rank_stability: pd.DataFrame | None = None,
    headline_run: Path | None = None,
) -> pd.DataFrame:
    """Assemble the consolidated registry of every emitted statistical test."""
    headline_run = Path(headline_run or st_cfg.HEADLINE_RUN)
    rows: list[dict] = []
    rows += _h1_rows(headline_run)
    rows += _fill_spec_rows(economic)
    rows += _paired_rows(paired)
    rows += _h2_rows(h2)
    rows += _h2_union_rows(h2_union)
    rows += _h3_rows(h3_rank_stability)
    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    return out
