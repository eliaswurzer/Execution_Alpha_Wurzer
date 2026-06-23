"""
render_as_horizon_robustness.py -- AS-horizon summary to thesis assets.

Reads the validated ``as_horizon_summary.csv`` produced by the adverse-selection
horizon robustness aggregation and renders two copy-ready thesis assets in the
Journal-of-Finance style defined in the thesis preamble: a results figure
(PDF + PNG) and a LaTeX table snippet plus a figure snippet. It does not
recompute any panel metric and only reformats the already validated summary, so
it never re-triggers the aggregation acceptance gate.

Usage::

    python -m analysis.runners.render_as_horizon_robustness
    python -m analysis.runners.render_as_horizon_robustness \
        --summary-csv <path> --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from .. import config as cfg  # noqa: E402
from ..reporting.jof_latex import fmt_int, fmt_num, jof_figure, jof_table  # noqa: E402

_DEFAULT_SUMMARY = (
    cfg.ARTIFACTS_DIR / "as_horizon_robustness_20260619" / "as_horizon_summary.csv"
)
_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[3] / "thesis" / "figures" / "final_20260618_queue"
)
_FIG_INCLUDE_DIR = "figures/final_20260618_queue"
_SAMPLE_SPAN = "2018-07-02 to 2019-12-31"


def _save_fig(fig, out_dir: Path, stem: str) -> str:
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return pdf.name


def build_figure(summary: pd.DataFrame, out_dir: Path) -> str:
    df = summary.sort_values("horizon_seconds")
    horizons = df["horizon_seconds"].to_numpy(dtype=float)
    markout = df["as_markout_bps"].to_numpy(dtype=float)
    component = df["as_component_bps"].to_numpy(dtype=float)
    net_alpha = df["mean_net_alpha_vs_moc_bps"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax.axvline(30.0, color="0.8", linewidth=0.9, linestyle=":", zorder=0)
    ax.plot(horizons, markout, marker="o", linewidth=1.6, color="#1f77b4",
            label="Adverse-selection markout")
    ax.plot(horizons, component, marker="s", linewidth=1.6, color="#d62728",
            label="Adverse-selection component")
    ax.plot(horizons, net_alpha, marker="^", linewidth=1.4, linestyle="--",
            color="black", label="Net alpha vs. MOC")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{int(h)}" for h in horizons])
    ax.set_xlim(horizons.min() * 0.8, horizons.max() * 1.25)
    ax.set_xlabel("Adverse-selection horizon (seconds, log scale)")
    ax.set_ylabel("Basis points")
    ax.annotate("headline (30s)", xy=(30.0, ax.get_ylim()[1]), xytext=(0, -3),
                textcoords="offset points", ha="center", va="top",
                fontsize=8, color="0.4")
    ax.legend(fontsize=8, loc="center left")
    return _save_fig(fig, out_dir, "fig_as_horizon_robustness")


def build_table(summary: pd.DataFrame) -> str:
    df = summary.sort_values("horizon_seconds")
    body: list[str] = []
    for row in df.itertuples(index=False):
        h = int(row.horizon_seconds)
        label = f"{h}s (headline)" if h == 30 else f"{h}s"
        body.append(
            f"{label} & "
            f"{fmt_num(row.mean_net_alpha_vs_moc_bps, nd=2)} & "
            f"{fmt_num(row.primary_t, nd=2)} & "
            f"{fmt_num(row.mean_fill_rate, nd=3)} & "
            f"{fmt_num(row.as_markout_bps, nd=2)} & "
            f"{fmt_num(row.as_component_bps, nd=2)} & "
            f"{fmt_int(row.n)} \\\\"
        )
    legend = (
        "This table reports the S3-full net-execution-alpha differential against "
        "the Market-on-Close benchmark across five post-fill adverse-selection "
        "horizons, holding the headline one-percent parent size and the "
        "Window~B arrival schedule fixed. The thirty-second row is the validated "
        "headline run, and the remaining four horizons recompute the "
        "adverse-selection diagnostics on the same set of parent orders. The "
        "net-alpha differential, the adverse-selection markout, and the "
        "adverse-selection component are in basis points. The sample covers "
        f"{_SAMPLE_SPAN} and contains 187,309 parent orders."
    )
    notes = (
        "Net alpha is the differential of S3-full against the MOC benchmark, "
        "anchored to the official closing price, so it does not depend on the "
        "post-fill horizon and is reported here to confirm invariance. The "
        "associated $t$-statistic uses two-way clustered standard errors by "
        "symbol and date. The adverse-selection markout is the negated mean "
        "signed post-fill mid-quote drift over the stated horizon conditional on "
        "a passive fill, and the adverse-selection component is the corresponding "
        "fill-rate-weighted mean across all parent orders. Both are "
        "horizon-dependent diagnostics of realized post-fill drift and are not "
        "deducted again from the close-anchored net-alpha differential."
    )
    return jof_table(
        caption="Adverse-Selection Horizon Robustness at the Headline Parent Size",
        label="tab:as-horizon-robustness",
        legend=legend,
        column_format="lrrrrrr",
        header=(
            r"Horizon & Net alpha vs.\ MOC & $t$ & Fill rate & "
            r"AS markout & AS component & $N$"
        ),
        body_lines=body,
        notes=notes,
        resize=True,
        include_stars_note=False,
    )


def build_figure_snippet() -> str:
    legend = (
        "This figure plots three horizon-dependent quantities for the S3-full "
        "headline cell, namely the Window~B one-percent parent size, against the "
        "post-fill adverse-selection horizon on a logarithmic axis. The "
        "adverse-selection markout rises to a peak near thirty to sixty seconds "
        "and reverts substantially by three hundred seconds, which indicates that "
        "the realized post-fill drift against the passive provider is largely "
        "transitory. The adverse-selection component is the fill-rate-weighted "
        "analogue across all parent orders. The net-alpha differential against "
        "MOC is anchored to the close and is flat across horizons by "
        f"construction. The sample covers {_SAMPLE_SPAN}."
    )
    return jof_figure(
        graphics_file=f"{_FIG_INCLUDE_DIR}/fig_as_horizon_robustness.pdf",
        caption="Adverse-Selection Markout across Measurement Horizons",
        label="fig:as-horizon-robustness",
        legend=legend,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=_DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    summary = pd.read_csv(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_name = build_figure(summary, args.out_dir)
    table_tex = build_table(summary)
    figure_tex = build_figure_snippet()

    tab_path = args.out_dir / "tab_as_horizon_robustness.tex"
    fig_path = args.out_dir / "fig_as_horizon_robustness.tex"
    tab_path.write_text(table_tex + "\n", encoding="utf-8")
    fig_path.write_text(figure_tex + "\n", encoding="utf-8")

    print(f"Wrote figure : {args.out_dir / fig_name}")
    print(f"Wrote table  : {tab_path}")
    print(f"Wrote figtex : {fig_path}")
    print("\n--- table snippet ---\n")
    print(table_tex)


if __name__ == "__main__":
    main()
