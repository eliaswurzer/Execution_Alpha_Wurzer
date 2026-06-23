"""Preliminary thesis-reporting templates for validated hypothesis artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    name: str
    source: str
    columns: tuple[str, ...]
    caption: str


TABLE_SPECS: tuple[TableSpec, ...] = (

    TableSpec(
        "adv_spread_bucket_summary",
        "liquidity_buckets/h1_2018/adv_spread_bucket_summary.csv",
        ("adv_bucket", "spread_bucket", "adv_spread_bucket", "n_symbols", "median_adv_dollar", "median_quoted_spread_bps"),
        "H1 2018 fixed ADV x spread bucket construction summary.",
    ),
    TableSpec(
        "h1_adv_spread_grid",
        "tables/analysis/h1_strategy_by_adv_spread_bucket.csv",
        ("adv_bucket", "spread_bucket", "adv_spread_bucket", "strategy", "n", "moc_diff_bps", "fill_rate", "as_cost_bps"),
        "H1 strategy performance by fixed ADV x spread bucket.",
    ),
    TableSpec(
        "h1_strategy_comparison",
        "hypotheses/h1/h1_tev.csv",
        ("strategy", "mean_alpha", "tev", "n"),
        "H1 strategy-level net alpha and tracking-error variance.",
    ),
    TableSpec(
        "h1_primary_test",
        "hypotheses/h1/h1_primary_ttest.csv",
        ("label", "mean", "se", "t", "p", "n"),
        "H1 paired primary test against the MOC benchmark.",
    ),
    TableSpec(
        "h2_marginal_signal_effects",
        "hypotheses/h2/h2_pooled.csv",
        ("label", "mean", "se", "t", "p", "n", "matching_metric"),
        "H2 pooled marginal signal effects.",
    ),
    TableSpec(
        "h3_risk_adjusted_ranking",
        "hypotheses/h3/h3_raear.csv",
        ("strategy", "mean_alpha", "tev", "te", "ir", "raear_eta_0_01"),
        "H3 risk-adjusted execution-alpha ranking.",
    ),
    TableSpec(
        "side_split",
        "tables/analysis/side_split.csv",
        ("side", "strategy", "mean_alpha_bps", "n"),
        "Strategy performance split by parent-order side.",
    ),
    TableSpec(
        "tier_split",
        "hypotheses/h1/h1_subgroup_tier.csv",
        ("group", "label", "mean", "se", "t", "p", "n"),
        "H1 paired effect by liquidity tier.",
    ),
    TableSpec(
        "sector_split",
        "tables/analysis/sector_split.csv",
        ("sector", "strategy", "mean_alpha_bps", "n"),
        "Strategy performance split by sector.",
    ),
    TableSpec(
        "static_posting_curve",
        "posting_curve_summary.csv",
        ("side", "tier", "limit_offset_bps", "n", "fill_probability", "mean_value_bps"),
        "Static posting-distance fill probability and value diagnostic.",
    ),
)

FIGURE_SPECS: tuple[tuple[str, str], ...] = (
    ("fig_h1_strategy_comparison.png", "H1 strategy comparison"),
    ("fig_h2_marginal_signal_effects.png", "H2 marginal signal effects"),
    ("fig_h3_risk_adjusted_ranking.png", "H3 risk-adjusted ranking"),
    ("fig_side_split.png", "Side split"),
    ("fig_tier_split.png", "Tier split"),
    ("fig_h1_adv_spread_heatmap.png", "H1 ADV x spread heatmap"),
    ("fig_fill_rate_adv_spread_heatmap.png", "Fill rate by ADV x spread bucket"),
    ("fig_sector_split.png", "Sector split"),
    ("fig_static_posting_curve.png", "Static posting distance diagnostics"),
)


def _read_or_placeholder(run_root: Path | None, spec: TableSpec) -> pd.DataFrame:
    if run_root is not None:
        path = Path(run_root) / spec.source
        if path.exists():
            try:
                frame = pd.read_csv(path)
                for col in spec.columns:
                    if col not in frame.columns:
                        frame[col] = pd.NA
                return frame[list(spec.columns)]
            except Exception:
                pass
    return pd.DataFrame(columns=list(spec.columns))


def _latex_table(frame: pd.DataFrame, spec: TableSpec) -> str:
    latex = frame.to_latex(index=False, escape=True) if not frame.empty else pd.DataFrame(columns=spec.columns).to_latex(index=False, escape=True)
    return "\n".join([
        f"% {spec.caption}",
        "\\begin{table}[htbp]",
        "\\centering",
        latex.strip(),
        f"\\caption{{{spec.caption}}}",
        f"\\label{{tab:{spec.name}}}",
        "\\end{table}",
        "",
    ])


def _save_placeholder_figure(path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, "Pending validated final-run artifacts", ha="center", va="center", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_preliminary_templates(
    out_dir: Path,
    *,
    run_root: Path | None = None,
    posting_summary: pd.DataFrame | None = None,
    title: str = "Preliminary Results Template",
) -> dict[str, str]:
    """Write Markdown, LaTeX, table CSVs and placeholder figures.

    Empty tables are deliberate placeholders and must not be interpreted as
    empirical results. When ``run_root`` contains validated CSV artifacts, the
    matching tables are populated from those files only.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    latex_dir = out_dir / "latex"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    markdown_lines = [
        f"# {title}",
        "",
        "This file is a reporting scaffold. Empty cells indicate missing or not-yet-validated artifacts, not empirical zeroes.",
        "Every populated number must trace to validated final-run artifacts before thesis insertion.",
        "",
    ]
    all_latex: list[str] = []
    outputs: dict[str, str] = {}

    for spec in TABLE_SPECS:
        frame = posting_summary.copy() if spec.name == "static_posting_curve" and posting_summary is not None else _read_or_placeholder(run_root, spec)
        for col in spec.columns:
            if col not in frame.columns:
                frame[col] = pd.NA
        frame = frame[list(spec.columns)]
        csv_path = tables_dir / f"{spec.name}.csv"
        tex_path = latex_dir / f"{spec.name}.tex"
        frame.to_csv(csv_path, index=False)
        tex = _latex_table(frame, spec)
        tex_path.write_text(tex, encoding="utf-8")
        all_latex.append(tex)
        markdown_lines.extend([
            f"## {spec.caption}",
            "",
            frame.head(20).to_markdown(index=False),
            "",
        ])
        outputs[f"table_{spec.name}"] = str(csv_path)
        outputs[f"latex_{spec.name}"] = str(tex_path)

    for filename, caption in FIGURE_SPECS:
        fig_path = figures_dir / filename
        if filename == "fig_static_posting_curve.png" and posting_summary is not None and not posting_summary.empty:
            from .static_posting_curve import save_posting_curve_figure
            save_posting_curve_figure(posting_summary, fig_path)
        else:
            _save_placeholder_figure(fig_path, caption)
        markdown_lines.extend([f"![{caption}](figures/{filename})", ""])
        outputs[f"figure_{filename}"] = str(fig_path)

    md_path = out_dir / "README_preliminary_reporting_template.md"
    combined_tex_path = latex_dir / "preliminary_results_tables.tex"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    combined_tex_path.write_text("\n".join(all_latex), encoding="utf-8")
    outputs["markdown"] = str(md_path)
    outputs["latex_combined"] = str(combined_tex_path)
    return outputs

