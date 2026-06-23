"""
plots.py – Generic plotting helpers with consistent styling.

Used across H1, H2, H3 analyses.  All functions return the
``(fig, ax)`` tuple so callers can customise further.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

from .thesis_figures.style import (
    CATEGORICAL,
    THEME,
    apply_thesis_style,
    color_for,
    emphasis_color,
)

# ------------------------------------------------------------------
# Global style
# ------------------------------------------------------------------
apply_thesis_style()
PALETTE = list(CATEGORICAL)


def _apply_defaults(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", labelsize=8)


# ------------------------------------------------------------------
# Box / violin plot
# ------------------------------------------------------------------

def boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 5),
    showfliers: bool = True,
    hue: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Seaborn boxplot with consistent styling.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str   – column names for the categorical and numeric axes.
    title, xlabel, ylabel : str
    figsize : tuple
    showfliers : bool – whether to show outlier points.
    hue : str, optional – sub‑grouping variable.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_hue = hue if hue is not None else x
    n_groups = df[plot_hue].nunique()
    sns.boxplot(data=df, x=x, y=y, hue=plot_hue, ax=ax,
                palette=PALETTE[:n_groups], showfliers=showfliers, legend=False)
    _apply_defaults(ax, title, xlabel or x, ylabel or y)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Histogram / KDE
# ------------------------------------------------------------------

def histogram(
    series: pd.Series,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 40,
    kde: bool = True,
    figsize: tuple = (8, 5),
    vline: Optional[float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram with optional KDE overlay and vertical reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(series.dropna(), bins=bins, kde=kde, ax=ax, color=PALETTE[0])
    if vline is not None:
        ax.axvline(vline, color=THEME["rose"], ls="--", lw=1.5, label=f"ref = {vline:.4f}")
        ax.legend()
    _apply_defaults(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Scatter with regression line
# ------------------------------------------------------------------

def scatter_reg(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 5),
    hue: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot with a linear regression line."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.5, "s": 30},
                line_kws={"color": THEME["orange"]}, color=THEME["purple"])
    _apply_defaults(ax, title, xlabel or x, ylabel or y)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Grouped bar chart (mean ± se)
# ------------------------------------------------------------------

def grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    title: str = "",
    ylabel: str = "",
    figsize: tuple = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of *value_col* mean by *group_col* with error bars."""
    summary = df.groupby(group_col, observed=True)[value_col].agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=figsize)
    summary["mean"].plot.bar(yerr=summary["sem"], ax=ax, color=PALETTE[:len(summary)],
                             capsize=4, edgecolor=emphasis_color(), linewidth=0.5)
    ax.axhline(0, color=THEME["gray"], lw=0.8, ls="--")
    _apply_defaults(ax, title, group_col, ylabel or value_col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Line plot (time-series)
# ------------------------------------------------------------------

def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 5),
    marker: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Time-series line plot with optional grouping.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str -- column names for the x and y axes.
    hue : str, optional -- grouping variable for multiple lines.
    title, xlabel, ylabel : str
    figsize : tuple
    marker : str, optional -- marker style (e.g. ``"o"``).

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    if hue:
        for i, (name, grp) in enumerate(df.groupby(hue, sort=True)):
            color = PALETTE[i % len(PALETTE)]
            ax.plot(grp[x], grp[y], label=name, color=color,
                    marker=marker, markersize=4, linewidth=1.2)
        ax.legend(title=hue, fontsize=9)
    else:
        ax.plot(df[x], df[y], color=PALETTE[0],
                marker=marker, markersize=4, linewidth=1.2)
    _apply_defaults(ax, title, xlabel or x, ylabel or y)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# RAEAR-Curve-Plot (P4.4)
# ------------------------------------------------------------------

def raear_curve_plot(
    tev_table: pd.DataFrame,
    *,
    eta_grid: Optional[np.ndarray] = None,
    title: str = "RAEAR(eta) -- Strategie-Vergleich",
    figsize: tuple = (8, 5.5),
):
    """Plottet RAEAR(eta) = mean_alpha - eta * TEV fuer alle Strategien.

    ``tev_table`` braucht Spalten ``strategy``, ``mean_alpha``, ``tev``.

    Zusatz: vertikale Linien an den paarweisen Break-Even-Etas zwischen
    der besten Strategie und jeder anderen (= eta wo zwei Strategien sich
    schneiden = (a_i - a_j) / (TEV_i - TEV_j)).
    """
    import matplotlib.pyplot as plt

    if eta_grid is None:
        eta_grid = np.linspace(0.0, 0.5, 100)

    fig, ax = plt.subplots(figsize=figsize)

    rows = list(tev_table[["strategy", "mean_alpha", "tev"]].itertuples(index=False))
    rows.sort(key=lambda r: r.mean_alpha, reverse=True)

    for i, r in enumerate(rows):
        alpha_curve = r.mean_alpha - eta_grid * r.tev
        ax.plot(eta_grid, alpha_curve, label=r.strategy,
                color=color_for("strategy", str(r.strategy), i), lw=1.6)

    # Paarweise Break-Even-Etas (eta* zwischen Strategie 0 und jeder anderen)
    if len(rows) >= 2:
        a0, t0 = rows[0].mean_alpha, rows[0].tev
        for j, r in enumerate(rows[1:], start=1):
            denom = r.tev - t0
            if abs(denom) < 1e-12:
                continue
            eta_be = (r.mean_alpha - a0) / denom
            if np.isfinite(eta_be) and eta_grid.min() <= eta_be <= eta_grid.max():
                ax.axvline(eta_be, color=THEME["gray"], ls="--", alpha=0.5, lw=0.9)
                ax.text(eta_be, ax.get_ylim()[1] * 0.95,
                        f"  η*({rows[0].strategy}/{r.strategy})={eta_be:.3f}",
                        rotation=90, va="top", ha="left", fontsize=8, color=THEME["gray"])

    ax.axhline(0, color=emphasis_color(), lw=0.7, alpha=0.5)
    _apply_defaults(ax, title, "η (benchmark-risk aversion)", "RAEAR (bps)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig, ax
