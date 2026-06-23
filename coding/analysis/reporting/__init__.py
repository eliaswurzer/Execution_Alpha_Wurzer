"""Reporting helpers for plots, tables, and thesis scaffolds."""

from .plots import (
    boxplot, grouped_bar, histogram, line_plot, raear_curve_plot, scatter_reg,
)
from .preliminary_templates import write_preliminary_templates

__all__ = [
    "boxplot",
    "grouped_bar",
    "histogram",
    "line_plot",
    "raear_curve_plot",
    "scatter_reg",
    "write_preliminary_templates",
]
