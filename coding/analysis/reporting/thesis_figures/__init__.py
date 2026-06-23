"""Standardized thesis figure rendering suite."""

from .spec import FigureSpec, load_specs
from .suite import FigureRenderResult, render_figures

__all__ = [
    "FigureRenderResult",
    "FigureSpec",
    "load_specs",
    "render_figures",
]
