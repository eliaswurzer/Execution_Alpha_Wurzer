"""Shared Matplotlib style for thesis-ready figures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterator

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import seaborn as sns  # noqa: E402


THEME = {
    "black": "#1A1A1A",
    "purple": "#5B2C83",
    "orange": "#E08214",
    "teal": "#1B9E77",
    "slate": "#3D5A80",
    "rose": "#C51B8A",
    "gold": "#B8860B",
    "gray": "#6F6F6F",
    "light_orange": "#FEE6CE",
    "mid_orange": "#FDAE6B",
    "magenta": "#B5359C",
    "deep_purple": "#2D1160",
}

THEME_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "thesis_purple_orange_sequential",
    [
        THEME["light_orange"],
        THEME["mid_orange"],
        THEME["magenta"],
        THEME["deep_purple"],
    ],
)
THEME_DIVERGING = plt.get_cmap("PuOr")

CATEGORICAL = [
    THEME["purple"],
    THEME["orange"],
    THEME["teal"],
    THEME["slate"],
    THEME["rose"],
    THEME["gold"],
    THEME["gray"],
]

# Backward-compatible names used by older plotting helpers.
PALETTE = {
    "blue": THEME["purple"],
    "orange": THEME["orange"],
    "green": THEME["teal"],
    "red": THEME["rose"],
    "purple": THEME["purple"],
    "brown": THEME["gold"],
    "gray": THEME["gray"],
    "black": THEME["black"],
    "teal": THEME["teal"],
    "slate": THEME["slate"],
    "gold": THEME["gold"],
    "rose": THEME["rose"],
}

COLOR_CYCLE = list(CATEGORICAL)


def sample_sequential(n: int) -> list[str]:
    """Return perceptually ordered colors from the thesis sequential ramp."""
    if n <= 0:
        return []
    if n == 1:
        return [THEME["purple"]]
    positions = [0.18 + (0.74 - 0.18) * i / (n - 1) for i in range(n)]
    return [to_hex(THEME_SEQUENTIAL(pos)) for pos in positions]


STRATEGY_COLORS = {
    "S0 MOC": THEME["black"],
    "S0_MOC": THEME["black"],
    "S1 Static": THEME["teal"],
    "S1_STATIC": THEME["teal"],
    "S2 Time-Adaptive": THEME["orange"],
    "S2_TIME_ADAPTIVE": THEME["orange"],
    "S3 OFI": THEME["slate"],
    "S3_OFI": THEME["slate"],
    "S3 IMB": THEME["rose"],
    "S3_IMB": THEME["rose"],
    "S3 Full": THEME["purple"],
    "S3_FULL": THEME["purple"],
    "S4 TOD": THEME["gold"],
    "S4_TOD": THEME["gold"],
    "S5 Value-Aware": THEME["gray"],
    "S5_VALUE_AWARE_XGB": THEME["gray"],
}

SPEC_GROUP_COLORS = {
    "Tape replay": THEME["purple"],
    "Model-based": THEME["orange"],
}

TIER_COLORS = {
    "Tier 1": to_hex(THEME_SEQUENTIAL(0.85)),
    "Tier 2": to_hex(THEME_SEQUENTIAL(0.55)),
    "Tier 3": to_hex(THEME_SEQUENTIAL(0.28)),
    "All": THEME["black"],
}

COMPONENT_COLORS = {
    "gross_alpha": THEME["purple"],
    "Gross alpha": THEME["purple"],
    "maker_rebate": THEME["teal"],
    "Maker rebate": THEME["teal"],
    "commission": THEME["orange"],
    "Commission": THEME["orange"],
    "self_impact": THEME["gold"],
    "Self-impact": THEME["gold"],
}

METRIC_SERIES_COLORS = {
    "Net alpha": THEME["purple"],
    "Net alpha vs. MOC": THEME["purple"],
    "Net alpha vs. MOC (bps)": THEME["purple"],
    "Fill rate": THEME["teal"],
    "AS markout (bps)": THEME["orange"],
    "Adverse-selection markout": THEME["orange"],
    "Adverse-selection component": THEME["gold"],
}

SEMANTIC_MAPS = {
    "strategy": STRATEGY_COLORS,
    "spec_group": SPEC_GROUP_COLORS,
    "tier": TIER_COLORS,
    "component": COMPONENT_COLORS,
    "metric": METRIC_SERIES_COLORS,
}

FILL_ALPHA = 0.16

_FONTS_REGISTERED = False


@dataclass(frozen=True)
class FigureProfile:
    name: str
    figsize: tuple[float, float]
    dpi: int = 220


PROFILES = {
    "single": FigureProfile("single", (5.45, 4.0)),
    "wide": FigureProfile("wide", (5.75, 3.95)),
    "two_panel": FigureProfile("two_panel", (5.85, 5.0)),
    "two_panel_wide": FigureProfile("two_panel_wide", (6.25, 4.15)),
    "three_panel_wide": FigureProfile("three_panel_wide", (6.45, 4.55)),
    "appendix_four_panel": FigureProfile("appendix_four_panel", (9.8, 5.6)),
    "heatmap": FigureProfile("heatmap", (5.8, 3.15)),
}


RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "font.sans-serif": ["STIX Two Text", "STIXGeneral", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
}


def _kpsewhich(name: str) -> Path | None:
    try:
        result = subprocess.run(
            ["kpsewhich", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    path = Path(result.stdout.strip())
    return path if path.exists() else None


def register_fonts() -> None:
    """Register STIX Two Text for Matplotlib when TeX owns the font files."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    candidates = [
        Path(r"C:\Windows\Fonts\STIXTwoText-Regular.otf"),
        Path(r"C:\Windows\Fonts\STIXTwoText-Regular.ttf"),
    ]
    for name in (
        "STIXTwoText-Regular.otf",
        "STIXTwoText-Italic.otf",
        "STIXTwoText-Bold.otf",
        "STIXTwoText-BoldItalic.otf",
    ):
        found = _kpsewhich(name)
        if found is not None:
            candidates.append(found)
    for path in candidates:
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            continue
    _FONTS_REGISTERED = True


def color_for(role: str | None, value: str, index: int = 0) -> str:
    """Return a stable semantic color with a categorical fallback."""
    table = SEMANTIC_MAPS.get(role or "")
    key = str(value)
    if table and key in table:
        return table[key]
    return CATEGORICAL[index % len(CATEGORICAL)]


def emphasis_color() -> str:
    return THEME["black"]


def apply_thesis_style() -> None:
    """Apply the thesis plotting defaults globally."""
    register_fonts()
    plt.rcParams.update(RCPARAMS)
    sns.set_theme(style="white", context="paper", rc=RCPARAMS)


@contextmanager
def thesis_style() -> Iterator[None]:
    """Temporarily apply the thesis plotting defaults."""
    register_fonts()
    with plt.rc_context(RCPARAMS):
        sns.set_theme(style="white", context="paper", rc=RCPARAMS)
        yield


def profile_size(name: str | None) -> tuple[float, float]:
    profile = PROFILES.get(name or "single", PROFILES["single"])
    return profile.figsize
