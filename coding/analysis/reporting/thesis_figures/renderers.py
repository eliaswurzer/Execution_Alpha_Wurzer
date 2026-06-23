"""Reusable Matplotlib renderers for thesis figure specs."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.colors import to_rgba
from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from ..jof_latex import jof_figure
from .spec import FigureSpec
from .style import (
    FILL_ALPHA,
    THEME,
    THEME_DIVERGING,
    THEME_SEQUENTIAL,
    color_for,
    emphasis_color,
    profile_size,
    thesis_style,
)

_TEXTWIDTH_IN = 150.0 / 25.4
_LANDSCAPE_LINEWIDTH_IN = 237.0 / 25.4


def _resolve_cmap(name: str | None):
    if name == "sequential":
        return THEME_SEQUENTIAL
    if name == "diverging":
        return THEME_DIVERGING
    return name or THEME_DIVERGING


def _series_color(spec: FigureSpec, value: str, index: int = 0) -> str:
    return color_for(spec.aesthetics.get("color_role"), value, index)


def _shrink_bbox(bbox, pixels: float = 1.5):
    if bbox.width <= 2 * pixels or bbox.height <= 2 * pixels:
        return bbox
    return bbox.padded(-pixels)


def _intersects(left, right) -> bool:
    a = _shrink_bbox(left)
    b = _shrink_bbox(right)
    return a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0


def _text_bbox(text: mtext.Text, renderer):
    if not text.get_visible() or not text.get_text():
        return None
    try:
        bbox = text.get_window_extent(renderer)
    except Exception:
        return None
    if bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox


def _figure_bbox(fig: plt.Figure):
    width, height = fig.get_size_inches() * fig.dpi
    return Bbox.from_bounds(0, 0, width, height)


def _outside_figure(bbox, figure_bbox, *, tolerance: float = 2.0) -> bool:
    return (
        bbox.x0 < figure_bbox.x0 - tolerance
        or bbox.y0 < figure_bbox.y0 - tolerance
        or bbox.x1 > figure_bbox.x1 + tolerance
        or bbox.y1 > figure_bbox.y1 + tolerance
    )


def _latex_width_inches(fig: plt.Figure, spec: FigureSpec) -> float:
    latex = getattr(spec, "latex", {}) or {}
    include_width = str(latex.get("include_width", ""))
    match = re.fullmatch(r"\s*([0-9.]+)\s*\\(textwidth|linewidth)\s*", include_width)
    if match is None:
        return float(fig.get_size_inches()[0])
    factor = float(match.group(1))
    unit = match.group(2)
    native_width = float(fig.get_size_inches()[0])
    if unit == "linewidth" and native_width > _TEXTWIDTH_IN + 0.2:
        return factor * _LANDSCAPE_LINEWIDTH_IN
    return factor * _TEXTWIDTH_IN


def _effective_font_scale(fig: plt.Figure, spec: FigureSpec) -> float:
    native_width = float(fig.get_size_inches()[0])
    if native_width <= 0:
        return 1.0
    return _latex_width_inches(fig, spec) / native_width


def _axis_texts(ax: plt.Axes) -> list[mtext.Text]:
    texts: list[mtext.Text] = []
    texts.extend(label for label in ax.get_xticklabels() if label.get_visible())
    texts.extend(label for label in ax.get_yticklabels() if label.get_visible())
    texts.extend(text for text in (ax.xaxis.label, ax.yaxis.label, ax.title) if text.get_visible())
    texts.extend(text for text in ax.texts if text.get_visible())
    return texts


def _all_visible_texts(fig: plt.Figure) -> list[mtext.Text]:
    texts: list[mtext.Text] = []
    for ax in fig.axes:
        texts.extend(_axis_texts(ax))
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            texts.extend(text for text in legend.get_texts() if text.get_visible())
            title = legend.get_title()
            if title.get_visible():
                texts.append(title)
    return [text for text in texts if text.get_text()]


def _clip_checked_texts(fig: plt.Figure) -> list[mtext.Text]:
    texts: list[mtext.Text] = []
    for ax in fig.axes:
        texts.extend(text for text in (ax.xaxis.label, ax.yaxis.label, ax.title) if text.get_visible())
        texts.extend(text for text in ax.texts if text.get_visible())
    return [text for text in texts if text.get_text()]


def _assert_layout_quality(fig: plt.Figure, spec: FigureSpec) -> None:
    """Fail fast on clipped text and direct label/legend overlaps."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = _figure_bbox(fig)
    font_scale = _effective_font_scale(fig, spec)
    layout = getattr(spec, "layout", {}) or {}
    min_effective_font = float(layout.get("min_effective_font_pt", 7.0))
    problems: list[str] = []
    all_axis_text_boxes: list[tuple[int, mtext.Text, object]] = []

    for ax_i, ax in enumerate(fig.axes):
        tick_groups = [
            ("x", [label for label in ax.get_xticklabels() if label.get_visible()]),
            ("y", [label for label in ax.get_yticklabels() if label.get_visible()]),
        ]
        for axis_name, labels in tick_groups:
            boxes = [(label, _text_bbox(label, renderer)) for label in labels]
            boxes = [(label, bbox) for label, bbox in boxes if bbox is not None]
            for (left_label, left_box), (right_label, right_box) in combinations(boxes, 2):
                if _intersects(left_box, right_box):
                    problems.append(
                        f"overlapping {axis_name}-tick labels on axis {ax_i}: "
                        f"{left_label.get_text()!r} / {right_label.get_text()!r}"
                    )

        annotation_boxes = [
            (text, bbox)
            for text in ax.texts
            if (bbox := _text_bbox(text, renderer)) is not None
        ]
        for (left_text, left_box), (right_text, right_box) in combinations(annotation_boxes, 2):
            if _intersects(left_box, right_box):
                problems.append(
                    f"overlapping annotations on axis {ax_i}: "
                    f"{left_text.get_text()!r} / {right_text.get_text()!r}"
                )

        for text in _axis_texts(ax):
            bbox = _text_bbox(text, renderer)
            if bbox is not None:
                all_axis_text_boxes.append((ax_i, text, bbox))

    legends = []
    for ax_i, ax in enumerate(fig.axes):
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        legend_box = legend.get_window_extent(renderer)
        legends.append((ax_i, legend_box))
        if _outside_figure(legend_box, figure_bbox):
            problems.append(f"legend on axis {ax_i} is outside figure bounds")
        for text_ax_i, text, bbox in all_axis_text_boxes:
            if _intersects(legend_box, bbox):
                problems.append(
                    f"legend on axis {ax_i} overlaps label on axis {text_ax_i}: "
                    f"{text.get_text()!r}"
                )

    for (left_ax, left), (right_ax, right) in combinations(legends, 2):
        if _intersects(left, right):
            problems.append(f"legends overlap on axes {left_ax} and {right_ax}")

    clip_checked = set(_clip_checked_texts(fig))
    for text in _all_visible_texts(fig):
        bbox = _text_bbox(text, renderer)
        if bbox is None:
            continue
        if text in clip_checked and _outside_figure(bbox, figure_bbox):
            problems.append(f"text is outside figure bounds: {text.get_text()!r}")
        effective_size = float(text.get_fontsize()) * font_scale
        if effective_size < min_effective_font:
            problems.append(
                f"effective font below {min_effective_font:.1f} pt: "
                f"{text.get_text()!r} is {effective_size:.1f} pt"
            )

    if problems:
        shown = "; ".join(problems[:8])
        more = "" if len(problems) <= 8 else f"; plus {len(problems) - 8} more"
        raise RuntimeError(f"{spec.id} failed layout QA: {shown}{more}")


def _set_common(ax: plt.Axes, spec: FigureSpec, *, ylabel: str | None = None) -> None:
    layout = spec.layout
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(layout.get("xlabel", ""))
    ax.set_ylabel(ylabel if ylabel is not None else layout.get("ylabel", ""))
    if "ymin" in layout or "ymax" in layout:
        ax.set_ylim(bottom=layout.get("ymin"), top=layout.get("ymax"))
    if layout.get("xscale"):
        ax.set_xscale(layout["xscale"])
    if layout.get("yscale"):
        ax.set_yscale(layout["yscale"])
    if layout.get("x_major_step") is not None:
        ax.xaxis.set_major_locator(MultipleLocator(float(layout["x_major_step"])))
    if layout.get("x_minor_step") is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(float(layout["x_minor_step"])))
    ax.grid(True, axis="y", which="major", color="0.72", linewidth=0.42, alpha=0.28)
    if layout.get("x_grid", False):
        ax.grid(True, axis="x", which="major", color="0.70", linewidth=0.38, alpha=0.24)
    if layout.get("x_minor_grid", False):
        ax.grid(True, axis="x", which="minor", color="0.74", linewidth=0.30, alpha=0.18)
    if layout.get("hline") is not None:
        ax.axhline(float(layout["hline"]), color="0.55", linewidth=0.68, zorder=1)
    if layout.get("vline") is not None:
        ax.axvline(float(layout["vline"]), color="0.62", linewidth=0.60, linestyle=":", zorder=1)


def _finalize(fig: plt.Figure, spec: FigureSpec, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{spec.id}.pdf"
    png = out_dir / f"{spec.id}.png"
    tight_rect = spec.layout.get("tight_rect")
    if tight_rect is not None:
        fig.tight_layout(rect=tuple(float(v) for v in tight_rect))
    else:
        fig.tight_layout()
    _assert_layout_quality(fig, spec)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    graphics_file = spec.latex.get("graphics_file", f"figures/{out_dir.name}/{spec.id}.pdf")
    tex = jof_figure(
        graphics_file=graphics_file,
        caption=spec.latex["caption"],
        label=spec.latex["label"],
        legend=spec.latex["notes"],
        width=spec.latex.get("include_width", "0.88\\textwidth"),
    )
    tex_path = out_dir / f"{spec.id}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return {"pdf": pdf.name, "png": png.name, "tex": tex_path.name}


def _order_values(values: pd.Series, preferred: list[str] | None = None) -> list[str]:
    seen = [str(v) for v in values.dropna().drop_duplicates().tolist()]
    if not preferred:
        return seen
    return [v for v in preferred if v in seen] + [v for v in seen if v not in preferred]


def _axis_values(values: pd.Series, column: str) -> pd.Series:
    if "date" not in column.lower() and not column.lower().endswith("_end"):
        return values
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed if parsed.notna().any() else values


def _wrap_category_label(value: object, *, max_chars: int = 12) -> str:
    text = str(value)
    if len(text) <= max_chars or " " not in text:
        return text
    head, tail = text.split(" ", 1)
    return f"{head}\n{tail}"


def _display_category_label(spec: FigureSpec, value: object, *, max_chars: int = 12) -> str:
    text = str(value)
    labels = spec.layout.get("xtick_labels", {}) or {}
    if text in labels:
        text = str(labels[text])
    return _wrap_category_label(text, max_chars=max_chars)


def _apply_legend_handles(
    ax: plt.Axes,
    spec: FigureSpec,
    handles: list[object],
    labels: list[str],
    *,
    default_ncols: int = 1,
) -> None:
    if not handles:
        return
    label_map = spec.layout.get("legend_labels", {}) or {}
    labels = [str(label_map.get(label, label)) for label in labels]
    kwargs: dict[str, object] = {
        "handles": handles,
        "labels": labels,
        "title": spec.layout.get("legend_title", None),
        "frameon": False,
        "loc": spec.layout.get("legend_loc", "best"),
        "ncols": int(spec.layout.get("legend_ncols", default_ncols)),
    }
    bbox = spec.layout.get("legend_bbox_to_anchor")
    if bbox is not None:
        kwargs["bbox_to_anchor"] = tuple(float(v) for v in bbox)
    ax.legend(**kwargs)


def _apply_legend(ax: plt.Axes, spec: FigureSpec, *, default_ncols: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    _apply_legend_handles(ax, spec, list(handles), list(labels), default_ncols=default_ncols)


def _line_style_for(spec: FigureSpec, series: str) -> str:
    styles = spec.aesthetics.get("linestyles", {}) or {}
    if series in styles:
        return str(styles[series])
    if series == spec.aesthetics.get("emphasize") and spec.aesthetics.get("emphasize_dashed", False):
        return "--"
    return "-"


def _line_width_for(spec: FigureSpec, series: str, default: float) -> float:
    widths = spec.aesthetics.get("line_widths", {}) or {}
    if series in widths:
        return float(widths[series])
    if series == spec.aesthetics.get("emphasize") and spec.aesthetics.get("emphasize_line_width") is not None:
        return float(spec.aesthetics["emphasize_line_width"])
    return default


def _line_zorder_for(spec: FigureSpec, series: str) -> float:
    zorders = spec.aesthetics.get("zorders", {}) or {}
    if series in zorders:
        return float(zorders[series])
    if series == spec.aesthetics.get("emphasize"):
        return float(spec.aesthetics.get("emphasize_zorder", 4.0))
    return float(spec.aesthetics.get("default_zorder", 2.0))


def _apply_date_axis(ax: plt.Axes, spec: FigureSpec) -> None:
    layout = spec.layout
    if layout.get("date_major_month_interval") is not None:
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(layout["date_major_month_interval"]))
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter(layout.get("date_format", "%Y-%m")))
    if layout.get("date_minor_month_interval") is not None:
        ax.xaxis.set_minor_locator(
            mdates.MonthLocator(interval=int(layout["date_minor_month_interval"]))
        )


def _cluster_label_defaults(ax: plt.Axes, points: list[tuple[str, float, float]]) -> dict[str, dict[str, object]]:
    if not points:
        return {}
    coords = np.array([ax.transData.transform((x, y)) for _, x, y in points], dtype=float)
    threshold = 58.0
    neighbors = {i: set() for i in range(len(points))}
    for i, j in combinations(range(len(points)), 2):
        if np.linalg.norm(coords[i] - coords[j]) <= threshold:
            neighbors[i].add(j)
            neighbors[j].add(i)

    defaults: dict[str, dict[str, object]] = {}
    seen: set[int] = set()
    cluster_offsets = [
        (-10, -22),
        (-10, -6),
        (-10, 11),
        (-10, 28),
        (-10, 45),
        (-10, -39),
    ]
    for start in range(len(points)):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        component: list[int] = []
        seen.add(start)
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in neighbors[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        component.sort(key=lambda idx: points[idx][0])
        if len(component) == 1:
            idx = component[0]
            label, _, _ = points[idx]
            right_side = coords[idx, 0] > ax.bbox.x0 + 0.72 * ax.bbox.width
            defaults[label] = {
                "xytext": (-8, 5) if right_side else (6, 5),
                "ha": "right" if right_side else "left",
                "va": "center",
            }
            continue
        mean_x = float(coords[component, 0].mean())
        right_side = mean_x > ax.bbox.x0 + 0.58 * ax.bbox.width
        for pos, idx in enumerate(component):
            label, _, _ = points[idx]
            if right_side:
                xytext = cluster_offsets[pos % len(cluster_offsets)]
                ha = "right"
            else:
                xoff, yoff = cluster_offsets[pos % len(cluster_offsets)]
                xytext = (-xoff, yoff)
                ha = "left"
            defaults[label] = {"xytext": xytext, "ha": ha, "va": "center"}
    return defaults


def _coerce_label_offset(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        xytext = raw.get("xytext", raw.get("offset", (6, 5)))
        return {
            "xytext": tuple(xytext),
            "ha": raw.get("ha", "left"),
            "va": raw.get("va", "center"),
        }
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return {"xytext": (raw[0], raw[1]), "ha": "left", "va": "center"}
    return {"xytext": (6, 5), "ha": "left", "va": "center"}


def _annotation_label(spec: FigureSpec, label: str) -> str:
    aliases = spec.aesthetics.get("label_aliases", {}) or {}
    return str(aliases.get(label, label))


def _relax_annotation_overlaps(ax: plt.Axes, annotations: list[plt.Annotation]) -> None:
    if len(annotations) < 2:
        return
    fig = ax.figure
    directions = [1 if idx % 2 == 0 else -1 for idx in range(len(annotations))]
    for iteration in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [(text, _text_bbox(text, renderer)) for text in annotations]
        boxes = [(text, bbox) for text, bbox in boxes if bbox is not None]
        moved = False
        for left_i, (_left_text, left_box) in enumerate(boxes):
            for right_i, (right_text, right_box) in enumerate(boxes[left_i + 1 :], start=left_i + 1):
                if not _intersects(left_box, right_box):
                    continue
                xoff, yoff = right_text.get_position()
                step = 5 + iteration * 2
                direction = directions[right_i % len(directions)]
                right_text.set_position((xoff, yoff + direction * step))
                moved = True
        if not moved:
            break


def _annotate_points(ax: plt.Axes, df: pd.DataFrame, spec: FigureSpec) -> None:
    aes = spec.aesthetics
    label_col = aes.get("label")
    if label_col not in df.columns:
        return
    points = [
        (str(row[label_col]), float(row[aes["x"]]), float(row[aes["y"]]))
        for _, row in df.iterrows()
        if pd.notna(row[aes["x"]]) and pd.notna(row[aes["y"]])
    ]
    defaults = _cluster_label_defaults(ax, points)
    custom = aes.get("label_offsets", {}) or {}
    fontsize = float(aes.get("label_fontsize", 9))
    annotations: list[plt.Annotation] = []
    for label, x, y in points:
        options = (
            _coerce_label_offset(custom[label])
            if label in custom
            else defaults.get(label, {"xytext": (6, 5), "ha": "left", "va": "center"})
        )
        annotation = ax.annotate(
            _annotation_label(spec, label),
            (x, y),
            textcoords="offset points",
            xytext=options["xytext"],
            ha=str(options["ha"]),
            va=str(options["va"]),
            fontsize=fontsize,
            color=THEME["black"],
        )
        annotations.append(annotation)
    _relax_annotation_overlaps(ax, annotations)



def _render_line(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    fig, ax = plt.subplots(figsize=profile_size(spec.layout.get("profile")))
    x, y = aes["x"], aes["y"]
    hue = aes.get("hue")
    marker = aes.get("marker")
    line_width = float(spec.layout.get("line_width", 1.45))
    if hue:
        for i, name in enumerate(_order_values(df[hue], aes.get("order"))):
            sub = df[df[hue].astype(str) == name].sort_values(x)
            color = emphasis_color() if name == aes.get("emphasize") else _series_color(spec, name, i)
            ax.plot(_axis_values(sub[x], x), sub[y], label=name, marker=marker, linewidth=line_width, color=color)
        _apply_legend(ax, spec)
    else:
        sub = df.sort_values(x)
        ax.plot(_axis_values(sub[x], x), sub[y], marker=marker, linewidth=line_width, color=THEME["purple"])
    _set_common(ax, spec)
    if x.lower().endswith("date") or "date" in x.lower():
        fig.autofmt_xdate()
    return fig


def _render_line_ci(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    fig, ax = plt.subplots(figsize=profile_size(spec.layout.get("profile")))
    sub = df.sort_values(aes["x"]).copy()
    x = _axis_values(sub[aes["x"]], aes["x"])
    y = pd.to_numeric(sub[aes["y"]], errors="coerce")
    lo = pd.to_numeric(sub[aes["ci_low"]], errors="coerce")
    hi = pd.to_numeric(sub[aes["ci_high"]], errors="coerce")
    ax.plot(x, y, color=THEME["purple"], linewidth=float(spec.layout.get("line_width", 1.45)))
    ax.fill_between(x, lo, hi, color=THEME["purple"], alpha=FILL_ALPHA, linewidth=0)
    _set_common(ax, spec)
    fig.autofmt_xdate()
    return fig


def _render_line_panels(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    panels = _order_values(df[aes["panel"]])
    sharex = bool(spec.layout.get("sharex", True))
    fig, axes = plt.subplots(len(panels), 1, figsize=profile_size(spec.layout.get("profile")), sharex=sharex)
    if len(panels) == 1:
        axes = [axes]
    legend_mode = spec.layout.get("legend_mode", "each")
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    line_width = float(spec.layout.get("line_width", 1.35))
    for panel_i, (ax, panel) in enumerate(zip(axes, panels)):
        sub_panel = df[df[aes["panel"]].astype(str) == panel]
        for i, series in enumerate(_order_values(sub_panel[aes["hue"]], aes.get("order"))):
            sub = sub_panel[sub_panel[aes["hue"]].astype(str) == series].sort_values(aes["x"])
            x_values = _axis_values(sub[aes["x"]], aes["x"])
            color = emphasis_color() if series == aes.get("emphasize") else _series_color(spec, series, i)
            (line,) = ax.plot(
                x_values,
                sub[aes["y"]],
                marker=aes.get("marker"),
                label=series,
                linewidth=_line_width_for(spec, series, line_width),
                color=color,
                linestyle=_line_style_for(spec, series),
                zorder=_line_zorder_for(spec, series),
            )
            if series not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(series)
            if aes.get("ci_low") and aes.get("ci_high"):
                lo = pd.to_numeric(sub[aes["ci_low"]], errors="coerce")
                hi = pd.to_numeric(sub[aes["ci_high"]], errors="coerce")
                if lo.notna().any() and hi.notna().any():
                    ax.fill_between(x_values, lo, hi, color=color, alpha=FILL_ALPHA, linewidth=0)
        ax.set_title(panel)
        if "date" in aes["x"].lower() or aes["x"].lower().endswith("_end"):
            _apply_date_axis(ax, spec)
        _set_common(ax, spec, ylabel=panel if not spec.layout.get("ylabel") else spec.layout.get("ylabel"))
        if panel_i < len(panels) - 1:
            ax.set_xlabel("")
        if legend_mode == "each" or (legend_mode == "first" and panel_i == 0):
            _apply_legend(ax, spec, default_ncols=2)
    if legend_mode == "figure":
        _apply_legend_handles(axes[0], spec, legend_handles, legend_labels, default_ncols=2)
    axes[-1].set_xlabel(spec.layout.get("xlabel", ""))
    if "date" in aes["x"].lower() or aes["x"].lower().endswith("_end"):
        if sharex:
            fig.autofmt_xdate()
        else:
            for ax in axes:
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha("right")
    return fig


def _render_line_fit_panels(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    panels = _order_values(df[aes["panel"]], aes.get("order"))
    fig, axes = plt.subplots(len(panels), 1, figsize=profile_size(spec.layout.get("profile")), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for panel_i, (ax, panel) in enumerate(zip(axes, panels)):
        sub = df[df[aes["panel"]].astype(str) == panel].sort_values(aes["x"])
        x = _axis_values(sub[aes["x"]], aes["x"])
        color = emphasis_color() if panel == aes.get("emphasize") else _series_color(spec, panel, panel_i)
        ax.plot(x, sub[aes["y"]], color=color, alpha=0.62, linewidth=1.35, label="Daily share")
        ax.plot(x, sub[aes["fit"]], color=color, linewidth=1.0, linestyle="--", label="Fitted trend")
        ax.set_title(panel)
        _set_common(ax, spec, ylabel="")
        _apply_date_axis(ax, spec)
        if panel_i < len(panels) - 1:
            ax.set_xlabel("")
        if panel_i == 0:
            _apply_legend(ax, spec, default_ncols=2)
    axes[-1].set_xlabel(spec.layout.get("xlabel", ""))
    if spec.layout.get("ylabel"):
        fig.supylabel(spec.layout["ylabel"], x=0.01)
    if "date" in aes["x"].lower() or aes["x"].lower().endswith("_end"):
        fig.autofmt_xdate()
    return fig


def _render_heatmap(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    pivot = df.pivot_table(index=aes["index"], columns=aes["columns"], values=aes["value"], aggfunc="mean")
    fig, ax = plt.subplots(figsize=profile_size(spec.layout.get("profile")))
    fmt = aes.get("fmt", ".2f")
    annot: object = bool(aes.get("annotate", False))
    heatmap_fmt = fmt
    if annot and aes.get("annotate_below") is not None:
        threshold = float(aes["annotate_below"])
        annot_frame = pivot.copy().astype(object)
        numeric = pivot.apply(pd.to_numeric, errors="coerce")
        for row in pivot.index:
            for col in pivot.columns:
                value = numeric.loc[row, col]
                annot_frame.loc[row, col] = "" if pd.isna(value) or value >= threshold else format(float(value), fmt)
        annot = annot_frame
        heatmap_fmt = ""
    sns.heatmap(
        pivot,
        annot=annot,
        fmt=heatmap_fmt,
        cmap=_resolve_cmap(aes.get("cmap")),
        center=spec.layout.get("center"),
        vmin=spec.layout.get("vmin"),
        vmax=spec.layout.get("vmax"),
        cbar_kws={"label": spec.layout.get("colorbar_label", "")},
        annot_kws={"fontsize": float(aes.get("annotate_fontsize", 8))},
        ax=ax,
    )
    ax.set_xlabel(spec.layout.get("xlabel", ""))
    ax.set_ylabel(spec.layout.get("ylabel", ""))
    if spec.layout.get("title"):
        ax.set_title(str(spec.layout["title"]))
    if "ytick_rotation" in spec.layout:
        ax.tick_params(axis="y", labelrotation=float(spec.layout["ytick_rotation"]))
    return fig


def _render_scatter(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    fig, ax = plt.subplots(figsize=profile_size(spec.layout.get("profile")))
    marker_by = aes.get("marker_by")
    show_labels = bool(aes.get("show_labels", True))
    legend_by_label = bool(aes.get("legend_by_label", False))
    marker_size = float(aes.get("marker_size", 36))
    marker_alpha = float(aes.get("marker_alpha", 1.0))
    marker_edgecolor = aes.get("marker_edgecolor")
    marker_linewidth = float(aes.get("marker_linewidth", 0.0 if marker_edgecolor is None else 0.65))
    halo_enabled = bool(aes.get("marker_halo", False))
    halo_size = float(aes.get("marker_halo_size", marker_size + 16))
    halo_edgecolor = aes.get("marker_halo_edgecolor", "white")
    halo_linewidth = float(aes.get("marker_halo_linewidth", 2.0))
    point_nudges = aes.get("point_nudges", {}) or {}

    def plot_xy(row: pd.Series, label: str) -> tuple[float, float]:
        x_val = float(row[aes["x"]])
        y_val = float(row[aes["y"]])
        nudge = point_nudges.get(label, {}) or {}
        return (
            x_val + float(nudge.get("dx", 0.0)),
            y_val + float(nudge.get("dy", 0.0)),
        )

    def draw_point(x_val: float, y_val: float, *, color: str, marker: str = "o", label: str | None = None) -> None:
        if halo_enabled:
            ax.scatter(
                [x_val],
                [y_val],
                s=halo_size,
                marker=marker,
                facecolors="none",
                edgecolors=halo_edgecolor,
                linewidths=halo_linewidth,
                zorder=4,
            )
        kwargs = {
            "s": marker_size,
            "marker": marker,
            "facecolors": [to_rgba(color, marker_alpha)],
            "edgecolors": marker_edgecolor or color,
            "linewidths": marker_linewidth,
            "label": label,
            "zorder": 5 if halo_enabled else 3,
        }
        ax.scatter([x_val], [y_val], **kwargs)

    if marker_by and marker_by in df.columns:
        markers = {"Model-based": "s", "Tape replay": "o"}
        for i, group in enumerate(_order_values(df[marker_by])):
            sub = df[df[marker_by].astype(str) == group]
            color = _series_color(spec, group, i)
            marker = markers.get(group, "o")
            ax.scatter(
                sub[aes["x"]],
                sub[aes["y"]],
                label=group,
                marker=marker,
                s=marker_size,
                facecolors=[to_rgba(color, marker_alpha)],
                edgecolors=marker_edgecolor or color,
                linewidths=marker_linewidth,
                zorder=3,
            )
        _apply_legend(ax, spec)
    elif aes.get("label") in df.columns and aes.get("color_role"):
        for i, (_, row) in enumerate(df.iterrows()):
            label = str(row[aes["label"]])
            x_val, y_val = plot_xy(row, label)
            draw_point(
                x_val,
                y_val,
                color=_series_color(spec, label, i),
                label=label if legend_by_label else None,
            )
        if legend_by_label:
            _apply_legend(ax, spec, default_ncols=3)
    else:
        ax.scatter(
            df[aes["x"]],
            df[aes["y"]],
            s=marker_size,
            facecolors=[to_rgba(THEME["purple"], marker_alpha)],
            edgecolors=marker_edgecolor or THEME["purple"],
            linewidths=marker_linewidth,
            zorder=3,
        )
    _set_common(ax, spec)
    fig.canvas.draw()
    if show_labels:
        _annotate_points(ax, df, spec)
    return fig


def _render_stacked_bar(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    x_col = aes["x"]
    components = list(aes["components"])
    labels = aes.get("component_labels", {})
    fig, ax = plt.subplots(figsize=profile_size(spec.layout.get("profile")))
    x = np.arange(len(df))
    bottom_pos = np.zeros(len(df))
    bottom_neg = np.zeros(len(df))
    for i, col in enumerate(components):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        base = np.where(vals >= 0, bottom_pos, bottom_neg)
        ax.bar(x, vals, bottom=base, label=labels.get(col, col), width=0.65,
               color=color_for("component", col, i))
        bottom_pos = np.where(vals >= 0, bottom_pos + vals, bottom_pos)
        bottom_neg = np.where(vals < 0, bottom_neg + vals, bottom_neg)
    point_labels = aes.get("point_labels", {})
    markers = ["o", "v", "D"]
    for i, col in enumerate(aes.get("points", [])):
        ax.plot(x, df[col], linestyle="none", marker=markers[i % len(markers)],
                color=emphasis_color() if i == 0 else THEME["rose"],
                markerfacecolor="none" if i > 0 else emphasis_color(),
                label=point_labels.get(col, col), zorder=4)
    rotation = float(spec.layout.get("xtick_rotation", 0))
    ax.set_xticks(
        x,
        [_display_category_label(spec, value) for value in df[x_col]],
        rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.legend(
        fontsize=10,
        ncols=int(spec.layout.get("legend_ncols", 3)),
        frameon=False,
        loc=spec.layout.get("legend_loc", "lower center"),
        bbox_to_anchor=tuple(spec.layout.get("legend_bbox_to_anchor", (0.5, 1.02))),
        title=spec.layout.get("legend_title"),
    )
    _set_common(ax, spec)
    return fig


def _render_bar_panels(df: pd.DataFrame, spec: FigureSpec) -> plt.Figure:
    aes = spec.aesthetics
    panels = _order_values(df[aes["panel"]])
    fig, axes = plt.subplots(1, len(panels), figsize=profile_size(spec.layout.get("profile")), squeeze=False)
    axes_flat = axes[0]
    legend_mode = spec.layout.get("legend_mode", "each")
    legend_loc = spec.layout.get("legend_loc", "best")
    legend_bbox = spec.layout.get("legend_bbox_to_anchor")
    legend_ncols = int(spec.layout.get("legend_ncols", 1))
    figure_legend: tuple[list[object], list[str]] | None = None
    for panel_i, (ax, panel) in enumerate(zip(axes_flat, panels)):
        sub = df[df[aes["panel"]].astype(str) == panel]
        hue = aes.get("hue")
        if hue:
            hue_values = _order_values(sub[hue])
            palette = {value: _series_color(spec, value, i) for i, value in enumerate(hue_values)}
            sns.barplot(data=sub, x=aes["x"], y=aes["y"], hue=hue, hue_order=hue_values, ax=ax, palette=palette)
        else:
            sns.barplot(data=sub, x=aes["x"], y=aes["y"], ax=ax, color=THEME["purple"])
        ax.set_title(panel)
        rotation = float(spec.layout.get("xtick_rotation", 25))
        labels = [_display_category_label(spec, label.get_text(), max_chars=14) for label in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        _set_common(ax, spec, ylabel=panel if len(panels) > 1 else spec.layout.get("ylabel"))
        if ax.get_legend() is not None:
            show_legend = legend_mode == "each" or (legend_mode == "first" and panel_i == 0)
            handles, legend_labels = ax.get_legend_handles_labels()
            label_map = spec.layout.get("legend_labels", {}) or {}
            legend_labels = [str(label_map.get(label, label)) for label in legend_labels]
            if legend_mode == "figure" and figure_legend is None:
                figure_legend = (list(handles), legend_labels)
            if show_legend:
                kwargs = {
                    "handles": handles,
                    "labels": legend_labels,
                    "title": spec.layout.get("legend_title", ""),
                    "frameon": False,
                    "loc": legend_loc,
                    "ncols": legend_ncols,
                }
                if legend_bbox is not None:
                    kwargs["bbox_to_anchor"] = tuple(float(v) for v in legend_bbox)
                ax.legend(**kwargs)
            else:
                ax.get_legend().remove()
    if legend_mode == "figure" and figure_legend is not None:
        handles, legend_labels = figure_legend
        kwargs = {
            "handles": handles,
            "labels": legend_labels,
            "title": spec.layout.get("legend_title", ""),
            "frameon": False,
            "loc": legend_loc,
            "ncols": legend_ncols,
        }
        if legend_bbox is not None:
            kwargs["bbox_to_anchor"] = tuple(float(v) for v in legend_bbox)
        fig.legend(**kwargs)
    return fig


RENDERERS = {
    "bar_panels": _render_bar_panels,
    "heatmap": _render_heatmap,
    "line": _render_line,
    "line_ci": _render_line_ci,
    "line_fit_panels": _render_line_fit_panels,
    "line_panels": _render_line_panels,
    "scatter": _render_scatter,
    "stacked_bar": _render_stacked_bar,
}


def render_plot_table(df: pd.DataFrame, spec: FigureSpec, out_dir: Path) -> dict[str, str]:
    spec.validate_frame(list(df.columns))
    with thesis_style():
        fig = RENDERERS[spec.plot_type](df.copy(), spec)
        return _finalize(fig, spec, Path(out_dir))
