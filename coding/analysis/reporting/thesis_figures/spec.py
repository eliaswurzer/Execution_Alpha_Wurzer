"""Figure-spec registry loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import deep_merge, read_yaml


VALID_PLOT_TYPES = {
    "bar_panels",
    "heatmap",
    "line",
    "line_ci",
    "line_fit_panels",
    "line_panels",
    "scatter",
    "stacked_bar",
}

SPEC_VERSION = "thesis_figure_suite_v1"


@dataclass(frozen=True)
class FigureSpec:
    id: str
    plot_type: str
    source: dict[str, Any]
    transform: str
    table: dict[str, Any]
    aesthetics: dict[str, Any]
    layout: dict[str, Any]
    latex: dict[str, Any]
    required_columns: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FigureSpec":
        required = (
            "id",
            "plot_type",
            "source",
            "transform",
            "table",
            "aesthetics",
            "layout",
            "latex",
            "required_columns",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Figure spec missing fields {missing}: {payload.get('id')}")
        plot_type = str(payload["plot_type"])
        if plot_type not in VALID_PLOT_TYPES:
            raise ValueError(f"Invalid plot_type {plot_type!r} for {payload['id']}")
        return cls(
            id=str(payload["id"]),
            plot_type=plot_type,
            source=dict(payload["source"] or {}),
            transform=str(payload["transform"]),
            table=dict(payload["table"] or {}),
            aesthetics=dict(payload["aesthetics"] or {}),
            layout=dict(payload["layout"] or {}),
            latex=dict(payload["latex"] or {}),
            required_columns=tuple(str(c) for c in payload["required_columns"]),
        )

    def with_override(self, override: dict[str, Any]) -> "FigureSpec":
        payload = {
            "id": self.id,
            "plot_type": self.plot_type,
            "source": self.source,
            "transform": self.transform,
            "table": self.table,
            "aesthetics": self.aesthetics,
            "layout": self.layout,
            "latex": self.latex,
            "required_columns": list(self.required_columns),
        }
        merged = deep_merge(payload, override)
        merged["id"] = self.id
        merged["plot_type"] = self.plot_type
        merged["transform"] = self.transform
        return FigureSpec.from_dict(merged)

    def validate_frame(self, columns: list[str]) -> None:
        missing = [c for c in self.required_columns if c not in set(columns)]
        if missing:
            raise ValueError(f"{self.id}: plot table missing columns {missing}")


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "specs" / "registry.yaml"


def load_specs(path: Path | None = None) -> dict[str, FigureSpec]:
    registry = read_yaml(path or _default_registry_path())
    figures = registry.get("figures")
    if not isinstance(figures, list):
        raise ValueError("Figure registry must contain a 'figures' list")
    specs = [FigureSpec.from_dict(item) for item in figures]
    ids = [spec.id for spec in specs]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        raise ValueError(f"Duplicate figure ids: {duplicates}")
    return {spec.id: spec for spec in specs}
