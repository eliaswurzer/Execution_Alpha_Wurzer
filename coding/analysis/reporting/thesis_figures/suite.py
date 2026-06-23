"""Orchestration for the standardized thesis figure suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import FigureBuildContext, build_figure_data
from .io import hash_file, read_yaml, write_json
from .renderers import render_plot_table
from .spec import SPEC_VERSION, FigureSpec, load_specs


@dataclass
class FigureRenderResult:
    figure_id: str
    outputs: dict[str, str]
    data_source: str
    inputs: dict[str, str]
    skipped: str | None = None


def figure_input_path(out_dir: Path, spec: FigureSpec) -> Path:
    return Path(out_dir) / spec.table.get("path", f"figure_inputs/{spec.id}.csv")


def curated_input_path(out_dir: Path, spec: FigureSpec) -> Path:
    return Path(out_dir) / "figure_inputs_curated" / f"{spec.id}.csv"


def override_path(out_dir: Path, spec: FigureSpec) -> Path:
    return Path(out_dir) / "overrides" / f"{spec.id}.yaml"


def load_spec_with_override(spec: FigureSpec, out_dir: Path) -> tuple[FigureSpec, dict[str, str]]:
    path = override_path(out_dir, spec)
    if not path.exists():
        return spec, {}
    return spec.with_override(read_yaml(path)), {f"override:{spec.id}": hash_file(path)}


def prepare_figure_data(
    spec: FigureSpec,
    ctx: FigureBuildContext,
    *,
    refresh_data: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = figure_input_path(ctx.out_dir, spec)
    if refresh_data or not path.exists():
        built = build_figure_data(spec.transform, ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        built.frame.to_csv(path, index=False)
        inputs = dict(built.inputs)
        inputs[f"table:{spec.id}:artifact"] = hash_file(path)
        return built.frame, inputs
    frame = pd.read_csv(path)
    return frame, {f"table:{spec.id}:artifact": hash_file(path)}


def load_plot_table(
    spec: FigureSpec,
    ctx: FigureBuildContext,
    *,
    mode: str = "curated",
    refresh_data: bool = False,
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    artifact_frame, inputs = prepare_figure_data(spec, ctx, refresh_data=refresh_data)
    curated = curated_input_path(ctx.out_dir, spec)
    if mode == "curated" and curated.exists():
        frame = pd.read_csv(curated)
        inputs[f"table:{spec.id}:curated"] = hash_file(curated)
        return frame, "curated", inputs
    return artifact_frame, "artifact", inputs


def _select_specs(specs: dict[str, FigureSpec], figure: str | Iterable[str]) -> list[FigureSpec]:
    if figure == "all":
        return list(specs.values())
    if isinstance(figure, str):
        ids = [figure]
    else:
        ids = list(figure)
    missing = [fig for fig in ids if fig not in specs]
    if missing:
        raise ValueError(f"Unknown figure ids: {missing}")
    return [specs[fig] for fig in ids]


def render_figures(
    ctx: FigureBuildContext,
    *,
    figure: str | Iterable[str] = "all",
    mode: str = "curated",
    refresh_data: bool = False,
    specs_path: Path | None = None,
    fail_fast: bool = False,
) -> dict:
    if mode not in {"artifact", "curated"}:
        raise ValueError("mode must be 'artifact' or 'curated'")

    specs = load_specs(specs_path)
    selected = _select_specs(specs, figure)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[FigureRenderResult] = []
    manifest_inputs: dict[str, str] = {}
    outputs: dict[str, dict[str, str]] = {}
    skipped: dict[str, str] = {}

    for base_spec in selected:
        try:
            spec, override_inputs = load_spec_with_override(base_spec, ctx.out_dir)
            frame, data_source, inputs = load_plot_table(
                spec, ctx, mode=mode, refresh_data=refresh_data,
            )
            inputs.update(override_inputs)
            rendered = render_plot_table(frame, spec, ctx.out_dir)
            outputs[spec.id] = rendered
            manifest_inputs.update({f"{spec.id}:{k}": v for k, v in inputs.items()})
            results.append(FigureRenderResult(spec.id, rendered, data_source, inputs))
        except Exception as exc:
            skipped[base_spec.id] = str(exc)
            results.append(FigureRenderResult(base_spec.id, {}, mode, {}, skipped=str(exc)))
            if fail_fast:
                raise

    manifest = {
        "spec_version": SPEC_VERSION,
        "run_root": str(ctx.run_root),
        "mode": mode,
        "refresh_data": refresh_data,
        "outputs": outputs,
        "inputs_sha256": manifest_inputs,
        "skipped": skipped,
    }
    write_json(ctx.out_dir / "manifest.json", manifest)
    return manifest
