"""Local Streamlit studio for curating thesis figure plot tables and layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODING_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "runs" / "final_v4_20260618_queue"
DEFAULT_OUT_DIR = REPO_ROOT / "thesis" / "figures" / "final_20260618_queue"
if str(CODING_ROOT) not in sys.path:
    sys.path.insert(0, str(CODING_ROOT))

from analysis.reporting.thesis_figures.data import FigureBuildContext  # noqa: E402
from analysis.reporting.thesis_figures.io import read_yaml, write_yaml  # noqa: E402
from analysis.reporting.thesis_figures.spec import load_specs  # noqa: E402
from analysis.reporting.thesis_figures.suite import (  # noqa: E402
    curated_input_path,
    figure_input_path,
    load_plot_table,
    override_path,
    render_figures,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--as-horizon-csv", type=Path, default=None)
    parser.add_argument("--size-grid-root", type=Path, default=None)
    parser.add_argument("--volume-by-date-csv", type=Path, default=None)
    args, extras = parser.parse_known_args()
    args.ignored_args = extras
    return args


def main() -> None:
    import streamlit as st

    args = _parse_args()
    startup_warnings = []
    if args.ignored_args:
        startup_warnings.append(
            "Ignored extra command-line fragments. If your paths contain spaces, wrap them in quotes."
        )
    if args.run_root is None or not args.run_root.exists():
        if args.run_root is not None:
            startup_warnings.append(f"Run root not found: {args.run_root}. Using default run root.")
        args.run_root = DEFAULT_RUN_ROOT
    if args.out is None:
        args.out = DEFAULT_OUT_DIR

    specs = load_specs()
    ctx = FigureBuildContext(
        run_root=args.run_root,
        out_dir=args.out,
        stats_dir=args.stats_dir,
        as_horizon_csv=args.as_horizon_csv,
        size_grid_root=args.size_grid_root,
        volume_by_date_csv=args.volume_by_date_csv,
    )

    st.set_page_config(page_title="Thesis Figure Studio", layout="wide")
    st.title("Thesis Figure Studio")
    for warning in startup_warnings:
        st.warning(warning)
    if not args.run_root.exists():
        st.error(f"Run root does not exist: {args.run_root}")
        st.stop()

    with st.sidebar:
        fig_id = st.selectbox("Figure", list(specs), index=0)
        mode = st.radio("Mode", ["curated", "artifact"], horizontal=True)
        refresh = st.button("Refresh from artifacts")
        render_selected = st.button("Render selected")
        render_all = st.button("Render all")

    spec = specs[fig_id]
    if refresh:
        load_plot_table(spec, ctx, mode="artifact", refresh_data=True)

    frame, data_source, _inputs = load_plot_table(spec, ctx, mode=mode, refresh_data=False)
    override_file = override_path(ctx.out_dir, spec)
    override = read_yaml(override_file) if override_file.exists() else {}

    left, right = st.columns([1.25, 0.75])
    with left:
        st.caption(f"Source: {data_source} | artifact table: {figure_input_path(ctx.out_dir, spec)}")
        edited = st.data_editor(frame, num_rows="dynamic", width="stretch")
        if st.button("Save curated table"):
            path = curated_input_path(ctx.out_dir, spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(edited).to_csv(path, index=False)
            st.success(f"Saved {path}")

    with right:
        st.subheader("Layout override")
        latex = dict(override.get("latex", {}))
        layout = dict(override.get("layout", {}))
        layout["xlabel"] = st.text_input("X label", value=layout.get("xlabel", spec.layout.get("xlabel", "")))
        layout["ylabel"] = st.text_input("Y label", value=layout.get("ylabel", spec.layout.get("ylabel", "")))
        latex["include_width"] = st.text_input(
            "LaTeX width",
            value=latex.get("include_width", spec.latex.get("include_width", "0.88\\textwidth")),
        )
        if st.checkbox("Log x-axis", value=(layout.get("xscale", spec.layout.get("xscale")) == "log")):
            layout["xscale"] = "log"
        else:
            layout.pop("xscale", None)
        if st.button("Save layout override"):
            write_yaml(override_file, {"layout": layout, "latex": latex})
            st.success(f"Saved {override_file}")

        if render_selected:
            manifest = render_figures(ctx, figure=fig_id, mode=mode, refresh_data=False)
            st.json(manifest["outputs"].get(fig_id, {}))
        if render_all:
            manifest = render_figures(ctx, figure="all", mode=mode, refresh_data=False)
            st.json({"outputs": len(manifest["outputs"]), "skipped": manifest["skipped"]})

    preview = ctx.out_dir / f"{fig_id}.png"
    if preview.exists():
        st.image(str(preview), width="stretch")


if __name__ == "__main__":
    main()
