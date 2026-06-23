"""Render a standalone LaTeX compendium for a completed hypothesis run.

The compendium is an internal audit and reading document. It reuses the
validated thesis export snippets, copies the referenced figure PDFs next to the
standalone TeX file, and adds a concise interpretation plus raw CSV previews.

Example
-------
python -m analysis.runners.render_results_compendium \
  --run-root "<artifact-root>/runs/final_20260611_queue"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from ..reporting.jof_latex import escape_latex, fmt_int, fmt_num


class ResultsCompendiumError(RuntimeError):
    """Raised when the compendium cannot be rendered safely."""


REQUIRED_SNIPPETS = (
    "tab_h1_primary.tex",
    "tab_h2_pooled.tex",
    "tab_h3_raear.tex",
)

OPTIONAL_SNIPPETS = (
    "tab_h1_tier_subgroup.tex",
    "tab_fill_robustness.tex",
)

REQUIRED_FIGURES = (
    "fig_alpha_decomposition.pdf",
    "fig_alpha_fill_frontier.pdf",
    "fig_h2_heatmap.pdf",
    "fig_raear_curve.pdf",
)

OPTIONAL_FIGURES = (
    "fig_rolling_stability.pdf",
)

FIGURE_SNIPPETS = tuple(
    name.replace(".pdf", ".tex")
    for name in (*REQUIRED_FIGURES, *OPTIONAL_FIGURES)
)

RAW_PREVIEW_FILES = (
    "hypotheses/h1/h1_primary_ttest.csv",
    "hypotheses/h1/h1_tev.csv",
    "hypotheses/h2/h2_pooled.csv",
    "hypotheses/h2/h2_per_bin_differentials.csv",
    "hypotheses/h3/h3_raear.csv",
    "hypotheses/h3/h3_tev.csv",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_tex(path: Path) -> str:
    return str(path).replace("\\", "/")


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().items()
    }


def _failure_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "reason" not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame["reason"].value_counts(dropna=False).items()
    }


def _csv_preview(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return "MISSING"
    frame = pd.read_csv(path)
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
    return frame.to_csv(index=False).strip()


def _format_failure_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "No failure rows are present."
    return ", ".join(
        f"{escape_latex(key)}: {fmt_int(value)}"
        for key, value in counts.items()
    )


def _line_item(label: str, value: Any, *, raw_value: bool = False) -> str:
    body = str(value) if raw_value else escape_latex(value)
    return f"\\item \\textbf{{{escape_latex(label)}}}: {body}"


def _manifest_gate(run_root: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    problems: list[str] = []
    status_path = run_root / "run_status.json"
    thesis_manifest_path = run_root / "thesis_exports" / "manifest.json"
    status: dict[str, Any] = {}
    thesis_manifest: dict[str, Any] = {}

    if not status_path.exists():
        problems.append("run_status.json is missing")
    else:
        status = _read_json(status_path)
        if status.get("status") != "complete":
            problems.append(
                f"run_status.json status is {status.get('status')!r}, expected 'complete'"
            )

    if not thesis_manifest_path.exists():
        problems.append("thesis_exports/manifest.json is missing")
    else:
        thesis_manifest = _read_json(thesis_manifest_path)
        if thesis_manifest.get("draft") is not False:
            problems.append("thesis_exports/manifest.json is marked draft")

    for rel in RAW_PREVIEW_FILES:
        if not (run_root / rel).exists():
            problems.append(f"{rel} is missing")

    export_dir = run_root / "thesis_exports"
    for name in REQUIRED_SNIPPETS:
        if not (export_dir / name).exists():
            problems.append(f"thesis_exports/{name} is missing")
    for name in REQUIRED_FIGURES:
        if not (export_dir / name).exists():
            problems.append(f"thesis_exports/{name} is missing")

    return problems, status, thesis_manifest


def _copy_exports(run_root: Path, out_dir: Path) -> list[str]:
    export_dir = run_root / "thesis_exports"
    copied: list[str] = []
    for name in (
        *REQUIRED_SNIPPETS,
        *OPTIONAL_SNIPPETS,
        *FIGURE_SNIPPETS,
        *REQUIRED_FIGURES,
        *OPTIONAL_FIGURES,
    ):
        src = export_dir / name
        if not src.exists():
            continue
        dst = out_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def _interpret_h1(run_root: Path) -> str:
    primary = pd.read_csv(run_root / "hypotheses/h1/h1_primary_ttest.csv").iloc[0]
    mean = float(primary["mean"])
    t_stat = float(primary["t"])
    n_obs = primary["n"]
    direction = (
        "below" if mean < 0 else "above"
    )
    support = (
        "does not support" if mean < 0 else "supports"
    )
    return (
        "The primary paired test compares S3 Full with the Market-on-Close "
        "benchmark in Window B at the one percent parent size. The estimated "
        f"differential is {fmt_num(mean)} basis points with a t-statistic of "
        f"{fmt_num(t_stat)} and {fmt_int(n_obs)} paired observations. Because "
        f"the estimate is {direction} zero, the headline result {support} the "
        "hypothesis that signal-conditioned passive liquidity provision reduces "
        "execution costs relative to MOC in this specification."
    )


def _interpret_h2(run_root: Path) -> str:
    pooled = pd.read_csv(run_root / "hypotheses/h2/h2_pooled.csv")
    pieces: list[str] = []
    label_map = {
        "OFI_marginal": "OFI marginal",
        "IMB_marginal": "imbalance marginal",
        "FULL_vs_S2": "S3 Full versus S2",
        "interaction": "interaction",
    }
    for row in pooled.itertuples(index=False):
        label = label_map.get(str(row.label), str(row.label))
        pieces.append(f"{label}: {fmt_num(row.mean)} bps, t={fmt_num(row.t)}")
    return (
        "The H2 pooled decomposition asks whether OFI and closing-pressure "
        "signals improve on the time-adaptive S2 baseline after matching on "
        "realized S2 passive fill rates. The pooled estimates are "
        + "; ".join(pieces)
        + ". The average signal contribution is economically small and negative "
        "for the individual signal variants, while the within-bin table should "
        "be read as evidence on where the signal changes are helpful or harmful "
        "conditional on realized fill-rate exposure."
    )


def _interpret_h3(run_root: Path) -> str:
    raear = pd.read_csv(run_root / "hypotheses/h3/h3_raear.csv")
    passive = raear[raear["strategy"] != "S0_MOC"].copy()
    if passive.empty:
        return "The H3 table contains no passive strategy rows."
    best_mean = passive.sort_values("mean_alpha", ascending=False).iloc[0]
    s3 = raear[raear["strategy"] == "S3_FULL"]
    s3_text = ""
    if not s3.empty:
        row = s3.iloc[0]
        s3_text = (
            f" S3 Full has mean alpha versus MOC {fmt_num(row['mean_alpha'])} bps, "
            f"TEV {fmt_num(row['tev'])}, and TES {fmt_num(row['tes'])} bps."
        )
    return (
        "H3 evaluates whether any passive strategy remains attractive once "
        "benchmark-relative tracking-error variance is penalized. Among passive strategies, "
        f"{escape_latex(best_mean['strategy'])} has the highest mean alpha versus MOC at "
        f"{fmt_num(best_mean['mean_alpha'])} basis points, but its tracking-error "
        f"variance is {fmt_num(best_mean['tev'])}."
        + s3_text
        + " The negative information ratios and RAEAR values imply that the "
        "passive variants are dominated once benchmark-relative execution risk "
        "is given positive weight."
    )


def _validation_summary(run_root: Path, status: dict[str, Any]) -> tuple[str, dict[str, int]]:
    manifest_path = run_root / "metadata" / "simulation_manifest.csv"
    failures_path = run_root / "metadata" / "simulation_failures.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_counts = _status_counts(manifest, "status")
    failure_counts = _failure_counts(failures_path)
    sim = status.get("simulation") or {}
    lines: list[str] = []
    if status.get("status") == "complete":
        lines.append(
            "The run status gate is complete and the thesis export manifest is non-draft."
        )
    else:
        lines.append(
            "This interim document is marked DRAFT because the current "
            f"\\texttt{{run\\_status.json}} reports status {escape_latex(status.get('status', 'unknown'))}. "
            "Regenerate the compendium after the run returns to complete before using it as final evidence."
        )
    if sim:
        lines.extend([
            f"The simulation reports {fmt_int(sim.get('dates_with_valid_shards', 0))} valid date shards "
            f"out of {fmt_int(sim.get('dates_expected', 0))} expected dates.",
            f"Eligible coverage is {fmt_num(float(sim.get('eligible_coverage', 0.0)) * 100, nd=4)} percent, "
            f"with {fmt_int(sim.get('critical_failures', 0))} critical failures.",
        ])
    else:
        lines.append(
            "The current \\texttt{run\\_status.json} does not include the final simulation summary fields."
        )
    lines.extend([
        "Manifest date statuses are "
        + (", ".join(f"{k}: {fmt_int(v)}" for k, v in manifest_counts.items()) or "not available")
        + ".",
        "Failure reasons are " + _format_failure_counts(failure_counts) + ".",
    ])
    return "\n\n".join(lines), failure_counts


def _warnings_text(status: dict[str, Any], failure_counts: dict[str, int]) -> str:
    sim = status.get("simulation") or {}
    tier_fallback = sim.get("tier_fallback_symbols")
    adv_path = sim.get("adv_spread_bucket_map_path")
    if tier_fallback is None:
        tier_text = (
            "The current run_status.json does not report the tier fallback count. "
            "Inspect metadata/liquidity_tier_audit.csv before citing the fallback total."
        )
    else:
        tier_text = (
            f"{fmt_int(int(tier_fallback or 0))} symbols use the conservative liquidity-tier "
            "fallback. This affects passive posting distances and should be "
            "reported as a methodological caveat."
        )
    warnings = [
        (
            f"{fmt_int(failure_counts.get('missing_expected_vc', 0))} symbol-days "
            "are skipped because expected closing-auction volume is unavailable. "
            "These are primarily new index members or ticker-transition cases "
            "with too little trailing auction history."
        ),
        (
            f"{fmt_int(failure_counts.get('empty_after_filter', 0))} symbol-days "
            "remain partial after trade and quote filtering."
        ),
        tier_text,
        (
            "The ADV x spread bucket map is "
            + ("not attached to this run." if adv_path in (None, "", "null") else f"attached at {_path_tex(Path(str(adv_path)))}.")
        ),
    ]
    return "\n".join(f"\\item {escape_latex(text)}" for text in warnings)


def _raw_preview_sections(run_root: Path) -> str:
    parts: list[str] = []
    for rel in RAW_PREVIEW_FILES:
        path = run_root / rel
        parts.extend([
            f"\\subsection{{{escape_latex(rel)}}}",
            f"\\noindent Source: \\path{{{_path_tex(path)}}}",
            "\\begin{Verbatim}[fontsize=\\scriptsize]",
            _csv_preview(path),
            "\\end{Verbatim}",
            "",
        ])
    return "\n".join(parts)


def _document_preamble(draft: bool) -> str:
    title = "Standalone Results Compendium"
    if draft:
        title += " (DRAFT)"
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[protrusion=true,expansion=false]{{microtype}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{fancyvrb}}
\usepackage{{float}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.75em}}

\newcommand{{\joflegend}}[1]{{%
  \par\smallskip
  \begin{{minipage}}{{0.92\textwidth}}\small\itshape #1\end{{minipage}}%
  \par\smallskip}}
\newcommand{{\jofnotes}}[1]{{%
  \par\smallskip
  \begin{{minipage}}{{0.92\textwidth}}\footnotesize #1\end{{minipage}}}}
\newcommand{{\jofstars}}{{$^{{***}}$, $^{{**}}$, and $^{{*}}$ denote statistical
significance at the 1\%, 5\%, and 10\% level, respectively, based on two-way
clustered standard errors by symbol and date.}}
\newcommand{{\jofpanel}}[2]{{\multicolumn{{#1}}{{@{{}}l}}{{\textit{{#2}}}} \\ \addlinespace}}

\title{{{title}}}
\author{{Generated from validated run artifacts}}
\date{{\today}}
"""


def _render_tex(
    run_root: Path,
    out_dir: Path,
    *,
    draft: bool,
    status: dict[str, Any],
    thesis_manifest: dict[str, Any],
    problems: list[str],
) -> str:
    run_config_path = run_root / "metadata" / "run_config.json"
    run_config = _read_json(run_config_path) if run_config_path.exists() else {}
    sim = status.get("simulation") or {}
    validation_text, failure_counts = _validation_summary(run_root, status)
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    draft_banner = (
        "\\begin{center}\\fbox{\\textbf{DRAFT: acceptance gates failed.}}\\end{center}\n"
        if draft else ""
    )
    problem_text = (
        "\\section*{Gate Problems}\n\\begin{itemize}\n"
        + "\n".join(f"\\item {escape_latex(p)}" for p in problems)
        + "\n\\end{itemize}\n"
        if problems else ""
    )

    metadata_items = [
        _line_item("Run root", rf"\path{{{_path_tex(run_root)}}}", raw_value=True),
        _line_item("Run status", status.get("status", "unknown")),
        _line_item("Updated at", status.get("updated_at", "unknown")),
        _line_item("Simulation fingerprint", sim.get("fingerprint", thesis_manifest.get("simulation_fingerprint", "unknown"))),
        _line_item("Feature policy", thesis_manifest.get("feature_policy", "unknown")),
        _line_item("Fill specification", run_config.get("fill_specification", "unknown")),
        _line_item("Sample", f"{run_config.get('start', 'unknown')} to {run_config.get('end', 'unknown')}"),
        _line_item("Generated at", generated_at),
    ]

    return "\n".join([
        _document_preamble(draft),
        "\\begin{document}",
        "\\maketitle",
        draft_banner,
        "\\section*{Run Metadata}",
        "\\begin{itemize}",
        *metadata_items,
        "\\end{itemize}",
        problem_text,
        "\\section{Validation Summary}",
        validation_text,
        "\\section{H1: Passive Execution versus Market-on-Close}",
        _interpret_h1(run_root),
        "\\input{tab_h1_primary.tex}",
        "\\input{fig_alpha_decomposition.tex}",
        "\\input{fig_alpha_fill_frontier.tex}",
        "\\section{H2: Signal Contribution}",
        _interpret_h2(run_root),
        "\\input{tab_h2_pooled.tex}",
        "\\input{fig_h2_heatmap.tex}",
        "\\section{H3: Tracking-Error Trade-off}",
        _interpret_h3(run_root),
        "\\input{tab_h3_raear.tex}",
        "\\input{fig_raear_curve.tex}",
        "\\section{Methodological Warnings}",
        "\\begin{itemize}",
        _warnings_text(status, failure_counts),
        "\\end{itemize}",
        "\\section{Additional Thesis Exports}",
        "\\input{tab_h1_tier_subgroup.tex}" if (out_dir / "tab_h1_tier_subgroup.tex").exists() else "",
        "\\input{tab_fill_robustness.tex}" if (out_dir / "tab_fill_robustness.tex").exists() else "",
        "\\input{fig_rolling_stability.tex}" if (out_dir / "fig_rolling_stability.tex").exists() else "",
        "\\appendix",
        "\\section{Raw CSV Previews}",
        "The following previews are copied directly from the run CSV artifacts. "
        "They are intended for audit and orientation, not as publication tables.",
        _raw_preview_sections(run_root),
        "\\end{document}",
        "",
    ])


def _write_readme(out_dir: Path, run_root: Path, draft: bool, compiled_pdf: bool) -> None:
    lines = [
        "# Results compendium",
        "",
        "This folder contains a standalone reading document generated from a validated run bundle.",
        "",
        f"- run root: `{run_root}`",
        f"- draft: `{draft}`",
        f"- TeX: `results_compendium.tex`",
        f"- PDF compiled: `{compiled_pdf}`",
        "",
        "The compendium reuses `thesis_exports` snippets and adds interpretation plus raw CSV previews.",
    ]
    (out_dir / "README_results_compendium.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _compile_pdf(out_dir: Path, tex_name: str) -> tuple[bool, str]:
    latexmk = shutil.which("latexmk")
    if latexmk:
        cmd = [
            latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
            tex_name,
        ]
        proc = subprocess.run(
            cmd, cwd=out_dir, text=True, capture_output=True, timeout=120,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return False, "No latexmk or pdflatex executable found on PATH."
    log_parts: list[str] = []
    ok = True
    for _ in range(2):
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_name],
            cwd=out_dir, text=True, capture_output=True, timeout=120,
        )
        log_parts.append(proc.stdout + proc.stderr)
        ok = ok and proc.returncode == 0
        if proc.returncode != 0:
            break
    return ok, "\n".join(log_parts)


def render(
    run_root: Path,
    out_dir: Path | None = None,
    *,
    allow_incomplete: bool = False,
    compile_pdf: bool = False,
) -> dict[str, Any]:
    """Render the standalone compendium and return its manifest."""
    run_root = Path(run_root)
    out_dir = Path(out_dir) if out_dir else run_root / "results_compendium"
    problems, status, thesis_manifest = _manifest_gate(run_root)
    draft = bool(problems)
    if draft and not allow_incomplete:
        raise ResultsCompendiumError(
            "Run bundle failed the compendium acceptance gate:\n  - "
            + "\n  - ".join(problems)
            + "\nUse --allow-incomplete to render visibly marked DRAFT output."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_exports(run_root, out_dir)
    tex = _render_tex(
        run_root, out_dir, draft=draft, status=status,
        thesis_manifest=thesis_manifest, problems=problems,
    )
    tex_path = out_dir / "results_compendium.tex"
    tex_path.write_text(tex, encoding="utf-8")

    compiled_pdf = False
    compile_log = ""
    if compile_pdf:
        compiled_pdf, compile_log = _compile_pdf(out_dir, tex_path.name)
        if not compiled_pdf:
            (out_dir / "results_compendium_compile.log").write_text(
                compile_log, encoding="utf-8",
            )

    _write_readme(out_dir, run_root, draft, compiled_pdf)

    inputs = {
        rel: _sha256(run_root / rel)
        for rel in RAW_PREVIEW_FILES
        if (run_root / rel).exists()
    }
    for name in (
        *REQUIRED_SNIPPETS,
        *OPTIONAL_SNIPPETS,
        *FIGURE_SNIPPETS,
        *REQUIRED_FIGURES,
        *OPTIONAL_FIGURES,
    ):
        path = run_root / "thesis_exports" / name
        if path.exists():
            inputs[f"thesis_exports/{name}"] = _sha256(path)

    manifest = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "draft": draft,
        "gate_problems": problems,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tex": str(tex_path),
        "pdf": str(out_dir / "results_compendium.pdf") if compiled_pdf else None,
        "compiled_pdf": compiled_pdf,
        "copied_exports": copied,
        "inputs_sha256": inputs,
    }
    (out_dir / "results_compendium_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    manifest = render(
        args.run_root,
        out_dir=args.out_dir,
        allow_incomplete=args.allow_incomplete,
        compile_pdf=args.compile_pdf,
    )
    print(json.dumps({
        "out_dir": manifest["out_dir"],
        "tex": manifest["tex"],
        "pdf": manifest["pdf"],
        "draft": manifest["draft"],
    }, indent=2))


if __name__ == "__main__":
    main()
