"""Journal-of-Finance-style LaTeX snippet builders for thesis exports.

The generated snippets rely on the four macros defined in the thesis
preamble (``\\joflegend``, ``\\jofnotes``, ``\\jofstars``, ``\\jofpanel``)
and reproduce the template layout exactly, so a snippet replaces the
corresponding placeholder table or figure in ``thesis0506.tex`` wholesale.
"""

from __future__ import annotations

import numpy as np

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain-text cell content."""
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def fmt_num(value, nd: int = 2, dash: str = "--") -> str:
    """Format a number with ``nd`` decimals; math-mode minus; dash for NaN."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    body = f"{abs(v):,.{nd}f}"
    return f"$-${body}" if v < 0 else body


def fmt_int(value, dash: str = "--") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(v):
        return dash
    return f"{int(round(v)):,}"


def stars(p_value) -> str:
    """Significance stars at the 1/5/10 percent levels."""
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return r"$^{***}$"
    if p < 0.05:
        return r"$^{**}$"
    if p < 0.10:
        return r"$^{*}$"
    return ""


def with_stars(value, p_value, nd: int = 2) -> str:
    base = fmt_num(value, nd=nd)
    if base == "--":
        return base
    return base + stars(p_value)


def paren_t(t_value, nd: int = 2) -> str:
    """t-statistic in parentheses for the line below an estimate."""
    body = fmt_num(t_value, nd=nd)
    return f"({body})" if body != "--" else "(--)"


def jof_table(
    *,
    caption: str,
    label: str,
    legend: str,
    column_format: str,
    header: str,
    body_lines: list[str],
    notes: str,
    provenance: str = "",
    resize: bool = False,
    include_stars_note: bool = True,
) -> str:
    """Assemble a complete JoF-style table environment.

    ``header`` is the column-header row (without trailing ``\\\\``);
    ``body_lines`` are full tabular lines including their own terminators
    (so panel rows via ``\\jofpanel`` and (t)-rows are possible).
    """
    open_resize = "\\resizebox{\\textwidth}{!}{%\n" if resize else ""
    close_resize = "}%\n" if resize else ""
    star_note = " \\jofstars" if include_stars_note else ""
    parts = [
        provenance,
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\joflegend{{{legend}}}",
        open_resize + f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        header + r" \\" if header else "",
        "\\midrule" if header else "",
        *body_lines,
        "\\bottomrule",
        f"\\end{{tabular}}" + ("\n" + close_resize.rstrip("\n") if resize else ""),
        f"\\jofnotes{{{notes}{star_note}}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(part for part in parts if part != "")


def jof_figure(
    *,
    graphics_file: str,
    caption: str,
    label: str,
    legend: str,
    provenance: str = "",
    width: str = "0.88\\textwidth",
) -> str:
    """Assemble a complete JoF-style figure environment."""
    parts = [
        provenance,
        "\\begin{figure}[htbp]",
        "\\centering",
        f"\\includegraphics[width={width}]{{{graphics_file}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\jofnotes{{{legend}}}",
        "\\end{figure}",
        "",
    ]
    return "\n".join(part for part in parts if part != "")
