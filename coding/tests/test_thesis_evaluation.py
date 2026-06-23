from __future__ import annotations

import csv
from pathlib import Path

from analysis.reporting import thesis_evaluation as te


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parse_chapters_and_citations() -> None:
    tex = "\n".join([
        r"\begin{abstract}Text \parencite{alpha2020}.\end{abstract}",
        r"\chapter{Introduction}",
        r"\section{Motivation}",
        r"Claim \textcite{beta2021,gamma2022}.",
        r"\appendix",
        r"\chapter{Implementation and Reproducibility}",
        r"Appendix text.",
    ])
    chapters = te.parse_chapters(tex)
    citations = te.collect_citations(tex, chapters)
    assert [chapter.key for chapter in chapters] == [
        "Front Matter",
        "Introduction",
        "Implementation and Reproducibility",
    ]
    assert {use.key for use in citations} == {"alpha2020", "beta2021", "gamma2022"}
    assert [use.chapter for use in citations if use.key == "beta2021"] == ["Introduction"]


def test_bibtex_parser_handles_nested_braces() -> None:
    bib = """
@article{cont2014,
  author = {Cont, Rama and Kukanov, Arseniy},
  title = {The Price Impact of Order Book Events},
  year = {2014},
  journaltitle = {Journal of Financial Econometrics},
  doi = {10.1093/jjfinec/nbt003}
}
"""
    entries = te.parse_bibtex(bib)
    assert entries["cont2014"].fields["title"] == "The Price Impact of Order Book Events"
    assert entries["cont2014"].fields["doi"] == "10.1093/jjfinec/nbt003"


def test_numeric_checks_flag_old_h3_values() -> None:
    tex = (
        "S3 full $-$ S0 MOC & $-$0.12 & & $-$1.50 & 0.135 & 187,309 "
        "S3 full relative to S0 MOC is -0.117 basis points, with a two-way-clustered "
        "\\(t\\)-statistic of -1.50 and a two-sided \\(p\\)-value of 0.135. "
        "The dissemination-status subgroup in the primary H1 cell reports $-$0.12, $-$1.50, 0.135. "
        "The H3 analysis evaluates each strategy on MOC-relative net-alpha differentials in the same primary H1 cell. "
        "S3 full has a mean alpha of -0.05 basis points and TES 17.79. "
        "S3 Full & $-$0.12 & 317.71 & 17.82 "
        "S3 Full $-$ S2 & $-$0.01 & $-$1.18 & 0.881 & 1.000 & 0.316 & 561,927"
    )
    rows, findings = te.numeric_checks(tex)
    failed = [row["check"] for row in rows if row["status"] == "fail"]
    assert "H3 primary cell risk table" in failed
    assert any(finding.category == "numeric_consistency" for finding in findings)


def test_evaluate_thesis_writes_reports(tmp_path: Path) -> None:
    tex = r"""
\begin{abstract}The final evidence cites \parencite{ok2020}.\end{abstract}
\chapter{Introduction}
\section{Motivation}
The headline queue-aware replay run reports S3 full relative to S0 MOC is -0.117 basis points, with a two-way-clustered \(t\)-statistic of -1.50 and a two-sided \(p\)-value of 0.135.
\chapter{Empirical Results}
\section{H1}
S3 full $-$ S0 MOC & $-$0.12 & & $-$1.50 & 0.135 & 187,309 \\
The dissemination-status subgroup therefore contains only the pre-dissemination category in the primary H1 cell, with a mean S3-full differential against MOC of $-$0.12 basis points, a two-way-clustered \(t\)-statistic of $-$1.50, and a Holm-adjusted \(p\)-value of 0.135.
The H3 analysis evaluates each strategy on MOC-relative net-alpha differentials in the same primary H1 cell. S3 full has a mean alpha of -0.12 basis points relative to MOC and a tracking-error standard deviation of 17.82 basis points.
S3 Full & $-$0.12 & 317.71 & 17.82 & $-$0.007 \\
S3 Full $-$ S2 & $-$0.01 & $-$1.18 & 0.881 & 1.000 & 0.316 & 561,927 \\
\appendix
\chapter{Implementation and Reproducibility}
Appendix text.
"""
    bib = r"""
@article{ok2020,
  author = {Author, Ann},
  title = {A Verified Paper},
  year = {2020},
  journaltitle = {Journal}
}
@book{unused2021,
  author = {Writer, Bob},
  title = {Unused Book},
  year = {2021},
  publisher = {Press}
}
"""
    tex_path = tmp_path / "thesis.tex"
    bib_path = tmp_path / "references.bib"
    out_dir = tmp_path / "eval"
    _write(tex_path, tex)
    _write(bib_path, bib)

    summary = te.evaluate_thesis(tex_path, bib_path, out_dir)

    assert summary["chapters"] == 4
    assert (out_dir / "00_dashboard.md").exists()
    assert (out_dir / "literature_verification.csv").exists()
    assert (out_dir / "numeric_consistency.csv").exists()
    with (out_dir / "numeric_consistency.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["status"] == "pass" for row in rows)
