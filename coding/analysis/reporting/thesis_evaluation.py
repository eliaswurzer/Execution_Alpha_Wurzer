"""Read-only thesis evaluation suite.

The suite audits a LaTeX thesis source chapter by chapter and writes
machine-readable plus human-readable reports. It deliberately does not edit
the source document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


CHAPTER_RE = re.compile(r"^\s*\\chapter\*?\{(.+?)\}")
SECTION_RE = re.compile(r"^\s*\\(section|subsection|subsubsection)\*?\{(.+?)\}")
CITE_RE = re.compile(
    r"\\(?:textcite|parencite|cite|citep|citet|autocite|footcite|footfullcite|"
    r"citeauthor|citeyear)(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]+)\}"
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\{([^}]+)\}")
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")
LATEX_INLINE_RE = re.compile(r"\$([^$]+)\$")


SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


CHAPTER_CRITERIA = {
    "Front Matter": [
        "Abstract states the research object, headline evidence, and conservative interpretation.",
        "Front matter is free of draft markers and unresolved citations.",
    ],
    "Introduction": [
        "Motivation, scope, research question, and contributions are explicit.",
        "Headline result language matches Chapters 7 and 8.",
        "The chapter does not overpromise beyond validated evidence.",
    ],
    "Institutional Background and Related Literature": [
        "Institutional claims are cited and phrased as source-supported facts.",
        "Literature streams are synthesized rather than only listed.",
        "The gap maps clearly into the thesis design.",
    ],
    "Theoretical Framework": [
        "Notation is internally consistent.",
        "Equations map to strategy definitions and empirical objects.",
        "Risk and adverse-selection conventions are transparent.",
    ],
    "Hypothesis Development": [
        "Each hypothesis maps to a testable object.",
        "Directionality and multiple-testing discipline are explicit.",
        "Exploratory diagnostics are not framed as confirmatory tests.",
    ],
    "Data": [
        "Sample boundaries, exclusions, and data provenance are documented.",
        "DTAQ limitations are not overstated into full-book observability.",
        "Sample statistics agree with validated artifacts and results text.",
    ],
    "Empirical Methodology": [
        "Parent-order, fill, inference, and robustness protocols are reproducible.",
        "Metric definitions match tables and figures.",
        "Simulation assumptions are stated without turning into unsupported facts.",
    ],
    "Empirical Results": [
        "Prose, tables, figures, and artifacts agree numerically.",
        "H1, H2, H3, subgroup, and robustness interpretations stay within evidence.",
        "Null results are described as not rejected, not as evidence of absence.",
    ],
    "Conclusion": [
        "Summary repeats only validated headline findings.",
        "Limitations are concrete and not defensive filler.",
        "Future research does not introduce new empirical evidence.",
    ],
    "Implementation and Reproducibility": [
        "Artifact paths, commands, and acceptance criteria are internally consistent.",
        "Appendix material supports reproducibility without superseding thesis evidence.",
    ],
}


OLD_PHRASE_PATTERNS = {
    "old_rerender_path": r"rerender_20260616_remaining",
    "old_intended_full_sample": r"intended full-sample design",
    "old_unavailable_full_panel": r"not yet available",
    "old_intended_sample_split": r"intended 2018--2019 sample split",
}


CRITICAL_NUMERIC_CHECKS = [
    {
        "name": "H1 primary paired differential",
        "severity": "P0",
        "must_contain": [
            r"S3 full \$-\$ S0 MOC & \$-\$0\.12 .*187,309",
            r"S3 full relative to S0 MOC.*-0\.117.*t\\\)-statistic of -1\.50.*p\\\)-value of 0\.135",
        ],
        "must_not_contain": [],
        "rationale": "H1 headline result must stay on the primary Window-B, one-percent cell.",
    },
    {
        "name": "Dissemination subgroup primary cell",
        "severity": "P0",
        "must_contain": [
            r"dissemination-status subgroup.*primary H1 cell.*\$-\$0\.12.*\$-\$1\.50.*0\.135",
        ],
        "must_not_contain": [
            r"mean S3-full differential against MOC of \$-\$0\.05",
            r"t\s*\$?-\s*0\.79",
        ],
        "rationale": "The dissemination split should not report the old all-window diagnostic.",
    },
    {
        "name": "H3 primary cell risk table",
        "severity": "P0",
        "must_contain": [
            r"H3 analysis evaluates each strategy.*same primary H1 cell",
            r"S3 full has a mean alpha of -0\.12.*17\.82",
            r"S3 Full & \$-\$0\.12 & 317\.71 & 17\.82",
        ],
        "must_not_contain": [
            r"mean alpha of -0\.05",
            r"17\.79",
        ],
        "rationale": "H3 must use the same primary execution object as H1.",
    },
    {
        "name": "H2 all-window sample size retained",
        "severity": "P2",
        "must_contain": [
            r"S3 Full \$-\$ S2 .*561,927",
        ],
        "must_not_contain": [],
        "rationale": "H2 signal tests intentionally remain all-window diagnostics.",
    },
]


OVERMENTION_TERMS = [
    "public-data",
    "small-parent",
    "validated headline",
    "not rejected",
    "same primary H1 cell",
    "MOC",
    "queue-aware",
    "point-in-time",
]


@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    start_line: int
    end_line: int
    text: str
    sections: list[str]


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    location: str
    line: int | None
    message: str
    recommendation: str
    evidence: str = ""


@dataclass(frozen=True)
class CitationUse:
    key: str
    chapter: str
    line: int
    context: str


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


def _clean_title(title: str) -> str:
    title = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", title)
    title = title.replace("{", "").replace("}", "")
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def strip_latex(text: str) -> str:
    text = LATEX_INLINE_RE.sub(r"\1", text)
    text = text.replace(r"\&", "&").replace(r"\%", "%")
    text = text.replace("$-$", "-")
    text = LATEX_COMMAND_RE.sub(lambda m: m.group(1) or " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = strip_latex(text).lower()
    text = re.sub(r"[^a-z0-9.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_chapters(tex: str) -> list[Chapter]:
    lines = tex.splitlines()
    starts: list[tuple[int, str]] = []
    appendix_seen = False
    for idx, line in enumerate(lines, start=1):
        if line.strip() == r"\appendix":
            appendix_seen = True
        match = CHAPTER_RE.match(line)
        if match:
            title = strip_latex(match.group(1))
            key = title
            if appendix_seen and title == "Implementation and Reproducibility":
                key = "Implementation and Reproducibility"
            starts.append((idx, key))

    chapters: list[Chapter] = []
    first_start = starts[0][0] if starts else len(lines) + 1
    front_text = "\n".join(lines[: first_start - 1])
    chapters.append(Chapter(
        key="Front Matter",
        title="Front Matter",
        start_line=1,
        end_line=max(1, first_start - 1),
        text=front_text,
        sections=[],
    ))

    for pos, (start, title) in enumerate(starts):
        end = starts[pos + 1][0] - 1 if pos + 1 < len(starts) else len(lines)
        body_lines = lines[start - 1:end]
        sections = []
        for line in body_lines:
            section_match = SECTION_RE.match(line)
            if section_match:
                sections.append(strip_latex(section_match.group(2)))
        chapters.append(Chapter(
            key=title,
            title=title,
            start_line=start,
            end_line=end,
            text="\n".join(body_lines),
            sections=sections,
        ))
    return chapters


def find_chapter_for_line(chapters: list[Chapter], line_no: int) -> str:
    for chapter in chapters:
        if chapter.start_line <= line_no <= chapter.end_line:
            return chapter.key
    return "Unknown"


def collect_citations(tex: str, chapters: list[Chapter]) -> list[CitationUse]:
    uses: list[CitationUse] = []
    for idx, line in enumerate(tex.splitlines(), start=1):
        for match in CITE_RE.finditer(line):
            keys = [part.strip() for part in match.group(1).split(",") if part.strip()]
            context = strip_latex(line)[:240]
            chapter = find_chapter_for_line(chapters, idx)
            uses.extend(CitationUse(key=key, chapter=chapter, line=idx, context=context) for key in keys)
    return uses


def _find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{" and (idx == 0 or text[idx - 1] != "\\"):
            depth += 1
        elif char == "}" and (idx == 0 or text[idx - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_entry_header(entry: str) -> tuple[str, str, str] | None:
    match = re.match(r"@([A-Za-z]+)\s*\{\s*([^,]+)\s*,", entry, flags=re.DOTALL)
    if not match:
        return None
    entry_type = match.group(1).lower()
    key = match.group(2).strip()
    body = entry[match.end():].rstrip().rstrip("}")
    return entry_type, key, body


def parse_bibtex(bib_text: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}
    idx = 0
    while True:
        at = bib_text.find("@", idx)
        if at == -1:
            break
        open_idx = bib_text.find("{", at)
        if open_idx == -1:
            break
        close_idx = _find_matching_brace(bib_text, open_idx)
        if close_idx == -1:
            break
        raw = bib_text[at:close_idx + 1]
        idx = close_idx + 1
        parsed = _split_entry_header(raw)
        if not parsed:
            continue
        entry_type, key, body = parsed
        fields: dict[str, str] = {}
        pos = 0
        while pos < len(body):
            field_match = re.search(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[pos:])
            if not field_match:
                break
            name = field_match.group(1).lower()
            value_start = pos + field_match.end()
            while value_start < len(body) and body[value_start].isspace():
                value_start += 1
            if value_start >= len(body):
                break
            delimiter = body[value_start]
            if delimiter == "{":
                value_end = _find_matching_brace(body, value_start)
                if value_end == -1:
                    break
                value = body[value_start + 1:value_end]
                pos = value_end + 1
            elif delimiter == '"':
                value_end = value_start + 1
                while value_end < len(body):
                    if body[value_end] == '"' and body[value_end - 1] != "\\":
                        break
                    value_end += 1
                value = body[value_start + 1:value_end]
                pos = value_end + 1
            else:
                value_end = value_start
                while value_end < len(body) and body[value_end] not in ",\n":
                    value_end += 1
                value = body[value_start:value_end]
                pos = value_end + 1
            fields[name] = _clean_title(value)
        entries[key] = BibEntry(entry_type=entry_type, key=key, fields=fields)
    return entries


def marker_findings(tex: str, chapters: list[Chapter]) -> list[Finding]:
    findings: list[Finding] = []
    marker_re = re.compile(r"TODO|FIXME|TBD|\[To be completed\]|\?\?\?", re.IGNORECASE)
    loud_placeholder_re = re.compile(r"\bPLACEHOLDER\b")
    table_placeholder_re = re.compile(r"(^|&)\s*--\s*(&|\\\\)")
    for line_no, line in enumerate(tex.splitlines(), start=1):
        chapter = find_chapter_for_line(chapters, line_no)
        if marker_re.search(line) or loud_placeholder_re.search(line):
            findings.append(Finding(
                severity="P0",
                category="draft_marker",
                location=chapter,
                line=line_no,
                message="Draft marker or placeholder text remains in the thesis source.",
                recommendation="Replace the placeholder with final prose or remove the unfinished item.",
                evidence=strip_latex(line)[:240],
            ))
        if table_placeholder_re.search(line) and "S0 MOC" not in line:
            findings.append(Finding(
                severity="P0",
                category="table_placeholder",
                location=chapter,
                line=line_no,
                message="A table cell appears to contain a placeholder double dash.",
                recommendation="Replace placeholder cells with final values or remove the row/table.",
                evidence=line.strip()[:240],
            ))
        for name, pattern in OLD_PHRASE_PATTERNS.items():
            if re.search(pattern, line, flags=re.IGNORECASE):
                findings.append(Finding(
                    severity="P1",
                    category=name,
                    location=chapter,
                    line=line_no,
                    message="Old intermediate-run wording or path remains in the source.",
                    recommendation="Use the final validated-run convention instead.",
                    evidence=strip_latex(line)[:240],
                ))
    return findings


def label_ref_findings(tex: str, chapters: list[Chapter]) -> list[Finding]:
    labels: dict[str, int] = {}
    duplicates: list[tuple[str, int, int]] = []
    refs: list[tuple[str, int]] = []
    for line_no, line in enumerate(tex.splitlines(), start=1):
        for label in LABEL_RE.findall(line):
            if label in labels:
                duplicates.append((label, labels[label], line_no))
            else:
                labels[label] = line_no
        for ref in REF_RE.findall(line):
            refs.append((ref, line_no))

    findings: list[Finding] = []
    for label, first, second in duplicates:
        findings.append(Finding(
            severity="P0",
            category="duplicate_label",
            location=find_chapter_for_line(chapters, second),
            line=second,
            message=f"Duplicate LaTeX label `{label}`.",
            recommendation=f"Rename one label; first occurrence is on line {first}.",
            evidence=label,
        ))
    for ref, line_no in refs:
        if ref not in labels:
            findings.append(Finding(
                severity="P0",
                category="missing_label",
                location=find_chapter_for_line(chapters, line_no),
                line=line_no,
                message=f"Reference `{ref}` has no matching label.",
                recommendation="Add the missing label or correct the reference key.",
                evidence=ref,
            ))
    return findings


def numeric_checks(tex: str) -> tuple[list[dict[str, str]], list[Finding]]:
    joined = re.sub(r"\s+", " ", tex)
    rows: list[dict[str, str]] = []
    findings: list[Finding] = []
    for check in CRITICAL_NUMERIC_CHECKS:
        status = "pass"
        detail_parts: list[str] = []
        for pattern in check["must_contain"]:
            if re.search(pattern, joined, flags=re.IGNORECASE):
                detail_parts.append(f"present: {pattern}")
            else:
                status = "fail"
                detail_parts.append(f"missing: {pattern}")
        for pattern in check["must_not_contain"]:
            if re.search(pattern, joined, flags=re.IGNORECASE):
                status = "fail"
                detail_parts.append(f"forbidden-present: {pattern}")
            else:
                detail_parts.append(f"absent: {pattern}")
        rows.append({
            "check": str(check["name"]),
            "status": status,
            "severity": str(check["severity"]),
            "rationale": str(check["rationale"]),
            "details": " | ".join(detail_parts),
        })
        if status != "pass":
            findings.append(Finding(
                severity=str(check["severity"]),
                category="numeric_consistency",
                location="Cross-Chapter",
                line=None,
                message=f"Numeric check failed: {check['name']}.",
                recommendation=str(check["rationale"]),
                evidence=" | ".join(detail_parts),
            ))
    return rows, findings


def literature_rows(
    citation_uses: list[CitationUse],
    bib_entries: dict[str, BibEntry],
    *,
    online: bool = False,
    online_limit: int | None = None,
    cache_path: Path | None = None,
) -> tuple[list[dict[str, str]], list[Finding]]:
    cited_keys = sorted({use.key for use in citation_uses})
    uses_by_key: dict[str, list[CitationUse]] = {}
    for use in citation_uses:
        uses_by_key.setdefault(use.key, []).append(use)

    cache = _load_cache(cache_path)
    online_budget = online_limit if online_limit is not None else len(cited_keys)
    online_checked = 0
    rows: list[dict[str, str]] = []
    findings: list[Finding] = []

    for key in cited_keys:
        entry = bib_entries.get(key)
        first_use = uses_by_key[key][0]
        row = {
            "key": key,
            "cited": "yes",
            "uses": str(len(uses_by_key[key])),
            "first_chapter": first_use.chapter,
            "first_line": str(first_use.line),
            "entry_type": entry.entry_type if entry else "",
            "title": entry.fields.get("title", "") if entry else "",
            "year": _entry_year(entry) if entry else "",
            "doi": entry.fields.get("doi", "") if entry else "",
            "url": entry.fields.get("url", "") if entry else "",
            "local_status": "ok",
            "online_status": "not_run",
            "online_source": "",
            "title_similarity": "",
            "notes": "",
        }
        if entry is None:
            row["local_status"] = "missing_bib_entry"
            row["notes"] = "Citation key is used in the thesis but absent from references.bib."
            findings.append(Finding(
                severity="P0",
                category="missing_bib_entry",
                location=first_use.chapter,
                line=first_use.line,
                message=f"Citation key `{key}` has no BibTeX entry.",
                recommendation="Add the verified source to references.bib or remove/correct the citation.",
                evidence=first_use.context,
            ))
            rows.append(row)
            continue

        missing_fields = _missing_core_fields(entry)
        if missing_fields:
            row["local_status"] = "metadata_incomplete"
            row["notes"] = "Missing fields: " + ", ".join(missing_fields)
            findings.append(Finding(
                severity="P1",
                category="bib_metadata_incomplete",
                location=first_use.chapter,
                line=first_use.line,
                message=f"BibTeX entry `{key}` is missing core metadata.",
                recommendation=f"Complete fields: {', '.join(missing_fields)}.",
                evidence=row["title"],
            ))
        if _looks_suspicious(entry):
            row["local_status"] = "metadata_suspicious"
            row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Potential placeholder or fragile metadata."
            findings.append(Finding(
                severity="P1",
                category="bib_metadata_suspicious",
                location=first_use.chapter,
                line=first_use.line,
                message=f"BibTeX entry `{key}` contains suspicious metadata.",
                recommendation="Verify the entry against the primary source.",
                evidence=json.dumps(entry.fields, ensure_ascii=False)[:240],
            ))

        if online and online_checked < online_budget:
            online_checked += 1
            verification = verify_bib_entry_online(entry, cache)
            row.update(verification)
            if verification["online_status"] in {"metadata_mismatch", "source_not_found", "online_error"}:
                findings.append(Finding(
                    severity="P1",
                    category="literature_online_verification",
                    location=first_use.chapter,
                    line=first_use.line,
                    message=f"Online literature verification flagged `{key}` as {verification['online_status']}.",
                    recommendation="Open the cited source manually and confirm title, year, and claim support.",
                    evidence=verification.get("notes", ""),
                ))
        elif online:
            row["online_status"] = "not_checked_budget_exhausted"
        rows.append(row)

    unused = sorted(set(bib_entries) - set(cited_keys))
    for key in unused:
        entry = bib_entries[key]
        rows.append({
            "key": key,
            "cited": "no",
            "uses": "0",
            "first_chapter": "",
            "first_line": "",
            "entry_type": entry.entry_type,
            "title": entry.fields.get("title", ""),
            "year": _entry_year(entry),
            "doi": entry.fields.get("doi", ""),
            "url": entry.fields.get("url", ""),
            "local_status": "unused_reference",
            "online_status": "not_run",
            "online_source": "",
            "title_similarity": "",
            "notes": "BibTeX entry is not cited in the thesis.",
        })

    _save_cache(cache_path, cache)
    return rows, findings


def _missing_core_fields(entry: BibEntry) -> list[str]:
    missing = []
    fields = entry.fields
    if not fields.get("title"):
        missing.append("title")
    if not _entry_year(entry):
        missing.append("year")
    if not (fields.get("author") or fields.get("editor") or fields.get("institution") or fields.get("organization")):
        missing.append("author/editor/institution")
    source_fields = ["journaltitle", "journal", "booktitle", "publisher", "institution", "organization", "url", "doi"]
    if not any(fields.get(field) for field in source_fields):
        missing.append("publication/source")
    return missing


def _entry_year(entry: BibEntry) -> str:
    year = entry.fields.get("year", "")
    if year:
        return year
    date = entry.fields.get("date", "")
    match = re.search(r"\d{4}", date)
    return match.group(0) if match else ""


def _looks_suspicious(entry: BibEntry) -> bool:
    noisy_fields = {"abstract", "url", "rights"}
    joined = " ".join(
        value for field, value in entry.fields.items()
        if field not in noisy_fields
    ).lower()
    return bool(re.search(r"\b(todo|placeholder|unknown)\b|forthcoming\?|(^|\s)n/a($|\s)", joined))


def _load_cache(cache_path: Path | None) -> dict[str, dict]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache_path: Path | None, cache: dict[str, dict]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_json(url: str, cache: dict[str, dict]) -> dict | None:
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "thesis-evaluation-suite/1.0 (metadata verification)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        cache[cache_key] = {"_error": "fetch_failed", "_url": url}
        return cache[cache_key]
    cache[cache_key] = payload
    time.sleep(0.05)
    return payload


def verify_bib_entry_online(entry: BibEntry, cache: dict[str, dict]) -> dict[str, str]:
    title = entry.fields.get("title", "")
    year = _entry_year(entry)
    doi = entry.fields.get("doi", "").strip()
    if doi:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
        payload = _fetch_json(url, cache)
        return _verification_from_crossref(entry, payload, url, title, year)
    source_url = entry.fields.get("url", "").strip()
    if source_url:
        return _verification_from_url(entry, source_url, cache)
    if title:
        query = urllib.parse.urlencode({"search": title, "per-page": "1"})
        url = "https://api.openalex.org/works?" + query
        payload = _fetch_json(url, cache)
        return _verification_from_openalex(entry, payload, url, title, year)
    return {
        "online_status": "source_not_found",
        "online_source": "",
        "title_similarity": "",
        "notes": "No title or DOI available for online metadata verification.",
    }


def _title_similarity(local: str, remote: str) -> float:
    left = normalize_text(local)
    right = normalize_text(remote)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _title_compatible(local: str, remote: str, similarity: float) -> bool:
    if similarity >= 0.72:
        return True
    left_tokens = set(normalize_text(local).split())
    right_tokens = set(normalize_text(remote).split())
    left_tokens = {token for token in left_tokens if len(token) > 2}
    right_tokens = {token for token in right_tokens if len(token) > 2}
    if not left_tokens or not right_tokens:
        return False
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return len(smaller & larger) / len(smaller) >= 0.8


def _year_matches(local: str, remote: str | int | None) -> bool:
    if not local or remote in (None, ""):
        return True
    local_match = re.search(r"\d{4}", str(local))
    remote_match = re.search(r"\d{4}", str(remote))
    if not local_match or not remote_match:
        return True
    return abs(int(local_match.group(0)) - int(remote_match.group(0))) <= 1


def _verification_from_crossref(
    entry: BibEntry,
    payload: dict | None,
    url: str,
    title: str,
    year: str,
) -> dict[str, str]:
    if not payload or payload.get("_error"):
        return {
            "online_status": "online_error",
            "online_source": url,
            "title_similarity": "",
            "notes": "Crossref lookup failed.",
        }
    message = payload.get("message", {})
    remote_title = " ".join(message.get("title", [])[:1])
    issued = message.get("issued", {}).get("date-parts", [[None]])[0][0]
    sim = _title_similarity(title, remote_title)
    status = "ok_online" if _title_compatible(title, remote_title, sim) and _year_matches(year, issued) else "metadata_mismatch"
    return {
        "online_status": status,
        "online_source": url,
        "title_similarity": f"{sim:.3f}",
        "notes": f"Crossref title={remote_title[:160]}; year={issued}; local_key={entry.key}",
    }


def _verification_from_openalex(
    entry: BibEntry,
    payload: dict | None,
    url: str,
    title: str,
    year: str,
) -> dict[str, str]:
    if not payload or payload.get("_error"):
        return {
            "online_status": "online_error",
            "online_source": url,
            "title_similarity": "",
            "notes": "OpenAlex lookup failed.",
        }
    results = payload.get("results", [])
    if not results:
        return {
            "online_status": "source_not_found",
            "online_source": url,
            "title_similarity": "",
            "notes": "OpenAlex returned no candidate.",
        }
    candidate = results[0]
    remote_title = candidate.get("display_name", "")
    remote_year = candidate.get("publication_year", "")
    sim = _title_similarity(title, remote_title)
    status = "ok_online" if _title_compatible(title, remote_title, sim) and _year_matches(year, remote_year) else "metadata_mismatch"
    return {
        "online_status": status,
        "online_source": candidate.get("id", url),
        "title_similarity": f"{sim:.3f}",
        "notes": f"OpenAlex title={remote_title[:160]}; year={remote_year}; local_key={entry.key}",
    }


def _verification_from_url(entry: BibEntry, source_url: str, cache: dict[str, dict]) -> dict[str, str]:
    cache_key = "url_status:" + hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    if cache_key in cache:
        status_payload = cache[cache_key]
    else:
        status_payload = _probe_url(source_url)
        cache[cache_key] = status_payload
        time.sleep(0.05)
    status_code = status_payload.get("status_code")
    if isinstance(status_code, int) and status_code < 500 and status_code != 404:
        status = "ok_online"
    else:
        status = "source_not_found"
    return {
        "online_status": status,
        "online_source": source_url,
        "title_similarity": "",
        "notes": f"URL probe status={status_payload.get('status_code')}; method={status_payload.get('method')}; local_key={entry.key}",
    }


def _probe_url(source_url: str) -> dict[str, str | int]:
    for method in ["HEAD", "GET"]:
        request = urllib.request.Request(
            source_url,
            method=method,
            headers={"User-Agent": "thesis-evaluation-suite/1.0 (metadata verification)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return {"status_code": int(response.status), "method": method}
        except urllib.error.HTTPError as exc:
            if exc.code != 405:
                return {"status_code": int(exc.code), "method": method}
        except (urllib.error.URLError, TimeoutError):
            continue
    return {"status_code": 0, "method": "failed"}


def repetition_candidates(chapters: list[Chapter]) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    sentences: list[tuple[str, str, int, str]] = []
    for chapter in chapters:
        if chapter.key == "Front Matter":
            continue
        for offset, line in enumerate(chapter.text.splitlines(), start=chapter.start_line):
            plain = strip_latex(line)
            for sentence in re.split(r"(?<=[.!?])\s+", plain):
                sentence = sentence.strip()
                if len(sentence) < 90:
                    continue
                normalized = normalize_text(sentence)
                if len(normalized) < 90:
                    continue
                sentences.append((chapter.key, sentence, offset, normalized))

    candidates: list[dict[str, str]] = []
    exact_seen: dict[str, tuple[str, str, int]] = {}
    for chapter, sentence, line_no, normalized in sentences:
        if normalized in exact_seen:
            prev_chapter, prev_sentence, prev_line = exact_seen[normalized]
            candidates.append({
                "kind": "exact",
                "similarity": "1.000",
                "first_chapter": prev_chapter,
                "first_line": str(prev_line),
                "second_chapter": chapter,
                "second_line": str(line_no),
                "first_sentence": prev_sentence,
                "second_sentence": sentence,
            })
        else:
            exact_seen[normalized] = (chapter, sentence, line_no)

    buckets: dict[str, list[int]] = {}
    for idx, (_, _, _, normalized) in enumerate(sentences):
        tokens = normalized.split()
        if len(tokens) < 12:
            continue
        keys = {
            " ".join(tokens[:6]),
            " ".join(tokens[:4] + tokens[-4:]),
        }
        for key in keys:
            buckets.setdefault(key, []).append(idx)

    compared: set[tuple[int, int]] = set()
    for bucket_indices in buckets.values():
        if len(candidates) >= 120:
            break
        if len(bucket_indices) > 40:
            bucket_indices = bucket_indices[:40]
        for pos, left_idx in enumerate(bucket_indices):
            chapter_a, sentence_a, line_a, normalized_a = sentences[left_idx]
            for right_idx in bucket_indices[pos + 1:]:
                pair = (min(left_idx, right_idx), max(left_idx, right_idx))
                if pair in compared:
                    continue
                compared.add(pair)
                chapter_b, sentence_b, line_b, normalized_b = sentences[right_idx]
                if chapter_a == chapter_b and abs(line_a - line_b) < 10:
                    continue
                len_a = len(normalized_a)
                len_b = len(normalized_b)
                if min(len_a, len_b) / max(len_a, len_b) < 0.75:
                    continue
                sim = SequenceMatcher(None, normalized_a, normalized_b).ratio()
                if sim >= 0.92:
                    candidates.append({
                        "kind": "near_duplicate",
                        "similarity": f"{sim:.3f}",
                        "first_chapter": chapter_a,
                        "first_line": str(line_a),
                        "second_chapter": chapter_b,
                        "second_line": str(line_b),
                        "first_sentence": sentence_a,
                        "second_sentence": sentence_b,
                    })
                    break

    term_counts: dict[str, dict[str, int]] = {}
    for chapter in chapters:
        chapter_text = normalize_text(chapter.text)
        counts = {}
        for term in OVERMENTION_TERMS:
            counts[term] = len(re.findall(re.escape(term.lower()), chapter_text))
        term_counts[chapter.key] = counts
    return candidates, term_counts


def chapter_scores(
    chapters: list[Chapter],
    findings: list[Finding],
    citation_uses: list[CitationUse],
    repetition_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    findings_by_chapter: dict[str, list[Finding]] = {}
    for finding in findings:
        findings_by_chapter.setdefault(finding.location, []).append(finding)
    citations_by_chapter: dict[str, int] = {}
    for use in citation_uses:
        citations_by_chapter[use.chapter] = citations_by_chapter.get(use.chapter, 0) + 1
    repetition_by_chapter: dict[str, int] = {}
    for row in repetition_rows:
        repetition_by_chapter[row["first_chapter"]] = repetition_by_chapter.get(row["first_chapter"], 0) + 1
        repetition_by_chapter[row["second_chapter"]] = repetition_by_chapter.get(row["second_chapter"], 0) + 1

    for chapter in chapters:
        chapter_findings = findings_by_chapter.get(chapter.key, [])
        severity_counts = {severity: sum(1 for f in chapter_findings if f.severity == severity) for severity in SEVERITY_ORDER}
        score = 1
        if severity_counts["P0"]:
            score = 5
        elif severity_counts["P1"] >= 3:
            score = 4
        elif severity_counts["P1"]:
            score = 3
        elif severity_counts["P2"] or repetition_by_chapter.get(chapter.key, 0):
            score = 2
        rows.append({
            "chapter": chapter.key,
            "line_range": f"{chapter.start_line}-{chapter.end_line}",
            "criteria_count": str(len(CHAPTER_CRITERIA.get(chapter.key, []))),
            "citations": str(citations_by_chapter.get(chapter.key, 0)),
            "repetition_candidates": str(repetition_by_chapter.get(chapter.key, 0)),
            "P0": str(severity_counts["P0"]),
            "P1": str(severity_counts["P1"]),
            "P2": str(severity_counts["P2"]),
            "P3": str(severity_counts["P3"]),
            "score_1_best_5_worst": str(score),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def findings_to_rows(findings: list[Finding]) -> list[dict[str, str]]:
    rows = []
    for finding in sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.location, f.line or 0)):
        rows.append({
            "severity": finding.severity,
            "category": finding.category,
            "location": finding.location,
            "line": "" if finding.line is None else str(finding.line),
            "message": finding.message,
            "recommendation": finding.recommendation,
            "evidence": finding.evidence,
        })
    return rows


def _finding_bullet(finding: Finding) -> str:
    line = f":{finding.line}" if finding.line else ""
    evidence = f" Evidence: {finding.evidence}" if finding.evidence else ""
    return (
        f"- `{finding.severity}` `{finding.category}` {finding.location}{line}: "
        f"{finding.message} Recommendation: {finding.recommendation}{evidence}"
    )


def write_dashboard(
    out_dir: Path,
    tex_path: Path,
    bib_path: Path,
    chapters: list[Chapter],
    findings: list[Finding],
    chapter_score_rows: list[dict[str, str]],
    numeric_rows: list[dict[str, str]],
    literature_rows_out: list[dict[str, str]],
    repetition_rows: list[dict[str, str]],
    online: bool,
) -> None:
    severity_counts = {severity: sum(1 for f in findings if f.severity == severity) for severity in SEVERITY_ORDER}
    lines = [
        "# Thesis Evaluation Dashboard",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Thesis source: `{tex_path}`",
        f"Bibliography: `{bib_path}`",
        f"Online literature metadata verification: {'enabled' if online else 'disabled'}",
        "",
        "## Status",
        "",
        f"- Chapters audited: {len(chapters)}",
        f"- Findings: P0={severity_counts['P0']}, P1={severity_counts['P1']}, P2={severity_counts['P2']}, P3={severity_counts['P3']}",
        f"- Numeric checks failed: {sum(1 for row in numeric_rows if row['status'] != 'pass')}",
        f"- Literature rows: {len(literature_rows_out)}",
        f"- Repetition candidates: {len(repetition_rows)}",
        "",
        "## Highest Priority Findings",
        "",
    ]
    high_priority = [f for f in findings if f.severity in {"P0", "P1"}]
    if high_priority:
        lines.extend(_finding_bullet(finding) for finding in sorted(
            high_priority,
            key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.location, f.line or 0),
        )[:40])
    else:
        lines.append("- No P0/P1 findings.")
    lines.extend(["", "## Chapter Scores", ""])
    lines.append("| Chapter | Lines | Citations | P0 | P1 | P2 | Score |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in chapter_score_rows:
        lines.append(
            f"| {row['chapter']} | {row['line_range']} | {row['citations']} | "
            f"{row['P0']} | {row['P1']} | {row['P2']} | {row['score_1_best_5_worst']} |"
        )
    lines.extend([
        "",
        "## Report Files",
        "",
        "- `findings.csv`",
        "- `chapter_scores.csv`",
        "- `numeric_consistency.csv`",
        "- `literature_verification.csv`",
        "- `repetition_candidates.md`",
        "- `edit_queue.md`",
    ])
    (out_dir / "00_dashboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chapter_reports(
    out_dir: Path,
    chapters: list[Chapter],
    findings: list[Finding],
    citation_uses: list[CitationUse],
    term_counts: dict[str, dict[str, int]],
) -> None:
    findings_by_chapter: dict[str, list[Finding]] = {}
    citations_by_chapter: dict[str, list[CitationUse]] = {}
    for finding in findings:
        findings_by_chapter.setdefault(finding.location, []).append(finding)
    for use in citation_uses:
        citations_by_chapter.setdefault(use.chapter, []).append(use)

    for idx, chapter in enumerate(chapters):
        slug = re.sub(r"[^a-z0-9]+", "_", chapter.key.lower()).strip("_") or "front_matter"
        name = f"chapter_{idx:02d}_{slug}.md" if chapter.key != "Implementation and Reproducibility" else "appendix_reproducibility.md"
        lines = [
            f"# {chapter.title}",
            "",
            f"Lines: {chapter.start_line}-{chapter.end_line}",
            "",
            "## Criteria",
            "",
        ]
        for criterion in CHAPTER_CRITERIA.get(chapter.key, ["General coherence, evidence coverage, and source support."]):
            lines.append(f"- {criterion}")
        lines.extend(["", "## Sections", ""])
        if chapter.sections:
            lines.extend(f"- {section}" for section in chapter.sections)
        else:
            lines.append("- No section-level headings.")
        lines.extend(["", "## Findings", ""])
        chapter_findings = sorted(
            findings_by_chapter.get(chapter.key, []),
            key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.line or 0),
        )
        if chapter_findings:
            lines.extend(_finding_bullet(finding) for finding in chapter_findings)
        else:
            lines.append("- No chapter-local findings.")
        lines.extend(["", "## Citation Load", ""])
        chapter_citations = citations_by_chapter.get(chapter.key, [])
        lines.append(f"- Citation uses: {len(chapter_citations)}")
        if chapter_citations:
            top = sorted({use.key for use in chapter_citations})[:30]
            lines.append("- Cited keys: " + ", ".join(f"`{key}`" for key in top))
        lines.extend(["", "## Repetition Term Counts", ""])
        for term, count in term_counts.get(chapter.key, {}).items():
            if count:
                lines.append(f"- `{term}`: {count}")
        if not any(term_counts.get(chapter.key, {}).values()):
            lines.append("- No monitored overmention terms found.")
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repetition_report(path: Path, rows: list[dict[str, str]], term_counts: dict[str, dict[str, int]]) -> None:
    lines = ["# Repetition Candidates", ""]
    if rows:
        for row in rows[:120]:
            lines.extend([
                f"## {row['kind']} similarity {row['similarity']}",
                "",
                f"- First: {row['first_chapter']}:{row['first_line']}",
                f"- Second: {row['second_chapter']}:{row['second_line']}",
                f"- First sentence: {row['first_sentence']}",
                f"- Second sentence: {row['second_sentence']}",
                "",
            ])
    else:
        lines.append("No high-similarity sentence candidates were detected.")
        lines.append("")
    lines.extend(["# Monitored Term Counts", ""])
    for chapter, counts in term_counts.items():
        visible = {term: count for term, count in counts.items() if count}
        if not visible:
            continue
        lines.append(f"## {chapter}")
        for term, count in visible.items():
            lines.append(f"- `{term}`: {count}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_edit_queue(path: Path, findings: list[Finding]) -> None:
    lines = ["# Prioritized Edit Queue", ""]
    for severity in ["P0", "P1", "P2", "P3"]:
        lines.extend([f"## {severity}", ""])
        severity_findings = [
            finding for finding in sorted(findings, key=lambda f: (f.location, f.line or 0))
            if finding.severity == severity
        ]
        if not severity_findings:
            lines.append("- None.")
        else:
            lines.extend(_finding_bullet(finding) for finding in severity_findings)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_thesis(
    tex_path: Path,
    bib_path: Path,
    out_dir: Path,
    *,
    online_literature: bool = False,
    online_limit: int | None = None,
) -> dict[str, int]:
    tex = read_text(tex_path)
    bib_text = read_text(bib_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    chapters = parse_chapters(tex)
    citations = collect_citations(tex, chapters)
    bib_entries = parse_bibtex(bib_text)

    findings: list[Finding] = []
    findings.extend(marker_findings(tex, chapters))
    findings.extend(label_ref_findings(tex, chapters))
    numeric_rows, numeric_findings = numeric_checks(tex)
    findings.extend(numeric_findings)

    lit_rows, lit_findings = literature_rows(
        citations,
        bib_entries,
        online=online_literature,
        online_limit=online_limit,
        cache_path=out_dir / "online_literature_cache.json",
    )
    findings.extend(lit_findings)

    repetition_rows, term_counts = repetition_candidates(chapters)
    for row in repetition_rows[:40]:
        findings.append(Finding(
            severity="P2",
            category="repetition_candidate",
            location=row["second_chapter"],
            line=int(row["second_line"]),
            message="Potential repetitive sentence or near-duplicate passage.",
            recommendation="Keep only if the repeated sentence performs a distinct argumentative function.",
            evidence=f"{row['similarity']} vs {row['first_chapter']}:{row['first_line']}",
        ))

    score_rows = chapter_scores(chapters, findings, citations, repetition_rows)

    write_csv(out_dir / "findings.csv", findings_to_rows(findings))
    write_csv(out_dir / "chapter_scores.csv", score_rows)
    write_csv(out_dir / "numeric_consistency.csv", numeric_rows)
    write_csv(out_dir / "literature_verification.csv", lit_rows)
    write_repetition_report(out_dir / "repetition_candidates.md", repetition_rows, term_counts)
    write_edit_queue(out_dir / "edit_queue.md", findings)
    write_chapter_reports(out_dir, chapters, findings, citations, term_counts)
    write_dashboard(
        out_dir,
        tex_path,
        bib_path,
        chapters,
        findings,
        score_rows,
        numeric_rows,
        lit_rows,
        repetition_rows,
        online_literature,
    )

    summary = {
        "chapters": len(chapters),
        "findings": len(findings),
        "P0": sum(1 for f in findings if f.severity == "P0"),
        "P1": sum(1 for f in findings if f.severity == "P1"),
        "P2": sum(1 for f in findings if f.severity == "P2"),
        "P3": sum(1 for f in findings if f.severity == "P3"),
        "numeric_failures": sum(1 for row in numeric_rows if row["status"] != "pass"),
        "literature_rows": len(lit_rows),
        "repetition_candidates": len(repetition_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thesis evaluation suite.")
    parser.add_argument("--tex", type=Path, required=True, help="Thesis LaTeX source.")
    parser.add_argument("--bib", type=Path, required=True, help="BibTeX database.")
    parser.add_argument("--out", type=Path, required=True, help="Output report directory.")
    parser.add_argument(
        "--online-literature",
        action="store_true",
        help="Verify cited-source metadata with Crossref/OpenAlex where possible.",
    )
    parser.add_argument(
        "--online-limit",
        type=int,
        default=None,
        help="Maximum cited entries to check online. Default checks all cited entries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_thesis(
        args.tex,
        args.bib,
        args.out,
        online_literature=args.online_literature,
        online_limit=args.online_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
