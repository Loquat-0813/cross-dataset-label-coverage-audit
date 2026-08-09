#!/usr/bin/env python3
"""Verify BibTeX entries against Crossref and OpenAlex.

The script deliberately keeps verification evidence separate from the source
bibliography.  It parses a small, conventional BibTeX subset used by this
paper, resolves each DOI with Crossref where available, and otherwise performs
title searches against Crossref and OpenAlex.  Results are written as an audit
artifact that includes the queried endpoint and field-level discrepancies.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote


USER_AGENT = "P1CitationAudit/1.0 (academic reference verification)"


def _read_balanced(text: str, start: int) -> tuple[str, int]:
    """Return text inside a brace starting at *start* and the next position."""
    if text[start] != "{":
        raise ValueError("expected an opening brace")
    depth = 0
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "\\" and not escaped:
            escaped = True
            continue
        if not escaped:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : pos], pos + 1
        escaped = False
    raise ValueError("unbalanced BibTeX entry")


def _split_top_level(value: str, separator: str = ",") -> list[str]:
    items: list[str] = []
    start = depth = 0
    escaped = False
    for pos, char in enumerate(value):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if not escaped:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == separator and depth == 0:
                items.append(value[start:pos])
                start = pos + 1
        escaped = False
    items.append(value[start:])
    return items


def parse_bibtex(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []
    cursor = 0
    while True:
        match = re.search(r"@(\w+)\s*\{", text[cursor:])
        if not match:
            break
        entry_type = match.group(1).lower()
        brace_pos = cursor + match.end() - 1
        body, cursor = _read_balanced(text, brace_pos)
        parts = _split_top_level(body)
        key = parts[0].strip()
        fields: dict[str, str] = {"entry_type": entry_type, "key": key}
        for part in parts[1:]:
            if "=" not in part:
                continue
            name, raw = part.split("=", 1)
            value = raw.strip().rstrip(",").strip()
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[name.strip().lower()] = re.sub(r"\s+", " ", value).strip()
        entries.append(fields)
    return entries


def normalized(value: str | None) -> str:
    value = value or ""
    value = re.sub(r"\\['`\"^~=.]\s*", "", value)
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("{", "").replace("}", "")
    value = "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def title_similarity(left: str | None, right: str | None) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def family_names(author: str | None) -> list[str]:
    if not author:
        return []
    names = []
    for person in author.split(" and "):
        person = person.strip()
        if not person or person.startswith("{"):
            continue
        names.append(normalized(person.split(",", 1)[0] if "," in person else person.split()[-1]))
    return names


def get_json(url: str, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "curl.exe",
                "-L",
                "--max-time",
                str(timeout),
                "-sS",
                "-H",
                f"User-Agent: {USER_AGENT}",
                "-H",
                "Accept: application/json",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            return None, f"curl exit {completed.returncode}: {completed.stderr.strip()}"
        return json.loads(completed.stdout), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def get_cached_json(cache_dir: Path | None, key: str, source: str) -> tuple[dict[str, Any] | None, str | None]:
    if cache_dir is None:
        return None, "cache unavailable"
    path = cache_dir / f"{key}__{source}.json"
    if not path.is_file() or path.stat().st_size == 0:
        return None, f"cache unavailable: {path.name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as error:
        return None, f"invalid cached JSON {path.name}: {error}"


def crossref_doi(doi: str, timeout: int) -> tuple[dict[str, Any] | None, str | None, str]:
    endpoint = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
    payload, error = get_json(endpoint, timeout)
    return (payload or {}).get("message"), error, endpoint


def crossref_title(title: str, timeout: int) -> tuple[dict[str, Any] | None, str | None, str]:
    endpoint = f"https://api.crossref.org/works?query.bibliographic={quote(title)}&rows=5"
    payload, error = get_json(endpoint, timeout)
    items = ((payload or {}).get("message", {}).get("items", []))
    if not items:
        return None, error or "no results", endpoint
    return max(items, key=lambda item: title_similarity(title, (item.get("title") or [""])[0])), error, endpoint


def openalex_title(title: str, timeout: int) -> tuple[dict[str, Any] | None, str | None, str]:
    endpoint = f"https://api.openalex.org/works?search={quote(title)}&per-page=5"
    payload, error = get_json(endpoint, timeout)
    items = ((payload or {}).get("results", []))
    if not items:
        return None, error or "no results", endpoint
    return max(items, key=lambda item: title_similarity(title, item.get("title"))), error, endpoint


def published_year(record: dict[str, Any], source: str) -> str | None:
    if source == "openalex":
        value = record.get("publication_year")
        return str(value) if value else None
    for key in ("published-print", "published-online", "issued", "published"):
        parts = record.get(key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            return str(parts[0][0])
    return None


def record_title(record: dict[str, Any], source: str) -> str:
    if source == "openalex":
        return record.get("title", "")
    return (record.get("title") or [""])[0]


def record_authors(record: dict[str, Any], source: str) -> list[str]:
    if source == "openalex":
        return [normalized(a.get("author", {}).get("display_name")) for a in record.get("authorships", [])]
    return [normalized(a.get("family")) for a in record.get("author", [])]


def comparable_metadata(record: dict[str, Any], source: str) -> dict[str, Any]:
    if source == "openalex":
        location = record.get("primary_location", {}).get("source") or {}
        return {
            "title": record.get("title"),
            "authors": [a.get("author", {}).get("display_name") for a in record.get("authorships", [])],
            "year": record.get("publication_year"),
            "doi": record.get("doi"),
            "venue": location.get("display_name"),
            "volume": record.get("biblio", {}).get("volume"),
            "issue": record.get("biblio", {}).get("issue"),
            "pages": [record.get("biblio", {}).get("first_page"), record.get("biblio", {}).get("last_page")],
            "id": record.get("id"),
        }
    return {
        "title": record_title(record, source),
        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in record.get("author", [])],
        "year": published_year(record, source),
        "doi": record.get("DOI"),
        "venue": (record.get("container-title") or [""])[0],
        "volume": record.get("volume"),
        "issue": record.get("issue"),
        "pages": record.get("page"),
        "type": record.get("type"),
        "url": record.get("URL"),
    }


def external_family_names(record: dict[str, Any], source: str) -> list[str]:
    if source == "openalex":
        return [
            normalized((a.get("author", {}).get("display_name") or "").split()[-1])
            for a in record.get("authorships", [])
            if (a.get("author", {}).get("display_name") or "").strip()
        ]
    return [normalized(a.get("family")) for a in record.get("author", [])]


def external_field(record: dict[str, Any], source: str, field: str) -> str | None:
    if source == "openalex":
        biblio = record.get("biblio", {})
        if field == "volume":
            return str(biblio.get("volume")) if biblio.get("volume") else None
        if field == "issue":
            return str(biblio.get("issue")) if biblio.get("issue") else None
        if field == "pages":
            first, last = biblio.get("first_page"), biblio.get("last_page")
            return f"{first}-{last}" if first and last else str(first) if first else None
    if field == "doi":
            return record.get("doi")
    if field == "issue":
        return record.get("issue")
    if field == "doi":
        return record.get("DOI")
    if field == "pages":
        return record.get("page")
    return record.get(field)


def normalized_page_range(value: str | None) -> str:
    return re.sub(r"[^0-9]+", "-", value or "").strip("-")


def normalized_doi(value: str | None) -> str:
    value = normalized(value)
    for prefix in ("https doi org ", "http doi org ", "doi "):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def author_sequences_match(bib: list[str], external: list[str]) -> bool:
    if any(len(name.split()) == 1 and len(name) > 12 for name in external):
        return True
    bib_last = [name.split()[-1] for name in bib]
    external_last = [name.split()[-1] for name in external]
    return bib_last == external_last


def candidate_score(entry: dict[str, str], source: str, record: dict[str, Any]) -> float:
    record_source = "openalex" if source == "openalex-title" else "crossref"
    score = title_similarity(entry.get("title"), record_title(record, record_source)) * 0.70
    bib_authors = family_names(entry.get("author"))
    api_authors = external_family_names(record, record_source)
    if bib_authors and api_authors and bib_authors[0].split()[-1] == api_authors[0].split()[-1]:
        score += 0.20
    if entry.get("year") == published_year(record, record_source):
        score += 0.10
    return score


def cached_candidate(
    entry: dict[str, str], cache_dir: Path | None, source: str
) -> tuple[dict[str, Any] | None, str | None, str]:
    title = entry.get("title", "")
    if source == "crossref-doi":
        endpoint = f"https://api.crossref.org/works/{quote(entry.get('doi', ''), safe='/')}"
    elif source == "crossref-title":
        endpoint = f"https://api.crossref.org/works?query.bibliographic={quote(title, safe='')}&rows=5"
    else:
        endpoint = f"https://api.openalex.org/works?search={quote(title, safe='')}&per-page=5"
    payload, error = get_cached_json(cache_dir, entry["key"], source.replace("-", "_"))
    if payload is None and cache_dir is None:
        if source == "crossref-doi":
            return crossref_doi(entry["doi"], 30)
        if source == "crossref-title":
            return crossref_title(title, 30)
        return openalex_title(title, 30)
    if payload is None:
        return None, error, endpoint
    if source == "crossref-doi":
        return payload.get("message"), error, endpoint
    if source == "crossref-title":
        items = payload.get("message", {}).get("items", [])
        return (
            max(items, key=lambda item: title_similarity(title, (item.get("title") or [""])[0])) if items else None,
            error or ("no cached Crossref title result" if not items else None),
            endpoint,
        )
    items = payload.get("results", [])
    return (
        max(items, key=lambda item: title_similarity(title, item.get("title"))) if items else None,
        error or ("no cached OpenAlex title result" if not items else None),
        endpoint,
    )


def audit_entry(entry: dict[str, str], timeout: int, cache_dir: Path | None = None) -> dict[str, Any]:
    del timeout
    is_official_document = not entry.get("doi") and entry.get("entry_type") == "misc" and entry.get("url")
    if is_official_document:
        return {
            "key": entry["key"],
            "status": "manual_check_required",
            "source": "official-dataset-or-documentation-url",
            "endpoint": entry["url"],
            "discrepancies": [],
            "errors": ["official endpoint requires manual author-side access confirmation"],
            "bib": entry,
        }

    primary_source = "crossref-doi" if entry.get("doi") else "crossref-title"
    primary, primary_error, primary_endpoint = cached_candidate(entry, cache_dir, primary_source)
    secondary, secondary_error, secondary_endpoint = cached_candidate(entry, cache_dir, "openalex-title")
    candidates = [
        (primary_source, primary, primary_error, primary_endpoint),
        ("openalex-title", secondary, secondary_error, secondary_endpoint),
    ]
    valid = [item for item in candidates if item[1] is not None]
    if not valid:
        return {
            "key": entry["key"],
            "status": "unverifiable",
            "source": "none",
            "endpoint": primary_endpoint,
            "errors": [error for _, _, error, _ in candidates if error],
            "bib": entry,
        }
    primary_doi_match = (
        primary_source == "crossref-doi"
        and primary is not None
        and title_similarity(entry.get("title"), record_title(primary, "crossref")) >= 0.99
        and normalized_doi(entry.get("doi")) == normalized_doi(primary.get("DOI"))
    )
    if primary_doi_match:
        final_source, record, _, endpoint = primary_source, primary, primary_error, primary_endpoint
    else:
        final_source, record, _, endpoint = max(valid, key=lambda item: candidate_score(entry, item[0], item[1]))
    record_source = "openalex" if final_source == "openalex-title" else "crossref"
    matched_title = record_title(record, record_source)
    similarity = title_similarity(entry.get("title"), matched_title)
    api_author_names = external_family_names(record, record_source)
    bib_author_names = family_names(entry.get("author"))
    year = published_year(record, record_source)
    discrepancies: list[str] = []
    if similarity < 0.94:
        discrepancies.append(f"title similarity {similarity:.3f} below 0.940")
    if entry.get("year") and year and entry["year"] != year:
        discrepancies.append(f"year BibTeX={entry['year']} API={year}")
    if bib_author_names and api_author_names and bib_author_names[0] != api_author_names[0]:
        discrepancies.append(f"first author BibTeX={bib_author_names[0]} API={api_author_names[0]}")
    elif bib_author_names and api_author_names and not author_sequences_match(bib_author_names, api_author_names):
        discrepancies.append("author sequence differs between BibTeX and selected external record")
    for field in ("volume", "issue", "pages"):
        bib_value = entry.get(field)
        api_value = external_field(record, record_source, field)
        if bib_value and api_value:
            matches = (
                normalized_page_range(bib_value) == normalized_page_range(api_value)
                if field == "pages"
                else normalized(bib_value) == normalized(api_value)
            )
            if not matches:
                discrepancies.append(f"{field} BibTeX={bib_value} API={api_value}")
    if entry.get("doi") and external_field(record, record_source, "doi") and normalized_doi(entry["doi"]) != normalized_doi(external_field(record, record_source, "doi")):
        discrepancies.append("resolved DOI differs from BibTeX DOI")

    status = "verified" if not discrepancies else "suspicious"
    return {
        "key": entry["key"],
        "status": status,
        "source": final_source,
        "endpoint": endpoint,
        "title_similarity": round(similarity, 4),
        "discrepancies": discrepancies,
        "errors": [error for _, _, error, _ in candidates if error],
        "bib": entry,
        "external": comparable_metadata(record, record_source),
        "evidence_sources": [
            {
                "source": source,
                "endpoint": candidate_endpoint,
                "title_similarity": round(
                    title_similarity(entry.get("title"), record_title(candidate, "openalex" if source == "openalex-title" else "crossref")),
                    4,
                ) if candidate is not None else None,
            }
            for source, candidate, _, candidate_endpoint in candidates
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    entries = parse_bibtex(args.bib)
    if not entries:
        raise SystemExit(f"no entries parsed from {args.bib}")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(audit_entry, entry, args.timeout, args.cache_dir) for entry in entries]
        for future in as_completed(futures):
            results.append(future.result())
            time.sleep(0.1)
    results.sort(key=lambda item: item["key"])
    counts = {status: sum(item["status"] == status for item in results) for status in ("verified", "suspicious", "hallucinated", "unverifiable")}
    output = {
        "audit_skill": "citation-verification-gate",
        "scope": "external metadata verification of every active BibTeX entry",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "input": {"path": str(args.bib), "sha256": __import__("hashlib").sha256(args.bib.read_bytes()).hexdigest()},
        "total_entries": len(results),
        "counts": counts,
        "entries": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"total_entries": len(results), **counts, "output": str(args.output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
