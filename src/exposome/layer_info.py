"""Curated per-layer info-panel metadata (source, citation, specs, limitations).

The webapp's per-exposome info panel (FUENTE / ESPECIFICACIONES / METODOLOGIA /
CITACION / DESCARGAR) must render the same canonical schema for every study
bundle.  The single source of truth is ``config/layer_info/<layer_id>.yaml``:
one file per catalog layer holding curated identity and prose.  Per-study
facts (years actually covered, native resolution) come from the layer's
processed ``*_metadata.json`` sidecar at publish time and never live here.

Resolution precedence mirrors :mod:`exposome.settings`:

``default < countries.<ISO2> < studies.<study_id>``

Entries are emitted under the layer_id *and* every ``webapp_keys`` alias so
the browser lookup chain (``sourceMap[id] || sourceMap[source_id||parent] ||
sourceMap[exposomeDataId(id)]``) always resolves without palette changes.
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .studies import StudyConfigError

LAYER_INFO_DIR = Path("config") / "layer_info"

_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "layer_id",
        "webapp_keys",
        "source",
        "citation",
        "specs",
        "coverage_note",
        "limitations",
        "methodology_doc",
        "color",
        "primary",
        "countries",
        "studies",
    }
)
_REQUIRED_KEYS = frozenset({"schema_version", "layer_id", "source", "specs"})
_SOURCE_KEYS = frozenset({"name", "provider", "url", "dataset_id", "license"})
_CITATION_KEYS = frozenset(
    {
        "authors",
        "year",
        "title",
        "journal",
        "doi",
        "doi_status",
        "paper",
        "bibtex_key",
    }
)
_SPECS_KEYS = frozenset({"spatial_resolution", "temporal_coverage", "validation"})
_OVERRIDABLE_KEYS = _ALLOWED_KEYS - {"schema_version", "layer_id", "countries", "studies"}

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


def load_layer_info_registry(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate every ``config/layer_info/<layer_id>.yaml`` file."""
    registry_dir = Path(repo_root) / LAYER_INFO_DIR
    registry: dict[str, dict[str, Any]] = {}
    if not registry_dir.is_dir():
        return registry
    for path in sorted(registry_dir.glob("*.yaml")):
        # macOS can materialize AppleDouble metadata as ``._<name>.yaml``
        # when an archive is copied through Dropbox/SMB/NTFS.  It is binary
        # metadata, not a registry entry, and must never be parsed as UTF-8.
        if path.name.startswith("._"):
            continue
        entry = _load_info_file(path)
        registry[entry["layer_id"]] = entry
    return registry


def _load_info_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise StudyConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise StudyConfigError(f"Layer info must be a YAML mapping: {path}")
    unknown = sorted(set(raw).difference(_ALLOWED_KEYS))
    missing = sorted(_REQUIRED_KEYS.difference(raw))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown keys: {unknown}")
        if missing:
            details.append(f"missing keys: {missing}")
        raise StudyConfigError(f"Invalid layer info schema in {path}: {'; '.join(details)}")
    if raw.get("schema_version") != 1:
        raise StudyConfigError(f"Unsupported layer info schema_version in {path}")
    layer_id = str(raw.get("layer_id") or "").strip()
    if not layer_id or path.stem != layer_id:
        raise StudyConfigError(f"Layer info layer_id must match filename: {path}")
    _validate_block(raw, "source", _SOURCE_KEYS, path)
    _validate_block(raw, "citation", _CITATION_KEYS, path)
    _validate_block(raw, "specs", _SPECS_KEYS, path)
    webapp_keys = raw.get("webapp_keys", [])
    if not isinstance(webapp_keys, list) or not all(
        isinstance(key, str) and key for key in webapp_keys
    ):
        raise StudyConfigError(f"webapp_keys must be a list of ids: {path}")
    limitations = raw.get("limitations", [])
    if not isinstance(limitations, list):
        raise StudyConfigError(f"limitations must be a list: {path}")
    _validate_doi_status(raw.get("citation") or {}, path)
    for scope, key_check in (("countries", _is_iso2), ("studies", _is_identifier)):
        overrides = raw.get(scope, {}) or {}
        if not isinstance(overrides, Mapping):
            raise StudyConfigError(f"{scope} must be a mapping: {path}")
        for scope_key, override in overrides.items():
            if not key_check(str(scope_key)):
                raise StudyConfigError(f"Invalid {scope} key {scope_key!r} in {path}")
            if not isinstance(override, Mapping):
                raise StudyConfigError(f"{scope}.{scope_key} must be a mapping: {path}")
            bad = sorted(set(override).difference(_OVERRIDABLE_KEYS))
            if bad:
                raise StudyConfigError(
                    f"{scope}.{scope_key} overrides unknown keys {bad} in {path}"
                )
            _validate_block(override, "source", _SOURCE_KEYS, path)
            _validate_block(override, "citation", _CITATION_KEYS, path)
            _validate_doi_status(override.get("citation") or {}, path)
            _validate_block(override, "specs", _SPECS_KEYS, path)
    return deepcopy(raw)


def _validate_doi_status(citation: Mapping[str, Any], path: Path) -> None:
    doi = citation.get("doi")
    status = citation.get("doi_status")
    if doi is not None and not DOI_PATTERN.match(str(doi)):
        raise StudyConfigError(f"Malformed DOI {doi!r} in {path}")
    if status not in {None, "verified", "not_assigned"}:
        raise StudyConfigError(f"Invalid citation.doi_status {status!r} in {path}")
    if status == "verified" and not doi:
        raise StudyConfigError(f"citation.doi_status 'verified' requires a DOI in {path}")
    if status == "not_assigned" and doi:
        raise StudyConfigError(
            f"citation.doi_status 'not_assigned' conflicts with DOI {doi!r} in {path}"
        )


def _validate_block(
    raw: Mapping[str, Any], name: str, allowed: frozenset[str], path: Path
) -> None:
    block = raw.get(name)
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise StudyConfigError(f"{name} must be a mapping: {path}")
    unknown = sorted(set(block).difference(allowed))
    if unknown:
        raise StudyConfigError(f"{name} has unknown keys {unknown} in {path}")


def _is_iso2(value: str) -> bool:
    return len(value) == 2 and value.isalpha() and value.isupper()


def _is_identifier(value: str) -> bool:
    return bool(re.match(r"^[a-z0-9_]+$", value))


def resolve_layer_info(
    entry: Mapping[str, Any],
    *,
    country_code: str | None = None,
    study_id: str | None = None,
) -> dict[str, Any]:
    """Merge a registry entry with its country and study overrides."""
    resolved = {
        key: deepcopy(value)
        for key, value in entry.items()
        if key not in {"countries", "studies"}
    }
    for scope, scope_key in (("countries", country_code), ("studies", study_id)):
        if not scope_key:
            continue
        override = (entry.get(scope) or {}).get(
            scope_key.upper() if scope == "countries" else scope_key
        )
        if isinstance(override, Mapping):
            _deep_merge(resolved, override)
    return resolved


def _deep_merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def format_paper(citation: Mapping[str, Any] | None) -> str | None:
    """Compose the human-readable reference the FUENTE/CITACION tabs display."""
    if not citation:
        return None
    paper = citation.get("paper")
    if paper:
        return str(paper)
    authors = citation.get("authors")
    year = citation.get("year")
    journal = citation.get("journal") or citation.get("title")
    if authors and year and journal:
        short = _short_authors(str(authors))
        return f"{short} ({year}) {journal}"
    if citation.get("title"):
        return str(citation["title"])
    return None


def _short_authors(authors: str) -> str:
    first = re.split(r"\s+and\s+|;", authors)[0].strip()
    surname = first.split(",")[0].strip()
    many = ("and" in authors) or (";" in authors) or ("others" in authors)
    return f"{surname} et al." if many else surname


def build_bibtex(citation: Mapping[str, Any] | None, layer_id: str) -> str | None:
    """Build a curated BibTeX record from structured citation fields."""
    if not citation:
        return None
    title = citation.get("title")
    year = citation.get("year")
    if not title or not year:
        return None
    key = citation.get("bibtex_key")
    if not key:
        authors = str(citation.get("authors") or layer_id)
        surname = re.sub(r"[^a-z]", "", _short_authors(authors).split()[0].lower())
        key = f"{surname or layer_id}{year}"
    fields: list[tuple[str, Any]] = []
    if citation.get("authors"):
        fields.append(("author", citation["authors"]))
    fields.append(("title", title))
    if citation.get("journal"):
        fields.append(("journal", citation["journal"]))
    fields.append(("year", year))
    if citation.get("doi"):
        fields.append(("doi", citation["doi"]))
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@article{{{key},\n{body}\n}}"


def build_sources_entry(
    info: Mapping[str, Any],
    sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit one canonical sources.json entry (the schema main.js renders).

    ``info`` is a resolved registry entry; ``sidecar`` is the processed
    layer's ``*_metadata.json`` payload, whose measured facts (years,
    resolution) complement — never replace — the curated prose.
    """
    source = info.get("source") or {}
    citation = info.get("citation") or {}
    specs = info.get("specs") or {}
    sidecar = sidecar or {}
    entry: dict[str, Any] = {
        "name": source.get("name") or info.get("layer_id", ""),
        "url": source.get("url", ""),
        "license": source.get("license", ""),
        "spatial_resolution": specs.get("spatial_resolution", ""),
        "temporal_coverage": specs.get("temporal_coverage", ""),
        "validation": specs.get("validation", ""),
        "layer_id": info.get("layer_id", ""),
    }
    years = _sidecar_years(sidecar)
    if years:
        entry["years"] = years
        if not entry["temporal_coverage"]:
            entry["temporal_coverage"] = (
                str(years[0]) if years[0] == years[-1] else f"{years[0]}-{years[-1]}"
            )
    if not entry["temporal_coverage"] and sidecar.get("period"):
        entry["temporal_coverage"] = str(sidecar["period"])
    resolution_m = sidecar.get("resolution_m") or sidecar.get("native_resolution_m")
    if resolution_m is not None:
        entry["resolution_m"] = resolution_m
        if not entry["spatial_resolution"]:
            entry["spatial_resolution"] = f"~{resolution_m} m"
    if source.get("dataset_id"):
        entry["dataset_id"] = source["dataset_id"]
    paper = format_paper(citation)
    if paper:
        entry["paper"] = paper
    if citation.get("doi"):
        entry["doi"] = citation["doi"]
        entry["doi_status"] = "verified"
    elif citation.get("doi_status") == "not_assigned":
        entry["doi_status"] = "not_assigned"
    bibtex = build_bibtex(citation, str(info.get("layer_id", "")))
    if bibtex:
        entry["bibtex"] = bibtex
    if info.get("coverage_note"):
        entry["coverage_note"] = info["coverage_note"]
    if info.get("color"):
        entry["color"] = info["color"]
    if info.get("primary") is not None:
        entry["primary"] = bool(info["primary"])
    return entry


def _sidecar_years(sidecar: Mapping[str, Any]) -> list[int] | None:
    years = sidecar.get("years")
    if isinstance(years, list) and years and all(isinstance(y, int) for y in years):
        return [min(years), max(years)]
    return None


def parse_markdown_sections(text: str) -> list[dict[str, Any]]:
    """Split methodology Markdown into collapsible h1-h3 sections.

    The METODOLOGIA tab renders a minimal styling subset, so full CommonMark
    parsing is unnecessary; body lines are preserved verbatim.
    """
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] = {"title": "Resumen", "level": 1, "body": []}
    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)", line)
        if match:
            if current["body"] or current["title"] != "Resumen":
                sections.append(current)
            current = {"title": match.group(2).strip(), "level": len(match.group(1)), "body": []}
        else:
            current["body"].append(line)
    if current["body"]:
        sections.append(current)
    return sections


def build_methodology_json(
    layer_id: str,
    info: Mapping[str, Any],
    repo_root: str | Path,
    *,
    extra_limitations: Iterable[str] = (),
    fallback_markdown: str | None = None,
) -> dict[str, Any] | None:
    """Build the METODOLOGIA payload with a trailing Limitaciones section.

    A declared-but-missing ``methodology_doc`` raises so a publish can never
    silently ship an empty tab; a layer with no declared doc may still pass
    ``fallback_markdown`` (native metadata ``method`` prose).
    """
    doc_ref = info.get("methodology_doc")
    if doc_ref:
        source = Path(str(doc_ref))
        if not source.is_absolute():
            source = Path(repo_root) / source
        if not source.is_file():
            raise FileNotFoundError(
                f"Layer {layer_id!r} declares missing methodology_doc: {source}"
            )
        markdown = source.read_text(encoding="utf-8")
    elif fallback_markdown:
        markdown = fallback_markdown
    else:
        return None
    sections = parse_markdown_sections(markdown)
    limitations: list[str] = []
    for item in list(info.get("limitations") or []) + list(extra_limitations):
        text = str(item).strip()
        if text and text not in limitations:
            limitations.append(text)
    if limitations:
        sections.append(
            {
                "title": "Limitaciones",
                "level": 2,
                "body": [f"- {text}" for text in limitations],
            }
        )
    return {"id": layer_id, "sections": sections, "raw": markdown}


def expand_aliases(
    entries: Mapping[str, dict[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Duplicate each layer_id entry under its palette webapp_keys aliases."""
    expanded: dict[str, dict[str, Any]] = dict(entries)
    for layer_id, entry in entries.items():
        for alias in (registry.get(layer_id) or {}).get("webapp_keys", []):
            if alias == layer_id:
                continue
            existing = expanded.get(alias)
            if existing is not None and existing.get("layer_id") != layer_id:
                raise StudyConfigError(
                    f"webapp key {alias!r} claimed by {layer_id!r} collides with "
                    f"layer {existing.get('layer_id')!r}"
                )
            expanded[alias] = entry
    return expanded
