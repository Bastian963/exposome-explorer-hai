"""Configuration contract for multi-location exposome studies.

The historical pipeline stores layer configuration in
``config/cities/<city>.yaml``.  New studies split stable location metadata from
the study-specific polygons and enabled layers.  This module supports both
layouts so layers can migrate without requiring a flag day.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COUNTRY_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "argentina": ("AR", "ARG", "America/Argentina/Buenos_Aires"),
    "chile": ("CL", "CHL", "America/Santiago"),
}
_LEGACY_NON_LAYER_KEYS = {
    "name",
    "country",
    "country_code",
    "region_query",
    "admin_level",
    "expected_communes",
    "expected_units",
    "crs",
    "bbox",
    "outputs",
    "boundaries",
}
_LAYER_SETTINGS_ALIASES = {
    "air_quality_pm25": "pm25",
    "greenspace_coverage": "greenspace",
    "greenspace_multisource": "greenspace",
    "greenspace_access": "greenspace",
    "greenspace_cv": "greenspace",
}
CACHE_ROOT_ENV = "EXPOSOME_CACHE_ROOT"


class StudyConfigError(ValueError):
    """Raised when a location or study configuration is invalid."""


@dataclass(frozen=True)
class BoundingBox:
    """A WGS84 bounding box in west, south, east, north order."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180 <= self.west < self.east <= 180):
            raise StudyConfigError(
                "bbox longitude must satisfy -180 <= west < east <= 180"
            )
        if not (-90 <= self.south < self.north <= 90):
            raise StudyConfigError(
                "bbox latitude must satisfy -90 <= south < north <= 90"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundingBox":
        aliases = {
            "west": ("west", "lon_min"),
            "south": ("south", "lat_min"),
            "east": ("east", "lon_max"),
            "north": ("north", "lat_max"),
        }
        parsed: dict[str, float] = {}
        for target, candidates in aliases.items():
            raw = next((value[key] for key in candidates if key in value), None)
            if raw is None:
                raise StudyConfigError(
                    f"bbox is missing '{target}' (accepted aliases: {candidates})"
                )
            try:
                parsed[target] = float(raw)
            except (TypeError, ValueError) as exc:
                raise StudyConfigError(f"bbox '{target}' must be numeric") from exc
        return cls(**parsed)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north


@dataclass(frozen=True)
class LocationConfig:
    """Stable geographic metadata shared by studies in one city."""

    id: str
    name: str
    country: str
    country_code: str
    country_code3: str | None
    timezone: str | None
    bbox: BoundingBox
    geographic_crs: str
    metric_crs: str | None
    config_path: Path
    raw: Mapping[str, Any]

    @property
    def city(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "country_code": self.country_code,
            "country_code3": self.country_code3,
            "timezone": self.timezone,
            "bbox": {
                "west": self.bbox.west,
                "south": self.bbox.south,
                "east": self.bbox.east,
                "north": self.bbox.north,
            },
            "crs": {
                "geographic": self.geographic_crs,
                "metric": self.metric_crs or "auto",
            },
        }


@dataclass(frozen=True)
class TemporalException:
    """A documented, permanent gap in one layer's required annual series.

    Declared per study so an undeclared missing year still aborts publication
    elsewhere (ADR 0008). The check that actually skips these years operates
    on ``layer_id`` (see ``discover_temporal_specs``/``_write_temporal_assets``
    in publishing.py), so ``indicator`` is documentation, not a finer-grained
    filter: a layer with multiple required indicators would have every one of
    them excepted for the declared years.
    """

    layer_id: str
    indicator: str
    years: tuple[int, ...]
    reason: str
    doc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "indicator": self.indicator,
            "years": list(self.years),
            "reason": self.reason,
            "doc": self.doc,
        }


@dataclass(frozen=True)
class PublicationPolicy:
    """Study-level acceptance target for one published browser product.

    ``maximal_spatial`` preserves the repository-wide default: every available
    indicator must publish the finest honest product supported by its method.
    ``administrative`` is an explicit aggregate-only opt-in for a product whose
    acceptance unit is the configured administrative geography.  It never
    upgrades polygon values into pixels.
    """

    completion_target: str = "maximal_spatial"
    detail_policy: str = "verified_optional"
    provenance_policy: str = "canonical"
    reviewed_legacy_layers: tuple[str, ...] = ()
    rationale: str | None = None

    @property
    def is_administrative(self) -> bool:
        return self.completion_target == "administrative"

    @property
    def administrative_only(self) -> bool:
        return self.detail_policy == "administrative_only"

    @property
    def accepts_declared_historical_provenance(self) -> bool:
        return self.provenance_policy == "declared_historical"

    @property
    def accepts_reviewed_legacy_provenance(self) -> bool:
        """Whether explicitly audited historical bundles may be published.

        This is deliberately separate from ``declared_historical``.  The
        latter is an aggregate-only legacy policy; reviewed bundles can retain
        an honestly published native detail asset when that detail is itself
        verified.
        """
        return self.provenance_policy == "reviewed_legacy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion_target": self.completion_target,
            "detail_policy": self.detail_policy,
            "provenance_policy": self.provenance_policy,
            "reviewed_legacy_layers": list(self.reviewed_legacy_layers),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class StudyConfig:
    """Study-specific spatial mode, AOI/units input, and layer selection."""

    id: str
    location_ref: str
    spatial_path: Path
    mode: str
    aoi_path: Path | None
    aoi_source: str | None
    aoi_license: str | None
    spatial_layer: str | None
    id_column: str
    name_column: str | None
    unit_type: str
    expected_units: int | None
    native_resolution: str | None
    spatial_source: str | None
    spatial_license: str | None
    enabled_layers: tuple[str, ...]
    period: Mapping[str, Any]
    config_path: Path
    raw: Mapping[str, Any]
    hidden: bool = False
    temporal_exceptions: tuple[TemporalException, ...] = ()
    publication_policy: PublicationPolicy = PublicationPolicy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "location": self.location_ref,
            "mode": self.mode,
            "aoi": {
                "path": str(self.aoi_path) if self.aoi_path else None,
                "source": self.aoi_source,
                "license": self.aoi_license,
            },
            "spatial": {
                "path": str(self.spatial_path) if self.spatial_path else None,
                "layer": self.spatial_layer,
                "id_column": self.id_column,
                "name_column": self.name_column,
                "unit_type": self.unit_type,
                "expected_units": self.expected_units,
                "native_resolution": self.native_resolution,
                "source": self.spatial_source,
                "license": self.spatial_license,
            },
            "layers": list(self.enabled_layers),
            "period": dict(self.period),
            "hidden": self.hidden,
            "temporal_exceptions": [item.to_dict() for item in self.temporal_exceptions],
            "publication": self.publication_policy.to_dict(),
        }


@dataclass(frozen=True)
class StudyPaths:
    """Canonical data and cache directories for one study."""

    reference: Path
    raw: Path
    interim: Path
    processed: Path
    cache: Path

    def layer_raw(self, layer_id: str) -> Path:
        """Deprecated study-scoped raw path retained for compatibility.

        New provider downloads must use :meth:`provider_raw` so they can be
        reused across cities and studies.
        """
        return self.raw / "legacy-study-inputs" / _validate_slug(layer_id, "layer id")

    def provider_raw(self, provider: str, dataset: str, version: str) -> Path:
        """Immutable raw download root shared by all studies."""
        return self.raw / _validate_slug(provider, "provider") / _validate_slug(
            dataset, "dataset"
        ) / _validate_slug(version, "version")

    def provider_snapshot(self, provider: str, dataset: str, version: str) -> Any:
        """Resolve a repo manifest and optional external payload directory.

        ``GEMMA_RAW_PAYLOAD_ROOT`` is deliberately local-only. It mirrors
        ``data/raw`` by provider/dataset/version, while the small manifest
        remains versionable in this repository.
        """
        from .raw_sources import RAW_PAYLOAD_ROOT_ENV, RawSnapshotStore

        manifest_root = self.provider_raw(provider, dataset, version)
        configured = os.environ.get(RAW_PAYLOAD_ROOT_ENV)
        payload_root = (
            manifest_root
            if not configured
            else Path(configured).expanduser()
            / _validate_slug(provider, "provider")
            / _validate_slug(dataset, "dataset")
            / _validate_slug(version, "version")
        )
        return RawSnapshotStore(manifest_root=manifest_root, payload_root=payload_root)

    def layer_interim(self, layer_id: str) -> Path:
        return self.interim / _validate_slug(layer_id, "layer id")

    def layer_processed(self, layer_id: str) -> Path:
        return self.processed / _validate_slug(layer_id, "layer id")

    def layer_cache(self, layer_id: str) -> Path:
        return self.cache / _validate_slug(layer_id, "layer id")

    def to_dict(self) -> dict[str, str]:
        return {
            "reference": str(self.reference),
            "raw": str(self.raw),
            "interim": str(self.interim),
            "processed": str(self.processed),
            "cache": str(self.cache),
        }


@dataclass(frozen=True)
class StudyContext:
    """Resolved configuration and paths passed to layer runners."""

    repo_root: Path
    location: LocationConfig
    study: StudyConfig
    paths: StudyPaths
    legacy_config: Mapping[str, Any] | None = None

    @property
    def spatial_path(self) -> Path:
        return self.study.spatial_path

    @property
    def mode(self) -> str:
        return self.study.mode

    @property
    def is_native(self) -> bool:
        return self.mode == "native"

    @property
    def aoi_path(self) -> Path | None:
        return self.study.aoi_path

    @property
    def expected_units(self) -> int | None:
        return self.study.expected_units

    @property
    def enabled_layers(self) -> tuple[str, ...]:
        return self.study.enabled_layers

    @property
    def country_code(self) -> str:
        return self.location.country_code

    @property
    def city(self) -> str:
        return self.location.city

    @cached_property
    def settings(self) -> Any:
        """Resolved settings composed once for this Study."""
        from exposome.settings import resolve_settings

        return resolve_settings(self)

    def layer_settings(self, layer_id: str) -> Mapping[str, Any]:
        """Return the centrally resolved settings slice for one Layer."""
        key = _LAYER_SETTINGS_ALIASES.get(layer_id, layer_id)
        return self.settings.layer(key)

    @cached_property
    def input_paths(self) -> Mapping[str, Mapping[str, Path]]:
        """Resolve every declared Study input to an absolute path once."""
        raw_root = self.study.raw.get("layer_inputs", self.study.raw.get("inputs", {}))
        if raw_root in (None, {}):
            return {}
        if not isinstance(raw_root, Mapping):
            raise StudyConfigError(
                f"Study layer_inputs must be a mapping: {self.study.config_path}"
            )
        resolved: dict[str, Mapping[str, Path]] = {}
        for layer_id, values in raw_root.items():
            if not isinstance(values, Mapping):
                raise StudyConfigError(f"Study inputs for {layer_id!r} must be a mapping")
            paths: dict[str, Path] = {}
            for name, raw_path in values.items():
                if raw_path in (None, ""):
                    continue
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    path = self.repo_root / path
                paths[str(name)] = path.resolve()
            resolved[str(layer_id)] = paths
        return resolved

    def layer_inputs(self, layer_id: str) -> Mapping[str, Path]:
        return self.input_paths.get(layer_id, {})

    @cached_property
    def spatial_units(self) -> Any:
        """Canonical aggregate polygons shared by preflight and every Layer."""
        if self.is_native:
            raise StudyConfigError(
                f"Study {self.study.id!r} is native and has no analysis units"
            )
        from exposome.spatial import load_spatial_units

        return load_spatial_units(self)

    @cached_property
    def metric_crs(self) -> str | None:
        if self.is_native:
            return self.location.metric_crs
        value = getattr(self.spatial_units, "attrs", {}).get("metric_crs")
        if value:
            return str(value)
        from exposome.spatial import resolve_metric_crs

        return resolve_metric_crs(
            self.spatial_units,
            configured=self.location.metric_crs,
        ).to_string()

    @cached_property
    def config(self) -> dict[str, Any]:
        """Return a mapping usable by code transitioning from city configs."""
        cfg = dict(self.legacy_config or {})
        # Modern studies receive the same resolved settings as the CLI.  A
        # direct legacy city study may still seed its adapter mapping above,
        # but no location inherits a different city's configuration.
        try:
            cfg.update(self.settings.legacy_mapping())
        except (ImportError, FileNotFoundError):
            # Keep direct legacy-city loading available while bootstrapping a
            # source checkout that has not yet run the config migration.
            pass
        except StudyConfigError as exc:
            if not str(exc).startswith("No layer defaults found"):
                raise
        from exposome.config import _apply_study_period

        _apply_study_period(
            cfg,
            self.study.period,
            enabled_layers=self.study.enabled_layers,
        )
        cfg.update(
            {
                "name": self.city,
                "location_id": self.location.id,
                "country": self.location.country,
                "country_code": self.country_code,
                "country_code3": self.location.country_code3,
                "timezone": self.location.timezone,
                # Legacy runners still accept a place query as a fallback,
                # although modern OSM layers normally query the study AOI.
                # Keep it in the injected mapping too: resolved_config() is
                # the mapping actually passed by the importable runner.
                "region_query": self.location.raw.get(
                    "region_query",
                    f"{self.location.name}, {self.location.country}",
                ),
                "study_id": self.study.id,
                "mode": self.mode,
                "expected_units": self.expected_units,
                # Transitional alias used by existing layer implementations.
                "expected_communes": self.expected_units,
                "crs": {
                    "geographic": self.location.geographic_crs,
                    "metric": self.location.metric_crs or "auto",
                },
                "bbox": {
                    "lon_min": self.location.bbox.west,
                    "lat_min": self.location.bbox.south,
                    "lon_max": self.location.bbox.east,
                    "lat_max": self.location.bbox.north,
                },
                "study_paths": self.paths.to_dict(),
                "aoi_path": str(self.aoi_path) if self.aoi_path else "",
            }
        )
        return cfg

    def load_spatial_units(self, *, repair_invalid: bool = False) -> Any:
        """Load this study's canonical polygons without a module-level cycle."""
        if self.is_native:
            raise StudyConfigError(
                f"Study {self.study.id!r} is native and has no analysis units; "
                "load the AOI instead."
            )
        if not repair_invalid:
            return self.spatial_units.copy()
        from exposome.spatial import load_spatial_units

        return load_spatial_units(self, repair_invalid=True)

    def resolved_config(
        self,
        *,
        spatial_units: Any | None = None,
        repair_invalid: bool = False,
    ) -> dict[str, Any]:
        """Return the transitional mapping with a concrete projected CRS.

        Existing layer implementations cannot pass ``"auto"`` to
        ``GeoDataFrame.to_crs``.  Callers that bridge a StudyContext into those
        layers should use this method after loading the study polygons.
        """
        if self.is_native:
            cfg = deepcopy(self.config)
            cfg["mode"] = "native"
            cfg["aoi_path"] = str(self.aoi_path) if self.aoi_path else ""
            self._apply_runner_portable_postprocess(cfg)
            return cfg
        units = spatial_units if spatial_units is not None else self.load_spatial_units(
            repair_invalid=repair_invalid
        )
        metric_crs = getattr(units, "attrs", {}).get("metric_crs") or self.metric_crs
        cfg = deepcopy(self.config)
        cfg["crs"] = dict(cfg["crs"])
        cfg["crs"]["metric"] = str(metric_crs)
        cfg["spatial_path"] = str(self.spatial_path)
        cfg["spatial_id_column"] = "spatial_id"
        cfg["spatial_name_column"] = "spatial_name"
        # The runner injects this mapping via ``resolved_config_scope`` so legacy
        # builders see the same ``spatial_units`` metadata dict that the direct
        # ``config.load_config`` path emits (see config.py). Without it, layers
        # that resolve per-unit geometry from ``cfg["spatial_units"]`` --
        # ``fetch_green_areas`` (bbox tiling), ``fetch_healthcare_osm`` (compact
        # AOI query) -- silently fall back to un-tiled, per-place-name Overpass
        # queries and time out on large rural units (Bogota's Sumapaz/Usme
        # localidades). See docs/knowledge/runbooks/incidentes-multiciudad.md.
        cfg["spatial_units"] = {
            "path": str(self.spatial_path),
            "layer": self.study.spatial_layer,
            "id_column": self.study.id_column,
            "name_column": self.study.name_column,
            "unit_type": self.study.unit_type,
            "expected_units": len(units),
            "source": self.study.spatial_source,
        }
        # Healthcare walk-graph cache and output base are study-scoped (mirrors
        # ``config.load_config``); leaving the layer default reuses Santiago's
        # graph/output names for every other city.
        healthcare = cfg.get("healthcare")
        if isinstance(healthcare, dict):
            healthcare["output_base"] = f"{self.study.id}_healthcare_access"
            healthcare.setdefault("network", {})["graph_cache"] = str(
                self.paths.cache / f"{self.study.id}_walk_graph.graphml"
            )
        self._apply_runner_portable_postprocess(cfg)
        return cfg

    def _apply_runner_portable_postprocess(self, cfg: dict[str, Any]) -> None:
        """Apply the portable post-processing that ``config.load_config`` does.

        The runner injects this mapping via ``resolved_config_scope`` instead of
        letting ``config.load_config`` resolve the study, so any portable
        post-process that path applies must be applied here too or the runner
        ships the wrong value silently. Covers:

        * ``population.country`` -- ``WorldPop/GP/100m/pop`` is per-country; the
          ``CHL`` layer default degenerates pop-weighting to the area mean off
          Chile (silent Chilean-data proxy). ``pm25`` reuses ``alan``'s
          population when it declares none, so both follow the study country.
        * hemisphere summer convention -- Dec--Feb south of the equator, Jun--Aug
          north; the default (Dec--Feb) is wrong for northern cities (Bogota).

        A parity test (tests/test_study_context_runtime.py) fails if this drifts
        from ``config.load_config``. See docs/knowledge/runbooks/incidentes-multiciudad.md.
        """
        from .config import _apply_location_summer_convention

        country3 = self.location.country_code3
        if country3 and cfg.get("alan", {}).get("population"):
            cfg["alan"]["population"]["country"] = country3
        if country3 and cfg.get("pm25", {}).get("population"):
            cfg["pm25"]["population"]["country"] = country3
        _apply_location_summer_convention(cfg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "location": self.location.to_dict(),
            "study": self.study.to_dict(),
            "paths": self.paths.to_dict(),
        }


def repo_root() -> Path:
    """Return the repository root inferred from this package."""
    return Path(__file__).resolve().parents[2]


def load_location(
    ref: str | Path | LocationConfig,
    *,
    repo_root_path: str | Path | None = None,
) -> LocationConfig:
    """Load a location YAML or adapt a legacy ``config/cities`` YAML."""
    if isinstance(ref, LocationConfig):
        return ref
    root = _normalise_root(repo_root_path)
    path = _resolve_location_path(ref, root)
    raw = _read_yaml(path)
    return _parse_location(raw, path)


def load_study(
    ref: str | Path | StudyContext,
    *,
    repo_root_path: str | Path | None = None,
) -> StudyContext:
    """Resolve a study, its location, canonical directories and legacy config.

    If no new study YAML matches ``ref`` but ``config/cities/<ref>.yaml``
    exists, a transitional study is synthesized from that city config.
    """
    if isinstance(ref, StudyContext):
        return ref
    root = _normalise_root(repo_root_path)
    path = _resolve_study_path(ref, root)

    if _is_legacy_city_path(path, root):
        raw_legacy = _read_yaml(path)
        location = _parse_location(raw_legacy, path)
        study = _parse_legacy_study(raw_legacy, path, root, location)
        legacy_config: Mapping[str, Any] | None = raw_legacy
    else:
        raw_study = _read_yaml(path)
        study = _parse_study(raw_study, path, root)
        location = load_location(study.location_ref, repo_root_path=root)
        legacy_config = _load_legacy_config(location, root)

    paths = build_study_paths(root, location, study)
    return StudyContext(
        repo_root=root,
        location=location,
        study=study,
        paths=paths,
        legacy_config=legacy_config,
    )


def build_study_paths(
    root: str | Path,
    location: LocationConfig,
    study: StudyConfig,
) -> StudyPaths:
    """Build canonical paths without creating directories."""
    base = Path(root).resolve()
    parts = (
        location.country_code.lower(),
        _validate_slug(location.id, "location id"),
        _validate_slug(study.id, "study id"),
    )
    configured_cache = os.environ.get(CACHE_ROOT_ENV)
    cache_root = Path(configured_cache).expanduser() if configured_cache else base / "cache"
    if not cache_root.is_absolute():
        cache_root = base / cache_root
    return StudyPaths(
        reference=base.joinpath("data", "reference", *parts),
        raw=base.joinpath("data", "raw"),
        interim=base.joinpath("data", "interim", *parts),
        processed=base.joinpath("data", "processed", *parts),
        cache=cache_root.joinpath(*parts),
    )


def _normalise_root(value: str | Path | None) -> Path:
    return Path(value).expanduser().resolve() if value is not None else repo_root()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise StudyConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise StudyConfigError(f"Configuration must be a YAML mapping: {path}")
    return raw


def _resolve_location_path(ref: str | Path, root: Path) -> Path:
    direct = _direct_yaml_path(ref, root)
    if direct is not None:
        return direct

    value = str(ref)
    candidates = [root / "config" / "locations" / f"{value}.yaml"]
    if "/" not in value and "\\" not in value:
        candidates.extend(sorted((root / "config" / "locations").glob(f"*/{value}.yaml")))
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise StudyConfigError(
            f"Ambiguous location '{value}'; use an ISO/city reference. Matches: {existing}"
        )
    if "/" not in value and "\\" not in value:
        legacy = root / "config" / "cities" / f"{value}.yaml"
        if legacy.is_file():
            return legacy.resolve()
    raise FileNotFoundError(f"Location config not found for '{value}'")


def _resolve_study_path(ref: str | Path, root: Path) -> Path:
    direct = _direct_yaml_path(ref, root)
    if direct is not None:
        return direct

    value = str(ref)
    candidates = [root / "config" / "studies" / f"{value}.yaml"]
    if "/" not in value and "\\" not in value:
        candidates.extend(sorted((root / "config" / "studies").glob(f"*/{value}.yaml")))
        # Compatibility fallback; a new study config always wins.
        candidates.append(root / "config" / "cities" / f"{value}.yaml")
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not existing:
        raise FileNotFoundError(f"Study config not found for '{value}'")
    if len(existing) > 1:
        # The first candidate is the exact new-layout match.
        exact = (root / "config" / "studies" / f"{value}.yaml").resolve()
        if exact in existing:
            return exact
        raise StudyConfigError(f"Ambiguous study '{value}'. Matches: {existing}")
    return existing[0]


def _direct_yaml_path(ref: str | Path, root: Path) -> Path | None:
    value = Path(ref).expanduser()
    looks_like_path = value.suffix.lower() in {".yaml", ".yml"} or value.is_absolute()
    if not looks_like_path:
        return None
    candidate = value if value.is_absolute() else root / value
    if not candidate.is_file():
        raise FileNotFoundError(f"Configuration not found: {candidate}")
    return candidate.resolve()


def _parse_location(raw: Mapping[str, Any], path: Path) -> LocationConfig:
    location_id = _validate_slug(str(raw.get("id") or raw.get("name") or ""), "location id")
    name = str(raw.get("display_name") or raw.get("name") or "").strip()
    country = str(raw.get("country") or "").strip()
    if not name:
        raise StudyConfigError(f"Location is missing 'name' or 'display_name': {path}")
    if not country:
        raise StudyConfigError(f"Location is missing 'country': {path}")

    defaults = _COUNTRY_DEFAULTS.get(country.casefold())
    country_code = str(raw.get("country_code") or (defaults[0] if defaults else "")).upper()
    country_code3_raw = raw.get("country_code3") or (defaults[1] if defaults else None)
    country_code3 = str(country_code3_raw).upper() if country_code3_raw else None
    if not re.fullmatch(r"[A-Z]{2}", country_code):
        raise StudyConfigError(
            f"Location country_code must be a two-letter ISO code: {path}"
        )
    if country_code3 is not None and not re.fullmatch(r"[A-Z]{3}", country_code3):
        raise StudyConfigError(
            f"Location country_code3 must be a three-letter ISO code: {path}"
        )

    bbox_raw = raw.get("bbox")
    if not isinstance(bbox_raw, Mapping):
        raise StudyConfigError(f"Location requires a bbox mapping: {path}")
    bbox = BoundingBox.from_mapping(bbox_raw)

    crs = raw.get("crs", {})
    if not isinstance(crs, Mapping):
        raise StudyConfigError(f"Location crs must be a mapping: {path}")
    geographic_crs = str(crs.get("geographic") or "EPSG:4326")
    metric_raw = crs.get("metric")
    metric_crs = None if metric_raw in (None, "", "auto") else str(metric_raw)
    timezone_raw = raw.get("timezone") or (defaults[2] if defaults else None)

    return LocationConfig(
        id=location_id,
        name=name,
        country=country,
        country_code=country_code,
        country_code3=country_code3,
        timezone=str(timezone_raw) if timezone_raw else None,
        bbox=bbox,
        geographic_crs=geographic_crs,
        metric_crs=metric_crs,
        config_path=path,
        raw=dict(raw),
    )


def _parse_study(raw: Mapping[str, Any], path: Path, root: Path) -> StudyConfig:
    study_id = _validate_slug(str(raw.get("id") or path.stem), "study id")
    location_ref = str(raw.get("location") or "").strip()
    if not location_ref:
        raise StudyConfigError(f"Study is missing 'location': {path}")
    mode = str(raw.get("mode") or "aggregate").strip().casefold()
    if mode not in {"aggregate", "native"}:
        raise StudyConfigError(f"Study mode must be 'aggregate' or 'native': {path}")

    spatial = raw.get("spatial")
    if spatial is None:
        spatial = {}
    if not isinstance(spatial, Mapping):
        raise StudyConfigError(f"Study spatial config must be a mapping: {path}")

    aoi = raw.get("aoi")
    if aoi is None:
        aoi = {}
    if not isinstance(aoi, Mapping):
        raise StudyConfigError(f"Study AOI config must be a mapping: {path}")

    def resolve_geo_path(value: Any, *, label: str) -> Path | None:
        if value in (None, ""):
            return None
        resolved = Path(str(value)).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
        if resolved.suffix.lower() not in {".geojson", ".json", ".gpkg"}:
            raise StudyConfigError(
                f"Study {label} input must be GeoJSON (.geojson/.json) or GeoPackage (.gpkg)"
            )
        return resolved.resolve()

    spatial_path = resolve_geo_path(spatial.get("path"), label="spatial")
    aoi_path = resolve_geo_path(aoi.get("path"), label="AOI")
    if mode == "aggregate" and spatial_path is None:
        raise StudyConfigError(f"Aggregate study requires spatial.path: {path}")
    if mode == "native" and aoi_path is None:
        raise StudyConfigError(f"Native study requires aoi.path: {path}")
    # Keep one compatibility path for existing code; native runners use aoi_path.
    spatial_path = spatial_path or aoi_path
    assert spatial_path is not None

    id_column = str(spatial.get("id_column") or "spatial_id").strip()
    name_raw = spatial.get("name_column")
    name_column = str(name_raw).strip() if name_raw not in (None, "") else None
    unit_type = _validate_slug(
        str(spatial.get("unit_type") or ("native_product" if mode == "native" else "spatial_unit")),
        "spatial unit type",
    )
    expected_units = _optional_positive_int(spatial.get("expected_units"), "expected_units")
    enabled_layers = _parse_enabled_layers(raw.get("layers", []), path)
    period = raw.get("period", {})
    if not isinstance(period, Mapping):
        raise StudyConfigError(f"Study period must be a mapping: {path}")
    temporal_exceptions = _parse_temporal_exceptions(
        raw.get("temporal_exceptions"), enabled_layers, path
    )
    publication_policy = _parse_publication_policy(raw.get("publication"), mode, path)
    from .layers import load_layer_catalog

    catalog = load_layer_catalog()
    resolved_enabled_layers = {catalog.resolve_id(layer_id) for layer_id in enabled_layers}
    unknown_reviewed_layers = sorted(
        catalog.resolve_id(layer_id)
        for layer_id in publication_policy.reviewed_legacy_layers
        if catalog.resolve_id(layer_id) not in resolved_enabled_layers
    )
    if unknown_reviewed_layers:
        raise StudyConfigError(
            "publication.reviewed_legacy_layers must be enabled Layers: "
            + ", ".join(unknown_reviewed_layers)
        )
    if publication_policy.is_administrative and expected_units is None:
        raise StudyConfigError(
            f"Administrative publication requires spatial.expected_units: {path}"
        )

    return StudyConfig(
        id=study_id,
        location_ref=location_ref,
        spatial_path=spatial_path.resolve(),
        mode=mode,
        aoi_path=aoi_path,
        aoi_source=_optional_string(aoi.get("source")),
        aoi_license=_optional_string(aoi.get("license")),
        spatial_layer=_optional_string(spatial.get("layer")),
        id_column=id_column,
        name_column=name_column,
        unit_type=unit_type,
        expected_units=expected_units,
        native_resolution=_optional_string(spatial.get("native_resolution")),
        spatial_source=_optional_string(spatial.get("source")),
        spatial_license=_optional_string(spatial.get("license")),
        enabled_layers=enabled_layers,
        period=dict(period),
        config_path=path,
        raw=dict(raw),
        hidden=bool(raw.get("hidden", False)),
        temporal_exceptions=temporal_exceptions,
        publication_policy=publication_policy,
    )


def _parse_legacy_study(
    raw: Mapping[str, Any],
    path: Path,
    root: Path,
    location: LocationConfig,
) -> StudyConfig:
    boundaries = raw.get("boundaries", {})
    if not isinstance(boundaries, Mapping):
        boundaries = {}
    spatial_path_raw = boundaries.get("path")
    if spatial_path_raw:
        spatial_path = Path(str(spatial_path_raw)).expanduser()
        if not spatial_path.is_absolute():
            spatial_path = root / spatial_path
    elif location.id == "santiago":
        spatial_path = root / "data" / "processed" / "socioeconomic_exposome_rm_santiago.geojson"
    else:
        spatial_path = root / "data" / "processed" / f"{location.id}_boundaries.geojson"

    expected = raw.get("expected_units", raw.get("expected_communes"))
    layers = tuple(
        key
        for key, value in raw.items()
        if key not in _LEGACY_NON_LAYER_KEYS and isinstance(value, Mapping)
    )
    return StudyConfig(
        id=location.id,
        location_ref=str(path),
        spatial_path=spatial_path.resolve(),
        mode="aggregate",
        aoi_path=None,
        aoi_source=None,
        aoi_license=None,
        spatial_layer=_optional_string(boundaries.get("layer")),
        id_column=str(boundaries.get("id_column") or "name"),
        name_column=str(boundaries.get("name_column") or "name"),
        unit_type=str(boundaries.get("unit_type") or "commune"),
        expected_units=_optional_positive_int(expected, "expected_units"),
        native_resolution=_optional_string(boundaries.get("native_resolution")),
        spatial_source=_optional_string(boundaries.get("source")) or "legacy city config",
        spatial_license=_optional_string(boundaries.get("license")),
        enabled_layers=layers,
        period={},
        config_path=path,
        raw=dict(raw),
        publication_policy=PublicationPolicy(),
    )


def _parse_publication_policy(
    value: Any,
    mode: str,
    path: Path,
) -> PublicationPolicy:
    if value in (None, ""):
        return PublicationPolicy()
    if not isinstance(value, Mapping):
        raise StudyConfigError(f"Study publication config must be a mapping: {path}")

    completion_target = str(
        value.get("completion_target") or "maximal_spatial"
    ).strip().casefold()
    if completion_target not in {"maximal_spatial", "administrative"}:
        raise StudyConfigError(
            "publication.completion_target must be 'maximal_spatial' or "
            f"'administrative': {path}"
        )
    detail_default = (
        "administrative_only"
        if completion_target == "administrative"
        else "verified_optional"
    )
    detail_policy = str(value.get("detail_policy") or detail_default).strip().casefold()
    if detail_policy not in {"verified_optional", "administrative_only"}:
        raise StudyConfigError(
            "publication.detail_policy must be 'verified_optional' or "
            f"'administrative_only': {path}"
        )
    provenance_policy = str(
        value.get("provenance_policy") or "canonical"
    ).strip().casefold()
    if provenance_policy not in {"canonical", "declared_historical", "reviewed_legacy"}:
        raise StudyConfigError(
            "publication.provenance_policy must be 'canonical', "
            f"'declared_historical' or 'reviewed_legacy': {path}"
        )
    raw_reviewed_layers = value.get("reviewed_legacy_layers") or []
    if not isinstance(raw_reviewed_layers, list) or not all(
        isinstance(layer_id, str) and layer_id.strip() for layer_id in raw_reviewed_layers
    ):
        raise StudyConfigError(
            f"publication.reviewed_legacy_layers must be a list of layer IDs: {path}"
        )
    reviewed_legacy_layers = tuple(dict.fromkeys(item.strip() for item in raw_reviewed_layers))
    rationale = _optional_string(value.get("rationale"))

    if completion_target == "administrative":
        if mode != "aggregate":
            raise StudyConfigError(
                f"Administrative publication is only valid for aggregate studies: {path}"
            )
        if detail_policy != "administrative_only":
            raise StudyConfigError(
                "Administrative publication requires detail_policy "
                f"'administrative_only': {path}"
            )
        if not rationale:
            raise StudyConfigError(
                f"Administrative publication requires a non-empty rationale: {path}"
            )
    elif detail_policy != "verified_optional":
        raise StudyConfigError(
            "maximal_spatial publication requires detail_policy "
            f"'verified_optional': {path}"
        )
    if provenance_policy == "declared_historical" and completion_target != "administrative":
        raise StudyConfigError(
            "declared_historical provenance is only valid for an explicit "
            f"administrative publication: {path}"
        )
    if provenance_policy == "reviewed_legacy":
        if not reviewed_legacy_layers:
            raise StudyConfigError(
                f"reviewed_legacy provenance requires reviewed_legacy_layers: {path}"
            )
        if "wind" in reviewed_legacy_layers:
            raise StudyConfigError(
                "wind cannot use reviewed_legacy provenance; it must be recomputed"
            )
    elif reviewed_legacy_layers:
        raise StudyConfigError(
            "publication.reviewed_legacy_layers requires provenance_policy 'reviewed_legacy'"
        )

    return PublicationPolicy(
        completion_target=completion_target,
        detail_policy=detail_policy,
        provenance_policy=provenance_policy,
        reviewed_legacy_layers=reviewed_legacy_layers,
        rationale=rationale,
    )


def _parse_temporal_exceptions(
    value: Any, enabled_layers: tuple[str, ...], path: Path
) -> tuple[TemporalException, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise StudyConfigError(f"temporal_exceptions must be a list: {path}")
    exceptions: list[TemporalException] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise StudyConfigError(f"temporal_exceptions[{index}] must be a mapping: {path}")
        layer_id = _validate_slug(
            str(entry.get("layer_id") or ""), "temporal_exceptions layer_id"
        )
        if layer_id not in enabled_layers:
            raise StudyConfigError(
                f"temporal_exceptions[{index}] references layer {layer_id!r}, which is "
                f"not enabled for this study -- the exception would be dead code: {path}"
            )
        indicator = str(entry.get("indicator") or "").strip()
        if not indicator:
            raise StudyConfigError(
                f"temporal_exceptions[{index}] requires a non-empty 'indicator': {path}"
            )
        raw_years = entry.get("years")
        if not isinstance(raw_years, list) or not raw_years:
            raise StudyConfigError(
                f"temporal_exceptions[{index}] requires a non-empty 'years' list: {path}"
            )
        years: list[int] = []
        for raw_year in raw_years:
            text = str(raw_year)
            if not (text.isdigit() and len(text) == 4):
                raise StudyConfigError(
                    f"temporal_exceptions[{index}] year {raw_year!r} must be a 4-digit "
                    f"year: {path}"
                )
            years.append(int(text))
        reason = str(entry.get("reason") or "").strip()
        if not reason:
            raise StudyConfigError(
                f"temporal_exceptions[{index}] requires a non-empty 'reason': {path}"
            )
        doc = str(entry.get("doc") or "").strip()
        if not doc:
            raise StudyConfigError(
                f"temporal_exceptions[{index}] requires a non-empty 'doc': {path}"
            )
        exceptions.append(
            TemporalException(
                layer_id=layer_id,
                indicator=indicator,
                years=tuple(sorted(set(years))),
                reason=reason,
                doc=doc,
            )
        )
    return tuple(exceptions)


def _parse_enabled_layers(value: Any, path: Path) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        items = [
            key
            for key, settings in value.items()
            if not isinstance(settings, Mapping) or settings.get("enabled", True)
        ]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise StudyConfigError(f"Study layers must be a list or mapping: {path}")
    parsed = tuple(_validate_slug(str(item), "layer id") for item in items)
    if len(parsed) != len(set(parsed)):
        raise StudyConfigError(f"Study layers contain duplicates: {path}")
    return parsed


def _load_legacy_config(
    location: LocationConfig,
    root: Path,
) -> Mapping[str, Any] | None:
    legacy_ref = location.raw.get("legacy_city_config")
    if not legacy_ref:
        return None
    path = Path(str(legacy_ref)).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Legacy city config not found: {path}")
    return _read_yaml(path)


def _is_legacy_city_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to((root / "config" / "cities").resolve())
    except ValueError:
        return False
    return True


def _validate_slug(value: str, label: str) -> str:
    normalised = value.strip().lower()
    if not _SLUG_RE.fullmatch(normalised):
        raise StudyConfigError(
            f"{label} must use lowercase ASCII letters, numbers, '-' or '_': {value!r}"
        )
    return normalised


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StudyConfigError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise StudyConfigError(f"{label} must be a positive integer")
    return parsed


def _optional_string(value: Any) -> str | None:
    return str(value).strip() if value not in (None, "") else None
