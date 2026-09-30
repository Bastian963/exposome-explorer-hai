"""Variable catalog shared by the study reports and the interactive explorer.

One place defines which exposome variables can be mapped for a study, how they are
drawn (raster / 1 km grid / administrative choropleth), and the English text that
accompanies them.  A variable is included for a study only if its data exist in that
study's published web bundle, so a new study needs no code change.

Reads only local, already-published artifacts (the study's web bundle,
``config/layer_info`` and ``config/reports/variable_report.yaml``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import report_figures as rf
from .layer_info import load_layer_info_registry

REPO = Path(__file__).resolve().parents[2]
STATUS_CSV = REPO / "docs/exposome_status.csv"
TEXT_YAML = REPO / "config/reports/variable_report.yaml"

DOMAINS = {
    "aire": "Air quality",
    "luz": "Light and noise",
    "verde": "Vegetation and water",
    "clima": "Climate and extreme events",
    "construido": "Built environment and mobility",
    "servicios": "Services and food environment",
    "social": "Socioeconomic and demographic",
}


@dataclass(frozen=True)
class Var:
    key: str  # manifest spatial_indicators key (or a report-only key for master columns)
    layer_id: str
    domain: str
    label: str
    unit: str
    kind: str  # raster | grid | choropleth
    source: str  # bundle-relative path (raster/grid) or master.geojson column
    cmap: str = "sequential"
    limits: tuple[float, float] | None = None
    prop: str = "value"  # property to color for kind == grid


# Order matters: contact sheets, variable sheets and the inventory follow it. The least striking maps
# (near-empty binary water masks, the sparse heavy-metals index) come last on purpose.
VARS: list[Var] = [
    Var("pm25", "air_quality_pm25", "aire", "PM$_{2.5}$ annual mean", "µg/m³", "raster", "detail/pm25.tif", "sequential_hot"),
    Var("no2", "air_quality_satellite", "aire", "Tropospheric NO$_2$ column", "mol/m²", "raster", "detail/no2.tif", "sequential_hot"),
    Var("alan", "alan", "luz", "Artificial light at night", "nW/cm²/sr", "raster", "detail/alan.tif", "sequential"),
    Var("noise", "noise", "luz", "Population exposed to noise", "% population", "choropleth", "noise_combined_pct", "sequential_hot"),
    Var("canopy", "greenspace_multisource", "verde", "Tree canopy fraction", "fraction", "raster", "detail/canopy.tif", "sequential_inv", (0.0, 1.0)),
    Var("green", "greenspace_multisource", "verde", "Green cover (Dynamic World)", "%", "grid", "subcomuna/green.geojson", "sequential_inv", (0.0, 100.0)),
    Var("greenspace_access", "greenspace_access", "verde", "Distance to nearest park", "m", "choropleth", "dist_to_nearest_park_m", "sequential_hot"),
    Var("heat_summer_tmax", "climate_heat", "clima", "Summer maximum temperature", "°C", "raster", "detail/heat_summer_tmax.tif", "sequential_hot"),
    Var("heat_hot_days", "climate_heat", "clima", "Hot days", "days/yr", "raster", "detail/heat_hot_days.tif", "sequential_hot"),
    Var("heat_tropical_nights", "climate_heat", "clima", "Tropical nights", "nights/yr", "raster", "detail/heat_tropical_nights.tif", "sequential_hot"),
    Var("heat_index", "climate_heat", "clima", "Heat exposure index", "index", "choropleth", "heat_exposure_index", "sequential_hot"),
    Var("rain_annual", "precipitation", "clima", "Annual precipitation", "mm", "raster", "detail/rain_annual.tif", "sequential_inv"),
    Var("rain_dry_spell", "precipitation", "clima", "Longest dry spell", "days", "raster", "detail/rain_dry_spell.tif", "sequential_hot"),
    Var("rain_heavy", "precipitation", "clima", "Heavy-rain days", "days/yr", "raster", "detail/rain_heavy.tif", "sequential_inv"),
    Var("rain_index", "precipitation", "clima", "Rainfall extremes index", "index", "choropleth", "precip_extremes_index", "sequential_hot"),
    Var("wind", "wind", "clima", "Mean wind speed", "m/s", "raster", "detail/wind.tif", "sequential"),
    Var("precipitation_spi", "precipitation_spi", "clima", "Mean SPI-12 (drought)", "index", "choropleth", "spi_12_mean", "diverging"),
    Var("wildfire", "wildfire", "clima", "Mean annual burned area", "% of {unit}", "choropleth", "fire_burned_pct_mean_annual", "sequential_hot"),
    Var("built_surface", "built_environment_ghsl", "construido", "Built-up surface (2020)", "m²/pixel", "raster", "detail/built_surface.tif", "sequential"),
    Var("built_volume", "built_environment_ghsl", "construido", "Built-up volume (2020)", "m³/pixel", "raster", "detail/built_volume.tif", "sequential"),
    Var("building_height", "built_environment_ghsl", "construido", "Building height (2018)", "m", "raster", "detail/building_height.tif", "sequential"),
    Var("street_intersection_density_km2", "street_network_topology", "construido", "Intersection density", "int./km²", "grid", "subcomuna/street_network_topology.geojson", "sequential", None, "street_intersection_density_km2"),
    Var("street_sne_nature_nats", "street_network_topology", "construido", "Street-network entropy (SNE)", "nats", "grid", "subcomuna/street_network_topology.geojson", "sequential", None, "street_sne_nature_nats"),
    Var("street_orientation_order", "street_network_topology", "construido", "Street orientation order", "0–1", "grid", "subcomuna/street_network_topology.geojson", "sequential", None, "street_orientation_order"),
    Var("street_orientation_entropy_nats", "street_network_topology", "construido", "Street orientation entropy", "nats", "choropleth", "street_orientation_entropy_nats", "sequential"),
    Var("street_largest_component_pct", "street_network_topology", "construido", "Largest connected component", "%", "choropleth", "street_largest_component_pct", "sequential_inv"),
    Var("walkability", "walkability", "construido", "Walkability index", "index", "choropleth", "walk_index", "sequential_inv"),
    Var("public_transport", "public_transport", "construido", "Public-transport index", "index", "choropleth", "transit_index", "sequential_inv"),
    Var("healthcare", "healthcare", "servicios", "Access to health facilities", "m", "grid", "subcomuna/healthcare.geojson", "sequential_hot"),
    Var("social_infrastructure", "social_infrastructure", "servicios", "Social-infrastructure index", "index", "choropleth", "social_index", "sequential_inv"),
    Var("food_environment", "food_environment", "servicios", "Food-environment index", "index", "choropleth", "food_index", "sequential_inv"),
    Var("nse", "socioeconomic", "social", "Socioeconomic index (NSE)", "index", "choropleth", "nse_index", "sequential_inv"),
    Var("poverty_income", "pobreza_sae", "social", "Income poverty (SAE 2022)", "%", "choropleth", "pobreza_ing_2022", "sequential_hot"),
    Var("poverty_multi", "pobreza_sae", "social", "Multidimensional poverty (SAE 2024)", "%", "choropleth", "pobreza_multi_2024", "sequential_hot"),
    Var("food_insecurity", "food_insecurity", "social", "Food insecurity (SAE 2022)", "%", "choropleth", "food_insec_2022", "sequential_hot"),
    Var("crime_property", "socioeconomic", "social", "Property-crime rate", "per 100,000", "choropleth", "tasa_delitos_propiedad", "sequential_hot"),
    Var("crime_violent", "socioeconomic", "social", "Violent-crime rate", "per 100,000", "choropleth", "tasa_delitos_violentos", "sequential_hot"),
    Var("demography", "demography", "social", "Population aged 65 and over", "%", "choropleth", "demo_pct_pop_65_plus", "sequential"),
    Var("heavy_metals", "heavy_metals", "aire", "Heavy-metals emission index (RETC)", "index", "choropleth", "hm_index", "sequential_hot"),
    Var("surface_water_total", "surface_water", "verde", "Surface water, total (2024)", "0 / 100", "raster", "annual/detail/surface_water_total_2024.tif", "water", (0.0, 100.0)),
    Var("surface_water_permanent", "surface_water", "verde", "Surface water, permanent (2024)", "0 / 100", "raster", "annual/detail/surface_water_permanent_2024.tif", "water", (0.0, 100.0)),
    Var("surface_water_seasonal", "surface_water", "verde", "Surface water, seasonal (2024)", "0 / 100", "raster", "annual/detail/surface_water_seasonal_2024.tif", "water", (0.0, 100.0)),
]

# Composite families that are published only as a summary; their components carry the maps.
COMPOSITE_FAMILIES = [
    ("heat", "Thermal family (composite summary)", "Each component keeps its own support; see the heat rasters and the heat index."),
    ("rain", "CHIRPS family (composite summary)", "Each component keeps its own support; see the rainfall rasters and the extremes index."),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _status_map() -> dict[str, dict[str, str]]:
    if not STATUS_CSV.is_file():
        return {}
    with STATUS_CSV.open(encoding="utf-8") as handle:
        return {row["layer_id"]: row for row in csv.DictReader(handle)}


def _raster_pixel_m(tif: Path) -> float:
    import rasterio

    with rasterio.open(tif) as src:
        return abs(src.transform.a)


def resolution_text(var: Var, entry: dict, tif: Path | None, ctx) -> tuple[str, str]:
    """(pixel/grid label, native-source label) for the tables and variable sheets."""
    if var.kind == "raster" and tif is not None:
        pixel = rf.pixel_size_label(None, _raster_pixel_m(tif))
        source = rf.pixel_size_label((entry.get("source") or {}).get("resolution") or None)
        source_native = (entry.get("detail") or {}).get("source_native_resolution_m")
        if source_native:
            source = rf.pixel_size_label(None, float(source_native))
        return pixel, source
    if var.kind == "grid":
        source = (entry.get("source") or {}).get("resolution")
        has_res = bool(source and (source.get("value") or source.get("x")))
        return "stable 1 km cell", rf.pixel_size_label(source) if has_res else "vector (OSM)"
    return f"1 {ctx['unit']} = 1 value ({ctx['n_units']} units)", "zonal summary"


def read_geo(path: Path):
    import geopandas as gpd

    return gpd.read_file(path).to_crs(3857)


def is_available(var: Var, ctx) -> bool:
    """A variable is mapped only if its data exist in this study's bundle."""
    if var.kind in ("raster", "grid"):
        if not (ctx["bundle"] / var.source).is_file():
            return False
        if var.kind == "grid":
            frame = ctx["grids"].setdefault(var.source, read_geo(ctx["bundle"] / var.source))
            return var.prop in frame and bool(frame[var.prop].notna().any())
        return True
    frame = ctx["communes"]
    return var.source in frame and bool(frame[var.source].notna().any())


def build_context(study_id: str) -> dict:
    import geopandas as gpd

    text = yaml.safe_load(TEXT_YAML.read_text(encoding="utf-8"))
    if study_id not in text["studies"]:
        raise ValueError(f"unknown study {study_id!r}; known: {sorted(text['studies'])}")
    study = text["studies"][study_id]
    bundle = REPO / study["bundle"]
    communes = gpd.read_file(bundle / "master.geojson").to_crs(3857)
    manifest = load_json(bundle / "manifest.json")
    return {
        "study_id": study_id,
        "study": study,
        "unit": study["unit"],
        "units": study["units"],
        "n_units": len(communes),
        "bundle": bundle,
        "out": REPO / f"output/{study_id}_variable_report",
        "communes": communes,
        "outline": communes,
        "colormaps": rf.load_colormaps(bundle / "palette.json"),
        "grids": {},
        "manifest_doc": manifest,
        "manifest": manifest.get("spatial_indicators", {}),
        "registry": load_layer_info_registry(REPO),
        "status": _status_map() if study.get("status") else {},
        "text": text,
        "info": {},
    }


def fmt(value: str, ctx) -> str:
    return value.format(unit=ctx["unit"], units=ctx["units"], n_units=ctx["n_units"])


def collect_info(ctx: dict) -> None:
    """Select the mapped variables and fill per-variable text/resolution facts."""
    text = ctx["text"]
    study_layers = ctx["study"].get("layers", {})
    ctx["vars"] = [v for v in VARS if is_available(v, ctx)]
    for var in ctx["vars"]:
        entry = ctx["manifest"].get(var.key, {})
        layer = ctx["registry"].get(var.layer_id, {})
        tif = ctx["bundle"] / var.source if var.kind == "raster" else None
        pixel, source_res = resolution_text(var, entry, tif, ctx)
        over = {
            **text["layers"].get(var.layer_id, {}),
            **study_layers.get(var.layer_id, {}),
            **(text["variables"].get(var.key) or {}),
        }
        over = {k: fmt(v, ctx) for k, v in over.items()}
        source = layer.get("source", {})
        citation = layer.get("citation", {})
        status = ctx["status"].get(var.layer_id, {})
        ctx["info"][var.key] = {
            "pixel": pixel,
            "source_res": source_res,
            "source_name": over.get("source_name") or source.get("name", "—"),
            "provider": over.get("provider") or source.get("provider", ""),
            "years": over.get("years") or (layer.get("specs") or {}).get("temporal_coverage", "—"),
            "limitation": over.get("limitation") or next(iter(layer.get("limitations") or []), ""),
            "paper": over.get("paper") or citation.get("paper") or citation.get("title") or "",
            "doi": over["doi"] if "doi" in over else (citation.get("doi") or ""),
            "method": over.get("method", ""),
            "note": over.get("note", ""),
            "final_check": status.get("final_check", "—"),
        }


def unmapped_rows(ctx) -> list[tuple[str, str, str]]:
    """Appendix-C rows: manifest indicators marked unavailable (grouped by layer) plus composites."""
    text = ctx["text"]
    place = ctx["study"]["place"]
    groups: dict[tuple, list[str]] = {}
    for key, entry in ctx["manifest"].items():
        avail = entry.get("availability") or {}
        if avail.get("status") == "available":
            continue
        group = (entry["layer_id"], avail.get("reason"), tuple(avail.get("supported_countries") or ()))
        groups.setdefault(group, []).append(key)
    rows = []
    for (layer_id, reason, supported), keys in groups.items():
        title = text["layer_titles"].get(layer_id, layer_id)
        template = text["reason_text"].get(reason, "Unavailable in this study ({reason}).")
        rows.append((layer_id, f"{title}, {len(keys)} indicator{'s' if len(keys) > 1 else ''}",
                     template.format(supported=", ".join(supported), place=place, reason=reason)))
    mapped_layers = {v.layer_id for v in ctx["vars"]}
    rows += [r for r in COMPOSITE_FAMILIES if r[0] in ctx["manifest"] and
             ctx["manifest"][r[0]]["layer_id"] in mapped_layers]
    return rows


