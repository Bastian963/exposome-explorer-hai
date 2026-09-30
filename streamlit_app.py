"""Interactive exposome explorer: toggle each layer of a study and click the map for its values.

Run from the repository root (study paths are resolved relative to it):

    uv sync --extra app --inexact
    .venv/bin/streamlit run apps/exposome_explorer/streamlit_app.py

Layers come from the same catalog as the variable reports (``exposome.variable_catalog``);
a layer is offered only if its data exist in the study's published web bundle.  Rasters are
drawn pixel by pixel, and values on click are read from the native raster or polygon.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from streamlit_folium import st_folium

# In this repo the app lives in apps/exposome_explorer/; a deployed copy has src/ next to the app.
for _root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]):
    if (_root / "src" / "exposome").is_dir():
        sys.path.insert(0, str(_root / "src"))
        break

from exposome import explorer_data as ed  # noqa: E402
from exposome import report_figures as rf  # noqa: E402
from exposome.variable_catalog import (  # noqa: E402
    DOMAINS,
    REPO,
    TEXT_YAML,
    build_context,
    collect_info,
)


def available_studies() -> dict[str, str]:
    """Studies configured in the report YAML whose bundle is present in this checkout."""
    studies = yaml.safe_load(TEXT_YAML.read_text(encoding="utf-8"))["studies"]
    return {
        sid: cfg["place"] for sid, cfg in studies.items() if (REPO / cfg["bundle"] / "master.geojson").is_file()
    }


STUDIES = available_studies()
DISPLAY_MAX_DIM = 1800  # raster pictures are decimated beyond this; click values stay native
DEFAULT_LAYER = "pm25"

st.set_page_config(page_title="Exposome explorer", page_icon=":material/public:", layout="wide")


@st.cache_resource(max_entries=2, show_spinner="Loading study…")
def load_study(study: str) -> dict:
    ctx = build_context(study)
    collect_info(ctx)
    ctx["var_by_key"] = {v.key: v for v in ctx["vars"]}
    return ctx


@st.cache_data(max_entries=64, show_spinner="Rendering layer…")
def layer_payload(study: str, key: str) -> dict:
    """Coloured overlay (raster PNG or GeoJSON) for one layer, computed once per study."""
    ctx = load_study(study)
    var = ctx["var_by_key"][key]
    cmap = rf.make_cmap(var.cmap, ctx["colormaps"])
    bounds = tuple(ctx["outline"].total_bounds)
    if var.kind == "raster":
        out = ed.raster_rgba(
            ctx["bundle"] / var.source, bounds, cmap, limits=var.limits, max_dim=DISPLAY_MAX_DIM
        )
        out["kind"] = "raster"
        return out
    frame = ctx["grids"][var.source] if var.kind == "grid" else ctx["communes"]
    column = var.prop if var.kind == "grid" else var.source
    geojson, limits = ed.polygon_geojson(frame, column, cmap, limits=var.limits, name_column="spatial_name")
    return {"kind": var.kind, "geojson": geojson, "limits": limits}


def colour_strip(var, ctx, width: int = 160) -> np.ndarray:
    cmap = rf.make_cmap(var.cmap, ctx["colormaps"])
    return (cmap(np.tile(np.linspace(0, 1, width), (10, 1))) * 255).astype("uint8")


def build_map(ctx: dict, study: str, selected: list[str], opacity: float, basemap: str):
    minx, miny, maxx, maxy = ctx["communes"].to_crs(4326).total_bounds
    fmap = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2],
        tiles="OpenStreetMap" if basemap == "OpenStreetMap" else None,
        zoom_control=True,
        control_scale=True,
        zoom_snap=0.25,
    )
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])
    for key in selected:
        var = ctx["var_by_key"][key]
        payload = layer_payload(study, key)
        if payload["kind"] == "raster":
            data = base64.b64encode(payload["png"]).decode()
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{data}", bounds=payload["bounds"],
                opacity=opacity, name=var.label, zindex=2,
            ).add_to(fmap)
        else:
            folium.GeoJson(
                payload["geojson"], name=var.label,
                style_function=lambda f, o=opacity: {
                    "fillColor": f["properties"]["color"], "fillOpacity": o, "weight": 0, "color": "#ffffff",
                },
            ).add_to(fmap)
    outline = ctx["communes"][["geometry", "spatial_name"]].to_crs(4326)
    folium.GeoJson(
        outline.to_json(), name=ctx["units"].capitalize(),
        style_function=lambda _f: {"fill": False, "color": "#222222", "weight": 1},
        tooltip=folium.GeoJsonTooltip(fields=["spatial_name"], aliases=[ctx["unit"].capitalize()]),
    ).add_to(fmap)
    return fmap


def values_at(ctx: dict, selected: list[str], lat: float, lon: float) -> pd.DataFrame:
    rows = []
    for key in selected:
        var = ctx["var_by_key"][key]
        if var.kind == "raster":
            value, where = ed.sample_raster(ctx["bundle"] / var.source, lat, lon), None
        elif var.kind == "grid":
            value, where = ed.lookup_polygon(ctx["grids"][var.source], var.prop, lat, lon)
        else:
            value, where = ed.lookup_polygon(ctx["communes"], var.source, lat, lon, name_column="spatial_name")
        rows.append({
            "Layer": var.label.replace("$", "").replace("_", "").replace("{", "").replace("}", ""),
            "Value": None if value is None else float(f"{value:.4g}"),
            "Unit": var.unit.format(unit=ctx["unit"]),
            "Support": ctx["info"][key]["pixel"].replace(
                f"1 {ctx['unit']} = 1 value ({ctx['n_units']} units)", f"{ctx['unit']} ({ctx['n_units']})"
            ),
        })
    return pd.DataFrame(rows)


# ---- sidebar: study + layer selection (fast UI first) ---------------------------------
with st.sidebar:
    st.title(":material/public: Exposome explorer")
    if len(STUDIES) > 1:
        study = st.segmented_control(
            "Study", list(STUDIES), format_func=STUDIES.get, default=next(iter(STUDIES)), required=True
        )
    else:
        study = next(iter(STUDIES))
ctx = load_study(study)

with st.sidebar:
    st.caption("Global Exposome Modeling, Mapping & Analytics (GEMMA)")
    selected: list[str] = []
    for code, domain in DOMAINS.items():
        members = [v for v in ctx["vars"] if v.domain == code]
        if not members:
            continue
        with st.expander(domain, expanded=code == "aire"):
            for var in members:
                label = var.label.replace("$_{2.5}$", "₂.₅").replace("$_2$", "₂")
                if st.checkbox(label, value=var.key == DEFAULT_LAYER, key=f"{study}:{var.key}"):
                    selected.append(var.key)
    opacity = st.slider("Layer opacity", 0.2, 1.0, 0.8, 0.05)
    basemap = st.segmented_control(
        "Basemap", ["None (offline)", "OpenStreetMap"], default="None (offline)", required=True
    )

# ---- main: map + values at the clicked point -------------------------------------------
st.header(f"{STUDIES[study]}: {len(selected)} layer{'s' if len(selected) != 1 else ''} shown")
st.caption(
    f"{ctx['study']['scope']}. Click the map to read the value of every active layer at that point. "
    "Rasters show native pixels; values come from the native data."
)
map_col, info_col = st.columns([3, 2])
with map_col:
    fmap = build_map(ctx, study, selected, opacity, basemap)
    out = st_folium(
        fmap, key=f"map:{study}:{','.join(selected)}:{opacity}:{basemap}",
        height=640, width=None, returned_objects=["last_clicked"], pixelated=True,
    )
with info_col:
    click = (out or {}).get("last_clicked")
    if not selected:
        st.info("Select at least one layer in the sidebar.", icon=":material/layers:")
    elif not click:
        st.info("Click the map to see values.", icon=":material/touch_app:")
    else:
        lat, lon = click["lat"], click["lng"]
        _, name = ed.lookup_polygon(ctx["communes"], "area_km2", lat, lon, name_column="spatial_name")
        st.subheader(f"{ctx['unit'].capitalize()}: {name or 'outside the study area'}")
        st.caption(f"{lat:.5f}, {lon:.5f}")
        st.dataframe(values_at(ctx, selected, lat, lon), hide_index=True, width="stretch")
    if selected:
        with st.expander("Legend and sources", expanded=False):
            for key in selected:
                var, info = ctx["var_by_key"][key], ctx["info"][key]
                low, high = layer_payload(study, key)["limits"]
                st.markdown(f"**{var.label.replace('$', '')}**: {low:.4g} – {high:.4g} {var.unit.format(unit=ctx['unit'])}")
                st.image(colour_strip(var, ctx), width=160)
                st.caption(f"{info['source_name']}. {info['pixel']}. {info['years']}")
                if info["method"]:
                    st.caption(info["method"])
                if info["limitation"]:
                    st.caption(f"Limitation: {info['limitation']}")
