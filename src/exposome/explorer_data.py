"""Data helpers for the interactive exposome explorer (Streamlit app).

Rasters are coloured pixel by pixel (nearest neighbour, no resampling to a finer grid)
and overlaid on a web map.  Values shown on click are always read from the *native*
raster or polygon, never from the picture, so a display that had to be reduced for speed
(:func:`raster_rgba` ``max_dim``) still reports the true pixel value.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from .report_figures import robust_limits

WEB_MERCATOR = "EPSG:3857"
WGS84 = "EPSG:4326"


def _window(src, bounds_3857: tuple[float, float, float, float]):
    """Window covering ``bounds_3857`` plus one pixel of margin, clipped to the raster."""
    from rasterio.windows import Window, from_bounds

    pixel = abs(src.transform.a)
    minx, miny, maxx, maxy = bounds_3857
    window = from_bounds(
        minx - pixel, miny - pixel, maxx + pixel, maxy + pixel, src.transform
    ).round_offsets().round_lengths()
    return window.intersection(Window(0, 0, src.width, src.height))


def read_masked(tif: str | Path, bounds_3857: tuple[float, float, float, float]):
    """Read band 1 over the study area. Returns ``(data, window_bounds, pixel_m)``.

    Non-finite values and the raster's nodata become ``nan``.
    """
    import rasterio

    with rasterio.open(tif) as src:
        window = _window(src, bounds_3857)
        data = src.read(1, window=window).astype("float32")
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == src.nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        return data, src.window_bounds(window), abs(src.transform.a)


def raster_rgba(
    tif: str | Path,
    bounds_3857: tuple[float, float, float, float],
    cmap,
    *,
    limits: tuple[float, float] | None = None,
    max_dim: int | None = None,
) -> dict[str, Any]:
    """Colour a raster window as an RGBA PNG for a web-map image overlay.

    Returns ``png`` (bytes), ``bounds`` (``[[south, west], [north, east]]`` in WGS84),
    ``pixel_m`` (native pixel edge), ``limits`` and ``reduced`` (True when the picture was
    decimated to fit ``max_dim``; the native raster is unchanged).
    """
    from PIL import Image
    from rasterio.warp import transform_bounds

    data, (left, bottom, right, top), pixel_m = read_masked(tif, bounds_3857)
    vmin, vmax = limits or robust_limits(data)
    reduced = False
    if max_dim and max(data.shape) > max_dim:
        step = math.ceil(max(data.shape) / max_dim)
        data = data[::step, ::step]
        reduced = True
    norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isnan(data), 0, 255)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", optimize=True)
    west, south, east, north = transform_bounds(WEB_MERCATOR, WGS84, left, bottom, right, top)
    return {
        "png": buffer.getvalue(),
        "bounds": [[south, west], [north, east]],
        "pixel_m": pixel_m,
        "limits": (float(vmin), float(vmax)),
        "reduced": reduced,
    }


def sample_raster(tif: str | Path, lat: float, lon: float) -> float | None:
    """Native pixel value at a WGS84 point, or ``None`` outside the raster / on nodata."""
    import rasterio
    from pyproj import Transformer

    x, y = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform(lon, lat)
    with rasterio.open(tif) as src:
        left, bottom, right, top = src.bounds
        if not (left <= x <= right and bottom <= y <= top):
            return None
        value = float(next(src.sample([(x, y)]))[0])
    if not np.isfinite(value) or (src.nodata is not None and value == src.nodata):
        return None
    return value


def value_to_hex(values, cmap, limits: tuple[float, float]) -> list[str]:
    """Hex colour per value; ``nan`` maps to light grey."""
    from matplotlib.colors import to_hex

    vmin, vmax = limits
    out = []
    for value in values:
        if value is None or not np.isfinite(value):
            out.append("#e6e6e6")
        else:
            out.append(to_hex(cmap(float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)))))
    return out


def polygon_geojson(
    frame, column: str, cmap, *, limits: tuple[float, float] | None = None, name_column: str | None = None
) -> tuple[dict[str, Any], tuple[float, float]]:
    """WGS84 GeoJSON of ``frame`` (EPSG:3857) with a ``value`` and a precomputed ``color``.

    Coordinates are rounded to about a metre to keep the payload small.
    Returns ``(geojson, limits)``.
    """
    import json

    import shapely

    values = frame[column].astype("float64").to_numpy()
    lims = limits or robust_limits(values)
    out = frame[["geometry"]].copy()
    out["geometry"] = shapely.set_precision(frame.to_crs(WGS84).geometry.values, 1e-5)
    out["value"] = values
    out["color"] = value_to_hex(values, cmap, lims)
    if name_column and name_column in frame:
        out["name"] = frame[name_column].astype(str).to_numpy()
    out = out.set_geometry("geometry", crs=WGS84)
    return json.loads(out.to_json(na="null")), lims


def lookup_polygon(frame, column: str, lat: float, lon: float, *, name_column: str | None = None):
    """Value (and unit name) of the polygon of ``frame`` (EPSG:3857) containing a WGS84 point."""
    from pyproj import Transformer
    from shapely.geometry import Point

    x, y = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform(lon, lat)
    hits = frame.iloc[list(frame.sindex.query(Point(x, y), predicate="intersects"))]
    if hits.empty:
        return None, None
    row = hits.iloc[0]
    value = row[column]
    name = str(row[name_column]) if name_column and name_column in hits else None
    return (None if value is None or not np.isfinite(float(value)) else float(value)), name
