"""Figure and text helpers for the Santiago variable-availability report.

Rasters are drawn with nearest-neighbour interpolation so every native pixel is
visible; nothing is resampled to a finer grid than the source product.  Each
``draw_*`` function paints onto a caller-supplied axes so the same code serves
the per-variable figures and the contact sheets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Web-mercator pixel edges are only drawn when pixels are big enough to read.
GRID_EDGE_MIN_PIXEL_M = 3000.0


def load_colormaps(palette_path: str | Path) -> dict[str, list[str]]:
    """Return ``palette.json`` ``data_colormaps`` as ``{name: [hex, ...]}``."""
    palette = json.loads(Path(palette_path).read_text(encoding="utf-8"))
    return dict(palette["data_colormaps"])


def make_cmap(name: str, colormaps: Mapping[str, Sequence[str]]):
    from matplotlib.colors import LinearSegmentedColormap

    if name == "water":
        return LinearSegmentedColormap.from_list("water", ["#dcdcdc", "#3b5dc9"])
    return LinearSegmentedColormap.from_list(name, list(colormaps[name]))


def apply_report_style() -> None:
    """Astro-paper style (``astro-paper-plot-style`` skill): serif, inward ticks, thick axes."""
    import matplotlib.pyplot as plt

    from .paper_plot_style import apply_astro_paper_style

    apply_astro_paper_style("compact", use_seaborn=False)
    plt.rcParams.update({"axes.titlesize": 9, "savefig.bbox": "tight"})


def add_scale_bar(ax, bounds: tuple[float, float, float, float], km: float = 10.0) -> None:
    """Ground-distance scale bar on a Web-Mercator axes (corrects the 1/cos(lat) stretch)."""
    minx, miny, maxx, maxy = bounds
    lat = math.degrees(2 * math.atan(math.exp((miny + maxy) / 2 / 6378137.0)) - math.pi / 2)
    length = km * 1000.0 / math.cos(math.radians(lat))
    x1 = maxx - (maxx - minx) * 0.03
    x0 = x1 - length
    y = miny + (maxy - miny) * 0.04
    ax.plot([x0, x1], [y, y], color="black", lw=1.6, solid_capstyle="butt", zorder=5)
    ax.text((x0 + x1) / 2, y + (maxy - miny) * 0.012, f"{km:g} km", ha="center", va="bottom",
            fontsize=6.5, zorder=5)


def pixel_size_label(resolution: Mapping[str, Any] | None, transform_m: float | None = None) -> str:
    """Human label for the native pixel, e.g. ``1.19 km`` or ``30 m``."""
    metres = transform_m
    if metres is None and resolution:
        value = resolution.get("value", resolution.get("x"))
        unit = resolution.get("unit")
        if value is not None and unit == "m":
            metres = float(value)
        elif value is not None and unit == "degree":
            metres = float(value) * 111_320.0
    if metres is None:
        return "no pixel (administrative unit)"
    if metres >= 1000:
        return f"{metres / 1000:.2f} km"
    return f"{metres:.0f} m"


def robust_limits(
    values: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> tuple[float, float]:
    """Percentile stretch on finite values; widen a degenerate range."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _style_map_axes(ax, bounds: tuple[float, float, float, float], pad: float = 0.03) -> None:
    minx, miny, maxx, maxy = bounds
    dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")


def _add_colorbar(ax, image, label: str) -> None:
    import matplotlib.pyplot as plt

    bar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02, shrink=0.8)
    bar.ax.tick_params(labelsize=6, length=2, direction="in")
    bar.outline.set_linewidth(0.4)
    if label:
        bar.set_label(label, fontsize=6.5)


def draw_raster(
    ax,
    tif_path: str | Path,
    outline,
    cmap,
    *,
    unit: str = "",
    limits: tuple[float, float] | None = None,
    pixel_label: str = "",
    colorbar: bool = True,
    scalebar: bool = False,
):
    """Draw a single-band COG at native pixels, with the commune outline on top.

    ``outline`` is a GeoDataFrame already in the raster CRS.  Returns the pixel
    edge length in metres taken from the raster transform.
    """
    import rasterio

    from rasterio.windows import Window, from_bounds

    with rasterio.open(tif_path) as src:
        pixel_m = abs(src.transform.a)
        # Read only the study area plus one pixel of margin (no resampling).
        minx, miny, maxx, maxy = outline.total_bounds
        window = from_bounds(
            minx - pixel_m, miny - pixel_m, maxx + pixel_m, maxy + pixel_m, src.transform
        ).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=window).astype("float32")
        left, bottom, right, top = src.window_bounds(window)
        if src.nodata is not None and np.isfinite(src.nodata):
            data[data == src.nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
    vmin, vmax = limits or robust_limits(data)
    image = ax.imshow(
        np.ma.masked_invalid(data),
        extent=(left, right, bottom, top),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    if pixel_m >= GRID_EDGE_MIN_PIXEL_M:
        for x in np.arange(left, right + 1, pixel_m):
            ax.axvline(x, color="white", lw=0.25, alpha=0.6)
        for y in np.arange(top, bottom - 1, -pixel_m):
            ax.axhline(y, color="white", lw=0.25, alpha=0.6)
    outline.boundary.plot(ax=ax, color="#222222", linewidth=0.35)
    _style_map_axes(ax, tuple(outline.total_bounds))
    if scalebar:
        add_scale_bar(ax, tuple(outline.total_bounds))
    if colorbar:
        _add_colorbar(ax, image, unit)
    if pixel_label:
        ax.text(
            0.02, 0.02, f"1 pixel = {pixel_label}", transform=ax.transAxes, fontsize=6.5,
            color="#222222", bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5},
        )
    return pixel_m


def draw_polygons(
    ax,
    frame,
    column: str,
    outline,
    cmap,
    *,
    unit: str = "",
    pixel_label: str = "",
    colorbar: bool = True,
    limits: tuple[float, float] | None = None,
    scalebar: bool = False,
):
    """Choropleth of ``column`` over polygons (communes or a stable analysis grid)."""
    values = frame[column].astype("float64").to_numpy()
    vmin, vmax = limits or robust_limits(values)
    frame.plot(
        ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0,
        missing_kwds={"color": "#e6e6e6"},
    )
    outline.boundary.plot(ax=ax, color="#222222", linewidth=0.3)
    _style_map_axes(ax, tuple(outline.total_bounds))
    if scalebar:
        add_scale_bar(ax, tuple(outline.total_bounds))
    if colorbar:
        sm = _scalar_mappable(cmap, vmin, vmax)
        _add_colorbar(ax, sm, unit)
    if pixel_label:
        ax.text(
            0.02, 0.02, pixel_label, transform=ax.transAxes, fontsize=6.5,
            color="#222222", bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5},
        )


def _scalar_mappable(cmap, vmin: float, vmax: float):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    return sm


def save_figure(fig, out_base: Path, formats: Sequence[str] = ("png",)) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path)
        written.append(path)
    return written


# --- text helpers -----------------------------------------------------------

_TEX_SPECIALS = {
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
_UNICODE_TEX = {"₂": r"\textsubscript{2}", "₃": r"\textsubscript{3}", "³": r"\textsuperscript{3}"}


def tex_escape(text: Any) -> str:
    out = []
    for char in str(text):
        out.append(_TEX_SPECIALS.get(char) or _UNICODE_TEX.get(char) or char)
    return "".join(out)
