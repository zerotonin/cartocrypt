# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — viz_aegina                                         ║
# ║  « four panels, one story »                                      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Publication-quality multi-panel figure for the Aegina           ║
# ║  reference case.  Wong (2011) colourblind-safe palette,          ║
# ║  SVG-first with editable text, 300 dpi PNG companion,            ║
# ║  and a CSV data dump for reviewers.                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Publication-quality multi-panel figure for the Aegina demo.

« Four panels, one story: terrain, infrastructure, fictional
overlays, anonymised result »

Figure target: Nature / PNAS double column (~180 mm) at 300 dpi,
SVG-first with editable text (``svg.fonttype=none``), Wong (2011)
colourblind-safe palette throughout.

Panel layout::

    ┌───────────────┬───────────────┐
    │ (a) Terrain   │ (b) Infra     │
    ├───────────────┼───────────────┤
    │ (c) Fake      │ (d) Anonymised│
    └───────────────┴───────────────┘

Each panel is self-contained (scale bar, title, axes off) so it
can be extracted for supplementary material.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cartocrypt.aegina import LayerBundle
from cartocrypt.constants import (
    COLOURS,
    FIGURE_DPI,
    POKEMON_COLOURS,
    SVG_FONTTYPE,
    Coords,
)

matplotlib.rcParams["svg.fonttype"] = SVG_FONTTYPE
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Panel size: 180 mm ≈ 7.08 in width, 2×2 at 1:1 aspect
_FIG_W_IN = 7.08
_FIG_H_IN = 7.08

_ROAD_STYLE: dict[str, tuple[str, float]] = {
    "motorway":    (COLOURS["black"],        1.4),
    "primary":     (COLOURS["vermilion"],    1.2),
    "secondary":   (COLOURS["orange"],       1.0),
    "tertiary":    (COLOURS["blue"],         0.8),
    "residential": ("#707070",               0.5),
    "service":     ("#a0a0a0",               0.4),
    "track":       ("#a0a0a0",               0.4),
    "path":        ("#a0a0a0",               0.3),
    "unclassified":("#808080",               0.4),
}
_ROAD_DEFAULT = ("#909090", 0.4)

_HATCH = {
    "Charizard": "///",
    "Pikachu":   "...",
    "Eevee":     "xx",
}


# ┌────────────────────────────────────────────────────────────┐
# │ Public API  « plot_aegina_figure + CSV companion »          │
# └────────────────────────────────────────────────────────────┘

def plot_aegina_figure(
    bundle: LayerBundle,
    habitats: gpd.GeoDataFrame,
    tubes: gpd.GeoDataFrame,
    *,
    anon_coords: Coords | None = None,
    anon_graph: nx.Graph | None = None,
    out_path: Path | None = None,
    title: str = "CartoCrypt — Aegina reference case",
) -> matplotlib.figure.Figure:
    """Build the 2×2 publication figure.

    Args:
        bundle: Populated LayerBundle from :class:`AeginaIngestor`.
        habitats: Pokémon habitat GeoDataFrame.
        tubes: Futurama tube GeoDataFrame.
        anon_coords: Optional (N, 2) anonymised node coordinates
            matching ``anon_graph``.  When provided, panel (d) is
            populated; otherwise panel (d) shows a placeholder.
        anon_graph: Graph matching ``anon_coords`` (simple nx.Graph
            with sequential int node ids).
        out_path: If given, saves ``<stem>.svg`` and ``<stem>.png``.
        title: Super-title for the figure.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(_FIG_W_IN, _FIG_H_IN),
        constrained_layout=True,
    )
    (ax_a, ax_b), (ax_c, ax_d) = axes

    _panel_terrain(ax_a, bundle)
    _panel_infra(ax_b, bundle)
    _panel_fake(ax_c, bundle, habitats, tubes)
    _panel_anon(ax_d, anon_graph, anon_coords, habitats, tubes, bundle)

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd", strict=True):
        ax.text(
            0.02, 0.97, f"({letter})",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

    fig.suptitle(title, fontsize=12, fontweight="bold")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), format="png",
                    dpi=FIGURE_DPI, bbox_inches="tight")
        _write_csv_companions(out_path, habitats, tubes)

    return fig


def _write_csv_companions(
    out_path: Path,
    habitats: gpd.GeoDataFrame,
    tubes: gpd.GeoDataFrame,
) -> None:
    """Emit the data behind the figure as reviewer-friendly CSVs.

    Produces two files alongside the SVG / PNG:

    * ``<stem>_habitats.csv`` — species, habitat_id, area_m2,
      centroid_lon, centroid_lat, n_vertices.
    * ``<stem>_tubes.csv`` — line_name, station_a, station_b,
      length_m, colour.
    """
    stem = out_path.with_suffix("")

    if habitats is not None and not habitats.empty:
        rows = []
        for _, row in habitats.iterrows():
            g = row.geometry
            c = g.centroid if g is not None and not g.is_empty else None
            rows.append({
                "species":      row.get("species"),
                "habitat_id":   row.get("habitat_id"),
                "area_m2":      row.get("area_m2"),
                "centroid_lon": round(c.x, 6) if c else None,
                "centroid_lat": round(c.y, 6) if c else None,
                "n_vertices":   len(g.exterior.coords) if g is not None
                                and g.geom_type == "Polygon" else None,
            })
        import pandas as pd
        pd.DataFrame(rows).to_csv(
            stem.with_name(stem.name + "_habitats.csv"), index=False,
        )

    if tubes is not None and not tubes.empty:
        out_cols = ["line_name", "station_a", "station_b",
                    "length_m", "colour"]
        tubes[[c for c in out_cols if c in tubes.columns]].to_csv(
            stem.with_name(stem.name + "_tubes.csv"), index=False,
        )


# ┌────────────────────────────────────────────────────────────┐
# │ Panels  « one builder per sub-plot »                        │
# └────────────────────────────────────────────────────────────┘

def _panel_terrain(ax: matplotlib.axes.Axes, bundle: LayerBundle) -> None:
    """Panel (a): hillshade + coastline + peaks."""
    dem = bundle.dem
    a, _b, c, _d, e, f = bundle.dem_transform
    extent = _dem_extent(dem, bundle.dem_transform)

    # Matplotlib's LightSource expects positive dy — if our grid is
    # top-down (e < 0), we flip for shading only.
    dem_shade = np.flipud(dem) if e < 0 else dem
    ls = LightSource(azdeg=315, altdeg=35)
    try:
        rgb = ls.shade(
            dem_shade, cmap=matplotlib.colormaps.get_cmap("gist_earth"),
            vert_exag=5, blend_mode="overlay",
        )
        if e < 0:
            rgb = np.flipud(rgb)
        ax.imshow(rgb, extent=extent, origin="upper")
    except Exception:
        ax.imshow(
            dem, extent=extent, origin="upper",
            cmap="gist_earth",
        )

    _plot_gdf_lines(ax, bundle.coastline, colour=COLOURS["blue"], linewidth=0.8)

    if not bundle.peaks.empty:
        for _, row in bundle.peaks.iterrows():
            g = row.geometry
            if g is None or g.geom_type != "Point":
                continue
            ax.plot(g.x, g.y, "^", color=COLOURS["black"],
                    markersize=4, markeredgewidth=0.4,
                    markerfacecolor=COLOURS["yellow"])
            name = row.get("name") if "name" in row.index else None
            if isinstance(name, str) and name:
                ax.annotate(
                    name, xy=(g.x, g.y), xytext=(3, 3),
                    textcoords="offset points", fontsize=6,
                )

    ax.set_title("Terrain", fontsize=9)
    _style_geo_axes(ax, bundle.bbox)
    _add_scale_bar(ax, bundle.bbox)
    _add_north_arrow(ax)


def _panel_infra(ax: matplotlib.axes.Axes, bundle: LayerBundle) -> None:
    """Panel (b): roads, buildings, settlements."""
    if bundle.landuse is not None and not bundle.landuse.empty:
        bundle.landuse.plot(
            ax=ax, facecolor="#f0e9d8", edgecolor="none", alpha=0.4,
        )
    if bundle.water is not None and not bundle.water.empty:
        bundle.water.plot(
            ax=ax, facecolor=COLOURS["sky_blue"], edgecolor="none",
            alpha=0.5,
        )
    _plot_gdf_lines(ax, bundle.coastline, colour=COLOURS["blue"], linewidth=0.6)

    _plot_roads(ax, bundle.roads)

    if bundle.buildings is not None and not bundle.buildings.empty:
        bundle.buildings.plot(
            ax=ax, facecolor="#606060", edgecolor="none", alpha=0.8,
            linewidth=0.0,
        )

    if bundle.settlements is not None and not bundle.settlements.empty:
        for _, row in bundle.settlements.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            c = g.centroid
            ax.plot(c.x, c.y, "o", color=COLOURS["reddish_purple"],
                    markersize=4, markeredgewidth=0.4,
                    markeredgecolor="white")
            name = row.get("name") if "name" in row.index else None
            if isinstance(name, str) and name:
                ax.annotate(
                    name, xy=(c.x, c.y), xytext=(4, 2),
                    textcoords="offset points", fontsize=6,
                    fontweight="bold",
                )

    ax.set_title("Infrastructure", fontsize=9)
    _style_geo_axes(ax, bundle.bbox)


def _panel_fake(
    ax: matplotlib.axes.Axes,
    bundle: LayerBundle,
    habitats: gpd.GeoDataFrame,
    tubes: gpd.GeoDataFrame,
) -> None:
    """Panel (c): Pokémon habitats + Futurama tubes."""
    # Faint base so overlays are legible
    _plot_gdf_lines(ax, bundle.coastline, colour="#909090", linewidth=0.4)
    if bundle.water is not None and not bundle.water.empty:
        bundle.water.plot(ax=ax, facecolor="#dde8f0", edgecolor="none", alpha=0.6)

    # Habitats
    if habitats is not None and not habitats.empty:
        for species, sub in habitats.groupby("species"):
            colour = POKEMON_COLOURS.get(species, COLOURS["black"])
            sub.plot(
                ax=ax,
                facecolor=colour,
                edgecolor=colour,
                alpha=0.35,
                hatch=_HATCH.get(species, ""),
                linewidth=0.8,
            )

    # Tubes
    if tubes is not None and not tubes.empty:
        for _, row in tubes.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            xs, ys = g.xy
            ax.plot(
                list(xs), list(ys),
                color=row["colour"], linewidth=1.4,
                linestyle=(0, (4, 2)), alpha=0.95, zorder=4,
            )
        # Stations at unique endpoints
        stations = {}
        for _, row in tubes.iterrows():
            for end_name, idx in [("station_a", 0), ("station_b", -1)]:
                g = row.geometry
                if g is None or g.is_empty:
                    continue
                pt = g.coords[idx]
                stations[row[end_name]] = pt
        for name, (x, y) in stations.items():
            ax.plot(x, y, "o", markersize=5,
                    markerfacecolor="white",
                    markeredgecolor=COLOURS["black"],
                    markeredgewidth=0.8, zorder=5)

    ax.set_title("Fictional overlays", fontsize=9)
    _style_geo_axes(ax, bundle.bbox)

    # Legend
    handles = [
        Patch(facecolor=POKEMON_COLOURS["Charizard"], alpha=0.35,
              hatch=_HATCH["Charizard"], edgecolor=POKEMON_COLOURS["Charizard"],
              label="Charizard (mountains)"),
        Patch(facecolor=POKEMON_COLOURS["Pikachu"], alpha=0.35,
              hatch=_HATCH["Pikachu"], edgecolor=POKEMON_COLOURS["Pikachu"],
              label="Pikachu (coast)"),
        Patch(facecolor=POKEMON_COLOURS["Eevee"], alpha=0.35,
              hatch=_HATCH["Eevee"], edgecolor=POKEMON_COLOURS["Eevee"],
              label="Eevee (valleys)"),
        Line2D([0], [0], color=COLOURS["reddish_purple"], linewidth=1.4,
               linestyle=(0, (4, 2)), label="Futurama tube"),
    ]
    ax.legend(
        handles=handles, loc="lower left", fontsize=6,
        framealpha=0.9, handlelength=2.0, borderpad=0.3,
    )


def _panel_anon(
    ax: matplotlib.axes.Axes,
    anon_graph: nx.Graph | None,
    anon_coords: Coords | None,
    habitats: gpd.GeoDataFrame,
    tubes: gpd.GeoDataFrame,
    bundle: LayerBundle,
) -> None:
    """Panel (d): anonymised re-embedding."""
    if anon_graph is None or anon_coords is None:
        ax.text(
            0.5, 0.5,
            "Anonymised layout\n(pass anon_graph + anon_coords to plot)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#606060",
        )
        ax.set_title("Anonymised", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d0d0")
        return

    nodes = list(anon_graph.nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    for u, v in anon_graph.edges():
        i, j = idx[u], idx[v]
        ax.plot(
            [anon_coords[i, 0], anon_coords[j, 0]],
            [anon_coords[i, 1], anon_coords[j, 1]],
            color="#606060", linewidth=0.5, alpha=0.7,
        )
    ax.scatter(
        anon_coords[:, 0], anon_coords[:, 1],
        s=4, color=COLOURS["sky_blue"], zorder=4,
    )

    # Overlay points already attached via cartocrypt.overlay ride
    # in the node dicts.  Draw habitat centroids and tube endpoints
    # from the graph's overlay payload if present.
    polys = anon_graph.graph.get("overlay_polygons", [])
    for poly in polys:
        species = poly.get("species") or poly.get("_layer", "")
        colour = POKEMON_COLOURS.get(species, COLOURS["reddish_purple"])
        # Polygon boundaries aren't re-embedded yet; plot a marker at centroid
        cx, cy = poly.get("_centroid_xy", [np.nan, np.nan])
        if np.isfinite(cx) and np.isfinite(cy):
            ax.plot(cx, cy, "s", markersize=4,
                    markerfacecolor=colour, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=5)

    ax.set_title("Anonymised", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#b0b0b0")


# ┌────────────────────────────────────────────────────────────┐
# │ Drawing helpers  « shared low-level matplotlib glue »       │
# └────────────────────────────────────────────────────────────┘

def _plot_roads(ax: matplotlib.axes.Axes, g: nx.MultiDiGraph) -> None:
    """Draw the OSM road graph with class-based styling."""
    try:
        import osmnx as ox
        edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
    except Exception:
        return
    if edges is None or edges.empty:
        return

    # Group by highway tag and plot in ascending importance
    order = ["path", "track", "service", "residential", "unclassified",
             "tertiary", "secondary", "primary", "motorway"]
    groups: dict[str, list[Any]] = {}
    for _, row in edges.iterrows():
        hw = row.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0] if hw else "unclassified"
        groups.setdefault(str(hw), []).append(row.geometry)

    for cls in [c for c in order if c in groups] + [c for c in groups if c not in order]:
        colour, lw = _ROAD_STYLE.get(cls, _ROAD_DEFAULT)
        for geom in groups[cls]:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax.plot(list(xs), list(ys), color=colour, linewidth=lw, alpha=0.9)
            elif geom.geom_type == "MultiLineString":
                for sub in geom.geoms:
                    xs, ys = sub.xy
                    ax.plot(list(xs), list(ys), color=colour, linewidth=lw, alpha=0.9)


def _plot_gdf_lines(
    ax: matplotlib.axes.Axes,
    gdf: gpd.GeoDataFrame | None,
    *,
    colour: str,
    linewidth: float,
) -> None:
    """Draw LineString / MultiLineString geometries from a GeoDataFrame."""
    if gdf is None or gdf.empty:
        return
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            xs, ys = g.xy
            ax.plot(list(xs), list(ys), color=colour, linewidth=linewidth)
        elif g.geom_type == "MultiLineString":
            for sub in g.geoms:
                xs, ys = sub.xy
                ax.plot(list(xs), list(ys), color=colour, linewidth=linewidth)
        elif g.geom_type in ("Polygon", "MultiPolygon"):
            gpd.GeoSeries([g], crs=gdf.crs).boundary.plot(
                ax=ax, color=colour, linewidth=linewidth,
            )


def _dem_extent(
    dem: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compute (left, right, bottom, top) image extent from affine."""
    a, _b, c, _d, e, f = transform
    h, w = dem.shape
    left = c
    right = c + a * w
    top = f
    bottom = f + e * h
    return (left, right, bottom, top) if e < 0 else (left, right, top, bottom)


def _style_geo_axes(
    ax: matplotlib.axes.Axes,
    bbox: tuple[float, float, float, float],
) -> None:
    """Apply common styling: equal aspect, bbox limits, thin spines."""
    n, s, e, w = bbox
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("equal")
    ax.tick_params(labelbottom=False, labelleft=False,
                   bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#606060")
        spine.set_linewidth(0.5)


def _add_scale_bar(
    ax: matplotlib.axes.Axes,
    bbox: tuple[float, float, float, float],
    length_km: float = 2.0,
) -> None:
    """Draw a simple km-scale bar in the lower-right corner."""
    n, s, e, w = bbox
    lat0 = 0.5 * (n + s)
    deg_per_km_lon = 1.0 / (111.320 * np.cos(np.radians(lat0)))
    dx = length_km * deg_per_km_lon
    x1 = e - 0.06 * (e - w)
    x0 = x1 - dx
    y = s + 0.05 * (n - s)
    ax.plot([x0, x1], [y, y], color=COLOURS["black"], linewidth=1.5)
    ax.plot([x0, x0], [y - 0.002, y + 0.002], color=COLOURS["black"], linewidth=1.5)
    ax.plot([x1, x1], [y - 0.002, y + 0.002], color=COLOURS["black"], linewidth=1.5)
    ax.text(
        0.5 * (x0 + x1), y + 0.004, f"{length_km:.0f} km",
        fontsize=6, ha="center", va="bottom",
    )


def _add_north_arrow(ax: matplotlib.axes.Axes) -> None:
    """Tiny N arrow in the upper-right corner."""
    ax.annotate(
        "N", xy=(0.93, 0.93), xytext=(0.93, 0.80),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", linewidth=1.0, color=COLOURS["black"]),
    )
