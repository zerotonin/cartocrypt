"""Attach non-canonical project layers to a CartoCrypt base graph.

« Your Pokemon, our cryptography »

Users bring their own project-specific geographic data — species
sightings, powerline surveys, habitat polygons, sensor grids,
whatever — in any standard geo format.  This module snaps those
features to the base graph so they co-transform during
anonymisation.

The user is responsible for getting their data into a
GeoDataFrame (or a file geopandas can read: GeoJSON, Shapefile,
GeoPackage).  CartoCrypt does not do format conversion.

Multiple layers can be attached to the same graph.  Each layer
is identified by a user-chosen name string.

How features are stored
-----------------------
Point features   → snapped to nearest graph node, stored in
                   ``g.nodes[n]["overlays"]``.
Line features    → snapped to nearest edge (start/end nodes),
                   stored in ``g.edges[u,v]["overlays"]`` or
                   in ``g.graph["overlay_orphan_lines"]`` if
                   no matching edge exists.
Polygon features → stored in ``g.graph["overlay_polygons"]``
                   with boundary coordinates for Fourier
                   perturbation during anonymisation.

All overlay attributes survive the re-embedding pipeline
unchanged.  Coordinates move with their host graph element.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from cartocrypt.constants import Coords

# ═════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════


def add_layer(
    g: nx.Graph,
    coords: Coords,
    data: gpd.GeoDataFrame | Path | str,
    name: str,
    *,
    snap_tolerance_m: float = 500.0,
) -> dict[str, Any]:
    """Attach a project layer to the base graph.

    Args:
        g: CartoCrypt base graph (modified in place).
        coords: (N, 2) base graph coordinate array.
        data: Project data as a GeoDataFrame, or a Path / string
            to a file that geopandas can read (GeoJSON, Shapefile,
            GeoPackage).
        name: Layer name.  Must be unique per graph.  Used to
            retrieve and filter features later.
        snap_tolerance_m: Maximum snap distance in metres.
            Features further than this from any graph node are
            still attached but flagged with ``_beyond_tolerance``.

    Returns:
        Summary dict with feature counts by geometry type.

    Raises:
        ValueError: If the layer name is already taken, or data
            is empty.
        FileNotFoundError: If a file path does not exist.
    """
    # ── Check for duplicate layer name ─────────────────────────
    existing = {lyr["name"] for lyr in g.graph.get("_overlay_meta", [])}
    if name in existing:
        msg = f"Layer name '{name}' already exists on this graph."
        raise ValueError(msg)

    # ── Resolve to GeoDataFrame ────────────────────────────────
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Layer file not found: {path}")
        gdf = gpd.read_file(path)
    elif isinstance(data, gpd.GeoDataFrame):
        gdf = data
    else:
        msg = (
            f"Expected GeoDataFrame or path to GeoJSON/Shapefile/"
            f"GeoPackage, got {type(data).__name__}"
        )
        raise TypeError(msg)

    if gdf.empty:
        raise ValueError(f"Layer '{name}' contains no features.")

    # ── Spatial index ──────────────────────────────────────────
    tree = cKDTree(coords)

    # ── Walk features and attach ───────────────────────────────
    counts = {"points": 0, "lines": 0, "polygons": 0, "beyond_tolerance": 0}

    for _, row in gdf.iterrows():
        geom = row.geometry
        attrs = _row_to_attrs(row, name)
        _dispatch(g, coords, tree, geom, attrs, name,
                  snap_tolerance_m, counts)

    # ── Store layer metadata ───────────────────────────────────
    meta = {"name": name, **counts,
            "total": counts["points"] + counts["lines"] + counts["polygons"]}
    if "_overlay_meta" not in g.graph:
        g.graph["_overlay_meta"] = []
    g.graph["_overlay_meta"].append(meta)

    return meta


def get_layers(g: nx.Graph) -> list[dict[str, Any]]:
    """List all project layers attached to the graph.

    Args:
        g: Graph with layers.

    Returns:
        List of summary dicts, one per layer.
    """
    return list(g.graph.get("_overlay_meta", []))


def get_points(
    g: nx.Graph,
    name: str | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Retrieve point features from graph nodes.

    Args:
        g: Graph with layers.
        name: Layer name filter (None = all layers).

    Returns:
        List of (node_id, attribute_dict) tuples.
    """
    out = []
    for n, data in g.nodes(data=True):
        for ov in data.get("overlays", []):
            if name is None or ov.get("_layer") == name:
                out.append((n, ov))
    return out


def get_lines(
    g: nx.Graph,
    name: str | None = None,
) -> list[tuple[int, int, dict[str, Any]]]:
    """Retrieve line features from graph edges.

    Args:
        g: Graph with layers.
        name: Layer name filter (None = all layers).

    Returns:
        List of (u, v, attribute_dict) tuples.  Includes orphan
        lines that didn't match any edge (u and v are the snapped
        node ids but no edge exists between them).
    """
    out = []
    for u, v, data in g.edges(data=True):
        for ov in data.get("overlays", []):
            if name is None or ov.get("_layer") == name:
                out.append((u, v, ov))
    # Include orphans
    for ov in g.graph.get("overlay_orphan_lines", []):
        if name is None or ov.get("_layer") == name:
            out.append((ov["_snap_start"], ov["_snap_end"], ov))
    return out


def get_polygons(
    g: nx.Graph,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve polygon features.

    Args:
        g: Graph with layers.
        name: Layer name filter (None = all layers).

    Returns:
        List of attribute dicts (each includes ``_boundary``
        coordinate array for Fourier perturbation).
    """
    all_p = g.graph.get("overlay_polygons", [])
    if name is None:
        return list(all_p)
    return [p for p in all_p if p.get("_layer") == name]


def attach_settlements(
    g: nx.Graph,
    coords: Coords,
    bundle: Any,
    *,
    layer_name: str = "settlement",
    snap_tolerance_m: float = 2000.0,
) -> dict[int, int]:
    """Snap each settlement Point in ``bundle`` to a graph node.

    Each settlement becomes an overlay on its nearest graph node,
    so it travels along when the graph is re-embedded.

    Args:
        g: Base graph (sequential integer node labels).
        coords: ``(N, 2)`` EPSG:4326 coordinate array.
        bundle: Anything with a ``settlements`` GeoDataFrame
            attribute — typically :class:`cartocrypt.aegina.LayerBundle`.
        layer_name: Overlay layer identifier; default
            ``"settlement"``.
        snap_tolerance_m: Maximum snap distance; settlements
            further than this flag ``_beyond_tolerance = True`` on
            the overlay but are still attached.

    Returns:
        ``{positional_settlement_index → graph_node_index}``
        mapping — rows are keyed by integer position (``iloc``), not
        by the source GDF's potentially-MultiIndex label.  Indexing
        the re-embedded coordinate array by the mapping's values
        then gives warped settlement positions.
    """
    if getattr(bundle, "settlements", None) is None:
        return {}
    settlements = bundle.settlements
    if settlements.empty:
        return {}
    add_layer(g, coords, settlements, layer_name,
              snap_tolerance_m=snap_tolerance_m)
    # Cache settlement centroids as (x, y) by positional index.
    centroids: list[tuple[int, float, float]] = []
    for pos, (_s_idx, s_row) in enumerate(settlements.iterrows()):
        sg = s_row.geometry
        if sg is None or sg.is_empty:
            continue
        c = sg.centroid
        centroids.append((pos, float(c.x), float(c.y)))

    mapping: dict[int, int] = {}
    for n_id, data in g.nodes(data=True):
        for ov in data.get("overlays", []):
            if ov.get("_layer") != layer_name:
                continue
            src_xy = ov.get("_original_xy")
            if src_xy is None:
                continue
            best_pos = None
            best_d = np.inf
            for pos, cx, cy in centroids:
                d = (cx - src_xy[0]) ** 2 + (cy - src_xy[1]) ** 2
                if d < best_d:
                    best_d = d
                    best_pos = pos
            if best_pos is not None:
                mapping[int(best_pos)] = int(n_id)
    return mapping


def polygons_as_gdf(
    g: nx.Graph,
    name: str | None = None,
) -> gpd.GeoDataFrame:
    """Return attached polygon overlays as a flat GeoDataFrame.

    Splicing and viz code want a GeoDataFrame with real Shapely
    geometry, not the raw ``_boundary`` coordinate lists stored
    inside ``g.graph['overlay_polygons']``.  This helper rebuilds
    the geometry and returns a GDF in EPSG:4326.

    Args:
        g: Graph with overlays attached.
        name: Optional layer-name filter.

    Returns:
        GeoDataFrame with columns from the stored attribute dicts
        plus a ``geometry`` column of Shapely Polygons.  Internal
        bookkeeping fields that start with ``_`` are dropped
        (except ``_layer``, which is renamed to ``layer``).
    """
    from shapely.geometry import Polygon

    rows: list[dict[str, Any]] = []
    geoms: list[Any] = []
    for poly in g.graph.get("overlay_polygons", []):
        if name is not None and poly.get("_layer") != name:
            continue
        boundary = poly.get("_boundary")
        if not boundary:
            continue
        geom = Polygon(boundary)
        if geom.is_empty:
            continue
        row = {
            k: v for k, v in poly.items()
            if not k.startswith("_") or k == "_layer"
        }
        if "_layer" in row:
            row["layer"] = row.pop("_layer")
        rows.append(row)
        geoms.append(geom)

    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return gdf


def summarise(g: nx.Graph) -> str:
    """Human-readable summary of all attached layers.

    Args:
        g: Graph with layers.

    Returns:
        Multi-line summary string.
    """
    layers = get_layers(g)
    if not layers:
        return "No project layers attached."
    lines = ["CartoCrypt Project Layers", "=" * 40]
    for la in layers:
        lines.append(f"  {la['name']}: "
                     f"{la['points']} pts, "
                     f"{la['lines']} lines, "
                     f"{la['polygons']} polys "
                     f"({la['total']} total)")
        if la["beyond_tolerance"] > 0:
            lines.append(f"    ({la['beyond_tolerance']} beyond snap tolerance)")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════
#  Private helpers
# ═════════════════════════════════════════════════════════════════


def _row_to_attrs(row: Any, layer_name: str) -> dict[str, Any]:
    """Extract serialisable attributes from a GeoDataFrame row."""
    attrs: dict[str, Any] = {"_layer": layer_name}
    for k, v in row.items():
        if k == "geometry":
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            attrs[k] = v
        elif isinstance(v, (np.integer,)):
            attrs[k] = int(v)
        elif isinstance(v, (np.floating,)):
            attrs[k] = float(v)
        else:
            attrs[k] = str(v)
    return attrs


def _dispatch(
    g: nx.Graph,
    coords: Coords,
    tree: cKDTree,
    geom: Any,
    attrs: dict[str, Any],
    name: str,
    tolerance: float,
    counts: dict[str, int],
) -> None:
    """Route a geometry to the appropriate attachment function."""
    gt = geom.geom_type

    if gt == "Point":
        _snap_point(g, coords, tree, geom, attrs, tolerance, counts)
    elif gt == "MultiPoint":
        for pt in geom.geoms:
            _snap_point(g, coords, tree, pt, attrs.copy(), tolerance, counts)
    elif gt == "LineString":
        _snap_line(g, coords, tree, geom, attrs, counts)
    elif gt == "MultiLineString":
        for ln in geom.geoms:
            _snap_line(g, coords, tree, ln, attrs.copy(), counts)
    elif gt == "Polygon":
        _store_polygon(g, geom, attrs, counts)
    elif gt == "MultiPolygon":
        for poly in geom.geoms:
            _store_polygon(g, poly, attrs.copy(), counts)
    elif hasattr(geom, "geoms"):
        for sub in geom.geoms:
            _dispatch(g, coords, tree, sub, attrs.copy(), name,
                      tolerance, counts)


def _snap_point(
    g: nx.Graph,
    coords: Coords,
    tree: cKDTree,
    geom: Any,
    attrs: dict[str, Any],
    tolerance: float,
    counts: dict[str, int],
) -> None:
    """Snap a Point to the nearest graph node."""
    xy = np.array([geom.x, geom.y])
    dist_deg, idx = tree.query(xy)
    idx = int(idx)

    dist_m = dist_deg * 111_000.0  # rough degree→metre
    attrs["_snap_node"] = idx
    attrs["_snap_dist_m"] = round(dist_m, 1)
    attrs["_original_xy"] = [float(geom.x), float(geom.y)]
    if dist_m > tolerance:
        attrs["_beyond_tolerance"] = True
        counts["beyond_tolerance"] += 1

    if "overlays" not in g.nodes[idx]:
        g.nodes[idx]["overlays"] = []
    g.nodes[idx]["overlays"].append(attrs)
    counts["points"] += 1


def _snap_line(
    g: nx.Graph,
    coords: Coords,
    tree: cKDTree,
    geom: Any,
    attrs: dict[str, Any],
    counts: dict[str, int],
) -> None:
    """Snap a LineString's endpoints to nearest nodes."""
    start = np.array(geom.coords[0][:2])
    end = np.array(geom.coords[-1][:2])
    _, si = tree.query(start)
    _, ei = tree.query(end)
    si, ei = int(si), int(ei)

    attrs["_snap_start"] = si
    attrs["_snap_end"] = ei

    if g.has_edge(si, ei):
        if "overlays" not in g.edges[si, ei]:
            g.edges[si, ei]["overlays"] = []
        g.edges[si, ei]["overlays"].append(attrs)
    else:
        if "overlay_orphan_lines" not in g.graph:
            g.graph["overlay_orphan_lines"] = []
        g.graph["overlay_orphan_lines"].append(attrs)

    counts["lines"] += 1


def _store_polygon(
    g: nx.Graph,
    geom: Any,
    attrs: dict[str, Any],
    counts: dict[str, int],
) -> None:
    """Store a Polygon with its boundary for later perturbation."""
    attrs["_boundary"] = [list(c[:2]) for c in geom.exterior.coords]
    attrs["_area_m2"] = round(geom.area * 111_000.0**2, 1)
    attrs["_centroid_xy"] = [geom.centroid.x, geom.centroid.y]

    if "overlay_polygons" not in g.graph:
        g.graph["overlay_polygons"] = []
    g.graph["overlay_polygons"].append(attrs)
    counts["polygons"] += 1
