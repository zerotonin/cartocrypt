"""Ingest cartographic data into labelled planar graphs.

« From maps to graphs — the first transformation »

Parses OpenStreetMap, GeoJSON, and Shapefile sources into a
NetworkX planar graph with node/edge/face attributes preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np

from cartocrypt.constants import DEFAULT_CRS, Coords


def from_osm(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """Download an OSM road network via Overpass API.

    Args:
        bbox: Bounding box as (north, south, east, west).
        network_type: OSM network type (drive, walk, bike, all).

    Returns:
        Raw OSM graph with geometry attributes on edges.
    """
    import osmnx as ox
    return ox.graph_from_bbox(bbox=bbox, network_type=network_type)


def from_geojson(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame.

    Args:
        path: Path to the .geojson file.

    Returns:
        GeoDataFrame with geometry and attributes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")
    return gpd.read_file(path, driver="GeoJSON")


def from_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Load an ESRI Shapefile into a GeoDataFrame.

    Args:
        path: Path to the .shp file.

    Returns:
        GeoDataFrame with geometry and attributes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")
    return gpd.read_file(path)


def to_labelled_graph(
    source: gpd.GeoDataFrame | nx.MultiDiGraph | nx.Graph,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert geodata to a labelled planar graph.

    Extracts a simple undirected graph where:
    - Nodes carry: elevation (if available), original coords, type.
    - Edges carry: length (metres), road class / feature type.

    Args:
        source: Input as GeoDataFrame, OSM MultiDiGraph, or
            pre-built nx.Graph (e.g. from a test fixture).

    Returns:
        Tuple of (graph, coordinate_array, metadata_dict).
    """
    if isinstance(source, nx.Graph) and not isinstance(
        source, (nx.MultiDiGraph, nx.MultiGraph, nx.DiGraph)
    ):
        return _simple_graph_to_labelled(source)

    if isinstance(source, (nx.MultiDiGraph, nx.MultiGraph)):
        return _osm_to_labelled(source)

    if isinstance(source, gpd.GeoDataFrame):
        return _gdf_to_labelled(source)

    raise TypeError(f"Unsupported source type: {type(source)}")


def graph_summary(
    g: nx.Graph,
    coords: Coords,
    metadata: dict[str, Any],
) -> str:
    """Return a human-readable summary of a labelled graph.

    Args:
        g: Labelled graph.
        coords: Coordinate array.
        metadata: Pipeline metadata.

    Returns:
        Multi-line summary string.
    """
    lengths = []
    for u, v, data in g.edges(data=True):
        lengths.append(data.get("length", 0.0))

    road_classes: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        rc = data.get("road_class", "unknown")
        road_classes[rc] = road_classes.get(rc, 0) + 1

    is_planar, _ = nx.check_planarity(g)

    lines = [
        "CartoCrypt Graph Summary",
        "=" * 40,
        f"Source:           {metadata.get('source', 'unknown')}",
        f"CRS:             {metadata.get('crs', 'unknown')}",
        f"Nodes:           {g.number_of_nodes()}",
        f"Edges:           {g.number_of_edges()}",
        f"Components:      {nx.number_connected_components(g)}",
        f"Planar:          {is_planar}",
        f"Coord range X:   [{coords[:, 0].min():.6f}, {coords[:, 0].max():.6f}]",
        f"Coord range Y:   [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]",
    ]
    if lengths:
        lines.append(
            f"Edge lengths:    min={min(lengths):.1f}, "
            f"max={max(lengths):.1f}, mean={np.mean(lengths):.1f} m"
        )
    if road_classes:
        rc_str = ", ".join(f"{k}: {v}" for k, v in sorted(road_classes.items()))
        lines.append(f"Road classes:    {rc_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
#  Private converters
# ─────────────────────────────────────────────────────────────────

def _simple_graph_to_labelled(
    g: nx.Graph,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Handle a pre-built simple nx.Graph (fixture or manual)."""
    nodes = sorted(g.nodes)
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    coords = np.zeros((n, 2), dtype=np.float64)
    for nd in nodes:
        i = node_idx[nd]
        data = g.nodes[nd]
        coords[i, 0] = data.get("x", data.get("lon", 0.0))
        coords[i, 1] = data.get("y", data.get("lat", 0.0))

    for u, v, data in g.edges(data=True):
        if "length" not in data:
            i, j = node_idx[u], node_idx[v]
            data["length"] = float(np.linalg.norm(coords[i] - coords[j]))

    if nodes != list(range(n)):
        g = nx.relabel_nodes(g, node_idx)

    metadata = {
        "source": "pre-built",
        "crs": DEFAULT_CRS,
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "node_index": {i: i for i in range(n)},
    }
    return g, coords, metadata


def _osm_to_labelled(
    mg: nx.MultiDiGraph,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert an OSM MultiDiGraph to a labelled simple graph."""
    ug = nx.Graph()
    for u, v, data in mg.edges(data=True):
        if not ug.has_edge(u, v):
            ug.add_edge(u, v, **data)
        elif data.get("length", float("inf")) < ug[u][v].get("length", float("inf")):
            ug[u][v].update(data)

    for nd, data in mg.nodes(data=True):
        if nd in ug:
            ug.nodes[nd].update(data)

    nodes = sorted(ug.nodes)
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    coords = np.zeros((n, 2), dtype=np.float64)
    for nd in nodes:
        i = node_idx[nd]
        coords[i, 0] = ug.nodes[nd].get("x", 0.0)
        coords[i, 1] = ug.nodes[nd].get("y", 0.0)

    for u, v, data in ug.edges(data=True):
        if "length" not in data:
            i, j = node_idx[u], node_idx[v]
            data["length"] = float(np.linalg.norm(coords[i] - coords[j]))

    g = nx.relabel_nodes(ug, node_idx)
    metadata = {
        "source": "osm",
        "crs": DEFAULT_CRS,
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "node_index": node_idx,
    }
    return g, coords, metadata


def _gdf_to_labelled(
    gdf: gpd.GeoDataFrame,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert a GeoDataFrame of LineStrings to a labelled graph."""
    from shapely.geometry import LineString

    g = nx.Graph()
    coord_to_node: dict[tuple[float, float], int] = {}
    node_id = 0

    for _, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue

        start = (geom.coords[0][0], geom.coords[0][1])
        end = (geom.coords[-1][0], geom.coords[-1][1])

        for pt in (start, end):
            if pt not in coord_to_node:
                coord_to_node[pt] = node_id
                g.add_node(node_id, x=pt[0], y=pt[1])
                node_id += 1

        u, v = coord_to_node[start], coord_to_node[end]
        if u != v and not g.has_edge(u, v):
            attrs = {k: val for k, val in row.items() if k != "geometry"}
            attrs["length"] = geom.length
            g.add_edge(u, v, **attrs)

    if g.number_of_edges() == 0:
        raise ValueError("No LineString geometries found in GeoDataFrame")

    nodes = sorted(g.nodes)
    n = len(nodes)
    coords = np.array(
        [[g.nodes[nd]["x"], g.nodes[nd]["y"]] for nd in nodes],
        dtype=np.float64,
    )
    metadata = {
        "source": "geodataframe",
        "crs": str(gdf.crs) if gdf.crs else DEFAULT_CRS,
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "node_index": {nd: i for i, nd in enumerate(nodes)},
    }
    return g, coords, metadata
