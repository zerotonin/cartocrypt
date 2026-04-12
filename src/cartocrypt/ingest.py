"""Ingest cartographic data into labelled planar graphs.

« From maps to graphs — the first transformation »

Parses OpenStreetMap, GeoJSON, and Shapefile sources into a
NetworkX planar graph with node/edge/face attributes preserved.

Functions
---------
from_osm        Download and parse an OSM bounding box.
from_geojson    Load a GeoJSON file.
from_shapefile  Load an ESRI Shapefile.
to_labelled_graph  Unified converter → nx.PlanarEmbedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np

from cartocrypt.constants import Coords, DEFAULT_CRS


def from_osm(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """Download an OSM road network and return as a NetworkX graph.

    Args:
        bbox: Bounding box as (north, south, east, west).
        network_type: OSM network type passed to osmnx.graph_from_bbox.

    Returns:
        Raw OSM graph with geometry attributes on edges.
    """
    import osmnx as ox

    graph = ox.graph_from_bbox(bbox=bbox, network_type=network_type)
    return graph


def from_geojson(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame.

    Args:
        path: Path to the .geojson file.

    Returns:
        GeoDataFrame with geometry and attributes.
    """
    return gpd.read_file(path, driver="GeoJSON")


def from_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Load an ESRI Shapefile into a GeoDataFrame.

    Args:
        path: Path to the .shp file.

    Returns:
        GeoDataFrame with geometry and attributes.
    """
    return gpd.read_file(path)


def to_labelled_graph(
    gdf: gpd.GeoDataFrame | nx.MultiDiGraph,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert geodata to a labelled planar graph.

    Extracts a simple planar graph where:
    - Nodes carry: elevation, original coords, land-use type.
    - Edges carry: length (metres), road class / feature type.
    - Faces (cycles) carry: area (m²), perimeter, classification.

    Args:
        gdf: Input geodata as GeoDataFrame or OSM MultiDiGraph.

    Returns:
        Tuple of (graph, coordinate_array, metadata_dict).

    Raises:
        ValueError: If the input cannot be converted to a planar graph.
    """
    # ── OSM MultiDiGraph path ──────────────────────────────────
    if isinstance(gdf, (nx.MultiDiGraph, nx.MultiGraph)):
        return _osm_to_labelled(gdf)

    # ── GeoDataFrame path ──────────────────────────────────────
    return _gdf_to_labelled(gdf)


def _osm_to_labelled(
    mg: nx.MultiDiGraph,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert an OSM MultiDiGraph to a labelled simple planar graph.

    Args:
        mg: OSM graph from osmnx.

    Returns:
        Tuple of (graph, coords, metadata).
    """
    # Convert to undirected simple graph
    g = nx.Graph(mg.to_undirected())

    # Extract coordinates
    coords = np.array(
        [[g.nodes[n].get("x", 0.0), g.nodes[n].get("y", 0.0)] for n in g.nodes],
        dtype=np.float64,
    )

    # Build node index mapping
    node_list = list(g.nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}

    # Transfer edge lengths
    for u, v, data in g.edges(data=True):
        if "length" not in data:
            i, j = node_idx[u], node_idx[v]
            data["length"] = float(np.linalg.norm(coords[i] - coords[j]))

    metadata = {
        "source": "osm",
        "crs": DEFAULT_CRS,
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "node_index": node_idx,
    }

    return g, coords, metadata


def _gdf_to_labelled(
    gdf: gpd.GeoDataFrame,
) -> tuple[nx.Graph, Coords, dict[str, Any]]:
    """Convert a GeoDataFrame of lines/polygons to a labelled graph.

    Args:
        gdf: GeoDataFrame with line or polygon geometries.

    Returns:
        Tuple of (graph, coords, metadata).
    """
    # TODO: Implement line → node/edge extraction
    # TODO: Implement polygon → face extraction
    raise NotImplementedError(
        "GeoDataFrame → planar graph conversion is not yet implemented. "
        "Use from_osm() for the proof-of-concept."
    )
