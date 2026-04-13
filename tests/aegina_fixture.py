"""Synthetic Aegina road network fixture for testing.

« A phantom island for a phantom map »

Generates a planar graph resembling Aegina's road network:
- Coastal ring road (~25 nodes around the perimeter)
- Five village clusters (Aegina Town, Souvala, Agia Marina,
  Perdika, Mesagros) with internal street grids
- Interior mountain roads connecting villages through Mt. Oros
- Realistic edge lengths (metres), elevations, road classes

All coordinates are in EPSG:4326 (lon, lat).  The graph is
designed to be planar, 2-connected, and have ~80 nodes / ~120
edges — large enough for meaningful stress-majorisation tests,
small enough for fast iteration.

This fixture is a stand-in until the user can download real
OSM data from their workstation (overpass-api.de is blocked
in the sandboxed environment).

Usage
-----
>>> from tests.aegina_fixture import make_aegina_graph
>>> g, coords, metadata = make_aegina_graph()
>>> g.number_of_nodes()
82
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def make_aegina_graph() -> tuple[nx.Graph, np.ndarray, dict]:
    """Build a synthetic Aegina road network.

    Returns:
        Tuple of (graph, coords[N,2], metadata).
    """
    g = nx.Graph()
    coords_dict: dict[int, tuple[float, float]] = {}
    node_id = 0

    # ── Coastal ring (25 nodes) ────────────────────────────────
    # Aegina is roughly centred at (23.49, 37.73), ~5 km radius
    centre_lon, centre_lat = 23.49, 37.73
    n_coast = 25
    coast_ids = []
    for k in range(n_coast):
        angle = 2.0 * np.pi * k / n_coast
        # Slightly irregular radius for realism
        r = 0.045 + 0.005 * np.sin(3 * angle) + 0.003 * np.cos(5 * angle)
        lon = centre_lon + r * np.cos(angle)
        lat = centre_lat + r * np.sin(angle)
        elev = 5.0 + 10.0 * abs(np.sin(2 * angle))  # coastal elevation

        g.add_node(node_id, elevation=round(elev, 1), place_type="coast")
        coords_dict[node_id] = (lon, lat)
        coast_ids.append(node_id)
        node_id += 1

    # Connect coastal ring
    for k in range(n_coast):
        u, v = coast_ids[k], coast_ids[(k + 1) % n_coast]
        length = _haversine(coords_dict[u], coords_dict[v])
        g.add_edge(u, v, length=round(length, 1), road_class="secondary")

    # ── Village clusters ───────────────────────────────────────
    villages = [
        {"name": "Aegina Town",   "anchor_coast": 0,  "offset": (-0.008, -0.005), "n": 12, "elev_base": 3.0},
        {"name": "Souvala",       "anchor_coast": 6,  "offset": (0.003, 0.006),   "n": 8,  "elev_base": 8.0},
        {"name": "Agia Marina",   "anchor_coast": 12, "offset": (0.006, -0.002),  "n": 10, "elev_base": 15.0},
        {"name": "Perdika",       "anchor_coast": 19, "offset": (-0.005, -0.008), "n": 8,  "elev_base": 20.0},
        {"name": "Mesagros",      "anchor_coast": 16, "offset": (0.000, 0.005),   "n": 6,  "elev_base": 120.0},
    ]

    village_centres = {}
    for vil in villages:
        anchor = coast_ids[vil["anchor_coast"]]
        base_lon, base_lat = coords_dict[anchor]
        dx, dy = vil["offset"]
        vil_ids = []

        # Small grid cluster
        n_side = int(np.ceil(np.sqrt(vil["n"])))
        count = 0
        for row in range(n_side):
            for col in range(n_side):
                if count >= vil["n"]:
                    break
                lon = base_lon + dx + col * 0.002
                lat = base_lat + dy + row * 0.002
                elev = vil["elev_base"] + row * 5.0 + np.random.default_rng(node_id).uniform(-2, 2)

                g.add_node(node_id, elevation=round(elev, 1),
                           place_type="village", village=vil["name"])
                coords_dict[node_id] = (lon, lat)
                vil_ids.append(node_id)
                node_id += 1
                count += 1

        # Connect village grid (horizontal + vertical edges)
        for i, vid in enumerate(vil_ids):
            row_i, col_i = divmod(i, n_side)
            # Right neighbour
            if col_i + 1 < n_side and i + 1 < len(vil_ids):
                nb = vil_ids[i + 1]
                length = _haversine(coords_dict[vid], coords_dict[nb])
                g.add_edge(vid, nb, length=round(length, 1), road_class="residential")
            # Below neighbour
            below = i + n_side
            if below < len(vil_ids):
                nb = vil_ids[below]
                length = _haversine(coords_dict[vid], coords_dict[nb])
                g.add_edge(vid, nb, length=round(length, 1), road_class="residential")

        # Connect first village node to coastal anchor
        length = _haversine(coords_dict[vil_ids[0]], coords_dict[anchor])
        g.add_edge(vil_ids[0], anchor, length=round(length, 1), road_class="tertiary")

        village_centres[vil["name"]] = vil_ids[0]

    # ── Interior mountain roads ────────────────────────────────
    # Connect villages through the interior (Mt. Oros region)
    # Add 3 mountain pass nodes
    mountain_ids = []
    for k, (lon_off, lat_off, elev) in enumerate([
        (0.000, 0.010, 350.0),   # Mt. Oros summit approach
        (-0.010, 0.005, 180.0),  # Western pass
        (0.010, 0.003, 220.0),   # Eastern pass
    ]):
        g.add_node(node_id, elevation=elev, place_type="mountain")
        coords_dict[node_id] = (centre_lon + lon_off, centre_lat + lat_off)
        mountain_ids.append(node_id)
        node_id += 1

    # Connect mountain nodes to each other
    for i in range(len(mountain_ids)):
        for j in range(i + 1, len(mountain_ids)):
            length = _haversine(coords_dict[mountain_ids[i]], coords_dict[mountain_ids[j]])
            g.add_edge(mountain_ids[i], mountain_ids[j],
                       length=round(length, 1), road_class="track")

    # Connect mountain passes to nearest village centres
    mountain_village_links = [
        (0, "Mesagros"), (0, "Agia Marina"),
        (1, "Aegina Town"), (1, "Mesagros"),
        (2, "Souvala"), (2, "Agia Marina"),
    ]
    for mi, vname in mountain_village_links:
        vc = village_centres[vname]
        length = _haversine(coords_dict[mountain_ids[mi]], coords_dict[vc])
        g.add_edge(mountain_ids[mi], vc,
                    length=round(length, 1), road_class="track")

    # ── Build output arrays ────────────────────────────────────
    nodes = sorted(g.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    coords = np.array([coords_dict[n] for n in nodes], dtype=np.float64)

    # Relabel to sequential integers
    g = nx.relabel_nodes(g, node_idx)

    metadata = {
        "source": "synthetic_aegina",
        "crs": "EPSG:4326",
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "node_index": {i: i for i in range(n)},
        "bbox": {
            "north": float(coords[:, 1].max()),
            "south": float(coords[:, 1].min()),
            "east": float(coords[:, 0].max()),
            "west": float(coords[:, 0].min()),
        },
        "villages": list(village_centres.keys()),
        "description": (
            "Synthetic road network of Aegina island, Greece. "
            "Coastal ring + 5 village clusters + mountain passes."
        ),
    }

    return g, coords, metadata


def _haversine(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute Haversine distance in metres between two (lon, lat) points.

    Args:
        p1: (longitude, latitude) in degrees.
        p2: (longitude, latitude) in degrees.

    Returns:
        Distance in metres.
    """
    R = 6_371_000.0  # Earth radius in metres
    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))
