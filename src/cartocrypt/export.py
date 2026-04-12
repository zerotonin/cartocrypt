"""Export anonymised graph data to standard geospatial formats.

« From graph back to map — the final transformation »
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from cartocrypt.constants import Coords


def to_geojson(
    g: nx.Graph,
    coords: Coords,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    """Write the anonymised graph as a GeoJSON FeatureCollection.

    Edges become LineString features; nodes become Point features.
    All original attributes are preserved; coordinates are replaced.

    Args:
        g: Labelled graph with attributes.
        coords: (N, 2) anonymised coordinate array.
        metadata: Pipeline metadata (checksum, key fingerprint, etc.).
        path: Output .geojson file path.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    features: list[dict[str, Any]] = []

    # ── Node features ──────────────────────────────────────────
    for nd in nodes:
        i = node_idx[nd]
        props = {k: _jsonable(v) for k, v in g.nodes[nd].items()}
        props["_cartocrypt_node_id"] = str(nd)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(coords[i, 0]), float(coords[i, 1])],
            },
            "properties": props,
        })

    # ── Edge features ──────────────────────────────────────────
    for u, v, data in g.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        props = {k: _jsonable(val) for k, val in data.items()}
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(coords[i, 0]), float(coords[i, 1])],
                    [float(coords[j, 0]), float(coords[j, 1])],
                ],
            },
            "properties": props,
        })

    collection = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "_cartocrypt_version": "0.1.0",
            "_cartocrypt_checksum": metadata.get("checksum", ""),
        },
    }

    path.write_text(json.dumps(collection, indent=2, default=str))


def to_svg(
    g: nx.Graph,
    coords: Coords,
    path: Path,
    *,
    width: int = 800,
    height: int = 600,
) -> None:
    """Write a simple SVG visualisation of the anonymised graph.

    Args:
        g: Labelled graph.
        coords: (N, 2) anonymised coordinates.
        path: Output .svg file path.
        width: SVG canvas width in pixels.
        height: SVG canvas height in pixels.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # Scale coordinates to canvas
    scaled = _scale_to_canvas(coords, width, height, margin=40)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'style="font-family: monospace;">',
    ]

    # Edges
    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        lines.append(
            f'  <line x1="{scaled[i,0]:.1f}" y1="{scaled[i,1]:.1f}" '
            f'x2="{scaled[j,0]:.1f}" y2="{scaled[j,1]:.1f}" '
            f'stroke="#56B4E9" stroke-width="1.5" opacity="0.7"/>'
        )

    # Nodes
    for nd in nodes:
        i = node_idx[nd]
        lines.append(
            f'  <circle cx="{scaled[i,0]:.1f}" cy="{scaled[i,1]:.1f}" '
            f'r="3" fill="#D55E00"/>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────


def _scale_to_canvas(
    coords: Coords,
    width: int,
    height: int,
    margin: int = 40,
) -> np.ndarray:
    """Scale coordinates to fit an SVG canvas."""
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = mx - mn
    span[span < 1e-12] = 1.0
    scaled = (coords - mn) / span
    scaled[:, 0] = margin + scaled[:, 0] * (width - 2 * margin)
    scaled[:, 1] = margin + scaled[:, 1] * (height - 2 * margin)
    return scaled


def _jsonable(v: Any) -> Any:
    """Convert a value to a JSON-serialisable type."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return str(v)
