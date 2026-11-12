"""Tests for attribute splicing."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from cartocrypt.constants import SPLICE_SIZE_RATIO_MAX
from cartocrypt.faces import extract_faces, face_adjacency_graph, face_areas
from cartocrypt.overlay import add_layer, polygons_as_gdf
from cartocrypt.splicing import splice_attributes
from tests.aegina_fixture import make_aegina_graph

# ───── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def aegina_with_habitats():
    g, coords, meta = make_aegina_graph()
    # Four moderately sized habitat polygons scattered across the island
    habitats = gpd.GeoDataFrame({
        "species": ["A", "B", "C", "D"],
        "geometry": [
            Polygon([(23.46, 37.72), (23.49, 37.72),
                     (23.49, 37.74), (23.46, 37.74)]),
            Polygon([(23.50, 37.72), (23.52, 37.72),
                     (23.52, 37.74), (23.50, 37.74)]),
            Polygon([(23.47, 37.75), (23.50, 37.75),
                     (23.50, 37.77), (23.47, 37.77)]),
            Polygon([(23.51, 37.75), (23.54, 37.75),
                     (23.54, 37.77), (23.51, 37.77)]),
        ],
    }, crs="EPSG:4326")
    add_layer(g, coords, habitats, "habitat")
    ov_gdf = polygons_as_gdf(g, name="habitat")
    return g, coords, ov_gdf


# ───── Face adjacency ──────────────────────────────────────

def test_face_adjacency_edges_iff_shared_boundary(aegina_with_habitats):
    g, coords, _ = aegina_with_habitats
    faces = extract_faces(g, coords)
    fa = face_adjacency_graph(faces)

    # For each pair of faces, adjacency ↔ shared undirected edge
    def face_edges(face):
        n = len(face)
        return {tuple(sorted([face[k], face[(k + 1) % n]]))
                for k in range(n)}
    edge_sets = [face_edges(f) for f in faces]

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            shared = edge_sets[i] & edge_sets[j]
            assert fa.has_edge(i, j) == (len(shared) > 0)


# ───── Splicing properties ─────────────────────────────────

def test_splice_preserves_node_edge_counts(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    _ = splice_attributes(g, coords, ov_gdf, key=b"\x11" * 32)
    assert g.number_of_nodes() == n_nodes
    assert g.number_of_edges() == n_edges


def test_splice_overlay_fragment_count_preserved(aegina_with_habitats):
    """Bijection on faces means total fragment count is unchanged.

    Two splice runs under different keys must produce the same
    total fragment count: the permutation is a bijection on faces
    so no fragment is created or destroyed, only relocated.
    """
    g, coords, ov_gdf = aegina_with_habitats
    result_a = splice_attributes(g, coords, ov_gdf, key=b"\x22" * 32)
    result_b = splice_attributes(g, coords, ov_gdf, key=b"\xbb" * 32)
    assert len(result_a.overlays) == len(result_b.overlays)
    assert len(result_a.overlays) > 0


def test_splice_rollup_multiset_preserved(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    result = splice_attributes(g, coords, ov_gdf, key=b"\x33" * 32,
                               rollup_fields=["elevation"])
    # Post-splice rollup collection is a permutation of the original
    input_rollups = []
    faces = extract_faces(g, coords)
    for face in faces:
        vals = [g.nodes[n].get("elevation") for n in face]
        vals = [v for v in vals if v is not None]
        if vals:
            input_rollups.append(round(float(np.median(vals)), 6))
    output_rollups = [
        round(float(r.get("elevation", 0.0)), 6)
        for r in result.face_rollups.values()
        if "elevation" in r
    ]
    assert sorted(input_rollups) == sorted(output_rollups)


def test_splice_deterministic_under_key(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    key = b"\x44" * 32
    a = splice_attributes(g, coords, ov_gdf, key=key)
    b = splice_attributes(g, coords, ov_gdf, key=key)
    assert list(a.face_rollups.items()) == list(b.face_rollups.items())
    assert len(a.overlays) == len(b.overlays)
    # Geometry parity
    for ga, gb in zip(a.overlays.geometry, b.overlays.geometry, strict=True):
        assert ga.equals_exact(gb, tolerance=1e-9)


def test_splice_differs_with_different_keys(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    a = splice_attributes(g, coords, ov_gdf, key=b"\x55" * 32)
    b = splice_attributes(g, coords, ov_gdf, key=b"\xaa" * 32)
    # Rollup dicts on at least one face should differ
    diffs = sum(1 for f in a.face_rollups
                if a.face_rollups[f] != b.face_rollups[f])
    assert diffs > 0


def test_splice_size_ratio_respected(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    faces = extract_faces(g, coords)
    areas = face_areas(coords, faces)

    result = splice_attributes(g, coords, ov_gdf, key=b"\x66" * 32)

    # For every spliced fragment, source and target face areas
    # must be within the size-ratio guard.
    ratio_max = SPLICE_SIZE_RATIO_MAX * 1.001  # float slack
    for _idx, row in result.overlays.iterrows():
        src = row.get("_spliced_from_face")
        tgt = row.get("_spliced_to_face")
        if src is None or tgt is None:
            continue  # identity fragment
        a_src, a_tgt = float(areas[src]), float(areas[tgt])
        ratio = max(a_src, a_tgt) / max(min(a_src, a_tgt), 1e-30)
        assert ratio <= ratio_max, (
            f"face {src} (area {a_src}) ↔ face {tgt} (area {a_tgt})"
            f" ratio {ratio:.2f} > {ratio_max}"
        )


def test_splice_face_geometry_unchanged(aegina_with_habitats):
    """Splicing must never touch node coordinates."""
    g, coords, ov_gdf = aegina_with_habitats
    original = coords.copy()
    _ = splice_attributes(g, coords, ov_gdf, key=b"\x77" * 32)
    assert np.array_equal(coords, original)


def test_splice_handles_empty_overlay(aegina_with_habitats):
    g, coords, _ = aegina_with_habitats
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    result = splice_attributes(g, coords, empty, key=b"\x88" * 32)
    assert result.overlays.empty
    # Rollups still computed
    assert len(result.face_rollups) > 0


def test_splice_distinguishable_from_identity(aegina_with_habitats):
    """At least one face's rollup should move to a different face."""
    g, coords, ov_gdf = aegina_with_habitats
    faces = extract_faces(g, coords)

    # Build input rollups explicitly for comparison
    input_rollups = {}
    for f_idx, face in enumerate(faces):
        vals = [g.nodes[n].get("elevation") for n in face]
        vals = [v for v in vals if v is not None]
        if vals:
            input_rollups[f_idx] = float(np.median(vals))

    result = splice_attributes(g, coords, ov_gdf, key=b"\x99" * 32,
                               rollup_fields=["elevation"])
    # Count faces whose rollup changed
    changed = sum(
        1 for f in input_rollups
        if abs(input_rollups[f]
               - float(result.face_rollups[f].get("elevation", np.inf))) > 1e-6
    )
    assert changed > 0, "splicing produced identity — no faces moved"
