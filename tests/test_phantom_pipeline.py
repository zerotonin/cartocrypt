"""Integration tests for the phantom-Aegina pipeline."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from cartocrypt.faces import extract_faces
from cartocrypt.fake_layers import (
    make_futurama_tubes,
    render_tubes_from_stations,
)
from cartocrypt.keygen import prf_coordinates_batch
from cartocrypt.overlay import add_layer, attach_settlements, polygons_as_gdf
from cartocrypt.reembed import (
    _find_outer_face,
    _resample_boundary,
    reembed_onto,
    tutte_embed,
)
from cartocrypt.splicing import reproject_to_embedding, splice_attributes
from tests.aegina_fixture import make_aegina_graph

# ───── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def aegina_with_habitats():
    g, coords, meta = make_aegina_graph()
    habitats = gpd.GeoDataFrame({
        "species": ["A", "B", "C"],
        "geometry": [
            Polygon([(23.46, 37.72), (23.49, 37.72),
                     (23.49, 37.74), (23.46, 37.74)]),
            Polygon([(23.50, 37.72), (23.52, 37.72),
                     (23.52, 37.74), (23.50, 37.74)]),
            Polygon([(23.47, 37.75), (23.50, 37.75),
                     (23.50, 37.77), (23.47, 37.77)]),
        ],
    }, crs="EPSG:4326")
    add_layer(g, coords, habitats, "habitat")
    ov_gdf = polygons_as_gdf(g)
    return g, coords, ov_gdf


# ───── Outer-boundary Tutte ─────────────────────────────────

def test_tutte_accepts_outer_boundary(aegina_with_habitats):
    g, _coords, _ = aegina_with_habitats
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    out = tutte_embed(g, outer_boundary=square)
    assert out.shape == (g.number_of_nodes(), 2)
    # Outer face nodes must lie on the square's axis-aligned bbox
    assert out[:, 0].min() >= -1e-9 and out[:, 0].max() <= 1 + 1e-9
    assert out[:, 1].min() >= -1e-9 and out[:, 1].max() <= 1 + 1e-9


def test_resample_boundary_length_matches_request(aegina_with_habitats):
    g, _coords, _ = aegina_with_habitats
    face = _find_outer_face(g)
    boundary = np.random.default_rng(0).random((500, 2))
    resampled = _resample_boundary(boundary, len(face))
    assert resampled.shape == (len(face), 2)


def test_tutte_without_boundary_stays_on_unit_circle(aegina_with_habitats):
    g, _coords, _ = aegina_with_habitats
    out = tutte_embed(g)
    face = _find_outer_face(g)
    node_list = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(node_list)}
    for nd in face:
        x, y = out[node_idx[nd]]
        assert abs(np.hypot(x, y) - 1.0) < 1e-9


def test_reembed_onto_interior_inside_boundary(aegina_with_habitats):
    g, coords, _ = aegina_with_habitats
    # Use the coords' convex hull as the phantom boundary — guaranteed
    # to contain all nodes after Tutte.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(coords)
    boundary = coords[hull.vertices]
    key = b"\x11" * 32
    seed = prf_coordinates_batch(key, g.number_of_nodes())
    anon = reembed_onto(
        g, coords, seed, boundary,
        preserve_lengths=False, preserve_areas=False,
    )
    poly = Polygon(boundary)
    # Tutte guarantees interior points lie within the convex hull of
    # outer-face vertex positions for the barycentric system.
    for pt in anon:
        p = Point(float(pt[0]), float(pt[1]))
        assert poly.contains(p) or poly.touches(p) or poly.distance(p) < 1e-6


# ───── Settlement attachment ────────────────────────────────

def test_attach_settlements_round_trip(aegina_with_habitats):
    g, coords, _ = aegina_with_habitats

    class _Stub:
        pass

    stub = _Stub()
    stub.settlements = gpd.GeoDataFrame({
        "name": ["Town-A", "Town-B"],
        "geometry": [
            Point(coords[5, 0] + 0.0003, coords[5, 1]),
            Point(coords[20, 0], coords[20, 1] + 0.0003),
        ],
    }, crs="EPSG:4326")

    mapping = attach_settlements(g, coords, stub)
    assert len(mapping) == 2
    # Node 5's position is within snap tolerance of Town-A's Point
    node_for_a = mapping[0]
    dist = np.hypot(
        coords[node_for_a, 0] - (coords[5, 0] + 0.0003),
        coords[node_for_a, 1] - coords[5, 1],
    )
    assert dist < 0.001


# ───── Tube rendering parity ────────────────────────────────

def test_render_tubes_from_stations_deterministic():
    stations = gpd.GeoDataFrame({
        "name": ["A", "B", "C", "D"],
        "geometry": [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
    }, crs="EPSG:4326")
    key = b"\x22" * 32
    a = render_tubes_from_stations(stations, key)
    b = render_tubes_from_stations(stations, key)
    assert list(a["station_a"]) == list(b["station_a"])
    assert list(a["station_b"]) == list(b["station_b"])
    for ga, gb in zip(a.geometry, b.geometry, strict=True):
        assert ga.equals_exact(gb, tolerance=1e-9)


def test_render_tubes_matches_make_futurama_tubes():
    """`make_futurama_tubes` wraps `render_tubes_from_stations`."""

    class _Bundle:
        pass

    bundle = _Bundle()
    bundle.settlements = gpd.GeoDataFrame({
        "name": ["X", "Y", "Z"],
        "geometry": [Point(23.45, 37.73),
                     Point(23.50, 37.75),
                     Point(23.48, 37.70)],
    }, crs="EPSG:4326")
    bundle.bbox = (37.78, 37.67, 23.58, 23.40)
    key = b"\x33" * 32

    via_wrapper = make_futurama_tubes(bundle, key)
    # Without the metre projector, distances are in degrees — a
    # different absolute scale, but the connection set should be
    # the same.  Strip metres-dependent fields and compare.
    via_direct = render_tubes_from_stations(bundle.settlements, key)
    assert set(via_wrapper["station_a"]) == set(via_direct["station_a"])
    assert set(via_wrapper["station_b"]) == set(via_direct["station_b"])


# ───── Spliced-overlay reprojection ─────────────────────────

def test_reproject_to_embedding_preserves_fragment_count(
    aegina_with_habitats,
):
    g, coords, ov_gdf = aegina_with_habitats
    faces = extract_faces(g, coords)
    spliced = splice_attributes(g, coords, ov_gdf, key=b"\x44" * 32)

    # Make a trivial "new embedding" that is coords scaled by 2
    # centred at the original centroid.
    centre = coords.mean(axis=0)
    dst = (coords - centre) * 2.0 + centre

    phantom = reproject_to_embedding(spliced, faces, coords, dst)
    assert len(phantom) == len(spliced.overlays)
    # All output geometries should still be polygons with positive area
    assert all(g_.area > 0 for g_ in phantom.geometry
               if g_ is not None and not g_.is_empty)


def test_reproject_identity_when_embeddings_match(aegina_with_habitats):
    g, coords, ov_gdf = aegina_with_habitats
    faces = extract_faces(g, coords)
    spliced = splice_attributes(g, coords, ov_gdf, key=b"\x55" * 32)
    same = reproject_to_embedding(spliced, faces, coords, coords)
    for ga, gb in zip(same.geometry, spliced.overlays.geometry, strict=True):
        if ga is None or gb is None:
            continue
        # Same embedding → the affine is identity, so geometries
        # round-trip within float tolerance.
        assert ga.equals_exact(gb, tolerance=1e-6)
