"""Face-area preservation tests for the Phase 3 reembed path."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from cartocrypt.constants import AREA_RTOL
from cartocrypt.faces import (
    extract_faces,
    face_area_residuals,
    face_areas,
    face_areas_signed,
)
from cartocrypt.keygen import prf_coordinates_batch
from cartocrypt.reembed import reembed
from cartocrypt.verify import verify_metrics

from tests.aegina_fixture import make_aegina_graph


@pytest.fixture
def aegina():
    g, coords, meta = make_aegina_graph()
    return g, coords, meta


# ───── face extraction ────────────────────────────────────────────

def test_face_extraction_aegina_fixture(aegina):
    g, coords, _ = aegina
    faces = extract_faces(g, coords)
    # Euler's formula on a connected planar graph: V - E + F = 2.
    # Our extractor drops the outer face, so expected interior
    # count is E - V + 1 for a connected graph.
    expected = g.number_of_edges() - g.number_of_nodes() + 1
    assert len(faces) == expected
    # Every face has ≥ 3 vertices.
    assert all(len(f) >= 3 for f in faces)


def test_face_extraction_requires_planar():
    import networkx as nx
    g = nx.complete_graph(5)  # K5 is not planar
    coords = np.random.default_rng(0).random((5, 2))
    with pytest.raises(ValueError, match="planar"):
        extract_faces(g, coords)


# ───── Shoelace vs shapely ────────────────────────────────────────

def test_shoelace_matches_shapely():
    rng = np.random.default_rng(42)
    for _ in range(12):
        n = rng.integers(3, 10)
        theta = np.sort(rng.uniform(0, 2 * np.pi, size=n))
        r = rng.uniform(0.5, 1.5, size=n)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        shoelace = float(face_areas(pts, [list(range(n))])[0])
        shapely_area = Polygon(pts).area
        assert abs(shoelace - shapely_area) < 1e-9


def test_signed_vs_unsigned_area():
    square_ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    faces = [[0, 1, 2, 3]]
    assert face_areas_signed(square_ccw, faces)[0] == pytest.approx(1.0)
    assert face_areas(square_ccw, faces)[0] == pytest.approx(1.0)
    # Reverse orientation → signed flips, unsigned unchanged.
    square_cw = square_ccw[::-1]
    faces_cw = [[0, 1, 2, 3]]
    assert face_areas_signed(square_cw, faces_cw)[0] == pytest.approx(-1.0)
    assert face_areas(square_cw, faces_cw)[0] == pytest.approx(1.0)


# ───── reembed preserves areas ────────────────────────────────────

def _reembed_once(g, coords, key):
    seed = prf_coordinates_batch(
        key, g.number_of_nodes(),
        bbox=(float(coords[:, 0].min()), float(coords[:, 1].min()),
              float(coords[:, 0].max()), float(coords[:, 1].max())),
    )
    return reembed(g, coords, seed, preserve_lengths=True, preserve_areas=True)


def test_reembed_preserves_areas_median_5pct(aegina):
    g, coords, _ = aegina
    anon = _reembed_once(g, coords, b"\x11" * 32)
    faces = extract_faces(g, coords)
    stats = face_area_residuals(coords, anon, faces)
    # Median relative face-area error stays below the 5 % design tolerance.
    assert stats["median_rel"] < AREA_RTOL
    assert stats["n_faces"] > 0


def test_reembed_preserves_edge_lengths(aegina):
    g, coords, _ = aegina
    anon = _reembed_once(g, coords, b"\x22" * 32)
    metrics = verify_metrics(g, coords, anon)
    # Mean relative edge-length error is small (the stress term
    # dominates the objective on this graph size).
    assert metrics["length_mean_rel_error"] < 0.30
    assert metrics["n_edges"] == g.number_of_edges()


def test_reembed_deterministic_under_key(aegina):
    g, coords, _ = aegina
    a = _reembed_once(g, coords, b"\x33" * 32)
    b = _reembed_once(g, coords, b"\x33" * 32)
    assert np.allclose(a, b, atol=1e-10)


def test_reembed_no_areas_flag(aegina):
    g, coords, _ = aegina
    seed = prf_coordinates_batch(
        b"\x44" * 32, g.number_of_nodes(),
        bbox=(float(coords[:, 0].min()), float(coords[:, 1].min()),
              float(coords[:, 0].max()), float(coords[:, 1].max())),
    )
    # Should run without touching faces at all.
    anon = reembed(g, coords, seed,
                   preserve_lengths=True, preserve_areas=False)
    assert anon.shape == coords.shape


# ───── verify_metrics carries area stats ──────────────────────────

def test_verify_metrics_reports_area_stats(aegina):
    g, coords, _ = aegina
    anon = _reembed_once(g, coords, b"\x55" * 32)
    m = verify_metrics(g, coords, anon)
    for key in ("n_faces", "area_median_rel_error",
                "area_max_rel_error", "area_p95_rel_error",
                "area_within_tol"):
        assert key in m
    assert m["n_faces"] > 0
    assert m["area_within_tol"] is True
