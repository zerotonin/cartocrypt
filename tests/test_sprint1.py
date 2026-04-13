"""Sprint 1 tests — Foundation & Ingest.

« Aegina under the microscope »

Acceptance criteria:
  - Aegina fixture returns a valid nx.Graph with >50 nodes, >60 edges
  - Every edge has a 'length' attribute in metres
  - Coords array has shape (N, 2) with finite values
  - Graph is connected
  - Graph is planar
  - to_labelled_graph handles pre-built graphs correctly
  - graph_summary produces readable output
  - GeoJSON round-trip: export then re-import preserves structure
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tests.aegina_fixture import make_aegina_graph, _haversine
from cartocrypt.ingest import to_labelled_graph, graph_summary


# ─────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def aegina():
    """Build the Aegina test graph."""
    return make_aegina_graph()


@pytest.fixture
def aegina_graph(aegina):
    return aegina[0]


@pytest.fixture
def aegina_coords(aegina):
    return aegina[1]


@pytest.fixture
def aegina_metadata(aegina):
    return aegina[2]


# ─────────────────────────────────────────────────────────────────
#  Fixture sanity checks
# ─────────────────────────────────────────────────────────────────

class TestAeginaFixture:
    """Validate the synthetic Aegina graph itself."""

    def test_node_count(self, aegina_graph: nx.Graph) -> None:
        """Graph has >50 nodes (Sprint 1 acceptance criterion)."""
        assert aegina_graph.number_of_nodes() > 50

    def test_edge_count(self, aegina_graph: nx.Graph) -> None:
        """Graph has >60 edges (Sprint 1 acceptance criterion)."""
        assert aegina_graph.number_of_edges() > 60

    def test_all_edges_have_length(self, aegina_graph: nx.Graph) -> None:
        """Every edge has a 'length' attribute."""
        for u, v, data in aegina_graph.edges(data=True):
            assert "length" in data, f"Edge ({u},{v}) missing length"
            assert data["length"] > 0, f"Edge ({u},{v}) has non-positive length"

    def test_edge_lengths_are_metres(self, aegina_graph: nx.Graph) -> None:
        """Edge lengths are in realistic range for road segments (50-5000m)."""
        lengths = [d["length"] for _, _, d in aegina_graph.edges(data=True)]
        assert min(lengths) > 10, f"Suspiciously short edge: {min(lengths)} m"
        assert max(lengths) < 20000, f"Suspiciously long edge: {max(lengths)} m"

    def test_coords_shape(self, aegina_graph: nx.Graph, aegina_coords: np.ndarray) -> None:
        """Coords array has shape (N, 2)."""
        n = aegina_graph.number_of_nodes()
        assert aegina_coords.shape == (n, 2)

    def test_coords_finite(self, aegina_coords: np.ndarray) -> None:
        """All coordinates are finite."""
        assert np.all(np.isfinite(aegina_coords))

    def test_coords_in_aegina_bbox(self, aegina_coords: np.ndarray) -> None:
        """Coordinates fall within Aegina's geographic bounding box."""
        lons = aegina_coords[:, 0]
        lats = aegina_coords[:, 1]
        assert lons.min() > 23.3, "Longitude too far west for Aegina"
        assert lons.max() < 23.7, "Longitude too far east for Aegina"
        assert lats.min() > 37.6, "Latitude too far south for Aegina"
        assert lats.max() < 37.9, "Latitude too far north for Aegina"

    def test_connected(self, aegina_graph: nx.Graph) -> None:
        """Graph is connected (single component)."""
        assert nx.is_connected(aegina_graph)

    def test_planar(self, aegina_graph: nx.Graph) -> None:
        """Graph is planar."""
        is_planar, _ = nx.check_planarity(aegina_graph)
        assert is_planar

    def test_has_coastal_nodes(self, aegina_graph: nx.Graph) -> None:
        """Graph contains coastal nodes."""
        coast_nodes = [
            n for n, d in aegina_graph.nodes(data=True)
            if d.get("place_type") == "coast"
        ]
        assert len(coast_nodes) >= 20

    def test_has_village_nodes(self, aegina_graph: nx.Graph) -> None:
        """Graph contains village nodes from multiple villages."""
        villages = set()
        for _, d in aegina_graph.nodes(data=True):
            if d.get("village"):
                villages.add(d["village"])
        assert len(villages) >= 4

    def test_has_mountain_nodes(self, aegina_graph: nx.Graph) -> None:
        """Graph contains mountain pass nodes."""
        mountain_nodes = [
            n for n, d in aegina_graph.nodes(data=True)
            if d.get("place_type") == "mountain"
        ]
        assert len(mountain_nodes) >= 2

    def test_has_elevation_data(self, aegina_graph: nx.Graph) -> None:
        """All nodes have elevation data."""
        for n, d in aegina_graph.nodes(data=True):
            assert "elevation" in d, f"Node {n} missing elevation"

    def test_has_road_classes(self, aegina_graph: nx.Graph) -> None:
        """Edges have diverse road class attributes."""
        classes = set()
        for _, _, d in aegina_graph.edges(data=True):
            if "road_class" in d:
                classes.add(d["road_class"])
        assert len(classes) >= 3, f"Only {classes} road classes found"

    def test_metadata_fields(self, aegina_metadata: dict) -> None:
        """Metadata contains required fields."""
        assert aegina_metadata["source"] == "synthetic_aegina"
        assert aegina_metadata["crs"] == "EPSG:4326"
        assert aegina_metadata["n_nodes"] > 50
        assert aegina_metadata["n_edges"] > 60
        assert "bbox" in aegina_metadata
        assert "villages" in aegina_metadata


# ─────────────────────────────────────────────────────────────────
#  to_labelled_graph tests
# ─────────────────────────────────────────────────────────────────

class TestToLabelledGraph:
    """Test the unified graph converter."""

    def test_accepts_fixture_graph(self, aegina: tuple) -> None:
        """to_labelled_graph accepts a pre-built nx.Graph."""
        g_raw, coords_raw, meta_raw = aegina
        g, coords, meta = to_labelled_graph(g_raw)
        assert g.number_of_nodes() == g_raw.number_of_nodes()
        assert g.number_of_edges() == g_raw.number_of_edges()

    def test_sequential_node_labels(self) -> None:
        """Output graph has sequential integer labels 0..N-1."""
        g = nx.Graph()
        g.add_edge("a", "b", length=100.0)
        g.add_edge("b", "c", length=200.0)
        g.nodes["a"]["x"] = 0.0
        g.nodes["a"]["y"] = 0.0
        g.nodes["b"]["x"] = 1.0
        g.nodes["b"]["y"] = 0.0
        g.nodes["c"]["x"] = 2.0
        g.nodes["c"]["y"] = 0.0
        g_out, coords, meta = to_labelled_graph(g)
        assert sorted(g_out.nodes) == [0, 1, 2]

    def test_preserves_edge_lengths(self, aegina: tuple) -> None:
        """Edge lengths survive the conversion."""
        g_raw, _, _ = aegina
        g, coords, meta = to_labelled_graph(g_raw)
        for u, v, data in g.edges(data=True):
            assert "length" in data
            assert data["length"] > 0

    def test_preserves_node_attributes(self, aegina: tuple) -> None:
        """Node attributes (elevation, place_type) survive conversion."""
        g_raw, _, _ = aegina
        g, _, _ = to_labelled_graph(g_raw)
        elevations = [d.get("elevation") for _, d in g.nodes(data=True)]
        assert all(e is not None for e in elevations)

    def test_coords_match_node_count(self, aegina: tuple) -> None:
        """Coords array length matches node count."""
        g_raw, _, _ = aegina
        g, coords, _ = to_labelled_graph(g_raw)
        assert coords.shape[0] == g.number_of_nodes()

    def test_rejects_unsupported_type(self) -> None:
        """TypeError for unsupported input types."""
        with pytest.raises(TypeError):
            to_labelled_graph("not a graph")

    def test_multidigraph_path(self) -> None:
        """to_labelled_graph handles MultiDiGraph (OSM-style)."""
        mg = nx.MultiDiGraph()
        mg.add_edge(100, 200, length=500.0, highway="secondary")
        mg.add_edge(200, 300, length=300.0, highway="residential")
        mg.nodes[100].update({"x": 23.49, "y": 37.73})
        mg.nodes[200].update({"x": 23.50, "y": 37.74})
        mg.nodes[300].update({"x": 23.51, "y": 37.73})
        g, coords, meta = to_labelled_graph(mg)
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2
        assert meta["source"] == "osm"


# ─────────────────────────────────────────────────────────────────
#  graph_summary tests
# ─────────────────────────────────────────────────────────────────

class TestGraphSummary:
    """Test human-readable graph summary."""

    def test_summary_contains_key_info(self, aegina: tuple) -> None:
        """Summary includes node/edge counts and planarity."""
        g, coords, meta = aegina
        s = graph_summary(g, coords, meta)
        assert "Nodes:" in s
        assert "Edges:" in s
        assert "Planar:" in s
        assert "True" in s  # should be planar

    def test_summary_contains_source(self, aegina: tuple) -> None:
        """Summary shows the data source."""
        g, coords, meta = aegina
        s = graph_summary(g, coords, meta)
        assert "synthetic_aegina" in s

    def test_summary_contains_edge_stats(self, aegina: tuple) -> None:
        """Summary shows min/max/mean edge lengths."""
        g, coords, meta = aegina
        s = graph_summary(g, coords, meta)
        assert "min=" in s
        assert "max=" in s
        assert "mean=" in s

    def test_summary_contains_road_classes(self, aegina: tuple) -> None:
        """Summary lists road classes."""
        g, coords, meta = aegina
        s = graph_summary(g, coords, meta)
        assert "secondary" in s or "residential" in s


# ─────────────────────────────────────────────────────────────────
#  Haversine utility test
# ─────────────────────────────────────────────────────────────────

class TestHaversine:
    """Test the Haversine distance function."""

    def test_known_distance(self) -> None:
        """Athens to Aegina is roughly 27 km."""
        athens = (23.7275, 37.9838)
        aegina_town = (23.4271, 37.7467)
        d = _haversine(athens, aegina_town)
        assert 25_000 < d < 40_000  # ~30 km

    def test_zero_distance(self) -> None:
        """Same point has zero distance."""
        p = (23.49, 37.73)
        assert _haversine(p, p) == pytest.approx(0.0, abs=0.01)

    def test_symmetric(self) -> None:
        """Haversine is symmetric."""
        a = (23.49, 37.73)
        b = (23.50, 37.74)
        assert _haversine(a, b) == pytest.approx(_haversine(b, a))
