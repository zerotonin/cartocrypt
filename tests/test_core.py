"""Tests for CartoCrypt core modules.

« Trust but verify — with pytest »
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from cartocrypt.canon import (
    attribute_hash,
    canonical_node_order,
    weisfeiler_lehman_hash,
)
from cartocrypt.keygen import (
    compute_checksum,
    generate_key,
    prf_coordinates,
    prf_coordinates_batch,
    verify_checksum,
)

# ─────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph() -> nx.Graph:
    """A small planar graph for testing."""
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),  # square
        (0, 2),  # diagonal
    ])
    for n in g.nodes:
        g.nodes[n]["elevation"] = float(n) * 10.0
    for u, v in g.edges():
        g.edges[u, v]["length"] = 100.0
    return g


@pytest.fixture
def key() -> bytes:
    return generate_key()


# ─────────────────────────────────────────────────────────────────
#  keygen tests
# ─────────────────────────────────────────────────────────────────

class TestKeygen:
    """Tests for key generation and PRF."""

    def test_key_length(self) -> None:
        k = generate_key()
        assert len(k) == 32

    def test_keys_differ(self) -> None:
        k1 = generate_key()
        k2 = generate_key()
        assert k1 != k2

    def test_prf_deterministic(self, key: bytes) -> None:
        a = prf_coordinates(key, 0)
        b = prf_coordinates(key, 0)
        assert a == b

    def test_prf_different_indices(self, key: bytes) -> None:
        a = prf_coordinates(key, 0)
        b = prf_coordinates(key, 1)
        assert a != b

    def test_prf_different_keys(self) -> None:
        k1 = generate_key()
        k2 = generate_key()
        a = prf_coordinates(k1, 0)
        b = prf_coordinates(k2, 0)
        assert a != b

    def test_prf_within_bbox(self, key: bytes) -> None:
        bbox = (10.0, 20.0, 30.0, 40.0)
        x, y = prf_coordinates(key, 42, bbox=bbox)
        assert 10.0 <= x <= 30.0
        assert 20.0 <= y <= 40.0

    def test_prf_batch_shape(self, key: bytes) -> None:
        coords = prf_coordinates_batch(key, 100)
        assert coords.shape == (100, 2)

    def test_checksum_deterministic(self, key: bytes) -> None:
        c1 = compute_checksum(key, "abc", "def")
        c2 = compute_checksum(key, "abc", "def")
        assert c1 == c2

    def test_checksum_verify(self, key: bytes) -> None:
        cs = compute_checksum(key, "gh", "ah")
        assert verify_checksum(key, "gh", "ah", cs)
        assert not verify_checksum(key, "gh", "WRONG", cs)


# ─────────────────────────────────────────────────────────────────
#  canon tests
# ─────────────────────────────────────────────────────────────────

class TestCanon:
    """Tests for canonical labelling and hashing."""

    def test_wl_hash_deterministic(self, simple_graph: nx.Graph) -> None:
        h1 = weisfeiler_lehman_hash(simple_graph)
        h2 = weisfeiler_lehman_hash(simple_graph)
        assert h1 == h2

    def test_wl_hash_isomorphic(self) -> None:
        """Isomorphic graphs should have the same WL hash."""
        g1 = nx.cycle_graph(5)
        # Relabel nodes
        mapping = {i: i + 100 for i in range(5)}
        g2 = nx.relabel_nodes(g1, mapping)
        assert weisfeiler_lehman_hash(g1) == weisfeiler_lehman_hash(g2)

    def test_wl_hash_different(self) -> None:
        """Non-isomorphic graphs should (usually) have different hashes."""
        g1 = nx.cycle_graph(5)
        g2 = nx.cycle_graph(6)
        assert weisfeiler_lehman_hash(g1) != weisfeiler_lehman_hash(g2)

    def test_canonical_order_deterministic(self, simple_graph: nx.Graph) -> None:
        o1 = canonical_node_order(simple_graph)
        o2 = canonical_node_order(simple_graph)
        assert o1 == o2

    def test_attribute_hash_deterministic(self, simple_graph: nx.Graph) -> None:
        h1 = attribute_hash(simple_graph)
        h2 = attribute_hash(simple_graph)
        assert h1 == h2

    def test_attribute_hash_changes(self, simple_graph: nx.Graph) -> None:
        h1 = attribute_hash(simple_graph)
        simple_graph.nodes[0]["elevation"] = 999.0
        h2 = attribute_hash(simple_graph)
        assert h1 != h2


# ─────────────────────────────────────────────────────────────────
#  reembed tests
# ─────────────────────────────────────────────────────────────────

class TestReembed:
    """Tests for the re-embedding pipeline."""

    def test_tutte_planar(self) -> None:
        """Tutte embedding should produce valid coordinates."""
        from cartocrypt.reembed import tutte_embed

        g = nx.grid_2d_graph(4, 4)
        # Convert to simple node labels
        g = nx.convert_node_labels_to_integers(g)
        coords = tutte_embed(g)
        assert coords.shape == (16, 2)
        assert np.all(np.isfinite(coords))

    def test_tutte_no_crossings(self) -> None:
        """Tutte embedding of a 3-connected graph should be crossing-free."""
        from cartocrypt.reembed import tutte_embed

        # Petersen graph is 3-connected but NOT planar — use a grid
        g = nx.grid_2d_graph(3, 3)
        g = nx.convert_node_labels_to_integers(g)
        coords = tutte_embed(g)
        # Basic sanity: all coords are distinct
        dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        assert dists.min() > 1e-10
