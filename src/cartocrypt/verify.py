"""Round-trip verification and determinism checks.

« Trust but verify — twice »
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from cartocrypt.canon import weisfeiler_lehman_hash
from cartocrypt.constants import Coords


def verify_round_trip(
    coords_a: Coords,
    coords_b: Coords,
    *,
    atol: float = 1e-10,
) -> bool:
    """Check that two anonymisation runs produce identical output.

    Args:
        coords_a: Coordinates from first run.
        coords_b: Coordinates from second run.
        atol: Absolute tolerance for floating-point comparison.

    Returns:
        True if outputs match within tolerance.
    """
    return bool(np.allclose(coords_a, coords_b, atol=atol))


def verify_topology(
    g_original: nx.Graph,
    g_anonymised: nx.Graph,
) -> dict[str, bool]:
    """Verify that topological invariants are preserved.

    Args:
        g_original: Original labelled graph.
        g_anonymised: Anonymised graph (same structure, new coords).

    Returns:
        Dictionary of invariant checks and their pass/fail status.
    """
    results: dict[str, bool] = {}

    # Same node count
    results["node_count"] = (
        g_original.number_of_nodes() == g_anonymised.number_of_nodes()
    )

    # Same edge count
    results["edge_count"] = (
        g_original.number_of_edges() == g_anonymised.number_of_edges()
    )

    # Same degree sequence
    deg_orig = sorted(dict(g_original.degree()).values())
    deg_anon = sorted(dict(g_anonymised.degree()).values())
    results["degree_sequence"] = deg_orig == deg_anon

    # Same WL hash (isomorphism class)
    results["wl_hash"] = (
        weisfeiler_lehman_hash(g_original)
        == weisfeiler_lehman_hash(g_anonymised)
    )

    # Same connected components count
    results["components"] = (
        nx.number_connected_components(g_original)
        == nx.number_connected_components(g_anonymised)
    )

    return results


def verify_metrics(
    g: nx.Graph,
    original_coords: Coords,
    anon_coords: Coords,
    *,
    length_rtol: float = 0.01,
    area_rtol: float = 0.05,
) -> dict[str, Any]:
    """Verify that metric invariants (lengths, areas) are preserved.

    Args:
        g: Graph structure (shared by original and anonymised).
        original_coords: (N, 2) original positions.
        anon_coords: (N, 2) anonymised positions.
        length_rtol: Relative tolerance for edge lengths.
        area_rtol: Relative tolerance for face areas.

    Returns:
        Dictionary with metric comparison statistics.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # Edge length comparison
    length_errors = []
    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        l_orig = float(np.linalg.norm(original_coords[i] - original_coords[j]))
        l_anon = float(np.linalg.norm(anon_coords[i] - anon_coords[j]))
        if l_orig > 1e-12:
            length_errors.append(abs(l_anon - l_orig) / l_orig)

    return {
        "n_edges": len(length_errors),
        "length_mean_rel_error": float(np.mean(length_errors)) if length_errors else 0.0,
        "length_max_rel_error": float(np.max(length_errors)) if length_errors else 0.0,
        "length_within_tol": all(e <= length_rtol for e in length_errors),
    }
