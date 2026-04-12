"""Canonical labelling and graph hashing.

« Every graph gets one true name »

Computes a canonical form for the labelled planar graph so that:
1. The checksum is independent of input vertex ordering.
2. The keyed PRF produces identical output regardless of parse order.

Uses Weisfeiler-Lehman hashing (pure Python, via networkx) for the
proof-of-concept.  Optional nauty/Traces backend via pynauty for
production-grade canonical forms.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import networkx as nx


def weisfeiler_lehman_hash(
    g: nx.Graph,
    iterations: int = 5,
    node_attr: str | None = None,
    edge_attr: str | None = None,
) -> str:
    """Compute the Weisfeiler-Lehman graph hash.

    Args:
        g: Input graph.
        iterations: Number of WL refinement rounds.
        node_attr: Node attribute key to include in labels.
        edge_attr: Edge attribute key to include in labels.

    Returns:
        Hex digest string uniquely identifying the graph's
        isomorphism class (up to WL distinguishing power).
    """
    return nx.weisfeiler_lehman_graph_hash(
        g,
        iterations=iterations,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )


def canonical_node_order(g: nx.Graph) -> list[Any]:
    """Return a deterministic node ordering for the graph.

    Uses degree-sequence sorting with WL colour refinement as
    tie-breaker.  Sufficient for the proof-of-concept; for
    production, replace with nauty canonical labelling.

    Args:
        g: Input graph.

    Returns:
        List of node identifiers in canonical order.
    """
    # Compute WL subtree labels for each node
    wl_labels = _wl_node_labels(g, iterations=5)

    # Sort by (degree, wl_label, node_id) for determinism
    nodes_sorted = sorted(
        g.nodes,
        key=lambda n: (g.degree(n), wl_labels.get(n, ""), str(n)),
    )
    return nodes_sorted


def attribute_hash(g: nx.Graph) -> str:
    """Hash the full attribute payload of the graph.

    Serialises all node and edge attributes to a canonical JSON
    string and returns its SHA-256 hex digest.

    Args:
        g: Input graph with attributes.

    Returns:
        Hex digest of the attribute payload.
    """
    payload: dict[str, Any] = {
        "nodes": {
            str(n): _serialise_attrs(d) for n, d in sorted(g.nodes(data=True))
        },
        "edges": {
            f"{u}-{v}": _serialise_attrs(d)
            for u, v, d in sorted(g.edges(data=True))
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ─────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────


def _wl_node_labels(g: nx.Graph, iterations: int = 5) -> dict[Any, str]:
    """Compute per-node WL colour labels.

    Args:
        g: Input graph.
        iterations: Number of refinement rounds.

    Returns:
        Mapping from node → WL colour string.
    """
    labels = {n: str(g.degree(n)) for n in g.nodes}

    for _ in range(iterations):
        new_labels: dict[Any, str] = {}
        for n in g.nodes:
            neighbour_labels = sorted(labels[nb] for nb in g.neighbors(n))
            combined = labels[n] + "|" + ",".join(neighbour_labels)
            new_labels[n] = hashlib.md5(combined.encode()).hexdigest()[:8]
        labels = new_labels

    return labels


def _serialise_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert attribute dict values to JSON-serialisable types.

    Args:
        attrs: Raw attribute dictionary.

    Returns:
        Cleaned dictionary with all values as strings/numbers.
    """
    clean: dict[str, Any] = {}
    for k, v in sorted(attrs.items()):
        if isinstance(v, (int, float, str, bool, type(None))):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
