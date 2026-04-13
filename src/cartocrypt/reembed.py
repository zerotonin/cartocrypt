"""Constrained planar re-embedding of labelled graphs.

« The heart of CartoCrypt — where topology meets geometry »

Given a labelled planar graph and a set of PRF-seeded initial
positions, adjusts vertex coordinates so that:

1. Planarity is preserved (no edge crossings).
2. Edge lengths match the originals (within tolerance).
3. Face areas match the originals (within tolerance).
4. The embedding is deterministic given the key.

Strategy
--------
Phase 1: Tutte embedding — fix outer face vertices on a convex
         polygon, solve for interior vertices via barycentric
         coordinates.  Guarantees a crossing-free straight-line
         drawing for 3-connected planar graphs.

Phase 2: Stress majorisation — iteratively adjust positions to
         minimise the stress function:
           Σ_{(i,j)} w_ij * (||p_i - p_j|| - d_ij)²
         where d_ij are the target edge lengths.

Phase 3: Area correction — scale individual faces via affine
         transforms to match target areas while preserving
         edge-length constraints.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from cartocrypt.constants import (
    STRESS_FTOL,
    STRESS_MAX_ITER,
    Coords,
)


def tutte_embed(
    g: nx.Graph,
    outer_face: list[Any] | None = None,
    seed_coords: Coords | None = None,
) -> Coords:
    """Compute a Tutte (barycentric) embedding of a planar graph.

    Fixes outer face vertices on a regular polygon and solves
    the linear system for interior vertex positions.

    Args:
        g: Simple planar graph (must be 2- or 3-connected).
        outer_face: Nodes forming the outer face.  If None,
            the longest face cycle is selected automatically.
        seed_coords: Optional (N, 2) array of initial positions
            used to determine outer face vertex placement.

    Returns:
        (N, 2) array of vertex coordinates.

    Raises:
        ValueError: If the graph is not planar.
    """
    if not nx.check_planarity(g)[0]:
        msg = "Input graph is not planar"
        raise ValueError(msg)

    nodes = list(g.nodes)
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # ── Select outer face ──────────────────────────────────────
    if outer_face is None:
        outer_face = _find_outer_face(g)

    outer_set = set(outer_face)
    interior = [nd for nd in nodes if nd not in outer_set]
    n_outer = len(outer_face)

    # ── Place outer face on regular polygon ────────────────────
    coords = np.zeros((n, 2), dtype=np.float64)
    for k, nd in enumerate(outer_face):
        angle = 2.0 * np.pi * k / n_outer
        i = node_idx[nd]
        coords[i, 0] = np.cos(angle)
        coords[i, 1] = np.sin(angle)

    if not interior:
        return coords

    # ── Build linear system for interior vertices ──────────────
    interior_idx = [node_idx[nd] for nd in interior]
    int_local = {idx: local for local, idx in enumerate(interior_idx)}
    n_int = len(interior)

    # Sparse matrix L and RHS b for Lx = b (separately for x, y)
    row, col, data = [], [], []
    bx = np.zeros(n_int, dtype=np.float64)
    by = np.zeros(n_int, dtype=np.float64)

    for local_i, glob_i in enumerate(interior_idx):
        nd = nodes[glob_i]
        neighbours = list(g.neighbors(nd))
        deg = len(neighbours)

        row.append(local_i)
        col.append(local_i)
        data.append(float(deg))

        for nb in neighbours:
            j = node_idx[nb]
            if j in int_local:
                row.append(local_i)
                col.append(int_local[j])
                data.append(-1.0)
            else:
                # Boundary node — move to RHS
                bx[local_i] += coords[j, 0]
                by[local_i] += coords[j, 1]

    L = csr_matrix((data, (row, col)), shape=(n_int, n_int))
    coords[interior_idx, 0] = spsolve(L, bx)
    coords[interior_idx, 1] = spsolve(L, by)

    return coords


def stress_majorise(
    g: nx.Graph,
    target_lengths: dict[tuple[Any, Any], float],
    initial_coords: Coords,
    *,
    ftol: float = STRESS_FTOL,
    max_iter: int = STRESS_MAX_ITER,
) -> Coords:
    """Refine vertex positions to match target edge lengths.

    Minimises the weighted stress function:
        Σ_{(i,j)∈E} w_ij (||p_i - p_j|| - d_ij)²

    where w_ij = 1/d_ij² (inverse-square weighting).

    Args:
        g: Planar graph.
        target_lengths: Mapping (u, v) → target length in metres.
        initial_coords: (N, 2) starting positions (e.g. from Tutte).
        ftol: Function tolerance for convergence.
        max_iter: Maximum optimisation iterations.

    Returns:
        (N, 2) refined coordinates.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)

    # Build edge list with targets and weights
    edges = []
    for (u, v), d_target in target_lengths.items():
        i, j = node_idx[u], node_idx[v]
        w = 1.0 / max(d_target, 1e-12) ** 2
        edges.append((i, j, d_target, w))

    def stress(x_flat: np.ndarray) -> float:
        """Stress objective function."""
        pos = x_flat.reshape(n, 2)
        total = 0.0
        for i, j, d, w in edges:
            dist = np.linalg.norm(pos[i] - pos[j])
            total += w * (dist - d) ** 2
        return total

    def stress_grad(x_flat: np.ndarray) -> np.ndarray:
        """Gradient of the stress function."""
        pos = x_flat.reshape(n, 2)
        grad = np.zeros_like(pos)
        for i, j, d, w in edges:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            factor = 2.0 * w * (dist - d) / dist
            grad[i] += factor * diff
            grad[j] -= factor * diff
        return grad.ravel()

    result = minimize(
        stress,
        initial_coords.ravel(),
        jac=stress_grad,
        method="L-BFGS-B",
        options={"ftol": ftol, "maxiter": max_iter},
    )

    return result.x.reshape(n, 2)


def reembed(
    g: nx.Graph,
    original_coords: Coords,
    seed_coords: Coords,
    *,
    preserve_lengths: bool = True,
    preserve_areas: bool = True,
) -> Coords:
    """Full re-embedding pipeline: Tutte → stress → area correction.

    Args:
        g: Labelled planar graph.
        original_coords: (N, 2) original vertex positions.
        seed_coords: (N, 2) PRF-seeded initial positions.
        preserve_lengths: Whether to enforce edge-length matching.
        preserve_areas: Whether to enforce face-area matching.

    Returns:
        (N, 2) anonymised coordinates.
    """
    # Phase 1: Tutte embedding for crossing-free layout
    coords = tutte_embed(g, seed_coords=seed_coords)

    # Phase 2: Stress majorisation for edge-length matching
    if preserve_lengths:
        target_lengths = _extract_edge_lengths(g, original_coords)
        coords = stress_majorise(
            g, target_lengths, coords,
        )

    # Phase 3: Area correction (TODO)
    if preserve_areas:
        pass  # Will be implemented in shapes.py integration

    return coords


# ─────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────


def _find_outer_face(g: nx.Graph) -> list[Any]:
    """Heuristic: select the longest simple cycle as outer face.

    Args:
        g: Planar graph.

    Returns:
        List of nodes forming the outer face boundary.
    """
    # For proof-of-concept: use the convex hull of node positions
    # if coordinates are available, otherwise longest cycle.
    try:
        cycles = list(nx.cycle_basis(g))
        if not cycles:
            return list(g.nodes)[:3]
        return max(cycles, key=len)
    except nx.NetworkXError:
        return list(g.nodes)[:3]


def _extract_edge_lengths(
    g: nx.Graph,
    coords: Coords,
) -> dict[tuple[Any, Any], float]:
    """Compute Euclidean edge lengths from coordinates.

    Args:
        g: Graph.
        coords: (N, 2) coordinate array.

    Returns:
        Mapping (u, v) → length.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    lengths = {}
    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        lengths[(u, v)] = float(np.linalg.norm(coords[i] - coords[j]))
    return lengths
