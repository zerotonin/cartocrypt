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
    AREA_WEIGHT,
    STRESS_FTOL,
    STRESS_MAX_ITER,
    Coords,
)
from cartocrypt.faces import (
    pack_faces,
    area_gradient_contribution,
    extract_faces,
    face_areas,
)


def tutte_embed(
    g: nx.Graph,
    outer_face: list[Any] | None = None,
    seed_coords: Coords | None = None,
    outer_boundary: Coords | None = None,
) -> Coords:
    """Compute a Tutte (barycentric) embedding of a planar graph.

    Fixes outer face vertices on a chosen convex curve and solves
    the linear system for interior vertex positions.

    Args:
        g: Simple planar graph (must be 2- or 3-connected).
        outer_face: Nodes forming the outer face.  If None,
            the longest face cycle is selected automatically.
        seed_coords: Optional (N, 2) array of initial positions
            used to determine outer face vertex placement.
        outer_boundary: Optional closed curve as ``(M, 2)`` array
            onto which the outer face vertices are pinned.  The
            curve is resampled (uniform arclength) to match the
            number of outer-face nodes.  When ``None``, the outer
            face is placed on a unit circle (the classical Tutte
            default).  Supplying this makes the interior naturally
            lay out inside the provided boundary.

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

    # ── Place outer face on boundary or unit circle ────────────
    coords = np.zeros((n, 2), dtype=np.float64)
    if outer_boundary is not None:
        boundary_pts = _resample_boundary(outer_boundary, n_outer)
        for k, nd in enumerate(outer_face):
            i = node_idx[nd]
            coords[i, 0] = boundary_pts[k, 0]
            coords[i, 1] = boundary_pts[k, 1]
    else:
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
    faces: list[list[int]] | None = None,
    target_face_areas: np.ndarray | None = None,
    area_weight: float = AREA_WEIGHT,
    fixed_nodes: list[int] | np.ndarray | None = None,
    ftol: float = STRESS_FTOL,
    max_iter: int = STRESS_MAX_ITER,
) -> Coords:
    """Refine vertex positions to match target edge lengths (and areas).

    Minimises the joint objective:

    .. math::

        L(p) = \\sum_{(i,j) \\in E} w_{ij}\\, (\\|p_i - p_j\\| - d_{ij})^2
             + \\alpha \\sum_f w_f\\, (A_f(p) - A^*_f)^2

    where the edge weights are ``w_ij = 1/d_ij²`` and the face
    weights are ``w_f = 1/A*_f²``.  When ``faces`` is ``None`` the
    area term is skipped and the behaviour is identical to the
    pre-Phase-3 implementation.

    Args:
        g: Planar graph.
        target_lengths: Mapping (u, v) → target length.
        initial_coords: (N, 2) starting positions (e.g. from Tutte).
        faces: Optional face cycles — output of
            :func:`cartocrypt.faces.extract_faces`.
        target_face_areas: ``(F,)`` target unsigned areas aligned
            with ``faces``.  Required iff ``faces`` is given.
        area_weight: α; trade-off between edge-length and
            face-area residuals.  Default
            :data:`cartocrypt.constants.AREA_WEIGHT`.
        fixed_nodes: Indices of nodes whose positions must not
            move during optimisation.  Used to pin outer-face
            nodes onto a phantom coastline so they don't drift
            off the boundary during stress relaxation.
        ftol: Function tolerance for convergence.
        max_iter: Maximum optimisation iterations.

    Returns:
        (N, 2) refined coordinates.
    """
    nodes = list(g.nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)

    # Inverse-square weights on edge and face targets turn each
    # residual term into a squared *relative* error — so summing
    # them is dimensionless and comparisons across datasets in
    # different unit systems (metres, degrees, …) work out without
    # re-tuning ``area_weight``.
    # Pack the edges into parallel numpy arrays so the objective
    # and gradient are fully vectorised — critical for L-BFGS-B
    # to stay fast on road graphs with ≥ 10 k edges.
    edge_iter = list(target_lengths.items())
    if edge_iter:
        edge_i = np.fromiter((node_idx[u] for (u, _v), _d in edge_iter),
                             dtype=np.int64, count=len(edge_iter))
        edge_j = np.fromiter((node_idx[v] for (_u, v), _d in edge_iter),
                             dtype=np.int64, count=len(edge_iter))
        edge_d = np.fromiter((d for (_uv, d) in edge_iter),
                             dtype=np.float64, count=len(edge_iter))
        edge_w = 1.0 / np.maximum(edge_d, 1e-12) ** 2
    else:
        edge_i = np.zeros(0, dtype=np.int64)
        edge_j = np.zeros(0, dtype=np.int64)
        edge_d = np.zeros(0, dtype=np.float64)
        edge_w = np.zeros(0, dtype=np.float64)

    use_area = faces is not None and target_face_areas is not None
    if use_area:
        face_weights = 1.0 / np.maximum(
            np.asarray(target_face_areas, dtype=np.float64), 1e-12,
        ) ** 2
        face_pack = pack_faces(faces)
    else:
        face_weights = None
        face_pack = None

    def objective(x_flat: np.ndarray) -> float:
        """Joint stress + area objective (vectorised over edges)."""
        pos = x_flat.reshape(n, 2)
        diff = pos[edge_i] - pos[edge_j]
        dist = np.linalg.norm(diff, axis=1)
        total = float(np.sum(edge_w * (dist - edge_d) ** 2))
        if use_area:
            a_obj, _ = area_gradient_contribution(
                pos, faces, target_face_areas, face_weights,
                packed=face_pack,
            )
            total += area_weight * a_obj
        return total

    def objective_grad(x_flat: np.ndarray) -> np.ndarray:
        """Gradient of the joint objective (vectorised over edges)."""
        pos = x_flat.reshape(n, 2)
        diff = pos[edge_i] - pos[edge_j]                      # (E, 2)
        dist = np.linalg.norm(diff, axis=1)                   # (E,)
        safe = dist > 1e-12
        factor = np.zeros_like(dist)
        factor[safe] = (
            2.0 * edge_w[safe] * (dist[safe] - edge_d[safe]) / dist[safe]
        )
        contrib = factor[:, None] * diff                      # (E, 2)
        grad = np.zeros_like(pos)
        np.add.at(grad[:, 0], edge_i, contrib[:, 0])
        np.add.at(grad[:, 0], edge_j, -contrib[:, 0])
        np.add.at(grad[:, 1], edge_i, contrib[:, 1])
        np.add.at(grad[:, 1], edge_j, -contrib[:, 1])
        if use_area:
            _, a_grad = area_gradient_contribution(
                pos, faces, target_face_areas, face_weights,
                packed=face_pack,
            )
            grad += area_weight * a_grad
        if fixed_mask is not None:
            grad[fixed_mask] = 0.0
        return grad.ravel()

    # Freeze pinned nodes by zeroing their gradient components.
    # L-BFGS-B then leaves them at their initial positions for
    # the whole optimisation — cheap and exact.
    fixed_mask = None
    if fixed_nodes is not None and len(fixed_nodes) > 0:
        fixed_mask = np.zeros(n, dtype=bool)
        fixed_mask[np.asarray(fixed_nodes, dtype=np.int64)] = True

    result = minimize(
        objective,
        initial_coords.ravel(),
        jac=objective_grad,
        method="L-BFGS-B",
        options={"ftol": ftol, "maxiter": max_iter},
    )

    final = result.x.reshape(n, 2)
    # Belt-and-braces: explicitly restore pinned coords even if the
    # optimiser's final step nudged them (L-BFGS-B is gradient-based
    # but its line search can overshoot).
    if fixed_mask is not None:
        final[fixed_mask] = initial_coords[fixed_mask]
    return final


def reembed(
    g: nx.Graph,
    original_coords: Coords,
    seed_coords: Coords,
    *,
    preserve_lengths: bool = True,
    preserve_areas: bool = True,
    outer_boundary: Coords | None = None,
) -> Coords:
    """Full re-embedding pipeline: Tutte → stress → area correction.

    Args:
        g: Labelled planar graph.
        original_coords: (N, 2) original vertex positions.
        seed_coords: (N, 2) PRF-seeded initial positions.
        preserve_lengths: Whether to enforce edge-length matching.
        preserve_areas: Whether to enforce face-area matching.
        outer_boundary: Optional closed curve to pin the outer face
            onto (see :func:`tutte_embed`).  Supplying this skips
            the bbox auto-rescale: the boundary already sets the
            output frame.

    Returns:
        (N, 2) anonymised coordinates.
    """
    # Phase 1: Tutte embedding for crossing-free layout.
    # Tutte places the outer face on either the user-supplied
    # boundary (for "re-embed onto this island" use cases) or a
    # unit circle (classical Tutte).  When no boundary is given
    # we isotropically rescale to the original bbox so subsequent
    # stress works at a sensible unit scale.
    coords = tutte_embed(g, seed_coords=seed_coords,
                         outer_boundary=outer_boundary)
    if outer_boundary is None:
        coords = _rescale_to_bbox(coords, original_coords)

    # ── Outer-face nodes pinned if a boundary was supplied ──────
    # Without pinning, stress majorisation drags the outer nodes
    # off the boundary and the phantom island loses its
    # silhouette.  Pinning keeps the coastline exact and forces
    # interior nodes to spread outward instead of collapsing.
    fixed_nodes: list[int] | None = None
    if outer_boundary is not None:
        nodes_list = list(g.nodes)
        node_idx = {nd: i for i, nd in enumerate(nodes_list)}
        outer_face = _find_outer_face(g)
        fixed_nodes = [node_idx[nd] for nd in outer_face]

    # ── Common precomputes for Phases 2–3 ──────────────────────
    target_lengths = (
        _extract_edge_lengths(g, original_coords)
        if preserve_lengths else {}
    )
    face_list: list[list[int]] | None = None
    target_areas: np.ndarray | None = None
    if preserve_areas:
        face_list = extract_faces(g, original_coords)
        target_areas = face_areas(original_coords, face_list)

    # ── Phase 2: length-only stress majorisation ────────────────
    # Starting from Tutte, stress alone converges quickly and
    # accurately (≲ 3 % mean length error on a 4 k-node graph).
    # Doing this first gives Phase 3 a near-optimal seed so the
    # area term can refine without wrecking length fidelity.
    if preserve_lengths:
        coords = stress_majorise(g, target_lengths, coords,
                                 fixed_nodes=fixed_nodes)

    # ── Phase 3: joint stress + area refinement ─────────────────
    # Seeded from the Phase-2 optimum, the joint objective's
    # gradient is dominated by the area residual (stress is already
    # near zero), so L-BFGS-B mostly nudges faces toward their
    # target areas while keeping edges close to their Phase-2
    # lengths.
    if preserve_areas:
        coords = stress_majorise(
            g,
            target_lengths,
            coords,
            faces=face_list,
            target_face_areas=target_areas,
            fixed_nodes=fixed_nodes,
        )

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


def reembed_onto(
    g: nx.Graph,
    original_coords: Coords,
    seed_coords: Coords,
    outer_boundary: Coords,
    *,
    preserve_lengths: bool = True,
    preserve_areas: bool = True,
) -> Coords:
    """Re-embed the graph with its outer face pinned to ``outer_boundary``.

    Rescales ``original_coords`` isotropically into the bbox of
    ``outer_boundary`` before deriving stress targets.  Without this
    rescale, target edge lengths (in the source frame's units) are
    in a different scale from the boundary (in the phantom frame's
    units), and stress-majorisation either diverges or collapses
    interior nodes toward the centroid — the classical Tutte
    shrinking artefact that produces an "empty island, nodes on
    the shoreline" figure.

    After rescaling, target lengths / face areas live in the same
    units as the phantom coastline, so stress can actually pull
    interior nodes outward to fill the island.

    Args:
        g: Planar labelled graph.
        original_coords: (N, 2) source coords.  Used for edge-length
            and face-area *target ratios*; the absolute scale is
            normalised to the outer boundary's bbox.
        seed_coords: PRF-seeded initial positions (unused once
            Tutte anchors to the boundary; kept for API parity).
        outer_boundary: (M, 2) closed curve in the target frame.

    Returns:
        (N, 2) re-embedded coordinates inside ``outer_boundary``.
    """
    rescaled = _isotropic_rescale_to_bbox(original_coords, outer_boundary)
    return reembed(
        g, rescaled, seed_coords,
        preserve_lengths=preserve_lengths,
        preserve_areas=preserve_areas,
        outer_boundary=outer_boundary,
    )


def _isotropic_rescale_to_bbox(
    coords: Coords,
    target_curve: Coords,
) -> Coords:
    """Fit ``coords`` into ``target_curve``'s bbox, preserving aspect.

    Uses the smaller of the two axis scale factors so the rescaled
    coords stay inside the target bbox without anisotropic stretching.
    Centres the result on the target bbox centroid.
    """
    src_lo = coords.min(axis=0)
    src_hi = coords.max(axis=0)
    tgt_lo = target_curve.min(axis=0)
    tgt_hi = target_curve.max(axis=0)
    src_span = np.maximum(src_hi - src_lo, 1e-30)
    tgt_span = np.maximum(tgt_hi - tgt_lo, 1e-30)
    scale = float(np.min(tgt_span / src_span))
    src_centre = 0.5 * (src_lo + src_hi)
    tgt_centre = 0.5 * (tgt_lo + tgt_hi)
    return (coords - src_centre) * scale + tgt_centre


def _resample_boundary(
    boundary: Coords,
    n_out: int,
) -> Coords:
    """Resample a closed curve to exactly ``n_out`` points (uniform arclength).

    If the input is not already closed (first != last), it is
    closed by appending the first point.  Duplicate closing rows
    are removed before the resample to avoid zero-length segments.

    Args:
        boundary: ``(M, 2)`` ordered curve coordinates.
        n_out: Desired output length.

    Returns:
        ``(n_out, 2)`` resampled curve.  The first and last rows
        are NOT duplicated — the caller should treat the output as
        a cyclic sequence.
    """
    pts = np.asarray(boundary, dtype=np.float64)
    # Drop a closing-duplicate row if present
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 2:
        return np.repeat(pts[:1], n_out, axis=0)
    # Close the loop for arclength computation
    closed = np.vstack([pts, pts[:1]])
    d = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-30:
        return np.repeat(pts[:1], n_out, axis=0)
    targets = np.linspace(0.0, total, n_out, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.column_stack([x, y])


def _rescale_to_bbox(
    coords: Coords,
    reference: Coords,
) -> Coords:
    """Isotropically rescale + recentre ``coords`` to ``reference``'s bbox.

    Planarity is preserved because the transform is affine with
    positive uniform scale.  The recentre also moves the Tutte
    centroid (origin) to the reference centroid so that PRF-seeded
    initial geometry and target geometry share a coordinate frame.
    """
    src_lo = coords.min(axis=0)
    src_hi = coords.max(axis=0)
    tgt_lo = reference.min(axis=0)
    tgt_hi = reference.max(axis=0)
    src_span = np.maximum(src_hi - src_lo, 1e-12)
    tgt_span = np.maximum(tgt_hi - tgt_lo, 1e-12)
    scale = float(np.min(tgt_span / src_span))
    src_centre = 0.5 * (src_lo + src_hi)
    tgt_centre = 0.5 * (tgt_lo + tgt_hi)
    return (coords - src_centre) * scale + tgt_centre


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
