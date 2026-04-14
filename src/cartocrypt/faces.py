# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — faces                                              ║
# ║  « where the plane becomes polygons »                            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Face extraction from a planar graph + vectorised Shoelace       ║
# ║  area computation.  Shared between reembed (area objective),     ║
# ║  verify (residual checks), and shapes (Fourier boundary ops).    ║
# ║                                                                  ║
# ║  Faces are returned as cyclic vertex-index lists; the outer      ║
# ║  (unbounded) face is dropped — it has no finite target area.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Planar-face extraction and Shoelace area utilities.

« where the plane becomes polygons »

Given a planar NetworkX graph with coordinates, this module:

* enumerates all *interior* faces via
  :meth:`networkx.PlanarEmbedding.traverse_face`, and
* computes signed polygon areas using a vectorised Shoelace
  formula — fast enough to call inside an L-BFGS-B objective.

The outer (unbounded) face is detected by its sign under Shoelace
(negative under the right-handed orientation `traverse_face`
returns) and dropped, because it has no finite target area.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from cartocrypt.constants import Coords

# ┌────────────────────────────────────────────────────────────┐
# │ Face extraction  « traverse_face over every half-edge »     │
# └────────────────────────────────────────────────────────────┘


def extract_faces(
    g: nx.Graph,
    coords: Coords,
) -> list[list[int]]:
    """Return the interior faces of a planar graph as vertex-index cycles.

    Args:
        g: A planar simple graph.  Nodes must be labelled
            ``0..N-1`` (i.e. the canonical form produced by
            :func:`cartocrypt.ingest.to_labelled_graph`).
        coords: ``(N, 2)`` coordinate array used solely to
            orient the planar embedding and classify the outer
            face by Shoelace sign.

    Returns:
        A list of faces; each face is a list of node indices in
        cyclic order.  The outer (unbounded) face is excluded.

    Raises:
        ValueError: If ``g`` is not planar.
    """
    is_planar, embedding = nx.check_planarity(g)
    if not is_planar:
        msg = "extract_faces: graph is not planar"
        raise ValueError(msg)

    # Every directed half-edge (u → v) in the planar embedding
    # bounds exactly one face on its right.  Walking `traverse_face`
    # from each half-edge visits every face once per vertex on its
    # boundary; we deduplicate by the frozen set of *directed*
    # edges in the cycle — orientation-aware, rotation-invariant,
    # and O(len(face)) per face with no string/tuple machinery.
    seen: set[frozenset[tuple[int, int]]] = set()
    faces: list[list[int]] = []
    for u in embedding.nodes:
        for v in embedding[u]:
            cycle = embedding.traverse_face(u, v)
            key = frozenset(
                zip(cycle, cycle[1:] + cycle[:1], strict=True),
            )
            if key in seen:
                continue
            seen.add(key)
            faces.append(list(cycle))

    # Drop the outer face: it is the one whose signed Shoelace area
    # is negative (clockwise under traverse_face's orientation) *and*
    # has the largest absolute area.  Using absolute-area as a
    # tiebreaker keeps this robust on graphs where a stray hole
    # happens to traverse clockwise.
    areas = face_areas_signed(coords, faces)
    if len(faces) >= 1:
        outer_idx = int(np.argmax(np.abs(areas)))
        if areas[outer_idx] < 0:
            faces = [f for i, f in enumerate(faces) if i != outer_idx]

    return faces


def _pack_faces(
    faces: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a ragged face list into a dense index array + mask.

    Returns:
        ``(idx, mask, lengths)`` where ``idx`` has shape
        ``(F, L_max)`` with padding filled by the last valid vertex
        (so Shoelace stays well-defined), ``mask`` is a ``(F, L_max)``
        boolean mask of "real" vertices, and ``lengths`` is the
        per-face vertex count.  All dtypes are ``int64`` or ``bool``.
    """
    if not faces:
        return (
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=bool),
            np.zeros(0, dtype=np.int64),
        )
    lengths = np.fromiter((len(f) for f in faces), dtype=np.int64,
                          count=len(faces))
    L_max = int(lengths.max())
    F = len(faces)
    idx = np.empty((F, L_max), dtype=np.int64)
    mask = np.zeros((F, L_max), dtype=bool)
    for k, face in enumerate(faces):
        n = len(face)
        idx[k, :n] = face
        # Pad with the last valid vertex so Shoelace contributions
        # from the padded positions are zero (consecutive-identical
        # vertices produce zero signed area).
        if n < L_max:
            idx[k, n:] = face[-1]
        mask[k, :n] = True
    return idx, mask, lengths


# ┌────────────────────────────────────────────────────────────┐
# │ Shoelace areas  « vectorised, analytic-gradient friendly »  │
# └────────────────────────────────────────────────────────────┘


def face_areas_signed(
    coords: Coords,
    faces: list[list[int]],
) -> np.ndarray:
    """Return the signed Shoelace area of each face.

    Positive under counter-clockwise orientation, negative under
    clockwise.  Used to detect and drop the outer face.

    Args:
        coords: ``(N, 2)`` coordinate array.
        faces: List of node-index cycles.

    Returns:
        ``(F,)`` float64 array of signed areas.
    """
    if not faces:
        return np.zeros(0, dtype=np.float64)
    idx, _mask, _lengths = _pack_faces(faces)
    return _face_areas_signed_packed(coords, idx)


def _face_areas_signed_packed(
    coords: Coords,
    idx: np.ndarray,
) -> np.ndarray:
    """Vectorised Shoelace over packed face indices.

    Padding vertices are identical to the face's last real vertex,
    so the extra Shoelace terms contribute zero.
    """
    pts = coords[idx]  # (F, L_max, 2)
    x = pts[..., 0]
    y = pts[..., 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    return 0.5 * np.sum(x * y_next - x_next * y, axis=1)


def face_areas(
    coords: Coords,
    faces: list[list[int]],
) -> np.ndarray:
    """Unsigned face areas — the quantity we actually target."""
    return np.abs(face_areas_signed(coords, faces))


# ┌────────────────────────────────────────────────────────────┐
# │ Analytic area gradient  « used by reembed.stress_majorise » │
# └────────────────────────────────────────────────────────────┘


def area_gradient_contribution(
    coords: Coords,
    faces: list[list[int]],
    target_areas: np.ndarray,
    weights: np.ndarray,
    *,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """Joint face-area objective and its gradient.

    Returns the scalar objective

    .. math::

        \\Phi(p) = \\sum_f w_f \\, (A_f(p) - A^*_f)^2

    and the ``(N, 2)`` gradient ``∂Φ/∂p`` via the closed-form
    Shoelace partials

    .. math::

        \\partial A_f / \\partial x_i &= \\tfrac{1}{2}
            (y_{i+1} - y_{i-1}) \\\\
        \\partial A_f / \\partial y_i &= \\tfrac{1}{2}
            (x_{i-1} - x_{i+1})

    where index arithmetic is modulo the face length.  The inner
    loop is vectorised over all faces by packing the ragged face
    list into a dense ``(F, L_max)`` array and padding each face
    with its own final vertex — the padded Shoelace contributions
    are self-cancelling so the maths stays exact.

    Args:
        coords: ``(N, 2)`` coordinates.
        faces: Face cycles (outer face excluded).
        target_areas: ``(F,)`` target unsigned areas ``A*_f``.
        weights: ``(F,)`` per-face weights ``w_f``.
        packed: Optional precomputed output of
            :func:`_pack_faces`.  Pass this from a caller that
            evaluates the objective many times (e.g. L-BFGS-B) to
            avoid re-packing on every call.

    Returns:
        ``(objective, gradient)`` where gradient has shape ``(N, 2)``.
    """
    if not faces:
        return 0.0, np.zeros_like(coords, dtype=np.float64)

    idx, _mask, _lengths = packed if packed is not None else _pack_faces(faces)
    target_areas = np.asarray(target_areas, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    pts = coords[idx]                           # (F, L, 2)
    x = pts[..., 0]
    y = pts[..., 1]

    # Signed Shoelace per face
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    a_signed = 0.5 * np.sum(x * y_next - x_next * y, axis=1)   # (F,)
    a_abs = np.abs(a_signed)
    sign = np.where(a_signed >= 0, 1.0, -1.0)                  # (F,)

    residual = a_abs - target_areas                            # (F,)
    total = float(np.sum(weights * residual * residual))

    # Per-face gradient factor
    factor = 2.0 * weights * residual * sign                   # (F,)

    # Analytic Shoelace partials at each (face, position)
    y_prev = np.roll(y, 1, axis=1)
    x_prev = np.roll(x, 1, axis=1)
    dAdx = 0.5 * (y_next - y_prev)                             # (F, L)
    dAdy = 0.5 * (x_prev - x_next)                             # (F, L)

    # Scale by per-face factor.  The padding trick — repeating the
    # last real vertex — means the padded positions scatter
    # self-cancelling contributions to the final real vertex, so
    # the gradient sum remains exact *without* masking.
    w_dx = factor[:, None] * dAdx
    w_dy = factor[:, None] * dAdy

    # Scatter-add back to node coords via np.add.at
    grad = np.zeros_like(coords, dtype=np.float64)
    np.add.at(grad[:, 0], idx.ravel(), w_dx.ravel())
    np.add.at(grad[:, 1], idx.ravel(), w_dy.ravel())

    return total, grad


# ┌────────────────────────────────────────────────────────────┐
# │ Convenience  « residual stats for verify + tests »          │
# └────────────────────────────────────────────────────────────┘


def face_area_residuals(
    coords_a: Coords,
    coords_b: Coords,
    faces: list[list[int]],
) -> dict[str, Any]:
    """Summary statistics for relative face-area error.

    Args:
        coords_a: "Original" coordinates (target).
        coords_b: "Anonymised" coordinates (candidate).
        faces: Face cycles.

    Returns:
        dict with ``n_faces``, ``median_rel``, ``max_rel``,
        ``p95_rel`` — all as floats.
    """
    a_target = face_areas(coords_a, faces)
    a_got = face_areas(coords_b, faces)
    eps = 1e-15
    rel = np.abs(a_got - a_target) / np.maximum(a_target, eps)
    return {
        "n_faces":    int(len(faces)),
        "median_rel": float(np.median(rel)) if len(rel) else 0.0,
        "max_rel":    float(np.max(rel)) if len(rel) else 0.0,
        "p95_rel":    float(np.quantile(rel, 0.95)) if len(rel) else 0.0,
    }
