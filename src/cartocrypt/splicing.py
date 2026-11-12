# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — splicing                                           ║
# ║  « same map, towns reshuffled »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Attribute splicing: keyed permutation of face payloads within   ║
# ║  topological-adjacency clusters.  Geometry is never modified;    ║
# ║  only the attributes attached to each face get reshuffled.       ║
# ║                                                                  ║
# ║  The metaphor is meiotic crossover — two adjacent faces pair     ║
# ║  up, exchange genes (overlay fragments, node rollups), and       ║
# ║  unfold with new payloads.  The bijection property means the     ║
# ║  global attribute multiset is preserved exactly, so the map      ║
# ║  still "looks right" statistically but no face carries its       ║
# ║  original fingerprint.                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Attribute splicing between topologically adjacent faces.

« same map, towns reshuffled »

Defeats the crocodile problem by permuting per-face attribute
payloads (overlay polygons, rolled-up node labels) among faces
that share a boundary, under a keyed RNG.  The island's outline,
road network, and face geometry are untouched — only *who lives
where* changes.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from cartocrypt.constants import (
    MIN_SPLICE_AREA_PCTILE,
    SPLICE_SIZE_RATIO_MAX,
    Coords,
    KeyBytes,
)
from cartocrypt.faces import (
    extract_faces,
    face_adjacency_graph,
    face_areas,
)

# ┌────────────────────────────────────────────────────────────┐
# │ Keyed RNG  « HMAC-derived numpy Generator »                 │
# └────────────────────────────────────────────────────────────┘


def _keyed_rng(key: KeyBytes, label: str) -> np.random.Generator:
    """Deterministic numpy Generator keyed by (key, label)."""
    digest = hmac.new(key, label.encode("utf-8"), hashlib.sha256).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed_int)


# ┌────────────────────────────────────────────────────────────┐
# │ Per-face payload                                            │
# └────────────────────────────────────────────────────────────┘

@dataclass
class FacePayload:
    """Everything splicing touches for a single face.

    Attributes:
        overlay_fragments: List of ``(row_dict, fragment_polygon)``
            tuples — one per overlay that intersects this face,
            already clipped to the face's geometry.
        rollup: Face-level summary derived from boundary-node
            attributes.  Free-form dict: consumers decide what to
            store.
    """

    overlay_fragments: list[tuple[dict[str, Any], Polygon]] = field(
        default_factory=list,
    )
    rollup: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpliceResult:
    """Returned by :func:`splice_attributes`.

    Attributes:
        overlays: GeoDataFrame of spliced overlay fragments.  Rows
            carry the original layer metadata (species, layer name,
            area) but their geometry has been reprojected onto the
            target face.
        face_rollups: ``face_id → rollup dict`` — permuted.
        cluster_count: Number of splice clusters processed.
        n_faces_spliced: How many faces participated (i.e. were
            above the size threshold).
    """

    overlays: gpd.GeoDataFrame
    face_rollups: dict[int, dict[str, Any]]
    cluster_count: int
    n_faces_spliced: int


# ┌────────────────────────────────────────────────────────────┐
# │ Public API                                                  │
# └────────────────────────────────────────────────────────────┘


def splice_attributes(
    g: nx.Graph,
    coords: Coords,
    overlays: gpd.GeoDataFrame,
    key: KeyBytes,
    *,
    rollup_fields: list[str] | None = None,
    min_area_pctile: float = MIN_SPLICE_AREA_PCTILE,
    size_ratio_max: float = SPLICE_SIZE_RATIO_MAX,
    faces: list[list[int]] | None = None,
) -> SpliceResult:
    """Permute face-level payloads among topologically adjacent faces.

    Args:
        g: Simple planar graph with integer node labels ``0..N-1``.
        coords: ``(N, 2)`` EPSG:4326 coordinate array.
        overlays: Polygon overlays to splice.  Typically the
            output of :func:`cartocrypt.overlay.polygons_as_gdf`.
            Must have a ``geometry`` column; any other columns are
            preserved on each fragment.
        key: 32-byte symmetric key — drives all randomness.
        rollup_fields: Node-attribute names to roll up per face
            (default: ``["elevation", "place_type"]``).  Rolled-up
            values for numeric fields use the median, categorical
            fields use the mode.
        min_area_pctile: Faces below this area quantile skip
            splicing (they are too small to swap meaningfully).
        size_ratio_max: Within a cluster, paired faces must have
            area ratio ≤ this value.
        faces: Optional pre-computed face list.  Computed if not
            given.

    Returns:
        :class:`SpliceResult`.
    """
    if faces is None:
        faces = extract_faces(g, coords)
    if rollup_fields is None:
        rollup_fields = ["elevation", "place_type"]

    # ── Face geometries + areas ────────────────────────────────
    face_polys = [Polygon(coords[face]) for face in faces]
    areas = face_areas(coords, faces)

    # ── Collect per-face payloads ──────────────────────────────
    payloads = _collect_payloads(
        g, coords, faces, face_polys, overlays, rollup_fields,
    )

    # ── Cluster by adjacency, pruning tiny faces ───────────────
    fa = face_adjacency_graph(faces)
    threshold = float(np.quantile(areas, min_area_pctile)) if len(areas) else 0.0
    active = [f for f in range(len(faces)) if areas[f] >= threshold]
    fa_pruned = fa.subgraph(active).copy()
    clusters = list(nx.connected_components(fa_pruned))

    # ── Permute payloads within each cluster ───────────────────
    perm_overlay = dict(zip(range(len(faces)), range(len(faces)), strict=True))
    perm_rollup = dict(zip(range(len(faces)), range(len(faces)), strict=True))

    for c_idx, cluster in enumerate(clusters):
        cluster_sorted = sorted(cluster)
        if len(cluster_sorted) < 2:
            continue
        rng_o = _keyed_rng(key, f"splice-overlay-{c_idx}-{cluster_sorted[0]}")
        rng_r = _keyed_rng(key, f"splice-rollup-{c_idx}-{cluster_sorted[0]}")

        # Size-banded sub-permutation: split cluster into buckets
        # such that max(area)/min(area) ≤ size_ratio_max within
        # each bucket, then permute inside each bucket.
        buckets = _size_banded_buckets(
            cluster_sorted, areas, size_ratio_max,
        )
        for bucket in buckets:
            if len(bucket) < 2:
                continue
            po = _derangement(rng_o, len(bucket))
            pr = _derangement(rng_r, len(bucket))
            for local, tgt_local in enumerate(po):
                perm_overlay[bucket[local]] = bucket[tgt_local]
            for local, tgt_local in enumerate(pr):
                perm_rollup[bucket[local]] = bucket[tgt_local]

    # ── Build spliced outputs ──────────────────────────────────
    spliced_rows: list[dict[str, Any]] = []
    spliced_geoms: list[Any] = []
    for f_tgt in range(len(faces)):
        f_src = perm_overlay[f_tgt]
        if f_src == f_tgt:
            # Identity — preserve original fragments as-is
            for row, geom in payloads[f_tgt].overlay_fragments:
                spliced_rows.append(dict(row))
                spliced_geoms.append(geom)
            continue
        src_poly = face_polys[f_src]
        tgt_poly = face_polys[f_tgt]
        affine = _affine_between_polygons(src_poly, tgt_poly)
        for row, geom in payloads[f_src].overlay_fragments:
            new_geom = affine_transform(geom, affine)
            if new_geom.is_empty:
                continue
            new_row = dict(row)
            new_row["_spliced_from_face"] = f_src
            new_row["_spliced_to_face"] = f_tgt
            spliced_rows.append(new_row)
            spliced_geoms.append(new_geom)

    if spliced_rows:
        out_gdf = gpd.GeoDataFrame(
            spliced_rows, geometry=spliced_geoms, crs="EPSG:4326",
        )
    else:
        out_gdf = gpd.GeoDataFrame(
            {"geometry": []}, crs="EPSG:4326",
        )

    rollup_out = {
        f_tgt: dict(payloads[perm_rollup[f_tgt]].rollup)
        for f_tgt in range(len(faces))
    }

    n_spliced = sum(len(c) for c in clusters if len(c) >= 2)
    return SpliceResult(
        overlays=out_gdf,
        face_rollups=rollup_out,
        cluster_count=sum(1 for c in clusters if len(c) >= 2),
        n_faces_spliced=n_spliced,
    )


# ┌────────────────────────────────────────────────────────────┐
# │ Reprojection helpers  « move spliced fragments into phantom »│
# └────────────────────────────────────────────────────────────┘


def reproject_spliced(
    spliced: gpd.GeoDataFrame,
    g: nx.Graph,
    old_coords: Coords,
    new_coords: Coords,
    *,
    faces: list[list[int]] | None = None,
) -> gpd.GeoDataFrame:
    """Reproject spliced fragments into a new graph embedding.

    After :func:`splice_attributes` has redistributed overlay
    fragments among faces (tagging each row with
    ``_spliced_to_face``), and after a fresh re-embedding has
    produced ``new_coords`` for every node, this helper maps each
    fragment from its current geometry (in ``old_coords``) onto the
    same target face's *new* vertex positions (in ``new_coords``).

    Each fragment is transported by the PCA-based affine that maps
    the target face's old vertex set onto its new vertex set.  Face
    identity is preserved; only the coordinate frame changes.

    Args:
        spliced: Output of :func:`splice_attributes`.  Must carry
            the ``_spliced_to_face`` column.
        g: Planar graph.
        old_coords: Coordinates in which ``spliced`` currently lives.
        new_coords: Target coordinates for each node of ``g``.
        faces: Optional pre-computed face list; defaults to
            :func:`cartocrypt.faces.extract_faces(g, old_coords)`.

    Returns:
        GeoDataFrame with the same columns as ``spliced`` but with
        each fragment's geometry transformed into ``new_coords``.
    """
    if spliced is None or spliced.empty:
        return spliced
    if faces is None:
        faces = extract_faces(g, old_coords)

    old_polys = [Polygon(old_coords[face]) for face in faces]
    new_polys = [Polygon(new_coords[face]) for face in faces]

    rows: list[dict[str, Any]] = []
    geoms: list[Any] = []
    for _idx, row in spliced.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        tgt = row.get("_spliced_to_face")
        if tgt is None or not 0 <= int(tgt) < len(faces):
            # Unmappable fragment — drop rather than mis-place
            continue
        tgt = int(tgt)
        affine = _affine_between_polygons(old_polys[tgt], new_polys[tgt])
        new_geom = affine_transform(geom, affine)
        if new_geom.is_empty:
            continue
        rows.append({k: v for k, v in row.items() if k != "geometry"})
        geoms.append(new_geom)

    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, crs=spliced.crs)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=spliced.crs)


def warped_settlement_gdf(
    settlements: gpd.GeoDataFrame,
    settle_to_node: dict[int, int],
    new_coords: Coords,
) -> gpd.GeoDataFrame:
    """Rebuild a settlements GeoDataFrame using warped node coords.

    Each settlement row whose host node survived the re-embedding
    is re-issued with its Point geometry moved to the node's new
    position.  Rows whose host nodes are missing from ``new_coords``
    (e.g. dropped during the largest-connected-component filter)
    are excluded.

    Args:
        settlements: Original settlements GeoDataFrame.
        settle_to_node: Mapping returned by
            :func:`cartocrypt.overlay.attach_settlements`.
        new_coords: (N, 2) re-embedded coordinate array.  The node
            indices keyed in ``settle_to_node`` must index into
            this array.

    Returns:
        GeoDataFrame with Point geometries at warped positions.
    """
    from shapely.geometry import Point

    if settlements is None or settlements.empty or not settle_to_node:
        return gpd.GeoDataFrame(
            {"name": [], "geometry": []},
            crs=settlements.crs if settlements is not None else None,
        )

    rows: list[dict[str, Any]] = []
    geoms: list[Any] = []
    n_max = int(new_coords.shape[0])
    for idx, node in settle_to_node.items():
        if not 0 <= int(node) < n_max:
            continue
        try:
            row = settlements.iloc[int(idx)]
        except IndexError:
            continue
        x, y = new_coords[int(node)]
        attrs = {k: v for k, v in row.items() if k != "geometry"}
        rows.append(attrs)
        geoms.append(Point(float(x), float(y)))

    if not rows:
        return gpd.GeoDataFrame(
            {"name": [], "geometry": []}, crs=settlements.crs,
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=settlements.crs)


# ┌────────────────────────────────────────────────────────────┐
# │ Payload collection                                          │
# └────────────────────────────────────────────────────────────┘


def reproject_to_embedding(
    spliced: SpliceResult,
    faces: list[list[int]],
    src_coords: Coords,
    dst_coords: Coords,
) -> gpd.GeoDataFrame:
    """Reproject spliced overlay fragments onto a new graph embedding.

    Each fragment carries ``_spliced_to_face`` — the index of the
    face it was placed into.  To move the fragment into a new
    embedding (e.g. the re-embedded phantom-island coords) we
    compute the per-face affine from the *source* face vertices
    to the *destination* face vertices and apply it.

    Args:
        spliced: Output of :func:`splice_attributes`.
        faces: Face cycles (same list used by splicing).
        src_coords: Coordinates under which splicing ran.
        dst_coords: Target coordinates (e.g. the re-embedded
            anonymised graph).

    Returns:
        New GeoDataFrame with the same rows as ``spliced.overlays``
        but with each geometry transformed into the ``dst_coords``
        frame.
    """
    if spliced.overlays.empty:
        return spliced.overlays.copy()

    rows: list[dict[str, Any]] = []
    geoms: list[Any] = []
    for _idx, row in spliced.overlays.iterrows():
        tgt_face = row.get("_spliced_to_face")
        if tgt_face is None or (isinstance(tgt_face, float)
                                and not np.isfinite(tgt_face)):
            tgt_face = row.get("_spliced_from_face")
        if tgt_face is None or (isinstance(tgt_face, float)
                                and not np.isfinite(tgt_face)):
            # Identity fragment with no recorded face — find the
            # face whose polygon contains the fragment centroid.
            tgt_face = _find_containing_face(row.geometry, faces, src_coords)
        geom = row.geometry
        if tgt_face is None or geom is None or geom.is_empty:
            rows.append(dict(row))
            geoms.append(geom)
            continue
        face_idx = int(tgt_face)
        src_poly = Polygon(src_coords[faces[face_idx]])
        dst_poly = Polygon(dst_coords[faces[face_idx]])
        affine = _affine_between_polygons(src_poly, dst_poly)
        new_geom = affine_transform(geom, affine)
        rows.append(dict(row))
        geoms.append(new_geom)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=spliced.overlays.crs)


def _find_containing_face(
    geom,
    faces: list[list[int]],
    coords: Coords,
) -> int | None:
    """Locate the first face whose polygon contains ``geom``'s centroid."""
    if geom is None or geom.is_empty:
        return None
    c = geom.centroid
    for f_idx, face in enumerate(faces):
        poly = Polygon(coords[face])
        if poly.contains(c) or poly.touches(c):
            return f_idx
    # Fallback: nearest-centroid face
    c_xy = np.array([c.x, c.y])
    best = None
    best_d = np.inf
    for f_idx, face in enumerate(faces):
        poly_c = coords[face].mean(axis=0)
        d = np.linalg.norm(poly_c - c_xy)
        if d < best_d:
            best = f_idx
            best_d = float(d)
    return best


def _collect_payloads(
    g: nx.Graph,
    coords: Coords,
    faces: list[list[int]],
    face_polys: list[Polygon],
    overlays: gpd.GeoDataFrame,
    rollup_fields: list[str],
) -> list[FacePayload]:
    """Build one FacePayload per face."""
    out: list[FacePayload] = [FacePayload() for _ in faces]

    # ── Overlay clipping: which fragments land in which face ──
    if overlays is not None and not overlays.empty:
        valid_face_polys = [p if p.is_valid else p.buffer(0)
                            for p in face_polys]
        for _idx, row in overlays.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            g_clean = geom if geom.is_valid else geom.buffer(0)
            row_attrs = {
                k: v for k, v in row.items() if k != "geometry"
            }
            for f_idx, face_poly in enumerate(valid_face_polys):
                if not face_poly.intersects(g_clean):
                    continue
                clipped = face_poly.intersection(g_clean)
                if clipped.is_empty:
                    continue
                # Normalise to simple polygons
                geoms = _as_polygons(clipped)
                for clip in geoms:
                    if clip.area <= 0:
                        continue
                    out[f_idx].overlay_fragments.append(
                        (row_attrs, clip),
                    )

    # ── Node-level rollups ─────────────────────────────────────
    for f_idx, face in enumerate(faces):
        rollup: dict[str, Any] = {}
        for field_name in rollup_fields:
            vals = [g.nodes[n].get(field_name) for n in face]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            if all(isinstance(v, (int, float, np.integer, np.floating))
                   for v in vals):
                rollup[field_name] = float(np.median(vals))
            else:
                # Modal categorical
                uniq, cnts = np.unique([str(v) for v in vals],
                                       return_counts=True)
                rollup[field_name] = str(uniq[int(np.argmax(cnts))])
        out[f_idx].rollup = rollup

    return out


# ┌────────────────────────────────────────────────────────────┐
# │ Cluster size-banding + keyed permutation                    │
# └────────────────────────────────────────────────────────────┘


def _size_banded_buckets(
    cluster: list[int],
    areas: np.ndarray,
    ratio_max: float,
) -> list[list[int]]:
    """Split a cluster into buckets with bounded area ratio.

    Sort faces by area; walk through them, starting a new bucket
    whenever the ratio of the current face's area to the bucket's
    minimum area would exceed ``ratio_max``.
    """
    by_area = sorted(cluster, key=lambda f: float(areas[f]))
    buckets: list[list[int]] = []
    current: list[int] = []
    bucket_min = 0.0
    for f in by_area:
        a = float(areas[f])
        if not current:
            current = [f]
            bucket_min = a
            continue
        if a <= bucket_min * ratio_max:
            current.append(f)
        else:
            buckets.append(current)
            current = [f]
            bucket_min = a
    if current:
        buckets.append(current)
    return buckets


def _derangement(
    rng: np.random.Generator,
    n: int,
    max_tries: int = 32,
) -> np.ndarray:
    """Return a permutation of ``range(n)`` with no fixed point.

    Falls back to a cyclic shift if random retries fail — always
    yields a fixed-point-free permutation for ``n ≥ 2``.
    """
    if n < 2:
        return np.arange(n)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


# ┌────────────────────────────────────────────────────────────┐
# │ Polygon geometry helpers                                    │
# └────────────────────────────────────────────────────────────┘


def _as_polygons(geom: Any) -> list[Polygon]:
    """Flatten any shapely result to a list of Polygon."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for sub in geom.geoms:
            out.extend(_as_polygons(sub))
        return out
    return []


def _affine_between_polygons(
    src: Polygon,
    tgt: Polygon,
) -> list[float]:
    """Compute a 2D affine that maps ``src`` onto ``tgt``.

    Uses centroid alignment plus a scale-and-rotate derived from
    the two polygons' principal axes (PCA on the exterior coords).
    Output is a shapely-style 6-tuple ``[a, b, d, e, xoff, yoff]``
    suitable for :func:`shapely.affinity.affine_transform`.
    """
    src_c = np.array([src.centroid.x, src.centroid.y])
    tgt_c = np.array([tgt.centroid.x, tgt.centroid.y])

    src_pts = np.asarray(src.exterior.coords)[:-1] - src_c
    tgt_pts = np.asarray(tgt.exterior.coords)[:-1] - tgt_c

    def _principal_axes(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(p) < 2:
            return np.eye(2), np.ones(2)
        cov = np.cov(p.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        return vecs[:, order], np.sqrt(np.maximum(vals[order], 1e-30))

    src_axes, src_scales = _principal_axes(src_pts)
    tgt_axes, tgt_scales = _principal_axes(tgt_pts)

    # Transform = T_tgt @ diag(tgt_scales / src_scales) @ T_src^T
    ratio = tgt_scales / np.maximum(src_scales, 1e-30)
    A = tgt_axes @ np.diag(ratio) @ src_axes.T

    offset = tgt_c - A @ src_c
    return [A[0, 0], A[0, 1], A[1, 0], A[1, 1], offset[0], offset[1]]
