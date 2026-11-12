"""End-to-end Aegina demo: phantom-island privacy pipeline.

Run from the repo root with the cartocrypt conda env active:

    python examples/aegina_demo.py

Set ``OPENTOPO_API_KEY`` for a real Copernicus GLO-30 hillshade;
without it the script falls back to a peak-IDW DEM.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point

from cartocrypt.aegina import AeginaIngestor
from cartocrypt.fake_layers import (
    make_futurama_tubes,
    make_pokemon_habitats,
    render_tubes_from_stations,
)
from cartocrypt.ingest import to_labelled_graph
from cartocrypt.keygen import generate_key, prf_coordinates_batch
from cartocrypt.overlay import add_layer, attach_settlements, polygons_as_gdf
from cartocrypt.splicing import splice_attributes
from cartocrypt.synthesis import resynthesise_dem
from cartocrypt.verify import verify_metrics
from cartocrypt.viz_aegina import plot_aegina_figure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s",
)


def main() -> int:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("▶  Ingesting Aegina layers from OSM…")
    bundle = AeginaIngestor().fetch_all(use_dem=True)
    print(bundle.summary())

    _ = generate_key  # silence unused import when pinned
    key = bytes.fromhex(
        "43617274 6f437279 70742041 6567696e 61204465 6d6f2053 65656421".replace(" ", "")
    ).ljust(32, b"\0")[:32]

    print("▶  Generating fictional overlays (real Aegina frame)…")
    habitats = make_pokemon_habitats(bundle, key)
    tubes = make_futurama_tubes(bundle, key)
    print(f"   habitats: {len(habitats)} polygons")
    print(f"   tubes:    {len(tubes)} segments")

    # ── Build the CC graph ────────────────────────────────────
    g_full, coords_full, _ = to_labelled_graph(bundle.roads)
    components = sorted(nx.connected_components(g_full), key=len, reverse=True)
    g_cc = g_full.subgraph(next(iter(components))).copy()
    coords_cc = _reorder(coords_full, g_full, g_cc)
    new_idx = {n: i for i, n in enumerate(g_cc.nodes)}
    g_cc = nx.relabel_nodes(g_cc, new_idx)
    print(f"   largest CC: {g_cc.number_of_nodes()} nodes, "
          f"{g_cc.number_of_edges()} edges")

    # ── Attach settlements + habitats so they travel with reembed
    print("▶  Attaching settlements + habitats to the graph…")
    settle_map = attach_settlements(g_cc, coords_cc, bundle)
    if not habitats.empty:
        add_layer(g_cc, coords_cc, habitats, "habitat")

    # ── Splice face attribute payloads on the real embedding ──
    print("▶  Splicing face attributes…")
    overlays_real = polygons_as_gdf(g_cc)
    spliced = splice_attributes(g_cc, coords_cc, overlays_real, key=key)
    print(f"   spliced fragments: {len(spliced.overlays)} across "
          f"{spliced.cluster_count} cluster(s), "
          f"{spliced.n_faces_spliced} faces participating")

    # ── Step 1: Reembed the graph freely ────────────────────────
    # Standard Tutte → stress → area.  The graph layout defines its
    # own shape; we derive the phantom coastline FROM it afterward.
    # No more forcing two independent shapes to agree.
    print("▶  Re-embedding graph (free layout)…")
    seed = prf_coordinates_batch(
        key, g_cc.number_of_nodes(),
        bbox=(float(coords_cc[:, 0].min()), float(coords_cc[:, 1].min()),
              float(coords_cc[:, 0].max()), float(coords_cc[:, 1].max())),
    )
    from cartocrypt.reembed import reembed
    t0 = time.perf_counter()
    anon_coords = reembed(
        g_cc, coords_cc, seed,
        preserve_lengths=True, preserve_areas=True,
    )
    print(f"   reembed: {time.perf_counter() - t0:.1f} s")

    # ── Step 2: Derive phantom coastline FROM the graph layout ─
    # The graph's concave hull IS the phantom island outline.
    # Fourier-perturb it to add coastline wiggliness.
    print("▶  Deriving phantom coastline from graph layout…")
    t0 = time.perf_counter()
    hull_pts = _concave_hull(anon_coords)
    phantom_coast = _add_coastline_detail(hull_pts, key)
    print(f"   coastline: {len(phantom_coast)} pts "
          f"({time.perf_counter() - t0:.2f} s)")

    # ── Step 3: Generate phantom DEM filling the derived coastline ─
    print("▶  Resynthesising phantom DEM inside graph outline…")
    t0 = time.perf_counter()
    ph_xmin = float(phantom_coast[:, 0].min()) - 0.002
    ph_xmax = float(phantom_coast[:, 0].max()) + 0.002
    ph_ymin = float(phantom_coast[:, 1].min()) - 0.002
    ph_ymax = float(phantom_coast[:, 1].max()) + 0.002
    ph_h, ph_w = bundle.dem.shape
    ph_pixel_x = (ph_xmax - ph_xmin) / ph_w
    ph_pixel_y = -(ph_ymax - ph_ymin) / ph_h
    phantom_transform = (ph_pixel_x, 0.0, ph_xmin, 0.0, ph_pixel_y, ph_ymax)
    phantom_bbox = (ph_ymax, ph_ymin, ph_xmax, ph_xmin)

    phantom_mask = _rasterise_direct(
        phantom_coast, (ph_h, ph_w), phantom_transform,
    )
    phantom_dem, h_used = resynthesise_dem(
        bundle.dem, key=key, mask=phantom_mask,
    )
    print(f"   DEM: {ph_h}×{ph_w}, Hurst={h_used:.3f} "
          f"({time.perf_counter() - t0:.1f} s)")

    metrics = verify_metrics(g_cc, coords_cc, anon_coords)
    print(f"   length err: mean={metrics['length_mean_rel_error']:.3f} "
          f"max={metrics['length_max_rel_error']:.3f}")
    print(f"   area   err: median={metrics['area_median_rel_error']:.3f} "
          f"p95={metrics['area_p95_rel_error']:.3f} "
          f"(within tol: {metrics['area_within_tol']})")

    # ── Regenerate habitats directly on the phantom terrain ──
    # Same make_pokemon_habitats algorithm, same key, but applied
    # to the phantom DEM — Charizard lands on phantom peaks,
    # Pikachu on phantom coast, Eevee in phantom valleys.
    # This is the terrain-driven approach: habitats correlate with
    # the terrain they sit on, not randomly spliced from elsewhere.
    print("▶  Regenerating habitats on phantom terrain…")
    from shapely.geometry import Polygon as ShPoly

    from cartocrypt.aegina import LayerBundle
    phantom_bundle = LayerBundle(
        roads=bundle.roads,
        buildings=bundle.buildings,
        coastline=gpd.GeoDataFrame(
            {"geometry": [ShPoly(phantom_coast)]},
            crs="EPSG:4326",
        ),
        landuse=bundle.landuse,
        water=bundle.water,
        settlements=bundle.settlements,
        peaks=bundle.peaks,
        dem=phantom_dem,
        dem_transform=phantom_transform,
        dem_source="phantom",
        bbox=phantom_bbox,
        meta={"phantom": True},
    )
    phantom_habitats = make_pokemon_habitats(phantom_bundle, key)
    print(f"   phantom habitats: {len(phantom_habitats)} polygons "
          f"({sorted(set(phantom_habitats['species'])) if not phantom_habitats.empty else []})")

    # ── Build phantom-frame settlements + tubes ──────────────
    print("▶  Rendering phantom tubes…")
    phantom_settlements = _warped_settlements(
        bundle.settlements, settle_map, anon_coords,
    )
    phantom_tubes = render_tubes_from_stations(
        phantom_settlements, key=key,
    ) if not phantom_settlements.empty else tubes.iloc[0:0]
    print(f"   phantom tubes: {len(phantom_tubes)} segments")

    # ── Render ────────────────────────────────────────────────
    out_stem = out_dir / "aegina"
    print(f"▶  Rendering figure → {out_stem}.svg / .png")
    fig = plot_aegina_figure(
        bundle, habitats, tubes,
        anon_graph=g_cc, anon_coords=anon_coords,
        phantom_coast=phantom_coast,
        phantom_dem=phantom_dem,
        spliced_overlays=phantom_habitats,
        phantom_tubes=phantom_tubes,
        phantom_settlements=phantom_settlements,
        out_path=out_stem,
    )
    plt.close(fig)

    print("Done.")
    return 0


def _reorder(coords_full, g_full, g_cc):
    idx_full = {n: i for i, n in enumerate(g_full.nodes)}
    rows = [idx_full[n] for n in g_cc.nodes]
    return np.asarray(coords_full)[rows]


def _warped_settlements(settlements_gdf, settle_map, anon_coords):
    """Build a GDF of settlements at their reembedded positions."""
    if settlements_gdf is None or settlements_gdf.empty or not settle_map:
        return gpd.GeoDataFrame(
            {"name": [], "geometry": []}, crs="EPSG:4326",
        )
    rows = []
    geoms = []
    for pos_idx, node_idx in settle_map.items():
        if pos_idx >= len(settlements_gdf):
            continue
        src_row = settlements_gdf.iloc[pos_idx]
        name = src_row.get("name") if "name" in settlements_gdf.columns \
            else f"Station-{pos_idx}"
        xy = anon_coords[node_idx]
        rows.append({"name": name})
        geoms.append(Point(float(xy[0]), float(xy[1])))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def _concave_hull(coords, alpha_ratio=0.3):
    """Compute a concave hull (alpha shape) of node coordinates.

    Uses Delaunay triangulation and removes triangles whose
    circumradius exceeds ``alpha_ratio × max_edge_length``.
    Falls back to convex hull if scipy is missing or the alpha
    shape is degenerate.
    """
    from scipy.spatial import ConvexHull, Delaunay
    from shapely.geometry import MultiLineString
    from shapely.ops import polygonize, unary_union

    if len(coords) < 4:
        hull = ConvexHull(coords)
        pts = coords[hull.vertices]
        return np.vstack([pts, pts[:1]])

    tri = Delaunay(coords)
    edges = set()
    # Compute threshold: alpha_ratio × median edge length
    all_lens = []
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            all_lens.append(np.linalg.norm(coords[a] - coords[b]))
    threshold = alpha_ratio * np.median(all_lens) * 10

    for simplex in tri.simplices:
        pts_tri = coords[simplex]
        # Circumradius
        a = np.linalg.norm(pts_tri[0] - pts_tri[1])
        b = np.linalg.norm(pts_tri[1] - pts_tri[2])
        c = np.linalg.norm(pts_tri[2] - pts_tri[0])
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        if area < 1e-15:
            continue
        circumr = (a * b * c) / (4.0 * area)
        if circumr < threshold:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                if edge in edges:
                    edges.discard(edge)
                else:
                    edges.add(edge)

    lines = [coords[[a, b]] for a, b in edges]
    if not lines:
        hull = ConvexHull(coords)
        pts = coords[hull.vertices]
        return np.vstack([pts, pts[:1]])

    ml = MultiLineString([line.tolist() for line in lines])
    polys = list(polygonize(unary_union(ml)))
    if not polys:
        hull = ConvexHull(coords)
        pts = coords[hull.vertices]
        return np.vstack([pts, pts[:1]])

    biggest = max(polys, key=lambda p: p.area)
    return np.asarray(biggest.exterior.coords)


def _add_coastline_detail(hull_pts, key, n_samples=256, n_preserve=3):
    """Add Fourier-driven wiggliness to a hull outline.

    Resamples the hull to ``n_samples`` points, computes the FFT,
    randomises phases above ``n_preserve`` with the key, inverse-FFTs.
    Result: a coastline-like curve whose low-frequency shape matches
    the graph hull but whose high-frequency detail is fresh and
    island-looking.
    """
    from cartocrypt.synthesis import _arclength_resample, _keyed_rng

    pts = np.asarray(hull_pts, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])

    resampled = _arclength_resample(pts, n_samples)
    z = resampled[:, 0] + 1j * resampled[:, 1]
    F = np.fft.fft(z)
    amp = np.abs(F)
    phase = np.angle(F)

    rng = _keyed_rng(key, "coastline-detail")
    n = len(F)
    for k in range(n_preserve, n // 2 + 1):
        new_phi = float(rng.uniform(0, 2 * np.pi))
        phase[k] = new_phi
        if n - k != k:
            phase[n - k] = -new_phi
    # Reduce high-freq amplitudes slightly to avoid spikes
    for k in range(n_preserve, n):
        amp[k] *= 0.6

    F_new = amp * np.exp(1j * phase)
    z_new = np.fft.ifft(F_new)
    xy = np.column_stack([z_new.real, z_new.imag])
    return np.vstack([xy, xy[:1]])


def _rasterise_direct(coast_pts, shape, transform):
    """Rasterise a coastline polygon onto a DEM grid in the SAME frame.

    No cross-frame rescaling — the coastline coords and the DEM
    affine share one coordinate system (both derived from the
    phantom coastline's bbox).
    """
    from matplotlib.path import Path as _MPath
    a, _b, c, _d, e, f = transform
    h, w = shape
    col_idx, row_idx = np.meshgrid(np.arange(w), np.arange(h))
    lon = a * col_idx + c
    lat = e * row_idx + f
    coast = np.asarray(coast_pts)
    # Close the polygon if not already closed
    if not np.allclose(coast[0], coast[-1]):
        coast = np.vstack([coast, coast[:1]])
    path = _MPath(coast)
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    mask = path.contains_points(pts).reshape(h, w)
    return mask


def _extract_coast_pts(bundle):
    """Flatten coastline GDF into an (M, 2) array."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.geometry import Polygon as ShPoly
    pts = []
    for g in bundle.coastline.geometry:
        if g is None or g.is_empty:
            continue
        if isinstance(g, LineString):
            pts.append(np.asarray(g.coords))
        elif isinstance(g, MultiLineString):
            for sub in g.geoms:
                pts.append(np.asarray(sub.coords))
        elif isinstance(g, ShPoly):
            pts.append(np.asarray(g.exterior.coords))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _fit_into_polygon_bbox(coords, polygon_pts, *, margin=0.05):
    """Isotropically rescale ``coords`` to fit inside ``polygon_pts``'s bbox."""
    src_lo = coords.min(axis=0)
    src_hi = coords.max(axis=0)
    tgt_lo = polygon_pts.min(axis=0)
    tgt_hi = polygon_pts.max(axis=0)
    src_span = np.maximum(src_hi - src_lo, 1e-30)
    tgt_span = np.maximum(tgt_hi - tgt_lo, 1e-30) * (1.0 - 2 * margin)
    s = float(np.min(tgt_span / src_span))
    src_c = 0.5 * (src_lo + src_hi)
    tgt_c = 0.5 * (tgt_lo + tgt_hi)
    return (coords - src_c) * s + tgt_c


def _rasterise_coast_to_dem(coast_pts, dem, transform):
    """Rasterise a closed coastline polygon onto the DEM grid."""
    from matplotlib.path import Path as _MPath
    a, _b, c, _d, e, f = transform
    h, w = dem.shape
    col_idx, row_idx = np.meshgrid(np.arange(w), np.arange(h))
    lon = a * col_idx + c
    lat = e * row_idx + f
    coast = np.asarray(coast_pts)
    src_xmin, src_xmax = coast[:, 0].min(), coast[:, 0].max()
    src_ymin, src_ymax = coast[:, 1].min(), coast[:, 1].max()
    tgt_xmin, tgt_xmax = lon.min(), lon.max()
    tgt_ymin, tgt_ymax = lat.min(), lat.max()
    dx_s = max(src_xmax - src_xmin, 1e-30)
    dy_s = max(src_ymax - src_ymin, 1e-30)
    dx_t = (tgt_xmax - tgt_xmin) * 0.85
    dy_t = (tgt_ymax - tgt_ymin) * 0.85
    s = min(dx_t / dx_s, dy_t / dy_s)
    src_cx = 0.5 * (src_xmin + src_xmax)
    src_cy = 0.5 * (src_ymin + src_ymax)
    tgt_cx = 0.5 * (tgt_xmin + tgt_xmax)
    tgt_cy = 0.5 * (tgt_ymin + tgt_ymax)
    mapped = np.column_stack([
        (coast[:, 0] - src_cx) * s + tgt_cx,
        (coast[:, 1] - src_cy) * s + tgt_cy,
    ])
    path = _MPath(mapped)
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    mask = path.contains_points(pts).reshape(h, w)
    return mask


if __name__ == "__main__":
    sys.exit(main())
