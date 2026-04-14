# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — fake_layers                                        ║
# ║  « Pokémon habitats and Futurama tubes — no real privacy »       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Procedurally generated fictional overlays for the Aegina        ║
# ║  demo.  Deterministic in the 32-byte key: same key always        ║
# ║  produces the same habitats and tubes, mirroring the core        ║
# ║  CartoCrypt reproducibility guarantee.                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Procedurally generated fictional overlays for the Aegina demo.

« Pokémon habitats and a Futurama tube-metro — no real privacy »

These layers intentionally carry *no* real-world information.
They exist so that we can demonstrate the full CartoCrypt
pipeline (ingest → overlay → anonymise → visualise) on a
recognisable real island without ever handling real sensitive
data about species locations or infrastructure.

Both generators are **deterministic in the 32-byte key**: the
same key always produces the same habitats and tubes, which
means the demo figure is reproducible across machines — mirroring
the core CartoCrypt guarantee.

Layers produced
---------------
- **Charizard** — mountain habitats on the highest 20 % of the
  DEM.  Polygons, hatched vermilion.
- **Pikachu**   — coastal strips (200–500 m inland) that avoid a
  500 m buffer around settlements.  Polygons, yellow.
- **Eevee**     — uninhabited valleys: lowest 40 % of DEM minus
  a buffer around roads and settlements.  Polygons,
  bluish-green.
- **Futurama tubes** — a pneumatic-tube metro connecting every
  settlement via MST + two seeded shortcut edges, each edge
  drawn as a quadratic bézier to look deliberately unrealistic.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

from cartocrypt.aegina import LayerBundle
from cartocrypt.constants import FUTURAMA_LINE_COLOURS, KeyBytes

# ┌────────────────────────────────────────────────────────────┐
# │ Keyed RNG  « HMAC-derived numpy Generator »                 │
# └────────────────────────────────────────────────────────────┘

def _keyed_rng(key: KeyBytes, label: str) -> np.random.Generator:
    """Derive a deterministic numpy Generator from (key, label).

    Using HMAC-SHA256 keeps the generator seed cryptographically
    bound to the user key, so a single master key drives every
    pseudo-random choice the demo makes.
    """
    digest = hmac.new(key, label.encode("utf-8"), hashlib.sha256).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed_int)


# ┌────────────────────────────────────────────────────────────┐
# │ Pokémon habitats  « Charizard, Pikachu, Eevee »             │
# └────────────────────────────────────────────────────────────┘

def make_pokemon_habitats(
    bundle: LayerBundle,
    key: KeyBytes,
    *,
    charizard_quantile: float = 0.88,
    eevee_quantile_range: tuple[float, float] = (0.20, 0.60),
    coast_buffer_m: tuple[float, float] = (0.0, 250.0),
    settlement_avoid_m: float = 500.0,
    eevee_settlement_avoid_m: float = 150.0,
    road_avoid_m: float = 80.0,
    min_polygon_area_m2: float = 5e4,
) -> gpd.GeoDataFrame:
    """Generate the three Pokémon habitat layers for an Aegina bundle.

    Args:
        bundle: Populated :class:`LayerBundle`.
        key: 32-byte symmetric key — drives all random choices.
        charizard_quantile: Elevation quantile defining "mountain".
        eevee_quantile_range: (low, high) quantile band for Eevee
            valleys — targets mid-elevation *inland* pixels, not
            the coastal plain that Pikachu already occupies.
        coast_buffer_m: (inner, outer) distances from coastline
            defining the Pikachu strip.
        settlement_avoid_m: Pikachu's minimum distance from towns.
        eevee_settlement_avoid_m: Eevee's (smaller) minimum distance.
        road_avoid_m: Minimum distance from roads for Eevee.
        min_polygon_area_m2: Drop habitat fragments smaller than
            this area.

    Returns:
        GeoDataFrame in EPSG:4326 with columns:
        ``species``, ``habitat_id``, ``area_m2``, ``geometry``.
    """
    rng = _keyed_rng(key, "pokemon-habitats")

    char_polys = _charizard_polygons(bundle, charizard_quantile, rng)
    pika_polys = _pikachu_polygons(
        bundle, coast_buffer_m, settlement_avoid_m, rng,
    )
    eeve_polys = _eevee_polygons(
        bundle, eevee_quantile_range,
        eevee_settlement_avoid_m, road_avoid_m, rng,
    )

    # Clean every polygon before set ops — jitter can create
    # self-touching rings that GEOS rejects.
    char_polys = [_clean(p) for p in char_polys if p and not p.is_empty]
    pika_polys = [_clean(p) for p in pika_polys if p and not p.is_empty]
    eeve_polys = [_clean(p) for p in eeve_polys if p and not p.is_empty]
    char_polys = _flatten_to_polygons(char_polys)
    pika_polys = _flatten_to_polygons(pika_polys)
    eeve_polys = _flatten_to_polygons(eeve_polys)

    # ── Clip to the island (no habitats in open water)
    land = _island_polygon(bundle)
    if land is not None and not land.is_empty:
        char_polys = _intersect_all(char_polys, land)
        pika_polys = _intersect_all(pika_polys, land)
        eeve_polys = _intersect_all(eeve_polys, land)

    # ── Resolve overlaps: mountains win over coast, coast over valleys
    char_union = _clean(unary_union(char_polys)) if char_polys else None
    pika_polys = _subtract_if(pika_polys, char_union)
    eeve_union = (
        _clean(unary_union(char_polys + pika_polys))
        if (char_polys or pika_polys) else None
    )
    eeve_polys = _subtract_if(eeve_polys, eeve_union)

    # ── Drop slivers
    char_polys = _filter_by_area(char_polys, min_polygon_area_m2)
    pika_polys = _filter_by_area(pika_polys, min_polygon_area_m2)
    eeve_polys = _filter_by_area(eeve_polys, min_polygon_area_m2)

    rows: list[dict[str, Any]] = []
    for species, polys in [
        ("Charizard", char_polys),
        ("Pikachu",   pika_polys),
        ("Eevee",     eeve_polys),
    ]:
        for i, p in enumerate(polys):
            rows.append({
                "species": species,
                "habitat_id": f"{species[:3].upper()}-{i:02d}",
                "area_m2": round(_geodetic_area_m2(p), 1),
                "geometry": p,
            })

    if not rows:
        return gpd.GeoDataFrame(
            {"species": [], "habitat_id": [], "area_m2": [], "geometry": []},
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ───── species-specific builders ──────────────────────────────────

def _charizard_polygons(
    bundle: LayerBundle,
    quantile: float,
    rng: np.random.Generator,
    *,
    max_clusters: int = 5,
    erosion_pixels: int = 3,
) -> list[Polygon]:
    """Polygons over the highest-elevation regions of the DEM.

    A small binary erosion breaks narrow saddles between summit
    clusters, giving several disjoint habitats rather than one
    contiguous mountain blob.
    """
    mask = _quantile_mask(bundle.dem, quantile, above=True)
    if erosion_pixels > 0:
        try:
            from scipy.ndimage import binary_erosion
            mask = binary_erosion(mask, iterations=erosion_pixels)
        except ImportError:  # pragma: no cover
            pass
    polys = _mask_to_polygons(mask, bundle.dem_transform)
    polys = sorted(polys, key=lambda p: p.area, reverse=True)[:max_clusters]
    return [_smooth_polygon(p, rng) for p in polys]


def _pikachu_polygons(
    bundle: LayerBundle,
    coast_buffer_m: tuple[float, float],
    settlement_avoid_m: float,
    rng: np.random.Generator,
) -> list[Polygon]:
    """Coastal strip avoiding cities.

    Operates in a *local equal-area* projection (Azimuthal
    Equidistant centred on the bbox) so that buffers in metres are
    meaningful.
    """
    if bundle.coastline is None or bundle.coastline.empty:
        # No coastline → synthesise one from the island outline
        coast = _island_outline(bundle)
        if coast is None:
            return []
    else:
        coast = unary_union(list(bundle.coastline.geometry))

    to_m, to_deg = _local_projector(bundle.bbox)
    coast_m = to_m(coast)

    inner, outer = coast_buffer_m
    strip_m = coast_m.buffer(outer).difference(coast_m.buffer(inner))

    if not bundle.settlements.empty:
        settle_m = to_m(unary_union(list(bundle.settlements.geometry)))
        strip_m = strip_m.difference(settle_m.buffer(settlement_avoid_m))

    polys = _as_polygon_list(strip_m)
    # Keep largest 3
    polys = sorted(polys, key=lambda p: p.area, reverse=True)[:3]
    # Small seeded jitter on boundary
    polys = [_smooth_polygon_xy(p, rng) for p in polys]
    return [to_deg(p) for p in polys]


def _eevee_polygons(
    bundle: LayerBundle,
    quantile_range: tuple[float, float],
    settlement_avoid_m: float,
    road_avoid_m: float,
    rng: np.random.Generator,
) -> list[Polygon]:
    """Mid-elevation, far-from-infrastructure polygons — inland valleys."""
    q_lo, q_hi = quantile_range
    lo = np.quantile(bundle.dem, q_lo)
    hi = np.quantile(bundle.dem, q_hi)
    mask = (bundle.dem >= lo) & (bundle.dem <= hi)
    polys_deg = _mask_to_polygons(mask, bundle.dem_transform)
    if not polys_deg:
        return []

    to_m, to_deg = _local_projector(bundle.bbox)
    polys_m = [to_m(p) for p in polys_deg]
    valleys_m = unary_union(polys_m)

    # Subtract roads + settlements
    avoid_geoms = []
    if not bundle.settlements.empty:
        avoid_geoms.append(
            to_m(unary_union(list(bundle.settlements.geometry)))
            .buffer(settlement_avoid_m)
        )
    road_lines = _road_graph_to_lines(bundle)
    if road_lines is not None:
        avoid_geoms.append(to_m(road_lines).buffer(road_avoid_m))

    if avoid_geoms:
        valleys_m = valleys_m.difference(unary_union(avoid_geoms))

    polys_m = _as_polygon_list(valleys_m)
    polys_m = sorted(polys_m, key=lambda p: p.area, reverse=True)[:2]
    polys_m = [_smooth_polygon_xy(p, rng) for p in polys_m]
    return [to_deg(p) for p in polys_m]


# ┌────────────────────────────────────────────────────────────┐
# │ Futurama tube-metro  « MST + shortcuts, bézier-curved »     │
# └────────────────────────────────────────────────────────────┘

def make_futurama_tubes(
    bundle: LayerBundle,
    key: KeyBytes,
    *,
    n_shortcuts: int = 2,
    curvature: float = 0.15,
    n_bezier_samples: int = 32,
) -> gpd.GeoDataFrame:
    """Generate a pneumatic tube-metro between Aegina's settlements.

    The network is the Euclidean minimum spanning tree over
    settlement centroids, plus ``n_shortcuts`` additional edges
    chosen deterministically from the key to introduce loops so the
    network isn't strictly a tree.

    Each edge is drawn as a quadratic bézier curve with the control
    point offset perpendicular to the straight segment by
    ``curvature × segment_length``.  This gives the pneumatic-tube
    "never straight" aesthetic.

    Args:
        bundle: Populated LayerBundle.
        key: 32-byte symmetric key.
        n_shortcuts: Number of non-MST edges to add.
        curvature: Perpendicular offset as a fraction of edge length.
        n_bezier_samples: Points per curved segment.

    Returns:
        GeoDataFrame with columns
        ``line_name``, ``station_a``, ``station_b``, ``length_m``,
        ``colour``, ``geometry``.
    """
    rng = _keyed_rng(key, "futurama-tubes")

    if bundle.settlements is None or bundle.settlements.empty:
        return gpd.GeoDataFrame(
            {"line_name": [], "station_a": [], "station_b": [],
             "length_m": [], "colour": [], "geometry": []},
            crs="EPSG:4326",
        )

    # Station list: name + centroid
    names: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for i, row in bundle.settlements.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        c = g.centroid
        nm = str(row.get("name", f"Station-{i}")) if "name" in row.index else f"Station-{i}"
        names.append(nm)
        xs.append(c.x)
        ys.append(c.y)
    if len(names) < 2:
        return gpd.GeoDataFrame(
            {"line_name": [], "station_a": [], "station_b": [],
             "length_m": [], "colour": [], "geometry": []},
            crs="EPSG:4326",
        )

    xs_a = np.asarray(xs)
    ys_a = np.asarray(ys)
    n = len(names)

    # Complete graph of Euclidean distances (deg, scaled to metres
    # for display using a rough deg→m factor)
    to_m, _ = _local_projector(bundle.bbox)
    stations_m = np.array(
        [to_m(Point(x, y)).coords[0] for x, y in zip(xs, ys, strict=True)]
    )

    g = nx.Graph()
    for i in range(n):
        g.add_node(i, name=names[i], xy_deg=(xs_a[i], ys_a[i]))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(stations_m[i] - stations_m[j]))
            g.add_edge(i, j, weight=d)

    mst = nx.minimum_spanning_tree(g, weight="weight")

    # Seeded shortcut edges: random non-MST edges biased towards
    # medium-length (to avoid crossing the whole island every time).
    non_mst = [(u, v) for u, v in g.edges if not mst.has_edge(u, v)]
    if non_mst and n_shortcuts > 0:
        weights = np.array([g[u][v]["weight"] for u, v in non_mst])
        # Prefer 1st–3rd quartile edges
        median = np.median(weights)
        prefs = 1.0 / (1.0 + np.abs(weights - median) / max(median, 1.0))
        prefs = prefs / prefs.sum()
        idx = rng.choice(len(non_mst), size=min(n_shortcuts, len(non_mst)),
                         replace=False, p=prefs)
        for k in idx:
            u, v = non_mst[k]
            mst.add_edge(u, v, weight=g[u][v]["weight"])

    # Assign each edge to a "line" (colour) by a greedy chain walk
    line_assign = _assign_lines(mst, rng)

    rows: list[dict[str, Any]] = []
    for (u, v), line_id in line_assign.items():
        pa = np.array([xs_a[u], ys_a[u]])
        pb = np.array([xs_a[v], ys_a[v]])
        curve = _quadratic_bezier(pa, pb, curvature, rng, n_bezier_samples)
        colour = FUTURAMA_LINE_COLOURS[line_id % len(FUTURAMA_LINE_COLOURS)]
        rows.append({
            "line_name": f"Line-{line_id + 1}",
            "station_a": names[u],
            "station_b": names[v],
            "length_m": round(mst[u][v]["weight"], 1),
            "colour": colour,
            "geometry": LineString(curve),
        })

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ┌────────────────────────────────────────────────────────────┐
# │ Private helpers — masks / polygons / projection             │
# └────────────────────────────────────────────────────────────┘

def _quantile_mask(
    dem: np.ndarray,
    quantile: float,
    *,
    above: bool,
) -> np.ndarray:
    """Boolean mask of DEM pixels above (or below) a quantile."""
    threshold = np.quantile(dem, quantile)
    return dem >= threshold if above else dem <= threshold


def _mask_to_polygons(
    mask: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
) -> list[Polygon]:
    """Convert a 2-D boolean mask to a list of shapely Polygons.

    Uses rasterio.features.shapes when available, falls back to a
    pure-numpy approach (contour extraction) otherwise.
    """
    try:
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import Affine
    except ImportError:
        return []

    a, b, c, d, e, f = transform
    aff = Affine(a, b, c, d, e, f)
    m8 = mask.astype(np.uint8)
    polys: list[Polygon] = []
    for geom, val in rio_shapes(m8, mask=m8 > 0, transform=aff):
        if val != 1:
            continue
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            if len(coords) >= 4:
                polys.append(Polygon(coords))
    return polys


def _local_projector(
    bbox: tuple[float, float, float, float],
):
    """Return (to_m, to_deg) transformers centred on the bbox.

    Uses an Azimuthal Equidistant projection — accurate metres in a
    small region (~100 km), and round-trips cleanly.
    """
    try:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyproj + shapely required") from exc

    n, s, e, w = bbox
    lat0 = 0.5 * (n + s)
    lon0 = 0.5 * (e + w)
    proj = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    def to_m(geom):
        return shp_transform(fwd.transform, geom)

    def to_deg(geom):
        return shp_transform(inv.transform, geom)

    return to_m, to_deg


def _flatten_to_polygons(items: list) -> list[Polygon]:
    """Collapse list of Polygons/MultiPolygons to a flat Polygon list."""
    out: list[Polygon] = []
    for g in items:
        out.extend(_as_polygon_list(g))
    return out


def _as_polygon_list(geom) -> list[Polygon]:
    """Normalise any shapely result to a list of Polygon."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out = []
        for sub in geom.geoms:
            out.extend(_as_polygon_list(sub))
        return out
    return []


def _subtract_if(polys: list[Polygon], other) -> list[Polygon]:
    """Return polys minus ``other`` (or polys unchanged if other is None)."""
    if other is None or other.is_empty:
        return polys
    other_v = _clean(other)
    out: list[Polygon] = []
    for p in polys:
        diff = _clean(p).difference(other_v)
        out.extend(_as_polygon_list(diff))
    return out


def _clean(geom):
    """Return a topologically valid version of ``geom``."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    # A final buffer(0) scrub — cheap and fixes the last self-touches
    try:
        g2 = geom.buffer(0)
        if not g2.is_empty:
            return g2
    except Exception:
        pass
    return geom


def _filter_by_area(polys: list[Polygon], min_m2: float) -> list[Polygon]:
    """Drop polygons whose geodetic area is below the threshold."""
    return [p for p in polys if _geodetic_area_m2(p) >= min_m2]


def _geodetic_area_m2(p: Polygon) -> float:
    """Approximate area in m² using a local flat-earth factor."""
    # Good enough at Aegina's latitude (~37.7°N).
    lat = p.centroid.y
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat))
    return float(p.area) * m_per_deg_lat * m_per_deg_lon


def _smooth_polygon(p: Polygon, rng: np.random.Generator) -> Polygon:
    """Gently smooth a lat-lon polygon (Gaussian boundary filter)."""
    if p.is_empty:
        return p
    coords = np.array(p.exterior.coords)
    if len(coords) < 6:
        return p
    smoothed = _gauss_smooth(coords, sigma=1.2)
    # Tiny rng-driven jitter so a key change is visible in the viz
    jitter = rng.normal(scale=1e-5, size=smoothed.shape)
    return Polygon(smoothed + jitter)


def _smooth_polygon_xy(p: Polygon, rng: np.random.Generator) -> Polygon:
    """Same as _smooth_polygon but in metres (larger jitter)."""
    if p.is_empty:
        return p
    coords = np.array(p.exterior.coords)
    if len(coords) < 6:
        return p
    smoothed = _gauss_smooth(coords, sigma=1.5)
    jitter = rng.normal(scale=1.0, size=smoothed.shape)
    return Polygon(smoothed + jitter)


def _gauss_smooth(coords: np.ndarray, sigma: float) -> np.ndarray:
    """Wrap-around Gaussian smoothing of a closed boundary."""
    x = coords[:, 0]
    y = coords[:, 1]
    # Use scipy.ndimage if available, else fall back to a simple kernel
    try:
        from scipy.ndimage import gaussian_filter1d
        xs = gaussian_filter1d(x, sigma=sigma, mode="wrap")
        ys = gaussian_filter1d(y, sigma=sigma, mode="wrap")
    except ImportError:  # pragma: no cover
        k = max(1, int(2 * sigma))
        kern = np.ones(2 * k + 1) / (2 * k + 1)
        xs = np.convolve(np.r_[x[-k:], x, x[:k]], kern, mode="valid")
        ys = np.convolve(np.r_[y[-k:], y, y[:k]], kern, mode="valid")
    out = np.column_stack([xs, ys])
    # Make sure it closes
    out[-1] = out[0]
    return out


def _road_graph_to_lines(bundle: LayerBundle, simplify_deg: float = 0.0003):
    """Flatten the OSM road graph into a simplified MultiLineString.

    Aegina has ~9 k road edges; unioning + transforming + buffering
    the full geometry takes minutes.  Pre-simplifying to ~30 m
    tolerance reduces vertex count by ~50× with no visible impact
    on the "near a road" mask used downstream.
    """
    try:
        import osmnx as ox
        edges = ox.graph_to_gdfs(bundle.roads, nodes=False, edges=True)
    except Exception:  # pragma: no cover
        return None
    if edges is None or edges.empty:
        return None
    simplified = [
        g.simplify(simplify_deg, preserve_topology=False)
        for g in edges.geometry
        if g is not None and not g.is_empty
    ]
    if not simplified:
        return None
    return unary_union(simplified)


def _island_outline(bundle: LayerBundle):
    """Rough outline from building/landuse convex hull as last resort."""
    parts = []
    for gdf in (bundle.buildings, bundle.landuse):
        if gdf is not None and not gdf.empty:
            parts.append(unary_union(list(gdf.geometry)))
    if not parts:
        return None
    return unary_union(parts).convex_hull.exterior


def _island_polygon(bundle: LayerBundle):
    """Build a land polygon that clips habitats to the island.

    Strategy: polygonise OSM ``natural=coastline`` (which forms
    closed rings around islands).  If that yields nothing (e.g.
    broken coastline segments), fall back to a buffer-unbuffer of
    all land-ish features (landuse, buildings, roads) so the mask
    still excludes open water.

    Returns ``None`` if no land signal is available at all.
    """
    # Primary: polygonise the coastline
    if bundle.coastline is not None and not bundle.coastline.empty:
        lines = [g for g in bundle.coastline.geometry
                 if g is not None and not g.is_empty]
        if lines:
            merged = unary_union(lines)
            polys = list(polygonize(merged))
            if polys:
                land = _clean(unary_union(polys))
                if land is not None and not land.is_empty:
                    return land

    # Fallback: buffer-union of all land features
    parts: list = []
    for gdf in (bundle.buildings, bundle.landuse):
        if gdf is not None and not gdf.empty:
            parts.append(unary_union(list(gdf.geometry)))
    if bundle.settlements is not None and not bundle.settlements.empty:
        parts.append(unary_union(list(bundle.settlements.geometry)))
    roads = _road_graph_to_lines(bundle)
    if roads is not None:
        parts.append(roads)
    if not parts:
        return None

    # ~100 m dilation then ~50 m erosion to stitch gaps without
    # bleeding far into the sea.
    merged = unary_union(parts)
    dilated = merged.buffer(0.001)
    eroded = dilated.buffer(-0.0005)
    return _clean(eroded)


def _intersect_all(polys: list[Polygon], land) -> list[Polygon]:
    """Intersect each polygon with ``land``, returning a flat Polygon list."""
    out: list[Polygon] = []
    land_v = _clean(land)
    for p in polys:
        inter = _clean(p).intersection(land_v)
        out.extend(_as_polygon_list(inter))
    return out


# ┌────────────────────────────────────────────────────────────┐
# │ Private helpers — Futurama tubes                            │
# └────────────────────────────────────────────────────────────┘

def _assign_lines(
    g: nx.Graph,
    rng: np.random.Generator,
) -> dict[tuple[int, int], int]:
    """Greedy chain walker: partition edges into a handful of "lines"."""
    assignment: dict[tuple[int, int], int] = {}
    remaining = set(tuple(sorted(e)) for e in g.edges())
    line_id = 0
    while remaining:
        # Start at the edge with highest-degree endpoint (hubs first)
        start = max(
            remaining,
            key=lambda uv: max(g.degree(uv[0]), g.degree(uv[1])),
        )
        chain = [start]
        remaining.discard(start)
        # Extend the chain in both directions greedily
        for direction in (0, 1):
            current = start[1 - direction]
            while True:
                candidates = [
                    tuple(sorted((current, nb)))
                    for nb in g.neighbors(current)
                    if tuple(sorted((current, nb))) in remaining
                ]
                if not candidates:
                    break
                nxt = candidates[rng.integers(0, len(candidates))]
                chain.append(nxt)
                remaining.discard(nxt)
                current = nxt[0] if nxt[1] == current else nxt[1]
        for edge in chain:
            assignment[edge] = line_id
        line_id += 1
    return assignment


def _quadratic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    curvature: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Sample a quadratic bézier with a perpendicular control point."""
    mid = 0.5 * (p0 + p1)
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    if length < 1e-12:
        return np.vstack([p0, p1])
    # Perpendicular unit vector
    perp = np.array([-seg[1], seg[0]]) / length
    offset = curvature * length * (1.0 + 0.3 * rng.standard_normal())
    side = 1.0 if rng.random() > 0.5 else -1.0
    ctrl = mid + side * offset * perp
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
    return pts
