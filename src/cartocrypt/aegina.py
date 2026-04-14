# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — aegina                                             ║
# ║  « one small island, every layer downloaded »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reference test case: Aegina (~28 km², off Athens).              ║
# ║  Pulls OSM roads, buildings, coastline, landuse, water,          ║
# ║  settlements, peaks, plus a COP30 DEM (OpenTopography)           ║
# ║  with a peak-IDW fallback when no API key is set.                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Aegina test-case ingestion: download all layers for one small island.

« A human-readable reference case — real geography, fake privacy »

Aegina (~28 km², off Athens) is CartoCrypt's canonical demo subject.
Small enough to pull every OSM layer in seconds, recognisable
enough that anonymisation is visually striking, and — because the
"sensitive" overlays we test with are Pokémon habitats and
Futurama tubes — free of any real-world privacy concern.

This module fetches the *real* layers:
  - street network (osmnx)
  - buildings, coastline, landuse, water, settlements, peaks
  - a Copernicus GLO-30 DEM tile from OpenTopography

…and packages them into a :class:`LayerBundle` that the fake-layer
generators and the viz class can consume.

DEM access
----------
OpenTopography's GlobalDEM API is throttled for unauthenticated
users.  Set the ``OPENTOPO_API_KEY`` environment variable to a
free key (https://opentopography.org/developers) for real DEM
downloads.  Without a key we fall back to inverse-distance
weighted interpolation over OSM ``natural=peak`` points, which is
coarse but keeps the demo frictionless.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np

from cartocrypt.constants import (
    AEGINA_BBOX,
    OPENTOPO_API_URL,
    OPENTOPO_DEMTYPE,
)

logger = logging.getLogger(__name__)

# ┌────────────────────────────────────────────────────────────┐
# │ OSM layer tags  « one dict per downloadable feature class » │
# └────────────────────────────────────────────────────────────┘

_LAYER_TAGS: dict[str, dict[str, Any]] = {
    "buildings":   {"building": True},
    "coastline":   {"natural": ["coastline"]},
    "landuse":     {"landuse": True, "natural": ["wood", "scrub", "grassland", "bare_rock"]},
    "water":       {"natural": ["water"], "waterway": True},
    "settlements": {"place": ["town", "village", "hamlet", "suburb"]},
    "peaks":       {"natural": ["peak"]},
}


# ┌────────────────────────────────────────────────────────────┐
# │ LayerBundle  « the container returned by fetch_all »        │
# └────────────────────────────────────────────────────────────┘

@dataclass
class LayerBundle:
    """Collected Aegina layers as returned by :class:`AeginaIngestor`.

    All GeoDataFrames are in EPSG:4326.  ``dem`` is a 2-D float32
    array of elevations in metres; ``dem_transform`` is the affine
    transform mapping pixel (row, col) → (lon, lat).  When no DEM
    is available, ``dem`` carries an IDW-synthesised grid and
    ``dem_source`` is ``"peak-idw"``.

    Attributes:
        roads: OSM MultiDiGraph from :func:`osmnx.graph_from_bbox`.
        buildings: Polygon features.
        coastline: LineString / MultiLineString features.
        landuse: Polygon features (forest, scrub, residential…).
        water: Polygon / LineString features.
        settlements: Point features (place=town/village/hamlet).
        peaks: Point features (natural=peak).  Carries ``ele`` tag
            when present.
        dem: (H, W) float32 elevation grid in metres.
        dem_transform: Rasterio-style affine 6-tuple (a, b, c, d, e, f).
        dem_source: ``"cop30"`` | ``"peak-idw"`` | ``"none"``.
        bbox: (north, south, east, west).
    """

    roads: nx.MultiDiGraph
    buildings: gpd.GeoDataFrame
    coastline: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    settlements: gpd.GeoDataFrame
    peaks: gpd.GeoDataFrame
    dem: np.ndarray
    dem_transform: tuple[float, float, float, float, float, float]
    dem_source: str
    bbox: tuple[float, float, float, float]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-screen human-readable summary."""
        lines = [
            "Aegina LayerBundle",
            "=" * 40,
            f"  roads:       {self.roads.number_of_nodes():>5d} nodes / "
            f"{self.roads.number_of_edges():>5d} edges",
            f"  buildings:   {len(self.buildings):>5d} polygons",
            f"  coastline:   {len(self.coastline):>5d} lines",
            f"  landuse:     {len(self.landuse):>5d} polygons",
            f"  water:       {len(self.water):>5d} features",
            f"  settlements: {len(self.settlements):>5d} points",
            f"  peaks:       {len(self.peaks):>5d} points",
            f"  dem:         {self.dem.shape[0]}×{self.dem.shape[1]} "
            f"({self.dem_source})",
            f"  bbox (N,S,E,W): {self.bbox}",
        ]
        return "\n".join(lines)


# ┌────────────────────────────────────────────────────────────┐
# │ AeginaIngestor  « download + cache every layer »            │
# └────────────────────────────────────────────────────────────┘

class AeginaIngestor:
    """Download and cache every layer needed for the Aegina demo.

    Parameters
    ----------
    bbox
        Bounding box as ``(north, south, east, west)``.  Defaults
        to :data:`cartocrypt.constants.AEGINA_BBOX`.
    cache_dir
        Location for osmnx + pooch caches.  Created if missing.
    network_type
        osmnx ``network_type`` argument for the road graph.

    Notes
    -----
    The ingestor is stateless between calls — each
    :meth:`fetch_all` invocation is independent.  Reuse caching is
    handled by osmnx / pooch on disk, not by in-memory state.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float] = AEGINA_BBOX,
        cache_dir: Path | None = None,
        network_type: str = "all",
    ) -> None:
        self.bbox = bbox
        self.cache_dir = Path(
            cache_dir or Path("~/.cache/cartocrypt").expanduser()
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.network_type = network_type

    # ───── public API ─────────────────────────────────────────

    def fetch_all(
        self,
        *,
        use_dem: bool = True,
        dem_resolution_deg: float = 0.0003,
    ) -> LayerBundle:
        """Download every layer and return a :class:`LayerBundle`.

        Args:
            use_dem: If True (default), try to download a real DEM.
                Falls back to peak-IDW on any failure.
            dem_resolution_deg: Grid step for the IDW fallback only.
                ~0.0003° ≈ 30 m at this latitude.

        Returns:
            Populated LayerBundle.
        """
        import osmnx as ox  # lazy so test mocks can inject

        ox.settings.cache_folder = str(self.cache_dir / "osmnx")
        ox.settings.use_cache = True

        left, bottom, right, top = self._bbox_osmnx()

        logger.info("Fetching Aegina road network…")
        roads = ox.graph_from_bbox(
            bbox=(left, bottom, right, top),
            network_type=self.network_type,
            simplify=True,
        )

        layers: dict[str, gpd.GeoDataFrame] = {}
        for name, tags in _LAYER_TAGS.items():
            logger.info("Fetching OSM layer: %s", name)
            layers[name] = self._safe_features(
                ox, (left, bottom, right, top), tags, name,
            )

        # DEM
        if use_dem:
            dem, transform, source = self._fetch_dem()
        else:
            dem, transform, source = self._peak_idw(
                layers["peaks"], dem_resolution_deg,
            )

        return LayerBundle(
            roads=roads,
            buildings=layers["buildings"],
            coastline=layers["coastline"],
            landuse=layers["landuse"],
            water=layers["water"],
            settlements=layers["settlements"],
            peaks=layers["peaks"],
            dem=dem,
            dem_transform=transform,
            dem_source=source,
            bbox=self.bbox,
            meta={"network_type": self.network_type},
        )

    # ───── internals ──────────────────────────────────────────

    def _bbox_osmnx(self) -> tuple[float, float, float, float]:
        """Convert our (N, S, E, W) to osmnx v2's (left, bottom, right, top)."""
        n, s, e, w = self.bbox
        return (w, s, e, n)

    @staticmethod
    def _safe_features(
        ox_mod: Any,
        bbox: tuple[float, float, float, float],
        tags: dict[str, Any],
        name: str,
    ) -> gpd.GeoDataFrame:
        """Call features_from_bbox, returning an empty GDF on any failure."""
        try:
            gdf = ox_mod.features_from_bbox(bbox=bbox, tags=tags)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("OSM layer %s failed: %s", name, exc)
            return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        return gdf

    def _fetch_dem(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], str]:
        """Download a COP30 tile from OpenTopography, else fall back."""
        api_key = os.environ.get("OPENTOPO_API_KEY", "").strip()
        if not api_key:
            logger.info("OPENTOPO_API_KEY not set — using peak-IDW fallback.")
            return self._peak_idw_from_bundle()

        try:
            import pooch
            import rasterio
        except ImportError as exc:  # pragma: no cover
            logger.warning("rasterio / pooch missing (%s) — peak-IDW fallback.", exc)
            return self._peak_idw_from_bundle()

        n, s, e, w = self.bbox
        url = (
            f"{OPENTOPO_API_URL}?demtype={OPENTOPO_DEMTYPE}"
            f"&south={s}&north={n}&west={w}&east={e}"
            f"&outputFormat=GTiff&API_Key={api_key}"
        )
        fname = f"aegina_{OPENTOPO_DEMTYPE}_{n}_{s}_{e}_{w}.tif"
        try:
            path = pooch.retrieve(
                url=url,
                known_hash=None,
                fname=fname,
                path=self.cache_dir / "dem",
            )
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("OpenTopography fetch failed: %s — peak-IDW fallback.", exc)
            return self._peak_idw_from_bundle()

        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)
            t = src.transform
            transform = (t.a, t.b, t.c, t.d, t.e, t.f)
        return dem, transform, "cop30"

    def _peak_idw_from_bundle(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], str]:
        """Standalone IDW fallback that fetches peaks on its own."""
        import osmnx as ox

        left, bottom, right, top = self._bbox_osmnx()
        peaks = self._safe_features(
            ox, (left, bottom, right, top),
            _LAYER_TAGS["peaks"], "peaks",
        )
        return self._peak_idw(peaks, 0.0003)

    def _peak_idw(
        self,
        peaks: gpd.GeoDataFrame,
        step: float,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], str]:
        """Inverse-distance-weighted elevation grid from peak points.

        Uses the ``ele`` tag when present, otherwise a synthetic
        radial gradient from coastline (0 m) to island centre so
        the hillshade is at least visually plausible.

        Args:
            peaks: GeoDataFrame of Point features.
            step: Grid step in degrees.

        Returns:
            (dem, transform, "peak-idw").
        """
        n, s, e, w = self.bbox
        lons = np.arange(w, e, step, dtype=np.float32)
        lats = np.arange(s, n, step, dtype=np.float32)[::-1]  # top-down
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        dem = np.zeros_like(lon_grid, dtype=np.float32)

        pts: list[tuple[float, float, float]] = []
        if not peaks.empty:
            for _, row in peaks.iterrows():
                g = row.geometry
                if g is None or g.is_empty or g.geom_type != "Point":
                    continue
                try:
                    ele = float(row.get("ele", 0) or 0)
                except (TypeError, ValueError):
                    ele = 0.0
                if ele <= 0:
                    ele = 300.0  # reasonable default for a named peak
                pts.append((g.x, g.y, ele))

        if not pts:
            # Synthetic radial hill — keeps the viz working offline.
            cx = 0.5 * (e + w)
            cy = 0.5 * (n + s)
            r_max = 0.5 * max(e - w, n - s)
            d = np.hypot(lon_grid - cx, lat_grid - cy) / r_max
            dem = np.clip(530.0 * (1.0 - d**1.5), 0.0, None).astype(np.float32)
        else:
            # Vectorised IDW with power = 2 (broadcast over all pixels)
            xs = np.asarray([p[0] for p in pts], dtype=np.float32)
            ys = np.asarray([p[1] for p in pts], dtype=np.float32)
            zs = np.asarray([p[2] for p in pts], dtype=np.float32)
            eps = np.float32(1e-9)
            dx = lon_grid[:, :, None] - xs[None, None, :]
            dy = lat_grid[:, :, None] - ys[None, None, :]
            d2 = dx * dx + dy * dy + eps
            w_ij = 1.0 / d2
            dem = ((w_ij * zs[None, None, :]).sum(axis=-1)
                   / w_ij.sum(axis=-1)).astype(np.float32)

        # Affine (rasterio-style: a b c d e f):
        #   lon = a*col + b*row + c;  lat = d*col + e*row + f
        a = step
        b = 0.0
        c = float(lons[0])
        d = 0.0
        e_ = -step
        f = float(lats[0])
        return dem, (a, b, c, d, e_, f), "peak-idw"


# ┌────────────────────────────────────────────────────────────┐
# │ Convenience  « one-liner for scripts »                      │
# └────────────────────────────────────────────────────────────┘

def fetch_aegina(**kwargs: Any) -> LayerBundle:
    """Shortcut for ``AeginaIngestor().fetch_all()``."""
    ingestor = AeginaIngestor(
        bbox=kwargs.pop("bbox", AEGINA_BBOX),
        cache_dir=kwargs.pop("cache_dir", None),
        network_type=kwargs.pop("network_type", "all"),
    )
    return ingestor.fetch_all(**kwargs)
