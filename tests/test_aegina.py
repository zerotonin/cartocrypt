"""Tests for the Aegina ingestion + fake-layer + viz pipeline.

Network-hitting tests are gated behind ``@pytest.mark.network`` so
they stay off by default (see ``pyproject.toml``).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from cartocrypt import aegina as aegina_mod
from cartocrypt.aegina import AeginaIngestor, LayerBundle
from cartocrypt.constants import AEGINA_BBOX
from cartocrypt.fake_layers import (
    make_futurama_tubes,
    make_pokemon_habitats,
)
from cartocrypt.viz_aegina import plot_aegina_figure


# ═════════════════════════════════════════════════════════════════
#  Synthetic bundle fixture (no network)
# ═════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_bundle() -> LayerBundle:
    """Build a small but complete LayerBundle without any network."""
    n, s, e, w = AEGINA_BBOX

    # DEM: radial hill on a 30×30 grid
    h, width = 30, 30
    step = (e - w) / width
    lons = np.linspace(w, e, width, dtype=np.float32)
    lats = np.linspace(n, s, h, dtype=np.float32)  # top-down
    lon_g, lat_g = np.meshgrid(lons, lats)
    cx, cy = 0.5 * (e + w), 0.5 * (n + s)
    r = np.hypot(lon_g - cx, lat_g - cy)
    dem = (530.0 * np.clip(1.0 - r / 0.08, 0.0, None)).astype(np.float32)
    transform = (step, 0.0, float(lons[0]), 0.0, -step, float(lats[0]))

    # Road graph: simple 5-node MultiDiGraph with x/y attrs + highway tag
    roads = nx.MultiDiGraph()
    pts = [
        (w + 0.02, s + 0.02),
        (w + 0.06, s + 0.03),
        (w + 0.10, s + 0.05),
        (w + 0.08, s + 0.08),
        (w + 0.04, s + 0.07),
    ]
    for i, (x, y) in enumerate(pts):
        roads.add_node(i, x=x, y=y, street_count=2)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    for u, v in edges:
        geom = LineString([pts[u], pts[v]])
        roads.add_edge(u, v, 0, geometry=geom, highway="residential",
                       length=geom.length * 111000.0)

    coastline = gpd.GeoDataFrame(
        {"geometry": [LineString([
            (w + 0.005, s + 0.005),
            (e - 0.005, s + 0.005),
            (e - 0.005, n - 0.005),
            (w + 0.005, n - 0.005),
            (w + 0.005, s + 0.005),
        ])]},
        crs="EPSG:4326",
    )
    buildings = gpd.GeoDataFrame(
        {"geometry": [
            Polygon([(w + 0.03, s + 0.03), (w + 0.035, s + 0.03),
                     (w + 0.035, s + 0.035), (w + 0.03, s + 0.035)]),
        ]},
        crs="EPSG:4326",
    )
    landuse = gpd.GeoDataFrame(
        {"landuse": ["forest"],
         "geometry": [Polygon([(w + 0.01, s + 0.06), (w + 0.06, s + 0.06),
                               (w + 0.06, s + 0.09), (w + 0.01, s + 0.09)])]},
        crs="EPSG:4326",
    )
    water = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    settlements = gpd.GeoDataFrame(
        {"name": ["Aegina Town", "Souvala", "Perdika"],
         "place": ["town", "village", "village"],
         "geometry": [
             Point(w + 0.03, s + 0.02),
             Point(w + 0.08, s + 0.05),
             Point(w + 0.05, s + 0.08),
         ]},
        crs="EPSG:4326",
    )
    peaks = gpd.GeoDataFrame(
        {"name": ["Oros"],
         "ele": [531.0],
         "geometry": [Point(cx, cy)]},
        crs="EPSG:4326",
    )

    return LayerBundle(
        roads=roads,
        buildings=buildings,
        coastline=coastline,
        landuse=landuse,
        water=water,
        settlements=settlements,
        peaks=peaks,
        dem=dem,
        dem_transform=transform,
        dem_source="synthetic",
        bbox=AEGINA_BBOX,
        meta={"synthetic": True},
    )


# ═════════════════════════════════════════════════════════════════
#  Ingestor (mocked)
# ═════════════════════════════════════════════════════════════════

class _FakeOx:
    """Minimal osmnx stand-in for offline ingestor tests."""

    def __init__(self) -> None:
        self.settings = type("S", (), {"cache_folder": "", "use_cache": True})()
        self.calls: list[str] = []

    def graph_from_bbox(self, bbox, network_type="all", simplify=True):
        self.calls.append(f"graph:{bbox}")
        g = nx.MultiDiGraph()
        left, bottom, right, top = bbox
        g.add_node(0, x=left + 0.01, y=bottom + 0.01, street_count=1)
        g.add_node(1, x=right - 0.01, y=top - 0.01, street_count=1)
        g.add_edge(0, 1, 0, length=100.0, highway="residential")
        return g

    def features_from_bbox(self, bbox, tags):
        self.calls.append(f"features:{sorted(tags)}")
        key = tuple(sorted(tags))
        if "natural" in tags and tags.get("natural") == ["peak"]:
            return gpd.GeoDataFrame(
                {"name": ["Oros"], "ele": [531.0],
                 "geometry": [Point(0.5 * (bbox[0] + bbox[2]),
                                    0.5 * (bbox[1] + bbox[3]))]},
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    def graph_to_gdfs(self, g, nodes=True, edges=True):
        import osmnx as ox_real
        return ox_real.graph_to_gdfs(g, nodes=nodes, edges=edges)


def test_ingestor_fetch_all_offline(monkeypatch, tmp_path):
    """Fetching with a mocked osmnx returns a fully populated bundle."""
    fake = _FakeOx()
    monkeypatch.setitem(__import__("sys").modules, "osmnx", fake)

    ing = AeginaIngestor(cache_dir=tmp_path)
    bundle = ing.fetch_all(use_dem=False)

    assert isinstance(bundle, LayerBundle)
    assert bundle.roads.number_of_nodes() == 2
    assert bundle.dem.shape[0] > 0 and bundle.dem.shape[1] > 0
    assert bundle.dem_source == "peak-idw"
    assert "graph:" in fake.calls[0]


def test_layerbundle_summary_runs(synthetic_bundle):
    s = synthetic_bundle.summary()
    assert "Aegina" in s
    assert "dem:" in s


# ═════════════════════════════════════════════════════════════════
#  Pokémon habitats
# ═════════════════════════════════════════════════════════════════

def test_pokemon_habitats_deterministic(synthetic_bundle):
    key = b"\x11" * 32
    a = make_pokemon_habitats(synthetic_bundle, key)
    b = make_pokemon_habitats(synthetic_bundle, key)
    assert len(a) == len(b)
    for g1, g2 in zip(a.geometry, b.geometry, strict=True):
        assert g1.equals_exact(g2, tolerance=1e-9)


def test_pokemon_habitats_changes_with_key(synthetic_bundle):
    a = make_pokemon_habitats(synthetic_bundle, b"\x01" * 32)
    b = make_pokemon_habitats(synthetic_bundle, b"\x02" * 32)
    if a.empty or b.empty:
        pytest.skip("habitat generation produced no polygons on this fixture")
    # At least one polygon's coordinates must differ
    diffs = [not g1.equals_exact(g2, tolerance=1e-9)
             for g1, g2 in zip(a.geometry, b.geometry, strict=False)]
    assert any(diffs)


def test_pokemon_habitats_species_present(synthetic_bundle):
    habitats = make_pokemon_habitats(synthetic_bundle, b"\x42" * 32)
    assert set(habitats["species"]).issubset({"Charizard", "Pikachu", "Eevee"})
    # Charizard should hit the central peak region → at least 1 polygon
    assert (habitats["species"] == "Charizard").any()


def test_pokemon_habitats_disjoint(synthetic_bundle):
    habitats = make_pokemon_habitats(synthetic_bundle, b"\x77" * 32)
    if habitats.empty:
        pytest.skip("no habitats produced")
    # Pairwise intersections should be empty (within tolerance)
    groups = {s: g for s, g in habitats.groupby("species")}
    species = list(groups.keys())
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            a = groups[species[i]].geometry.union_all()
            b = groups[species[j]].geometry.union_all()
            assert a.intersection(b).area < 1e-8


# ═════════════════════════════════════════════════════════════════
#  Futurama tubes
# ═════════════════════════════════════════════════════════════════

def test_futurama_tubes_connect_settlements(synthetic_bundle):
    tubes = make_futurama_tubes(synthetic_bundle, b"\x55" * 32)
    assert not tubes.empty
    n_settle = len(synthetic_bundle.settlements)
    # MST on n nodes has n-1 edges, plus shortcuts
    assert len(tubes) >= n_settle - 1


def test_futurama_tubes_connected_graph(synthetic_bundle):
    tubes = make_futurama_tubes(synthetic_bundle, b"\x99" * 32)
    g = nx.Graph()
    for _, row in tubes.iterrows():
        g.add_edge(row["station_a"], row["station_b"])
    assert nx.is_connected(g)


def test_futurama_tubes_deterministic(synthetic_bundle):
    key = b"\xaa" * 32
    a = make_futurama_tubes(synthetic_bundle, key)
    b = make_futurama_tubes(synthetic_bundle, key)
    assert list(a["station_a"]) == list(b["station_a"])
    assert list(a["station_b"]) == list(b["station_b"])
    for g1, g2 in zip(a.geometry, b.geometry, strict=True):
        assert g1.equals_exact(g2, tolerance=1e-9)


# ═════════════════════════════════════════════════════════════════
#  Viz smoke
# ═════════════════════════════════════════════════════════════════

def test_plot_aegina_figure_smoke(synthetic_bundle, tmp_path):
    key = b"\x33" * 32
    habitats = make_pokemon_habitats(synthetic_bundle, key)
    tubes = make_futurama_tubes(synthetic_bundle, key)

    out = tmp_path / "aegina"
    fig = plot_aegina_figure(
        synthetic_bundle, habitats, tubes, out_path=out,
    )
    assert fig is not None
    assert (tmp_path / "aegina.svg").stat().st_size > 2000
    assert (tmp_path / "aegina.png").stat().st_size > 2000
    import matplotlib.pyplot as plt
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════
#  Live-network tests (opt-in)
# ═════════════════════════════════════════════════════════════════

@pytest.mark.network
def test_real_aegina_download(tmp_path):
    """Actually hit OSM + (optionally) OpenTopography.  ~30 s."""
    ing = AeginaIngestor(cache_dir=tmp_path)
    bundle = ing.fetch_all(use_dem=True)
    assert bundle.roads.number_of_nodes() > 50
    assert bundle.dem_source in {"cop30", "peak-idw"}
