"""Tests for CartoCrypt overlay module.

« Weta sightings, powerlines, and Pokemon — all welcome »

Tests the single-interface pattern: users bring a GeoDataFrame
(or a file geopandas can read), call add_layer(), done.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from cartocrypt.overlay import (
    add_layer,
    get_layers,
    get_lines,
    get_points,
    get_polygons,
    summarise,
)
from tests.aegina_fixture import make_aegina_graph


# ─────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base():
    """Aegina base graph + coords."""
    g, coords, _ = make_aegina_graph()
    return g, coords


@pytest.fixture
def weta_sightings():
    """15 simulated weta sightings scattered across Aegina."""
    rng = np.random.default_rng(42)
    n = 15
    return gpd.GeoDataFrame(
        {
            "species": rng.choice(
                ["H. maori", "H. crassidens", "H. thoracica"], n
            ),
            "count": rng.integers(1, 10, n).tolist(),
            "observer": [f"obs_{i}" for i in range(n)],
        },
        geometry=[
            Point(23.44 + rng.uniform(0, 0.09),
                  37.69 + rng.uniform(0, 0.08))
            for _ in range(n)
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def powerlines():
    """3 powerline segments crossing Aegina."""
    return gpd.GeoDataFrame(
        {
            "voltage_kv": [110, 33, 110],
            "operator": ["DEDDIE", "DEDDIE", "ADMIE"],
        },
        geometry=[
            LineString([(23.44, 37.72), (23.48, 37.73), (23.52, 37.74)]),
            LineString([(23.46, 37.70), (23.49, 37.73)]),
            LineString([(23.50, 37.75), (23.53, 37.72)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def habitats():
    """2 habitat polygons on Aegina."""
    return gpd.GeoDataFrame(
        {
            "habitat_type": ["pine_forest", "maquis_scrub"],
            "protection": ["NATURA2000", "local"],
        },
        geometry=[
            Polygon([(23.44, 37.71), (23.46, 37.71),
                     (23.46, 37.73), (23.44, 37.73)]),
            Polygon([(23.50, 37.74), (23.53, 37.74),
                     (23.53, 37.76), (23.50, 37.76)]),
        ],
        crs="EPSG:4326",
    )


# ─────────────────────────────────────────────────────────────────
#  Points
# ─────────────────────────────────────────────────────────────────

class TestPointLayer:
    def test_attach_counts(self, base, weta_sightings):
        g, coords = base
        r = add_layer(g, coords, weta_sightings, "weta")
        assert r["points"] == 15
        assert r["total"] == 15

    def test_retrieve_by_name(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        pts = get_points(g, "weta")
        assert len(pts) == 15

    def test_attributes_preserved(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        pts = get_points(g, "weta")
        species = {a["species"] for _, a in pts}
        assert "H. maori" in species or "H. crassidens" in species

    def test_snap_distance_recorded(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        for _, a in get_points(g, "weta"):
            assert "_snap_dist_m" in a
            assert a["_snap_dist_m"] >= 0

    def test_original_xy_stored(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        for _, a in get_points(g, "weta"):
            assert "_original_xy" in a
            assert len(a["_original_xy"]) == 2

    def test_beyond_tolerance_flagged(self, base):
        g, coords = base
        far = gpd.GeoDataFrame(
            {"note": ["very far away"]},
            geometry=[Point(25.0, 39.0)],
            crs="EPSG:4326",
        )
        r = add_layer(g, coords, far, "remote", snap_tolerance_m=100.0)
        assert r["beyond_tolerance"] == 1
        pts = get_points(g, "remote")
        assert pts[0][1]["_beyond_tolerance"] is True


# ─────────────────────────────────────────────────────────────────
#  Lines
# ─────────────────────────────────────────────────────────────────

class TestLineLayer:
    def test_attach_counts(self, base, powerlines):
        g, coords = base
        r = add_layer(g, coords, powerlines, "power")
        assert r["lines"] == 3
        assert r["total"] == 3

    def test_retrieve_lines(self, base, powerlines):
        g, coords = base
        add_layer(g, coords, powerlines, "power")
        lns = get_lines(g, "power")
        assert len(lns) == 3

    def test_attributes_preserved(self, base, powerlines):
        g, coords = base
        add_layer(g, coords, powerlines, "power")
        lns = get_lines(g, "power")
        voltages = {a.get("voltage_kv") for _, _, a in lns}
        assert 110 in voltages
        assert 33 in voltages


# ─────────────────────────────────────────────────────────────────
#  Polygons
# ─────────────────────────────────────────────────────────────────

class TestPolygonLayer:
    def test_attach_counts(self, base, habitats):
        g, coords = base
        r = add_layer(g, coords, habitats, "habitat")
        assert r["polygons"] == 2
        assert r["total"] == 2

    def test_boundary_stored(self, base, habitats):
        g, coords = base
        add_layer(g, coords, habitats, "habitat")
        polys = get_polygons(g, "habitat")
        assert len(polys) == 2
        for p in polys:
            assert "_boundary" in p
            assert len(p["_boundary"]) >= 4

    def test_area_computed(self, base, habitats):
        g, coords = base
        add_layer(g, coords, habitats, "habitat")
        for p in get_polygons(g, "habitat"):
            assert p["_area_m2"] > 0

    def test_attributes_preserved(self, base, habitats):
        g, coords = base
        add_layer(g, coords, habitats, "habitat")
        types = {p["habitat_type"] for p in get_polygons(g, "habitat")}
        assert types == {"pine_forest", "maquis_scrub"}


# ─────────────────────────────────────────────────────────────────
#  File loading
# ─────────────────────────────────────────────────────────────────

class TestFileLoading:
    def test_geojson_file(self, base):
        g, coords = base
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [23.48, 37.73]},
                    "properties": {"pokemon": "Pikachu", "cp": 1200},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [23.50, 37.74]},
                    "properties": {"pokemon": "Snorlax", "cp": 3100},
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as f:
            json.dump(fc, f)
            path = Path(f.name)

        r = add_layer(g, coords, path, "pokemon")
        assert r["points"] == 2
        pts = get_points(g, "pokemon")
        names = {a["pokemon"] for _, a in pts}
        assert "Pikachu" in names
        path.unlink()

    def test_string_path(self, base):
        g, coords = base
        fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [23.49, 37.73]},
                "properties": {"sensor": "A1"},
            }],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as f:
            json.dump(fc, f)
            path_str = f.name

        r = add_layer(g, coords, path_str, "sensors")
        assert r["points"] == 1
        Path(path_str).unlink()

    def test_missing_file_raises(self, base):
        g, coords = base
        with pytest.raises(FileNotFoundError):
            add_layer(g, coords, "/no/such/file.geojson", "nope")


# ─────────────────────────────────────────────────────────────────
#  Multi-layer
# ─────────────────────────────────────────────────────────────────

class TestMultiLayer:
    def test_two_layers(self, base, weta_sightings, habitats):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        add_layer(g, coords, habitats, "habitat")
        assert len(get_layers(g)) == 2

    def test_duplicate_name_rejected(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        with pytest.raises(ValueError, match="already exists"):
            add_layer(g, coords, weta_sightings, "weta")

    def test_filter_by_name(self, base, weta_sightings, habitats):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        add_layer(g, coords, habitats, "habitat")
        assert len(get_points(g, "weta")) == 15
        assert len(get_points(g, "habitat")) == 0
        assert len(get_polygons(g, "habitat")) == 2
        assert len(get_polygons(g, "weta")) == 0

    def test_unfiltered_returns_all(self, base, weta_sightings, habitats):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        add_layer(g, coords, habitats, "habitat")
        assert len(get_points(g)) == 15
        assert len(get_polygons(g)) == 2

    def test_three_layers(self, base, weta_sightings, powerlines, habitats):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        add_layer(g, coords, powerlines, "power")
        add_layer(g, coords, habitats, "habitat")
        layers = get_layers(g)
        assert len(layers) == 3
        names = {l["name"] for l in layers}
        assert names == {"weta", "power", "habitat"}


# ─────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────

class TestSummary:
    def test_empty(self, base):
        g, _ = base
        assert "No project layers" in summarise(g)

    def test_with_layers(self, base, weta_sightings):
        g, coords = base
        add_layer(g, coords, weta_sightings, "weta")
        s = summarise(g)
        assert "weta" in s
        assert "15 pts" in s


# ─────────────────────────────────────────────────────────────────
#  Error handling
# ─────────────────────────────────────────────────────────────────

class TestErrors:
    def test_empty_gdf(self, base):
        g, coords = base
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="no features"):
            add_layer(g, coords, empty, "empty")

    def test_wrong_type(self, base):
        g, coords = base
        with pytest.raises(TypeError, match="GeoDataFrame"):
            add_layer(g, coords, 42, "bad")
