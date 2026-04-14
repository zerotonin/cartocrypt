"""End-to-end Aegina demo: real ingest + fake overlays + viz.

Run from the repo root with the cartocrypt conda env active:

    python examples/aegina_demo.py

Writes ``examples/output/aegina.svg`` and ``aegina.png``.

Set ``OPENTOPO_API_KEY`` in your environment for a real Copernicus
GLO-30 hillshade.  Without it the script falls back to a peak-IDW
synthesised DEM — still runs, just coarser terrain.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import networkx as nx
import time

from cartocrypt.aegina import AeginaIngestor
from cartocrypt.fake_layers import make_futurama_tubes, make_pokemon_habitats
from cartocrypt.ingest import to_labelled_graph
from cartocrypt.keygen import generate_key, prf_coordinates_batch
from cartocrypt.reembed import reembed
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

    # Deterministic demo key so the figure is reproducible.  Replace
    # with ``generate_key()`` to see how a new key reshuffles the
    # habitats + tubes.
    _ = generate_key  # silence unused import in pinned-key mode
    key = bytes.fromhex(
        "43617274 6f437279 70742041 6567696e 61204465 6d6f2053 65656421".replace(" ", "")
    ).ljust(32, b"\0")[:32]

    print("▶  Generating fictional overlays…")
    habitats = make_pokemon_habitats(bundle, key)
    tubes = make_futurama_tubes(bundle, key)
    print(f"   habitats: {len(habitats)} polygons "
          f"({sorted(set(habitats['species']))})")
    print(f"   tubes:    {len(tubes)} segments")

    # ── Anonymise the road graph ────────────────────────────────
    # L-BFGS-B scales poorly on ~4 000-node graphs (roadmap item).
    # For the demo we reembed only the largest connected component;
    # isolated sub-islands fall through un-anonymised and are drawn
    # as-is.  Production scaling is a future sprint.
    print("▶  Re-embedding road network…")
    g_full, coords_full, _ = to_labelled_graph(bundle.roads)
    components = sorted(nx.connected_components(g_full), key=len, reverse=True)
    biggest = next(iter(components))
    g_cc = g_full.subgraph(biggest).copy()
    node_index = {n: i for i, n in enumerate(g_cc.nodes)}
    coords_cc = _reorder(coords_full, g_full, g_cc)
    g_cc = nx.relabel_nodes(g_cc, node_index)

    print(f"   largest CC: {g_cc.number_of_nodes()} nodes, "
          f"{g_cc.number_of_edges()} edges "
          f"({len(components)} component(s) total)")

    seed = prf_coordinates_batch(
        key, g_cc.number_of_nodes(),
        bbox=(float(coords_cc[:, 0].min()), float(coords_cc[:, 1].min()),
              float(coords_cc[:, 0].max()), float(coords_cc[:, 1].max())),
    )
    t0 = time.perf_counter()
    anon_coords = reembed(g_cc, coords_cc, seed,
                          preserve_lengths=True, preserve_areas=True)
    print(f"   reembed: {time.perf_counter() - t0:.1f} s")

    metrics = verify_metrics(g_cc, coords_cc, anon_coords)
    print(f"   length err: mean={metrics['length_mean_rel_error']:.3f} "
          f"max={metrics['length_max_rel_error']:.3f}")
    print(f"   area   err: median={metrics['area_median_rel_error']:.3f} "
          f"p95={metrics['area_p95_rel_error']:.3f} "
          f"(within tol: {metrics['area_within_tol']})")

    out_stem = out_dir / "aegina"
    print(f"▶  Rendering figure → {out_stem}.svg / .png")
    fig = plot_aegina_figure(
        bundle, habitats, tubes,
        anon_graph=g_cc, anon_coords=anon_coords,
        out_path=out_stem,
    )
    plt.close(fig)

    print("Done.")
    return 0


def _reorder(coords_full, g_full, g_cc):
    """Extract the rows of ``coords_full`` that correspond to ``g_cc``."""
    import numpy as np
    idx_full = {n: i for i, n in enumerate(g_full.nodes)}
    rows = [idx_full[n] for n in g_cc.nodes]
    return np.asarray(coords_full)[rows]


if __name__ == "__main__":
    sys.exit(main())
