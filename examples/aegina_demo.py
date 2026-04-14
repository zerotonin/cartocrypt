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

from cartocrypt.aegina import AeginaIngestor
from cartocrypt.fake_layers import make_futurama_tubes, make_pokemon_habitats
from cartocrypt.keygen import generate_key
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

    out_stem = out_dir / "aegina"
    print(f"▶  Rendering figure → {out_stem}.svg / .png")
    fig = plot_aegina_figure(
        bundle, habitats, tubes, out_path=out_stem,
    )
    plt.close(fig)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
