# CartoCrypt

**Topology-preserving cryptographic anonymisation of cartographic datasets.**

Publish geographic maps — endangered species habitats, electrical grids,
sensitive infrastructure — without revealing real-world locations.

## What it does

CartoCrypt transforms a cartographic dataset into a **random geometric
configuration** that preserves:

- **Adjacencies** — the graph structure (which nodes connect to which)
- **Edge lengths** — road/grid segment distances
- **Face areas** — land parcel / water body sizes
- **Attribute data** — elevation, road class, land use, etc.

While **destroying** the actual coordinates, so the published map cannot
be geolocated.

## Key features

- **Deterministic keyed transformation** — a 32-byte symmetric key
  produces the same anonymised map every time.  Share the key with
  trusted collaborators so they can work on the same "phantom" geography
  across studies.
- **HMAC-SHA256 checksum** — verify that two parties holding the same
  key applied the transformation to the same source data.
- **Publication-ready figures** — Wong (2011) colourblind-safe palette,
  SVG output with editable text.

## Installation

```bash
conda env create -f environment.yml
conda activate cartocrypt
pip install -e ".[dev]" --break-system-packages
```

## Quick start

```bash
# Generate a key
cartocrypt keygen -o my_secret.key

# Anonymise an OSM bounding box (Dunedin city centre)
cartocrypt anonymise \
    --bbox "-45.86,-45.88,170.52,170.50" \
    --key my_secret.key \
    --output dunedin_anon.geojson

# Verify a checksum
cartocrypt verify \
    --key my_secret.key \
    --checksum abc123... \
    --geojson dunedin_anon.geojson
```

## Project structure

```
src/cartocrypt/
├── __init__.py     # Package metadata + ASCII banner
├── constants.py    # Colours, type aliases, defaults
├── ingest.py       # OSM / GeoJSON / Shapefile → labelled graph
├── canon.py        # Canonical labelling (WL hash)
├── keygen.py       # Key generation, PRF, HMAC checksum
├── reembed.py      # Constrained planar re-embedding (research core)
├── shapes.py       # Fourier boundary perturbation for polygons
├── export.py       # Graph → GeoJSON / SVG
├── verify.py       # Round-trip & topology verification
├── viz.py          # Publication figures (Wong palette)
└── cli.py          # Click CLI
```

## Citation

If you use CartoCrypt in your research, please cite:

```bibtex
@software{geurten2026cartocrypt,
  author    = {Geurten, Bart},
  title     = {{CartoCrypt}: Topology-Preserving Cryptographic
               Anonymisation of Cartographic Datasets},
  year      = {2026},
  url       = {https://github.com/zerotonin/cartocrypt},
  license   = {MIT},
}
```

## License

MIT — see [LICENSE](LICENSE).
