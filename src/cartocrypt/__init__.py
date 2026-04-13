"""
╔═══════════════════════════════════════════════════════════════════╗
║   ██████╗ █████╗ ██████╗ ████████╗ ██████╗                      ║
║  ██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗                     ║
║  ██║     ███████║██████╔╝   ██║   ██║   ██║                     ║
║  ██║     ██╔══██║██╔══██╗   ██║   ██║   ██║                     ║
║  ╚██████╗██║  ██║██║  ██║   ██║   ╚██████╔╝                     ║
║   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝                     ║
║   ██████╗██████╗ ██╗   ██╗██████╗ ████████╗                      ║
║  ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝                     ║
║  ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║                        ║
║  ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║                        ║
║  ╚██████╗██║  ██║   ██║   ██║        ██║                         ║
║   ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝                         ║
║                                                                   ║
║  « Topology-preserving cryptographic map anonymisation »          ║
╚═══════════════════════════════════════════════════════════════════╝

CartoCrypt — publish geographic maps without revealing locations.

Transforms cartographic datasets into topology-preserving, key-
reproducible anonymous configurations.  Adjacencies, edge lengths,
face areas, and attribute distributions are retained; coordinates
are replaced by deterministic pseudorandom embeddings derived from
a symmetric key.

Modules
-------
ingest      Parse OSM / GeoJSON / Shapefile → labelled planar graph.
canon       Canonical labelling and graph hashing (WL / nauty).
keygen      Symmetric key generation, serialisation, HMAC checksum.
overlay     Attach project-specific layers (species, grids, etc.).
reembed     Constrained planar re-embedding (Tutte + stress opt).
shapes      Fourier-descriptor boundary perturbation for polygons.
export      Write anonymised data back to GeoJSON / Shapefile / SVG.
verify      Round-trip determinism and checksum verification.
cli         Click-based command-line interface.
viz         Publication-quality figure generation (Wong palette).
constants   Shared constants, colour definitions, type aliases.
"""

try:
    from cartocrypt._version import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"

__author__ = "Bart Geurten"
__email__ = "bart.geurten@otago.ac.nz"
__license__ = "MIT"
