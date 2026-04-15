"""Constants, colour palette, and type aliases for CartoCrypt.

« Shared definitions across all modules »
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# ─────────────────────────────────────────────────────────────────
#  Type aliases
# ─────────────────────────────────────────────────────────────────
Coords: TypeAlias = npt.NDArray[np.float64]  # (N, 2) array of xy
AdjMatrix: TypeAlias = npt.NDArray[np.int8]  # adjacency matrix
KeyBytes: TypeAlias = bytes  # 32-byte symmetric key

# ─────────────────────────────────────────────────────────────────
#  Wong (2011) colourblind-safe palette  « semantic mapping »
# ─────────────────────────────────────────────────────────────────
COLOURS: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",  # original graph / input
    "sky_blue":       "#56B4E9",  # anonymised graph / output
    "bluish_green":   "#009E73",  # preserved invariants
    "yellow":         "#F0E442",  # warnings / thresholds
    "blue":           "#0072B2",  # edges below breakpoint
    "vermilion":      "#D55E00",  # edges above breakpoint / errors
    "reddish_purple": "#CC79A7",  # key / crypto elements
}

# ─────────────────────────────────────────────────────────────────
#  Cryptographic defaults
# ─────────────────────────────────────────────────────────────────
KEY_LENGTH_BYTES: int = 32          # AES-256
HMAC_ALGORITHM: str = "sha256"
PRF_ALGORITHM: str = "aes-256-ctr"  # used for coordinate generation

# ─────────────────────────────────────────────────────────────────
#  Graph extraction defaults
# ─────────────────────────────────────────────────────────────────
DEFAULT_CRS: str = "EPSG:4326"
SIMPLIFICATION_TOLERANCE_M: float = 1.0  # Shapely simplify tolerance

# ─────────────────────────────────────────────────────────────────
#  Re-embedding optimisation
# ─────────────────────────────────────────────────────────────────
TUTTE_MAX_ITER: int = 5000
STRESS_FTOL: float = 1e-10
STRESS_MAX_ITER: int = 10000

# Joint stress+area objective: α balances the two soft constraints.
# α = 1.0 puts face-area and edge-length residuals on equal footing
# under the inverse-square weighting convention.
AREA_WEIGHT: float = 10.0
# Post-reembed sanity check: median relative face-area error above
# this value triggers a warning (not an exception) via verify.py.
AREA_RTOL: float = 0.05

# ─────────────────────────────────────────────────────────────────
#  Attribute splicing (chromosome-crossover mitigation)
# ─────────────────────────────────────────────────────────────────
# Faces below this area quantile are too small to carry meaningful
# payload and are excluded from splicing to avoid absurd swaps.
MIN_SPLICE_AREA_PCTILE: float = 0.05
# Within a splice cluster, paired faces must have area ratio ≤ this
# value.  Prevents a sliver from swapping payload with a landmass.
SPLICE_SIZE_RATIO_MAX: float = 4.0

# ─────────────────────────────────────────────────────────────────
#  Gestalt resynthesis (spectral coastline + fBm DEM)
# ─────────────────────────────────────────────────────────────────
SYNTHESIS_N_BOUNDARY_SAMPLES: int = 512
# The lowest k Fourier components of the coastline are preserved
# exactly (amplitude + phase) so the phantom island still "reads"
# as a blob of the right overall shape; higher frequencies get
# fresh keyed phases.
SYNTHESIS_PRESERVE_LOW_FREQ: int = 4
# Fallback Hurst exponent when radial power-spectrum estimation is
# too noisy (typical Mediterranean island terrain: ~0.7–0.8).
SYNTHESIS_FBM_HURST_DEFAULT: float = 0.75

# ── Universal-multifractal DEM parameters (Gagnon/Lovejoy) ───────
# α: Lévy index of the generator.  α = 2 is the log-normal cascade
# case, which is implementable with Gaussian noise.  α < 2 would
# need a Lévy-stable sampler (future sprint); for now we ship α = 2
# and document the upgrade path.
SYNTHESIS_MF_ALPHA: float = 2.0
# C₁: codimension of the mean.  Controls intermittency — sparseness
# of the extreme events that give ridges their bite.  Gagnon et al.
# (2006) report ~0.12 for global Earth topography.
SYNTHESIS_MF_C1: float = 0.12
# Default method for DEM resynthesis.  "multifractal" uses the
# universal-multifractal cascade; "fbm" keeps the plain fractional
# Brownian motion code path for comparison + backwards compat.
SYNTHESIS_DEM_METHOD: str = "multifractal"

# ─────────────────────────────────────────────────────────────────
#  Figure output
# ─────────────────────────────────────────────────────────────────
FIGURE_DPI: int = 300
SVG_FONTTYPE: str = "none"  # editable text in SVG
DEFAULT_FIG_SIZE: tuple[float, float] = (10.0, 8.0)

# ─────────────────────────────────────────────────────────────────
#  Aegina demo test case
# ─────────────────────────────────────────────────────────────────
# (north, south, east, west) — tight bbox around Aegina island
AEGINA_BBOX: tuple[float, float, float, float] = (37.78, 37.67, 23.58, 23.40)

OPENTOPO_API_URL: str = "https://portal.opentopography.org/API/globaldem"
OPENTOPO_DEMTYPE: str = "COP30"  # Copernicus GLO-30, 30 m resolution

# Semantic colours for the three Pokémon habitats — extension of
# the Wong palette, chosen so the three species remain distinct
# under the common forms of colour-vision deficiency.
POKEMON_COLOURS: dict[str, str] = {
    "Charizard": COLOURS["vermilion"],      # mountain fire
    "Pikachu":   COLOURS["yellow"],         # coastal sparks
    "Eevee":     COLOURS["bluish_green"],   # hidden valleys
}

# Futurama tube-metro line colours (Wong palette subset)
FUTURAMA_LINE_COLOURS: list[str] = [
    COLOURS["reddish_purple"],
    COLOURS["blue"],
    COLOURS["sky_blue"],
    COLOURS["orange"],
]
