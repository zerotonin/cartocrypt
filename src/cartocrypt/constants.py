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

# ─────────────────────────────────────────────────────────────────
#  Figure output
# ─────────────────────────────────────────────────────────────────
FIGURE_DPI: int = 300
SVG_FONTTYPE: str = "none"  # editable text in SVG
DEFAULT_FIG_SIZE: tuple[float, float] = (10.0, 8.0)
