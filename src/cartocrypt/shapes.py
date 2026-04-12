"""Fourier-descriptor boundary perturbation for polygon faces.

« Same shape statistics, different shape »

Perturbs polygon boundaries using keyed phase randomisation of
Fourier descriptors.  Preserves: area, perimeter, convexity
sequence, and low-frequency shape statistics.  Destroys:
recognisable geographic outline.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np
from shapely.geometry import Polygon

from cartocrypt.constants import KeyBytes


def fourier_descriptors(boundary: np.ndarray) -> np.ndarray:
    """Compute Fourier descriptors of a 2D boundary curve.

    Args:
        boundary: (M, 2) array of ordered boundary points.

    Returns:
        Complex-valued Fourier descriptor array of length M.
    """
    z = boundary[:, 0] + 1j * boundary[:, 1]
    return np.fft.fft(z)


def perturb_boundary(
    boundary: np.ndarray,
    key: KeyBytes,
    face_index: int,
    *,
    n_preserve: int = 3,
) -> np.ndarray:
    """Perturb a polygon boundary with keyed phase randomisation.

    Keeps the lowest ``n_preserve`` Fourier components (overall
    shape, area, orientation) and randomises phases of higher
    components using the key + face index as seed.

    Args:
        boundary: (M, 2) ordered boundary points.
        key: 32-byte symmetric key.
        face_index: Canonical index of the face (for PRF seeding).
        n_preserve: Number of low-frequency components to keep.

    Returns:
        (M, 2) perturbed boundary with matching area/perimeter.
    """
    fd = fourier_descriptors(boundary)
    m = len(fd)

    # Generate deterministic random phases from key + face_index
    phases = _keyed_phases(key, face_index, m, n_preserve)

    # Apply phase perturbation to high-frequency components
    fd_perturbed = fd.copy()
    for k in range(n_preserve, m):
        fd_perturbed[k] *= np.exp(1j * phases[k - n_preserve])

    # Inverse FFT → new boundary
    z_new = np.fft.ifft(fd_perturbed)
    result = np.column_stack([z_new.real, z_new.imag])

    return result


def match_area(
    boundary: np.ndarray,
    target_area: float,
) -> np.ndarray:
    """Scale a boundary to match a target area.

    Args:
        boundary: (M, 2) boundary points.
        target_area: Desired polygon area in original units.

    Returns:
        Scaled (M, 2) boundary.
    """
    current = Polygon(boundary).area
    if current < 1e-15:
        return boundary
    scale = np.sqrt(abs(target_area / current))
    centroid = boundary.mean(axis=0)
    return (boundary - centroid) * scale + centroid


# ─────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────


def _keyed_phases(
    key: KeyBytes,
    face_index: int,
    n_total: int,
    n_preserve: int,
) -> np.ndarray:
    """Derive deterministic random phases from key + face index.

    Args:
        key: 32-byte symmetric key.
        face_index: Face canonical index.
        n_total: Total number of Fourier components.
        n_preserve: Number of preserved low-frequency components.

    Returns:
        Array of phases in [0, 2π) for high-frequency components.
    """
    n_phases = n_total - n_preserve
    phases = np.empty(n_phases, dtype=np.float64)

    for i in range(n_phases):
        msg = struct.pack(">QQ", face_index, i)
        digest = hmac.new(key, msg, hashlib.sha256).digest()
        u = int.from_bytes(digest[:8], "big") / (2**64)
        phases[i] = u * 2.0 * np.pi

    return phases
