"""Symmetric key generation, PRF coordinate mapping, and HMAC checksums.

« One key to cloak them all »

Provides:
- Key generation and serialisation (32-byte AES-256 keys).
- Keyed pseudorandom function (PRF) mapping canonical node indices
  to deterministic (x, y) coordinate pairs.
- HMAC-SHA256 checksum binding key ↔ graph canonical form.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from pathlib import Path

import numpy as np

from cartocrypt.constants import (
    HMAC_ALGORITHM,
    KEY_LENGTH_BYTES,
    Coords,
    KeyBytes,
)


def generate_key() -> KeyBytes:
    """Generate a cryptographically secure random symmetric key.

    Returns:
        32-byte key suitable for AES-256 / HMAC-SHA256.
    """
    return os.urandom(KEY_LENGTH_BYTES)


def save_key(key: KeyBytes, path: Path) -> None:
    """Write a key to a binary file.

    Args:
        key: 32-byte symmetric key.
        path: Destination file path.
    """
    path.write_bytes(key)


def load_key(path: Path) -> KeyBytes:
    """Read a key from a binary file.

    Args:
        path: Path to the key file.

    Returns:
        32-byte symmetric key.

    Raises:
        ValueError: If the file does not contain exactly 32 bytes.
    """
    data = path.read_bytes()
    if len(data) != KEY_LENGTH_BYTES:
        msg = f"Expected {KEY_LENGTH_BYTES} bytes, got {len(data)}"
        raise ValueError(msg)
    return data


def prf_coordinates(
    key: KeyBytes,
    node_index: int,
    *,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> tuple[float, float]:
    """Derive a deterministic (x, y) pair for a node from the key.

    Uses HMAC-SHA256 as a PRF: feeds the key and the canonical node
    index to produce 16 bytes, which are split into two float64
    values and scaled to the target bounding box.

    Args:
        key: 32-byte symmetric key.
        node_index: Canonical integer index of the node.
        bbox: Target bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        Tuple (x, y) within the target bbox.
    """
    # Derive 16 bytes from HMAC(key, index)
    msg = struct.pack(">Q", node_index)  # 8-byte big-endian uint64
    digest = hmac.new(key, msg, hashlib.sha256).digest()

    # Take first 8 bytes → uniform float in [0, 1), next 8 → same
    u = int.from_bytes(digest[:8], "big") / (2**64)
    v = int.from_bytes(digest[8:16], "big") / (2**64)

    x_min, y_min, x_max, y_max = bbox
    x = x_min + u * (x_max - x_min)
    y = y_min + v * (y_max - y_min)
    return x, y


def prf_coordinates_batch(
    key: KeyBytes,
    n_nodes: int,
    *,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> Coords:
    """Derive coordinates for all nodes in batch.

    Args:
        key: 32-byte symmetric key.
        n_nodes: Number of nodes.
        bbox: Target bounding box.

    Returns:
        (N, 2) array of (x, y) pairs.
    """
    coords = np.empty((n_nodes, 2), dtype=np.float64)
    for i in range(n_nodes):
        coords[i] = prf_coordinates(key, i, bbox=bbox)
    return coords


def compute_checksum(
    key: KeyBytes,
    graph_hash: str,
    attribute_hash: str,
) -> str:
    """Compute an HMAC-SHA256 checksum binding key to graph identity.

    Args:
        key: 32-byte symmetric key.
        graph_hash: WL hash of the graph structure.
        attribute_hash: SHA-256 hash of graph attributes.

    Returns:
        Hex digest checksum string.
    """
    payload = f"{graph_hash}|{attribute_hash}".encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_checksum(
    key: KeyBytes,
    graph_hash: str,
    attribute_hash: str,
    expected: str,
) -> bool:
    """Verify a checksum against expected value.

    Args:
        key: 32-byte symmetric key.
        graph_hash: WL hash of the graph structure.
        attribute_hash: SHA-256 hash of graph attributes.
        expected: Previously computed checksum to compare against.

    Returns:
        True if checksum matches, False otherwise.
    """
    computed = compute_checksum(key, graph_hash, attribute_hash)
    return hmac.compare_digest(computed, expected)
