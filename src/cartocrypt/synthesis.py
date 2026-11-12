# ╔══════════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — synthesis                                          ║
# ║  « Aegina-like, but not Aegina »                                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Gestalt resynthesis: rebuild a coastline and DEM from the       ║
# ║  *statistical signatures* of the source (Fourier amplitudes of   ║
# ║  the boundary, radial power spectrum + hypsometric curve of      ║
# ║  the DEM) using fresh keyed phases.  The phantom island has the  ║
# ║  same flavour as the real one but a different specific shape.   ║
# ║                                                                  ║
# ║  Defeats shape-matching attacks while keeping topology           ║
# ║  (preserved by reembed) and attribute statistics (preserved by   ║
# ║  splicing) untouched.                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Gestalt resynthesis of coastline + DEM from statistical signatures.

« Aegina-like, but not Aegina »

Two independent primitives:

* :func:`resynthesise_boundary` — Fourier-descriptor resampling.
  Amplitudes from the source boundary; fresh keyed phases for the
  higher frequencies; low frequencies preserved so the output
  still "reads" as an island of the right overall blob shape.
* :func:`resynthesise_dem` — spectral fractional-Brownian-motion
  synthesis tuned to the source DEM's Hurst exponent, with
  hypsometric histogram matching so the elevation distribution is
  identical.
"""

from __future__ import annotations

import hashlib
import hmac

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon

from cartocrypt.constants import (
    SYNTHESIS_FBM_HURST_DEFAULT,
    SYNTHESIS_N_BOUNDARY_SAMPLES,
    SYNTHESIS_PRESERVE_LOW_FREQ,
    KeyBytes,
)

# ┌────────────────────────────────────────────────────────────┐
# │ Keyed RNG                                                   │
# └────────────────────────────────────────────────────────────┘


def _keyed_rng(key: KeyBytes, label: str) -> np.random.Generator:
    digest = hmac.new(key, label.encode("utf-8"), hashlib.sha256).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed_int)


# ┌────────────────────────────────────────────────────────────┐
# │ Coastline resynthesis  « Fourier amplitudes, fresh phases » │
# └────────────────────────────────────────────────────────────┘


def resynthesise_boundary(
    coastline: gpd.GeoDataFrame | LineString | MultiLineString | np.ndarray,
    key: KeyBytes,
    *,
    n_samples: int = SYNTHESIS_N_BOUNDARY_SAMPLES,
    preserve_low_freq: int = SYNTHESIS_PRESERVE_LOW_FREQ,
    preserve_perimeter: bool = True,
) -> np.ndarray:
    """Resample a closed boundary from its Fourier spectrum.

    The amplitudes ``|F_k|`` of the source boundary are kept; the
    phases at ``|k| >= preserve_low_freq`` are replaced with fresh
    keyed uniform draws (conjugate-symmetric so the inverse FFT
    stays real).  Phases at ``|k| < preserve_low_freq`` are
    retained, which locks in the overall blob shape.

    Args:
        coastline: Source boundary.  Accepted forms:

            * ``GeoDataFrame`` of Line / MultiLine / Polygon
              geometries (merged then polygonised).
            * Shapely ``LineString`` / ``MultiLineString``.
            * ``(M, 2)`` numpy array of ordered boundary coords.
        key: 32-byte symmetric key.
        n_samples: Resample length before FFT.
        preserve_low_freq: Number of low-frequency components whose
            phases are kept from the source.
        preserve_perimeter: If True, rescale the output so its
            perimeter matches the source perimeter.

    Returns:
        ``(n_samples, 2)`` array of boundary coords; first and last
        rows are equal (closed curve).
    """
    source_pts = _extract_closed_boundary(coastline)
    resampled = _arclength_resample(source_pts, n_samples)
    z = resampled[:, 0] + 1j * resampled[:, 1]

    F = np.fft.fft(z)
    amp = np.abs(F)
    phase = np.angle(F)

    # ── Fresh phases for |k| >= preserve_low_freq ──────────────
    rng = _keyed_rng(key, "synthesis-boundary")
    new_phase = phase.copy()
    n = len(F)
    half = n // 2
    for k in range(1, half + 1):
        if k < preserve_low_freq:
            continue
        new_phi = float(rng.uniform(0.0, 2.0 * np.pi))
        new_phase[k] = new_phi
        # Conjugate-symmetric partner
        if k != 0 and n - k != k:
            new_phase[n - k] = -new_phi

    F_new = amp * np.exp(1j * new_phase)
    z_new = np.fft.ifft(F_new)
    xy = np.column_stack([z_new.real, z_new.imag])

    if preserve_perimeter:
        src_perim = _closed_perimeter(resampled)
        new_perim = _closed_perimeter(xy)
        if new_perim > 1e-30:
            scale = src_perim / new_perim
            centre = xy.mean(axis=0)
            xy = (xy - centre) * scale + centre

    # Close the curve explicitly
    xy = np.vstack([xy, xy[:1]])
    return xy


# ┌────────────────────────────────────────────────────────────┐
# │ DEM resynthesis  « spectral multifractal + hypsometric match »│
# └────────────────────────────────────────────────────────────┘


def resynthesise_dem(
    dem: np.ndarray,
    key: KeyBytes,
    *,
    hurst: float | None = None,
    alpha: float | None = None,
    c1: float | None = None,
    mask: np.ndarray | None = None,
    method: str | None = None,
) -> tuple[np.ndarray, float]:
    """Generate a phantom DEM with matching statistical signature.

    Two methods are supported:

    * ``"multifractal"`` (default) — universal-multifractal cascade
      following Schertzer & Lovejoy (1987) and Gagnon, Lovejoy,
      Schertzer (2006).  A conservative log-normal flux is built
      in the spectral domain, exponentiated to create multiplicative
      intermittency, then fractionally integrated by the Hurst
      exponent to give the non-conservative terrain field.  Captures
      sharper ridges, steeper valleys, and non-Gaussian tails that
      plain fBm cannot produce.
    * ``"fbm"`` — original Mandelbrot-Voss fractional Brownian motion
      (single-parameter, Gaussian, isotropic).  Retained for
      comparison and backwards compatibility.

    Both methods histogram-match the result against the source's
    hypsometric curve so the full elevation distribution is
    preserved exactly.

    Args:
        dem: Source elevation grid.
        key: 32-byte symmetric key.
        hurst: Override the estimated Hurst exponent.  ``None`` →
            estimate from the source radial power spectrum; fall
            back to :data:`cartocrypt.constants.SYNTHESIS_FBM_HURST_DEFAULT`
            if estimation fails.
        alpha: Lévy index of the multifractal generator (``α`` in
            the universal-multifractal notation).  Default
            :data:`cartocrypt.constants.SYNTHESIS_MF_ALPHA` = 2.0
            (log-normal cascade).  ``α < 2`` would need a Lévy-stable
            sampler — not yet implemented.
        c1: Codimension of the mean (``C₁``).  Controls intermittency.
            Default
            :data:`cartocrypt.constants.SYNTHESIS_MF_C1` = 0.12,
            from Gagnon et al.'s fit to global Earth topography.
        mask: Optional 2-D boolean mask; cells where ``mask == False``
            are set to zero in the output.
        method: ``"multifractal"`` or ``"fbm"``.  Default from
            :data:`cartocrypt.constants.SYNTHESIS_DEM_METHOD`.

    Returns:
        ``(phantom_dem, hurst_used)`` — the synthetic grid plus the
        Hurst exponent that drove its spectrum.
    """
    from cartocrypt.constants import (
        SYNTHESIS_DEM_METHOD,
        SYNTHESIS_MF_ALPHA,
        SYNTHESIS_MF_C1,
    )
    method = (method or SYNTHESIS_DEM_METHOD).lower()
    if hurst is None:
        hurst = _estimate_hurst(dem)

    if method == "fbm":
        field = _synth_fbm_field(dem.shape, key, hurst)
    elif method == "multifractal":
        alpha = SYNTHESIS_MF_ALPHA if alpha is None else float(alpha)
        c1 = SYNTHESIS_MF_C1 if c1 is None else float(c1)
        if abs(alpha - 2.0) > 1e-9:
            # Lévy-stable sampler not yet implemented; fall back to
            # log-normal cascade with a warning rather than silently
            # producing the wrong distribution.
            import warnings
            warnings.warn(
                f"Lévy multifractal with α={alpha} not yet implemented; "
                "falling back to α=2 (log-normal cascade).",
                stacklevel=2,
            )
        field = _synth_multifractal_field(dem.shape, key, hurst, c1)
    else:
        msg = f"Unknown DEM synthesis method: {method!r}"
        raise ValueError(msg)

    field = _histogram_match(field, dem)

    if mask is not None:
        field = np.where(mask, field, 0.0)

    return field.astype(np.float32), float(hurst)


def _synth_fbm_field(
    shape: tuple[int, int],
    key: KeyBytes,
    hurst: float,
) -> np.ndarray:
    """Plain fractional Brownian motion via spectral synthesis."""
    h, w = int(shape[0]), int(shape[1])
    rng = _keyed_rng(key, "synthesis-dem-fbm")

    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    kxy = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    kxy[0, 0] = 1.0

    amp = kxy ** (-(hurst + 1.0))
    amp[0, 0] = 0.0

    noise = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    F = noise * amp
    return np.fft.ifft2(F).real


def _synth_multifractal_field(
    shape: tuple[int, int],
    key: KeyBytes,
    hurst: float,
    c1: float,
) -> np.ndarray:
    """Universal-multifractal (log-normal cascade) terrain synthesis.

    Implements the ``α = 2`` Schertzer-Lovejoy FIF (fractionally
    integrated flux) recipe:

    1. Build a Gaussian log-flux generator in the spectral domain
       with the characteristic ``|k|^{-1/α}`` envelope (here ``α = 2``).
    2. Rescale its variance so the multifractal codimension of the
       mean equals ``C₁`` at the image's outer scale.
    3. Exponentiate: ``ε = exp(γ)`` gives a positive conservative
       multifractal flux with log-normal statistics and heavy
       right-tail intermittency.
    4. Fractionally integrate the flux by ``|k|^{-H}`` to recover
       a non-conservative terrain field with the observed Hurst
       exponent.

    The result is histogram-matched downstream so any monotonic drift
    introduced by the cascade is absorbed — we rely on the cascade
    for *spatial* structure and on histogram matching for the
    hypsometric marginals.
    """
    h, w = int(shape[0]), int(shape[1])
    rng = _keyed_rng(key, "synthesis-dem-multifractal")

    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    kxy = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    kxy[0, 0] = 1.0

    # Step 1: log-flux generator — Gaussian with |k|^{-1/2} envelope
    # (the α=2 special case of |k|^{-1/α}).
    noise = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    gen_envelope = kxy ** (-0.5)  # 1/α with α=2
    gen_envelope[0, 0] = 0.0
    log_flux_spec = noise * gen_envelope
    log_flux = np.fft.ifft2(log_flux_spec).real

    # Step 2: scale log-flux variance so the multifractal codimension
    # of the mean matches C₁.  For α=2 the relation is
    #     Var[γ] = 2 C₁ ln(Λ_eff)
    # where Λ_eff is the effective outer/inner-scale ratio ≈ min(h, w).
    lambda_eff = max(min(h, w), 2)
    target_var = 2.0 * float(c1) * float(np.log(lambda_eff))
    current_var = float(np.var(log_flux))
    if current_var > 1e-30:
        log_flux *= np.sqrt(max(target_var, 0.0) / current_var)

    # Step 3: exponentiate → conservative multifractal flux.
    flux = np.exp(log_flux)
    # Normalise so E[ε] = 1 (conservative cascade by construction).
    flux /= max(float(flux.mean()), 1e-30)

    # Step 4: fractional integration by H.
    flux_spec = np.fft.fft2(flux)
    int_envelope = kxy ** (-float(hurst))
    int_envelope[0, 0] = 0.0
    field_spec = flux_spec * int_envelope
    return np.fft.ifft2(field_spec).real


# ┌────────────────────────────────────────────────────────────┐
# │ Terrain-role fields  « elevation + slope + coast distance » │
# └────────────────────────────────────────────────────────────┘


def compute_terrain_roles(
    dem: np.ndarray,
    coastline_pts: np.ndarray,
    dem_transform: tuple[float, float, float, float, float, float],
) -> dict[str, np.ndarray]:
    """Compute per-pixel terrain-role fields on a DEM.

    Returns three float32 grids, each normalised to ``[0, 1]`` by
    its own CDF (percentile rank):

    * ``elevation_pctile`` — 0.0 at sea level, 1.0 at the summit.
    * ``slope_pctile`` — 0.0 flat, 1.0 steepest.
    * ``coast_dist_pctile`` — 0.0 on the shoreline, 1.0 at the
      point furthest from any coastline vertex.

    These fields let us classify each graph node by its "terrain
    role" and match it to a phantom position with the same role.

    Args:
        dem: ``(H, W)`` elevation grid.
        coastline_pts: ``(M, 2)`` array of coastline vertices in
            the DEM's geographic CRS (lon, lat or x, y).
        dem_transform: Rasterio-style affine 6-tuple mapping
            pixel ``(col, row)`` to geographic ``(x, y)``.

    Returns:
        Dict with keys ``"elevation_pctile"``, ``"slope_pctile"``,
        ``"coast_dist_pctile"``; each value is a ``(H, W)`` float32
        array in ``[0, 1]``.
    """
    from scipy.spatial import cKDTree

    h, w = dem.shape
    a, _b, c, _d, e, f = dem_transform

    # ── Geographic coordinate grids ────────────────────────────
    col_idx, row_idx = np.meshgrid(np.arange(w), np.arange(h))
    lon = a * col_idx + c
    lat = e * row_idx + f

    # ── Elevation percentile ───────────────────────────────────
    elev_flat = dem.ravel().astype(np.float64)
    elev_pctile = _percentile_rank(elev_flat).reshape(h, w)

    # ── Slope (gradient magnitude) percentile ──────────────────
    dy, dx = np.gradient(dem.astype(np.float64))
    slope = np.sqrt(dx * dx + dy * dy)
    slope_pctile = _percentile_rank(slope.ravel()).reshape(h, w)

    # ── Coast distance percentile ──────────────────────────────
    coast = np.asarray(coastline_pts, dtype=np.float64)
    if coast.ndim != 2 or coast.shape[1] != 2:
        coast = coast.reshape(-1, 2)
    tree = cKDTree(coast)
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    dists, _ = tree.query(pts)
    coast_dist_pctile = _percentile_rank(dists).reshape(h, w)

    return {
        "elevation_pctile": elev_pctile.astype(np.float32),
        "slope_pctile":     slope_pctile.astype(np.float32),
        "coast_dist_pctile": coast_dist_pctile.astype(np.float32),
    }


def sample_terrain_role_at_coords(
    coords: np.ndarray,
    role_fields: dict[str, np.ndarray],
    dem_transform: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Look up terrain-role vectors for a set of geographic coords.

    Args:
        coords: ``(N, 2)`` geographic coordinates (lon/x, lat/y).
        role_fields: Output of :func:`compute_terrain_roles`.
        dem_transform: Same affine used for :func:`compute_terrain_roles`.

    Returns:
        ``(N, 3)`` array; columns are ``[elevation_pctile,
        slope_pctile, coast_dist_pctile]``.
    """
    a, _b, c, _d, e, f = dem_transform
    h, w = role_fields["elevation_pctile"].shape
    # Geographic → pixel
    cols = np.clip(((coords[:, 0] - c) / a).astype(int), 0, w - 1)
    rows = np.clip(((coords[:, 1] - f) / e).astype(int), 0, h - 1)

    out = np.empty((len(coords), 3), dtype=np.float32)
    out[:, 0] = role_fields["elevation_pctile"][rows, cols]
    out[:, 1] = role_fields["slope_pctile"][rows, cols]
    out[:, 2] = role_fields["coast_dist_pctile"][rows, cols]
    return out


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map values to their CDF rank in [0, 1]."""
    n = len(x)
    if n == 0:
        return x.copy()
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


# ┌────────────────────────────────────────────────────────────┐
# │ Hypsometric signature utilities                             │
# └────────────────────────────────────────────────────────────┘


def hypsometric_quantiles(
    dem: np.ndarray,
    n: int = 64,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the cumulative elevation distribution at ``n`` probes."""
    flat = dem[mask] if mask is not None else dem.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.zeros(n)
    qs = np.linspace(0.0, 1.0, n)
    return np.quantile(flat, qs)


# ┌────────────────────────────────────────────────────────────┐
# │ Internals — boundary                                        │
# └────────────────────────────────────────────────────────────┘


def _extract_closed_boundary(src) -> np.ndarray:
    """Normalise the many accepted input forms to an (M, 2) array."""
    import geopandas as _gpd
    if isinstance(src, np.ndarray):
        pts = np.asarray(src, dtype=np.float64)
    elif isinstance(src, (LineString, MultiLineString)):
        pts = _concatenate_linework(src)
    elif isinstance(src, _gpd.GeoDataFrame):
        if src.empty:
            raise ValueError("Empty coastline GeoDataFrame")
        geoms = [g for g in src.geometry if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError("No non-empty geometries in coastline")
        # Try to polygonise first; if that fails, concatenate linework
        try:
            from shapely.ops import polygonize, unary_union
            merged = unary_union(geoms)
            polys = list(polygonize(merged))
            if polys:
                # Use the largest polygon's exterior
                biggest = max(polys, key=lambda p: p.area)
                pts = np.asarray(biggest.exterior.coords, dtype=np.float64)
            else:
                pts = _concatenate_linework_list(geoms)
        except Exception:
            pts = _concatenate_linework_list(geoms)
    elif isinstance(src, Polygon):
        pts = np.asarray(src.exterior.coords, dtype=np.float64)
    else:
        raise TypeError(f"Unsupported coastline type: {type(src)}")

    # Ensure closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    return pts


def _concatenate_linework(geom) -> np.ndarray:
    if isinstance(geom, LineString):
        return np.asarray(geom.coords, dtype=np.float64)
    if isinstance(geom, MultiLineString):
        return _concatenate_linework_list(list(geom.geoms))
    raise TypeError(type(geom))


def _concatenate_linework_list(geoms) -> np.ndarray:
    arrays = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if isinstance(g, (LineString,)):
            arrays.append(np.asarray(g.coords, dtype=np.float64))
        elif isinstance(g, MultiLineString):
            for sub in g.geoms:
                arrays.append(np.asarray(sub.coords, dtype=np.float64))
        elif isinstance(g, Polygon):
            arrays.append(np.asarray(g.exterior.coords, dtype=np.float64))
    if not arrays:
        raise ValueError("No linework found")
    return np.vstack(arrays)


def _arclength_resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an ordered boundary to n points, uniform in arclength."""
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-30:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])
    return np.column_stack([x, y])


def _closed_perimeter(pts: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    return float(np.sum(d))


# ┌────────────────────────────────────────────────────────────┐
# │ Internals — DEM                                             │
# └────────────────────────────────────────────────────────────┘


def _estimate_hurst(dem: np.ndarray) -> float:
    """Estimate the Hurst exponent H via radial power-spectrum slope.

    Fits ``log S(k) = -(2H + 1) log k + c`` to the radially
    averaged power spectrum; solves for ``H``.  Returns the default
    constant if the fit is ill-conditioned.
    """
    h, w = dem.shape
    mean_removed = dem - float(np.mean(dem))
    F = np.fft.fft2(mean_removed)
    P = np.abs(F) ** 2

    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)

    k_flat = K.ravel()
    p_flat = P.ravel()
    # Radial averaging: bin by integer k
    k_bin = k_flat.astype(np.int64)
    valid = (k_bin >= 1) & (k_bin <= min(h, w) // 4)
    k_bin = k_bin[valid]
    p_flat = p_flat[valid]
    if k_bin.size < 8:
        return SYNTHESIS_FBM_HURST_DEFAULT
    bins = np.bincount(k_bin, weights=p_flat)
    counts = np.bincount(k_bin)
    keep = counts > 0
    radial = bins[keep] / np.maximum(counts[keep], 1)
    ks = np.arange(len(bins))[keep]
    if len(ks) < 4 or np.any(radial <= 0):
        return SYNTHESIS_FBM_HURST_DEFAULT
    try:
        slope, _ = np.polyfit(np.log(ks), np.log(radial), 1)
    except Exception:
        return SYNTHESIS_FBM_HURST_DEFAULT
    hurst = (-slope - 1.0) / 2.0
    # Clamp to the physically meaningful range for natural terrain
    return float(np.clip(hurst, 0.3, 0.95))


def _histogram_match(
    src: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """Remap ``src`` so its CDF matches ``ref``'s.

    Standard histogram-specification: rank-transform source, then
    read back values at matching ranks in the sorted reference.
    """
    src_flat = src.ravel()
    ref_flat = ref.ravel()
    # Rank of each src value as a fraction in [0, 1)
    order = np.argsort(src_flat)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(src_flat))
    u = ranks / max(len(src_flat) - 1, 1)
    # Read back from sorted reference
    ref_sorted = np.sort(ref_flat)
    idx = np.clip((u * (len(ref_sorted) - 1)).astype(np.int64),
                  0, len(ref_sorted) - 1)
    matched = ref_sorted[idx]
    return matched.reshape(src.shape).astype(src.dtype, copy=False)
