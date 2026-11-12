"""Tests for gestalt resynthesis (coastline + DEM)."""

from __future__ import annotations

import numpy as np
import pytest

from cartocrypt import synthesis
from cartocrypt.constants import (
    SYNTHESIS_FBM_HURST_DEFAULT,
    SYNTHESIS_N_BOUNDARY_SAMPLES,
    SYNTHESIS_PRESERVE_LOW_FREQ,
)

# ───── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def blob_coastline() -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    r = 1.0 + 0.15 * np.sin(3 * theta) + 0.08 * np.cos(5 * theta) \
        + 0.03 * np.sin(11 * theta)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


@pytest.fixture
def synthetic_dem() -> np.ndarray:
    """Poor-man's fBm: low-pass filtered Gaussian noise."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((128, 128)).astype(np.float32)
    # Crude spectral low-pass via FFT
    F = np.fft.fft2(noise)
    ky = np.fft.fftfreq(128) * 128
    kx = np.fft.fftfreq(128) * 128
    K = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    K[0, 0] = 1.0
    F = F / (K ** 1.8)
    F[0, 0] = 0.0
    field = np.fft.ifft2(F).real.astype(np.float32)
    return (field - field.min()) * 500.0 / (field.max() - field.min())


# ───── Boundary resynthesis ────────────────────────────────

def test_boundary_resynth_preserves_power_spectrum(blob_coastline):
    new = synthesis.resynthesise_boundary(blob_coastline, key=b"\x11" * 32,
                                          preserve_perimeter=False)
    # Match the function's own pre-processing: close the curve
    # then arclength-resample.  Comparing against a direct
    # resample of the unclosed input gives a biased DC bin.
    closed = synthesis._extract_closed_boundary(blob_coastline)
    resampled_src = synthesis._arclength_resample(
        closed, SYNTHESIS_N_BOUNDARY_SAMPLES,
    )
    z_src = resampled_src[:, 0] + 1j * resampled_src[:, 1]
    z_new = new[:-1, 0] + 1j * new[:-1, 1]
    amp_src = np.abs(np.fft.fft(z_src))
    amp_new = np.abs(np.fft.fft(z_new))
    # Amplitudes must match element-wise (the whole point is that
    # resynthesis only touches phases).
    assert np.allclose(amp_src, amp_new, rtol=1e-6, atol=1e-9)


def test_boundary_resynth_different_from_source_above_low_freq(blob_coastline):
    new = synthesis.resynthesise_boundary(blob_coastline, key=b"\x22" * 32)
    closed = synthesis._extract_closed_boundary(blob_coastline)
    resampled_src = synthesis._arclength_resample(
        closed, SYNTHESIS_N_BOUNDARY_SAMPLES,
    )
    diffs = np.linalg.norm(new[:-1] - resampled_src, axis=1)
    assert np.max(diffs) > 0.01, "output is identical to source"


def test_boundary_resynth_deterministic(blob_coastline):
    a = synthesis.resynthesise_boundary(blob_coastline, key=b"\x33" * 32)
    b = synthesis.resynthesise_boundary(blob_coastline, key=b"\x33" * 32)
    assert np.allclose(a, b, atol=1e-12)


def test_boundary_resynth_closes_curve(blob_coastline):
    new = synthesis.resynthesise_boundary(blob_coastline, key=b"\x44" * 32)
    assert np.allclose(new[0], new[-1])


def test_boundary_resynth_preserves_low_freq(blob_coastline):
    """The preserved low-frequency components should round-trip."""
    new = synthesis.resynthesise_boundary(blob_coastline, key=b"\x55" * 32,
                                          preserve_perimeter=False)
    closed = synthesis._extract_closed_boundary(blob_coastline)
    resampled_src = synthesis._arclength_resample(
        closed, SYNTHESIS_N_BOUNDARY_SAMPLES,
    )
    z_src = resampled_src[:, 0] + 1j * resampled_src[:, 1]
    z_new = new[:-1, 0] + 1j * new[:-1, 1]
    F_src = np.fft.fft(z_src)
    F_new = np.fft.fft(z_new)
    # Low-frequency bins 0 .. preserve_low_freq-1 should match
    for k in range(SYNTHESIS_PRESERVE_LOW_FREQ):
        assert np.isclose(F_src[k], F_new[k], atol=1e-8), (
            f"low-freq mismatch at k={k}"
        )


# ───── DEM resynthesis ──────────────────────────────────────

def test_dem_resynth_shape_matches(synthetic_dem):
    phantom, _h = synthesis.resynthesise_dem(synthetic_dem, key=b"\x66" * 32)
    assert phantom.shape == synthetic_dem.shape


def test_dem_resynth_hypsometric_match(synthetic_dem):
    """After histogram matching, quantile curves must be near-identical."""
    phantom, _h = synthesis.resynthesise_dem(synthetic_dem, key=b"\x77" * 32)
    src_q = synthesis.hypsometric_quantiles(synthetic_dem, n=32)
    got_q = synthesis.hypsometric_quantiles(phantom, n=32)
    max_err = float(np.max(np.abs(src_q - got_q)))
    # Should be near-zero because histogram matching is exact at
    # the rank level; small float drift is OK.
    assert max_err < 1.0, f"hypsometric mismatch {max_err}"


def test_dem_resynth_deterministic(synthetic_dem):
    a, h_a = synthesis.resynthesise_dem(synthetic_dem, key=b"\x88" * 32)
    b, h_b = synthesis.resynthesise_dem(synthetic_dem, key=b"\x88" * 32)
    assert np.array_equal(a, b)
    assert h_a == h_b


def test_dem_resynth_respects_mask(synthetic_dem):
    mask = np.zeros_like(synthetic_dem, dtype=bool)
    mask[32:96, 32:96] = True  # central square is "land"
    phantom, _ = synthesis.resynthesise_dem(synthetic_dem, key=b"\x99" * 32,
                                            mask=mask)
    assert np.all(phantom[~mask] == 0.0)
    assert np.any(phantom[mask] != 0.0)


def test_dem_resynth_overridden_hurst(synthetic_dem):
    _, h = synthesis.resynthesise_dem(synthetic_dem, key=b"\xaa" * 32,
                                      hurst=0.6)
    assert h == 0.6


def test_hypsometric_quantiles_monotone(synthetic_dem):
    q = synthesis.hypsometric_quantiles(synthetic_dem, n=16)
    assert np.all(np.diff(q) >= -1e-9)


def test_dem_resynth_handles_flat_input():
    flat = np.full((32, 32), 42.0, dtype=np.float32)
    phantom, _ = synthesis.resynthesise_dem(
        flat, key=b"\xbb" * 32,
        hurst=SYNTHESIS_FBM_HURST_DEFAULT,
    )
    # Every phantom cell gets the constant — hypsometric match is trivial
    assert np.allclose(phantom, 42.0)


def test_dem_resynth_multifractal_non_gaussian_log_flux():
    """The multifractal raw field (before histogram match) should be
    log-normally distributed with heavier tails than a Gaussian fBm."""
    shape = (128, 128)
    key = b"\xcc" * 32
    mf_field = synthesis._synth_multifractal_field(shape, key, hurst=0.7, c1=0.12)
    fbm_field = synthesis._synth_fbm_field(shape, key, hurst=0.7)

    # Intermittency shows up in kurtosis / log-variance.  Multifractal
    # fields have excess kurtosis (heavier tails) vs. fBm.
    def _kurt(x):
        x = x - x.mean()
        v = x.var()
        return 0.0 if v <= 0 else float(np.mean(x ** 4) / (v ** 2)) - 3.0

    assert _kurt(mf_field) > _kurt(fbm_field), (
        "multifractal field should have heavier-tailed distribution "
        "than plain fBm"
    )


def test_dem_resynth_method_switch(synthetic_dem):
    """Both DEM synthesis methods produce valid output."""
    fbm_out, _ = synthesis.resynthesise_dem(
        synthetic_dem, key=b"\xdd" * 32, method="fbm",
    )
    mf_out, _ = synthesis.resynthesise_dem(
        synthetic_dem, key=b"\xdd" * 32, method="multifractal",
    )
    assert fbm_out.shape == synthetic_dem.shape
    assert mf_out.shape == synthetic_dem.shape
    # Same key, different methods → different outputs
    assert not np.array_equal(fbm_out, mf_out)


def test_dem_resynth_unknown_method_raises(synthetic_dem):
    with pytest.raises(ValueError, match="Unknown DEM synthesis method"):
        synthesis.resynthesise_dem(
            synthetic_dem, key=b"\xee" * 32, method="wibble",
        )
