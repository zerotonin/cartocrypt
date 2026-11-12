# Refactor notes — better statistical island capture and reversibility

*« where the phantom gets its gestalt, and who gets to reverse it »*

Design note written during the splicing + resynthesis sprint, after
the first phantom-Aegina figure exposed two honest gaps:

1. The phantom island looks under-populated in the interior because
   Tutte + pure fBm DEM capture only a weak slice of what makes
   Aegina *feel* like Aegina.
2. It is unclear which of CartoCrypt's operators is reversible with
   the key, which is only statistically reversible, and which is
   irreversible by construction.

This note records the literature survey, the reversibility taxonomy,
and the sequenced upgrade plan that the two follow-up sprints will
execute.  It lives alongside `threat_model.rst` and should be cited
from the paper's discussion section.

---

## 1  How close can we come to a statistical twin of the source island?

Ranking the approaches by fidelity, from what we ship today upward.

### 1.1  Fractional Brownian motion (current)

* Voss 1988; Mandelbrot 1983.  One parameter: the Hurst exponent
  `H`, equivalent to fractal dimension `D = 2 − H` for coastlines
  or `D = 3 − H` for 2-D surfaces.
* Isotropic, Gaussian, stationary.  Captures scale-invariant
  roughness and a single characteristic coastline-fractal-dimension
  (Mediterranean islands sit around `H ≈ 0.7–0.9`, `D ≈ 1.1–1.3`).
* Misses: anisotropy (real islands have preferred geological
  axes), intermittency (real terrain is *spikier* than Gaussian —
  sharp ridges and steep valleys), dendritic drainage, sharp
  ridgelines.

What you see in the current phantom DEM is an unrealistically smooth
"soft cushion" surface.  The gestalt of *an island* is there, but
the gestalt of *a mountain* is not.

References:
* Mandelbrot B.B. (1983) *The Fractal Geometry of Nature*.
* Voss R. (1988) in *The Science of Fractal Images*.
* Gagnon J.-S., Lovejoy S., Schertzer D. (2006)
  *Multifractal Earth topography*, **Nonlinear Processes in
  Geophysics** 13, 541–570.

### 1.2  Universal multifractal cascades — the next rung

The Schertzer-Lovejoy universal-multifractal framework describes a
random field with *three* parameters instead of fBm's one:

| Symbol | Role | Typical terrain value |
|---|---|---|
| `α`   | Lévy index — tail heaviness of the generator.  `α = 2` is log-normal; `α < 2` gives power-law tails. | 1.7 – 2.0 |
| `C₁`  | Codimension of the mean — intermittency / sparseness of extremes. | 0.10 – 0.15 |
| `H`   | Non-conservation exponent (fractional-integration order).  Same as fBm's `H`. | 0.5 – 0.8 |

Plus an anisotropy matrix if you buy into Generalized Scale
Invariance.

The model is constructive:

1. Draw a generator `γ(r)` — Lévy-α noise filtered by `|k|^{-1/α}`
   (or Gaussian noise, for the `α = 2` log-normal case).
2. Exponentiate: `ε = exp(γ)` gives a positive **multiplicative
   cascade** — a conservative multifractal flux with heavy-tailed
   intermittency.
3. Fractionally integrate by `H`: `T = |k|^{-H} ε` gives the
   non-conservative terrain field.
4. Histogram-match to the source hypsometric curve (as we already
   do).

The payoff is visual and statistical: sharper ridges, steeper
valleys, non-Gaussian tails, scale-dependent roughness.  The
engineering cost is one extra parameter estimator (C₁ from
log-variance), one extra Fourier filter (the `|k|^{-1/α}`
pre-filter), and one `exp()`.  **This is the upgrade CartoCrypt
should take first**; the `α = 2` case is implementable in a day
and already beats plain fBm.

Parameter estimation is well-studied:

* `H` — from the slope of the radial power spectrum of the
  *integrated* field (same as fBm).  Our current estimator gives
  the fractional-integration part for free.
* `C₁` — from the slope of the Lovejoy–Schertzer *K(q)* moment
  function at `q = 1`, or (pragmatically) from the sample log-variance
  normalised by `ln(Λ_max / Λ_min)`.
* `α` — from *double trace moments*.  Default `α = 1.8` is a safe
  generic for terrain (Gagnon et al. 2006); upgrade to live
  estimation only if the demo needs it.

References:
* Schertzer D., Lovejoy S. (1987)
  *Physical modeling and analysis of rain and clouds by anisotropic
  scaling multiplicative processes*, **J. Geophys. Res.** 92,
  9693–9714.
* Lovejoy S., Schertzer D. (2013)
  *The Weather and Climate: Emergent Laws and Multifractal Cascades*,
  Cambridge University Press.
* Gagnon, Lovejoy, Schertzer (2006), as above.

### 1.3  Conditional Gaussian simulation (geostatistics)

* Kriging + sequential Gaussian simulation.  Input: a variogram
  learned from the source, optional conditioning points.
* Tunable privacy / fidelity dial: zero conditioning → free-form
  phantom; some conditioning → the phantom shares specific anchor
  features with the source (for reviewers' sanity checks); full
  conditioning → near-identity.
* Mature Python stack: `GStatSim` (2023).

This is the right framework once we want *both* statistical
fidelity *and* selective feature preservation.  For the first
multifractal sprint we skip it; the second sprint (or an IJGIS
ablation) could add it.

References:
* Yin Z., Strebelle S., Caers J. (2023)
  *GStatSim V1.0: a Python package for geostatistical interpolation
  and conditional simulation*, **GMD** 16, 3765–3779.
* Cressie N. (1993) *Statistics for Spatial Data*, Wiley.

### 1.4  Landscape evolution models

Physics-based: stream-power law for fluvial incision + linear
diffusion for hillslope.  Given an initial heightfield and a set of
uplift / precipitation / resistance rates, you get back a DEM with
realistic drainage networks and sharp ridges.

Pragmatic use for CartoCrypt: *post-process* a multifractal initial
condition by running a fake geological history of a few million
iterations of SPL-plus-diffusion.  This trades generality for
geomorphological realism and is the only method that produces
dendritic drainage organically.

Honest caveat: LEM tuning is an art and sits outside the statistical
framework — it's physics simulation with half a dozen knobs.

References:
* Tucker G., Hancock G. (2010)
  *Modelling landscape evolution*, **Earth Surface Processes and
  Landforms** 35, 28–50.
* Howard A.D. (1994)
  *A detachment-limited model of drainage basin evolution*,
  **Water Resources Research** 30, 2261–2285.

### 1.5  Learned priors — GANs and diffusion models

Trained generative models on global DEM tiles.  The current state
of the art (Hu et al. 2024) uses conditional diffusion with sketch
guidance to produce DEMs indistinguishable from real terrain at
multiple scales.

Strengths: visually unbeatable, captures non-Gaussian structure at
*all* scales without needing to specify a model.

Weaknesses: requires training data + weights; adversarial concerns
around memorisation of training tiles; depends on a model whose
behaviour on out-of-distribution coastlines is hard to bound.

For an IJGIS paper this is probably too heavy; for a long-term
roadmap it is the eventual ceiling.

References:
* Panagiotou E. et al. (2020)
  *Procedural 3-D terrain generation using generative adversarial
  networks*, arXiv:2010.06411.
* Hu Y. et al. (2024)
  *Interactive authoring of terrain using diffusion models*,
  HAL-04324336.

### 1.6  Recommendation

**Upgrade path for CartoCrypt:**

> `current fBm` → `universal multifractal (α=2 log-normal)` →
> optional: `conditional Gaussian simulation` +
> `anisotropic multifractal` → long-term: `diffusion model`.

The `α = 2` multifractal is the highest-ROI step: one extra
parameter, dramatic visual improvement, drop-in replacement for the
current DEM resynthesiser.  That is what the very next sprint
implements.

---

## 2  Reversibility — can a key-holder recover the source?

Each of CartoCrypt's operators falls into one of three
information-theoretic regimes.

### 2.1  Bijective operators — trivially reversible

| Operator | Mathematical form | Inverse |
|---|---|---|
| **Splicing** | Per-face permutation within each adjacency cluster, drawn from a keyed derangement. | Apply the keyed inverse permutation.  Each spliced fragment is already tagged with `_spliced_from_face` / `_spliced_to_face`, so the inverse is literally reading those columns backwards. |
| **Re-embedding** | Deterministic function of the graph + key via Tutte → stress → area.  PRF-seeded initial positions make the output a closed-form function of the input. | *Numerically* recoverable — store the PRF intermediate state + original coords.  *Analytically* not reversible because L-BFGS-B convergence loses precision. |
| **Fourier boundary perturbation** | Amplitudes untouched; phases replaced with a keyed random draw. | Store the original phases (or the seed that generated the new ones) and reapply the inverse mask. |

Implication: **a user who knows the key can fully reconstruct the
source** for these three operators, up to the numerical precision
of the optimisation.  For cryptographic publication they behave as
deterministic keyed substitution ciphers on spatial data.

Practically, CartoCrypt should expose `unsplice_attributes(...)` and
`unperturb_boundary(...)` in the next reversibility sprint.  For
`re-embed` the cleanest mechanism is a PRF snapshot header.

### 2.2  Lossy operators — irreversible by design

**Gestalt resynthesis** (coastline + DEM) is a designed *statistical
preimage attack* on the source.  The output preserves only:

* The Fourier amplitude spectrum of the coastline.
* The full elevation CDF (hypsometric curve), via histogram matching.
* The universal multifractal parameters once we upgrade.

Infinitely many source DEMs satisfy any given tuple
`(amplitude-spectrum, hypsometric-CDF, α, C₁, H)`.  That
many-to-one property *is the privacy guarantee*.  Consequently
there is no function `phantom → source` that depends only on the
published phantom + a key, because the information needed to
distinguish the true source from its statistical siblings is gone.

### 2.3  How to make lossy operators reversible — RDHEI

**Reversible Data Hiding in Encrypted Images** (Zhang 2011 onward)
solves exactly this problem for raster images:

* Alice owns the source.  She encrypts and publishes the phantom.
* Bob (data-hider) embeds the *reversal residual* `Δ = source −
  resynthesised` — or a keyed-encrypted version of it —
  steganographically in the phantom's unused entropy (e.g. LSBs,
  prediction-error residuals, MSB-block labels).
* Carol (key-holder) decrypts the residual and reconstructs the
  source exactly.  Without the key she sees only the phantom.

For CartoCrypt this maps cleanly onto the DEM: the phantom GeoTIFF
has ~24 bits per pixel of capacity, the compressed residual of a
well-matched multifractal resynthesis is typically 2–4 bits per
pixel, so a full reversible phantom is feasible.  The literature
survey below covers the three main schools (RRBE, VRBE, VRAE).

References:
* Zhang X. (2011)
  *Reversible data hiding in encrypted image*, **IEEE Signal
  Processing Letters** 18, 255–258.
* Wu H.-T., Dugelay J.-L., Cheung Y.-M. (2018)
  *Reversible data hiding in encrypted images based on RGB↔YCbCr
  conversion*.
* Qian Z., Zhang X. (2016)
  *Reversible data hiding in encrypted image with public-key
  cryptography*, **Signal Processing: Image Communication**.
* Puteaux P., Puech W. (2018)
  *An efficient MSB prediction-based method for high-capacity
  reversible data hiding in encrypted images*, **IEEE TIFS** 13,
  1670–1681.

### 2.4  Format-preserving encryption

An orthogonal route: treat the entire map (coords + attributes) as
a structured plaintext and encrypt with a keyed bijection whose
ciphertext has the same *format*.  Ciphertext → plaintext via the
key (lossless).  Wrong tool if you want a *publishable* phantom —
FPE ciphertexts look like random valid maps, not statistically
meaningful phantoms.  Right tool for archival where only
key-holders ever see the map.

References:
* Bellare M., Ristenpart T., Rogaway P., Stegers T. (2009)
  *Format-preserving encryption*, **Selected Areas in
  Cryptography**.
* Dworkin M. (2016)
  *Recommendation for Block Cipher Modes of Operation:
  Methods for Format-Preserving Encryption* (FF1, FF3), NIST
  Special Publication 800-38G.

### 2.5  Design taxonomy for CartoCrypt

Three composable regimes, explicit trade-offs:

| Regime | Output | Reversibility | Use case |
|---|---|---|---|
| **Bijective-only** (splicing + re-embedding + Fourier perturbation) | Aegina-shaped map, attributes shuffled, coords rotated | Lossless with the key | Collaboration between key-holders; reviewer copies. |
| **Lossy publishable** (+ gestalt resynthesis) | Phantom island | Statistically impossible to reverse from the published artefact alone | Public publication where specifics must not be recoverable. |
| **Lossy + RDHEI sidecar** | Phantom that *looks* publishable, carries encrypted reversal payload | Lossless with the key; indistinguishable from the pure lossy phantom without it | Publish the "same" phantom, but trusted parties can reconstruct. |

This is also the axis we should add to `threat_model.rst` as an
explicit column: **reversibility** becomes a first-class property of
every operator alongside *what-it-defeats*.

---

## 3  Sprint queue after the multifractal upgrade

In order of ROI:

1. **Multifractal DEM upgrade** — this sprint.  `α = 2`
   log-normal cascade + C₁ intermittency + fractional integration
   by H.  Parameters estimated from the source.  Drop-in replacement
   for `resynthesise_dem`.
2. **Reversibility audit** — write inverses for splicing and
   Fourier perturbation, snapshot PRF state for re-embedding, add a
   reversibility column to the threat-model table, and expose a
   `cartocrypt unsplice` CLI alongside the existing `anonymise`.
3. **RDHEI sidecar** — embed an encrypted source-minus-resynthesised
   residual in the phantom DEM's LSBs so that a key-holder can
   losslessly reconstruct the source while a public reader sees only
   the phantom.  Use Puteaux-Puech MSB prediction for capacity.
4. **Anisotropic multifractal** — extend to `α < 2` with Lévy-stable
   sampling and a Generalized-Scale-Invariance anisotropy matrix
   estimated from the source's directional power spectrum.  Adds
   sharp-ridge realism.
5. **Conditional Gaussian simulation** — selective-feature
   preservation for sanity checks.
6. **Diffusion-model terrain** — research project; not for the
   IJGIS submission.

---

## 4  Summary

Current CartoCrypt has three operators (re-embedding, splicing,
resynthesis) and one composed pipeline (phantom-Aegina demo).  The
pipeline's phantom looks under-populated in the interior because
Tutte has no metric on the phantom frame and fBm has no
non-Gaussian structure.  Both failures have standard fixes:

* Rescale target edge lengths by the phantom-perimeter / real-perimeter
  ratio so stress pulls interior nodes outward on the phantom frame.
* Replace fBm with a universal multifractal (Gagnon/Lovejoy) so the
  DEM has realistic intermittency.

Reversibility is a *per-operator* property.  Two of CartoCrypt's
three operators are bijections and their inverses are one
permutation-flip away; the third (resynthesis) is lossy by
construction, and making it reversible requires the standard
RDHEI trick of hiding the reversal residual in the phantom's
unused entropy.

This note's sprint queue points at the multifractal upgrade first
(highest visual ROI, lowest risk).  RDHEI and anisotropic
multifractal follow in order of diminishing marginal return.
