.. _threat_model:

Threat Model — the Crocodile Problem
====================================

*« What an adversary sees, and what they can still figure out. »*

CartoCrypt's core promise is that a published map preserves
topology and shape statistics while destroying geolocatable
coordinates.  Under a strong threat model, however, *coordinate
destruction is not the whole attack surface*.  This chapter
documents the known attack we call the **crocodile problem**, the
cryptographic analogues that motivate our mitigations, and the
mitigation roadmap that sits outside the current CartoCrypt
implementation.

The crocodile analogy
---------------------

Imagine a substitution cipher on ASCII.  A friend proves it is
unbreakable by a direct brute-force keyspace attack.  You still
break it, not by guessing the key, but by making the friend say
the word *crocodile* somehow — then looking for a nine-letter
ciphertext word with the same first-and-fourth-letter repeat.
Frequency analysis defeats the cipher because the plaintext space
has structured constraints the key cannot erase.

Maps have the same vulnerability.  If the published dataset is a
topology-preserving re-embedding of a *specific* real island —
say Aegina, with exactly one peak above 500 m, 19 settlements,
~28 km² landmass, and a coastline of a particular length — then
an adversary with access to any atlas can re-identify the dataset
regardless of how well the coordinates are scrambled.  The
**feature multiset itself is the crocodile**.

Attack statement
----------------

*Given:* a published CartoCrypt artefact (anonymised graph +
preserved shape statistics + checksum).

*Adversary knows:* the general domain (coastal Greece / species X
habitat / electrical grid of region R), and has access to
auxiliary geographic databases.

*Attack:* compute a multiset signature from the published artefact
(number of high-degree faces, face-area histogram, perimeter
distribution, edge-length distribution) and search auxiliary
databases for candidate regions whose multiset matches.  If the
equivalence class is small (often a single candidate),
re-identification succeeds.

What CartoCrypt's current design *does* defeat
----------------------------------------------

* **Coordinate inversion.**  PRF-seeded initial positions plus
  stress majorisation with face-area correction produce
  deterministic but unpredictable coordinates.  Without the
  symmetric key, recovering real (lon, lat) from published
  (x, y) is infeasible.
* **Contour matching.**  Fourier-descriptor boundary perturbation
  in :mod:`cartocrypt.shapes` randomises high-frequency shape
  components while preserving area, perimeter, and low-order
  moments — so the outline of any individual face cannot be
  matched by high-frequency correlation against a reference
  coastline.
* **Graph-structure probes.**  The Weisfeiler-Lehman canonical
  hash binds the checksum to the graph's isomorphism class, not
  to the specific node labelling — so label-permutation attacks
  against the checksum fail.

What it does *not* defeat
-------------------------

* **Feature-multiset matching** — the crocodile attack.
* **Graph-isomorphism matching against a calibrated atlas** — if
  auxiliary databases contain the graph structures of every
  island in the Aegean, WL-hash equality between the published
  checksum and a database entry is a direct re-identification.
* **Semantic-label leaks** — publishing elevation as ``ele=531``
  (Mt. Oros' actual altitude) rather than a band ``ele∈[500,600]``
  re-exposes feature identity.
* **Temporal correlation attacks** — publishing two snapshots of
  the same region under different keys lets an adversary
  cross-reference preserved invariants to confirm they are the
  same underlying map.

Cryptographic analogues
-----------------------

The crocodile problem is, in cryptographic terminology, a
**known-plaintext attack against a permutation cipher with a
structured plaintext space**.  Bart's crocodile anecdote is
exactly classical letter-frequency analysis on a monoalphabetic
substitution cipher.  The mitigations mirror those used in
cryptography for the same problem:

=============================  ==========================================
Cryptographic technique        Geographic analogue
=============================  ==========================================
Chaffing and winnowing         Decoy feature injection ("spawn mountains")
Mimic functions [Wayner92]_    Generating covers from the same statistical
                               distribution as the real data
Deniable encryption            Hidden-map cover: a second *plausible*
[Canetti97]_, VeraCrypt        geography that decrypts under a decoy key
k-anonymity [Sweeney02]_       Equivalence-class generalisation so the
                               feature multiset matches ≥ k candidates
Differential privacy           Geo-indistinguishability [Andres13]_:
                               provable ε-DP on coordinates
=============================  ==========================================

Mitigation roadmap
------------------

None of the following are implemented in the current CartoCrypt
release; they are the sequenced next steps after the face-area
sprint.

1. **Generalisation of semantic labels.**  Publish elevation as a
   band (``400-500 m``) rather than a point value.  Cheapest win;
   trivial to ship.
2. **Chaff / decoy feature injection.**  Procedurally synthesise
   plausible fake peaks, lakes, and coastline bumps keyed to the
   symmetric key, so the published feature multiset matches at
   least ``k`` candidate regions from a calibrated atlas.  This
   is the formalisation of Bart's "spawn mountains" intuition,
   and directly corresponds to steganographic chaff.
3. **k-graph-anonymity.**  Tune chaff density so the published
   graph's Weisfeiler-Lehman equivalence class contains at least
   ``k`` real-world regions drawn from a published atlas (e.g.
   the Aegean islands dataset).  The parameter ``k`` becomes a
   tunable privacy dial analogous to ``ε`` in differential
   privacy.
4. **Geo-indistinguishability wrapper.**  On top of topology
   preservation, add a planar-Laplace noise layer [Andres13]_ on
   final coordinates for formal ε-differential-privacy guarantees
   in the coordinate domain.  Composes cleanly with the current
   re-embedding because the noise is added after Phase 3.
5. **Temporal-correlation resistance.**  Re-key transformations
   must ensure that two publications of the same region under
   different keys are unlinkable.  This is the hardest item and
   likely requires a keyed oblivious-transfer style wrapper.

Why face-area correction matters here
-------------------------------------

The Phase 3 face-area correction landed in this sprint
(:mod:`cartocrypt.reembed`, :mod:`cartocrypt.faces`) is *not* a
mitigation for the crocodile problem.  It is a *precondition*:
without it, the published face-area distribution drifts and
becomes a worse adversary fingerprint than the original.  By
anchoring face areas to their original values we ensure that
future chaff injection can match the published distribution
exactly, so a decoy peak is statistically indistinguishable from
a real one.

References
----------

.. [Sweeney02] L. Sweeney, *k-Anonymity: A Model for Protecting
   Privacy*, International Journal on Uncertainty, Fuzziness and
   Knowledge-based Systems, 10(5), 2002.

.. [Andres13] M. Andrés, N. Bordenabe, K. Chatzikokolakis, C.
   Palamidessi, *Geo-Indistinguishability: Differential Privacy
   for Location-Based Systems*, ACM CCS 2013, arXiv:1212.1984.

.. [Hampton10] K. Hampton et al., *Mapping Health Data: Improved
   Privacy Protection with Donut Method Geomasking*, American
   Journal of Epidemiology, 172(9), 2010.

.. [Wayner92] P. Wayner, *Mimic Functions*, Cryptologia 16(3),
   1992.

.. [Canetti97] R. Canetti, C. Dwork, M. Naor, R. Ostrovsky,
   *Deniable Encryption*, CRYPTO '97.

.. [Gao17] J. Gao et al., *Towards Plausible Graph
   Anonymization*, arXiv:1711.05441, 2017.

.. [Swanlund20] D. Swanlund, N. Schuurman, M. Brussoni, *Street
   Masking: a Network-based Geographic Mask*, International
   Journal of Health Geographics, 19(1), 2020.

.. [Seidl21] D. Seidl et al., *Adaptive Voronoi Masking*,
   GIScience 2021.
