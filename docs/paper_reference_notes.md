---
title: PySeshat Software — Results, Design Choices, and Notes for the Paper
date: 2026-08-27
status: living reference document; update as validation continues
---

# PySeshat Software — Results, Design Choices, and Notes for the Paper

This document collects everything the PySeshat software has produced and
learned so far, organized for writing the paper rather than for
maintaining the code (that's `docs/project_decisions.md`, the running
engineering decision log this document is distilled from — cross-referenced
throughout below by section name). It covers: the full validation results
across all eight benchmark systems, the specific nuances and caveats of
each case, the design choices made throughout the software and why, the
bugs found and fixed along the way (relevant to any data-integrity claims
the paper makes), the documentation/plotting work done to make the
software paper-ready, and a prioritized list of future work — marked with
what has already been executed versus what remains genuinely open.

## 1. What the software validates, and how

PySeshat validates two largely independent capabilities against real,
published systems:

**Orbit/triple-star fitting** (`orbit/`, `triple/`, `inference/`) — fitting
visual astrometric and radial-velocity orbits for both simple binaries and
hierarchical triples, via least-squares (scipy) and Bayesian MCMC (emcee).
Two parallel triple formalisms are supported and kept deliberately
separate: the legacy ORBITT2 scale-parameter formulation, and the modern
20-parameter ORBIT3 (Tokovinin) wobble-factor formulation. Validated
against four systems: FIN379, GL765 (simple binaries), ζ Aquarii and HIP
111805 (hierarchical triples, both fit through the real ORBIT3 formalism).

**Al-Wardat unresolved-binary stellar-SED fitting**
(`photometry/`, `stellar/`, `inference/photometric_*`) — fitting a
two-component Castelli-Kurucz (CK04) synthetic spectral model to a binary's
*combined*, unresolved photometry, recovering each component's effective
temperature, radius, and the system's reddening. This is the method the
paper is built around (Abushattal, Georgakarakos, Al-Wardat, Algnamat,
Haboubi, Dewi, Abu-Alrob & Hussein 2025, AJ, in press; arXiv:2512.23652;
doi:10.3847/1538-3881/ae0582 — all four SED benchmarks below are drawn from
this paper, chosen specifically because Al-Wardat himself co-authored it).
Validated against all four Sun-like SB2 systems that paper's own Table 4/5
cover: HD214222, HD191854, HD130669, HD184467.

Every fit is run through `validation/benchmark_runner.py`
(`python validation/benchmark_runner.py` runs the whole registry and writes
`validation/results/*.json`), and every system's literature-sourced
starting point, fit mask, and comparison target lives in
`validation/expected_results/<system>.json` — nothing is hand-tuned per
run. A full, prose walkthrough of every capability above (with runnable
code) now lives in `docs/guide/`; a figure-by-figure/section-by-section map
from this software to the target paper's planned structure lives in
`docs/paper_reproduction_notes.md`.

## 2. Full benchmark results (2026-08-27)

| System | Domain | Reduced χ² | Worst-fit parameter | Worst rel. err. |
|---|---|---|---|---|
| FIN379 | binary (visual+RV) | 1.68 | V0 | 0.46% |
| GL765 | binary (visual+RV) | 1.46 | omega | 0.60% |
| ζ Aquarii | hierarchical triple (ORBIT3) | 1.07 | inner i | 19.4%* |
| HIP 111805 | hierarchical triple (ORBIT3) | 1.71 | P | 0.00%† |
| HD214222 | Al-Wardat SED | 0.143 | primary_temperature | 5.35% |
| HD191854 | Al-Wardat SED | 0.107 | primary_temperature | 5.01% |
| HD130669 | Al-Wardat SED | 0.152 | secondary_temperature | 4.40% |
| HD184467 | Al-Wardat SED | 0.112 | primary_temperature | 6.37% |

\* ζ Aqr's inner inclination error is well within the literature's own
stated uncertainty (σ_i = 6.7°) for this poorly-constrained inner orbit —
not a red flag; see §3.3.

† HIP 111805's `.in3` header already *is* the published fitted solution
(the optimizer starts at, and correctly stays at, the optimum) — see §3.4.

All four Al-Wardat SED systems additionally recover radius to ≤1.10% of
literature and E(B-V) to physically sensible small values (≤0.006 mag in
every case, consistent with these being nearby, lightly-reddened Sun-like
systems).

## 3. Per-system nuances (what the paper should know about each case)

### 3.1 FIN379 (visual + RV binary)

More a software/inference regression benchmark than a primary physics
validation case. `RA=2.4412, Dec=-25.30, P=6.69yr, e=0.499, a=0.100″,
K1=7.43, K2=8.05 km/s`. A known data-format wrinkle: the raw `.inp` file
uses an RV component-flag encoding that needed explicit handling in the
parser (`io/orbit_parser.py`); this is a data-ingestion detail, not
evidence against the underlying orbit equations. Fitted 10 parameters to
0.46% worst-case error, `reduced χ²=1.68` (mildly elevated above the
formal value of 1, consistent with real astrometric/RV scatter rather than
a systematic offset). A visual pass of its generated orbit/residual/RV
plots (§4.10) shows the fitted curve tracking the observation scatter
closely with no axis or transpose anomalies, and residuals scattering
around zero with no strong systematic trend.

### 3.2 GL765 (visual + RV binary)

Model-vs-data sanity check (catalog orbit at observed epochs). No
definitive published RMS is established as project knowledge for this
system — don't treat any specific number as a literature target beyond
the fit's own internal consistency. Fitted to 0.60% worst-case error,
`reduced χ²=1.46`. Independently reconfirmed the RV K1/K2 sign convention
(§4.6) alongside FIN379. Its phase-folded RV plot (§4.10) shows both
components' primary/secondary curves correctly anti-phased and tracking
their observations tightly — a useful sanity-check figure candidate.

### 3.3 ζ Aquarii (hierarchical triple, ORBIT3)

The project's key ORBIT3 stress test: highly eccentric inner orbit
(e≈0.87) + a very long outer orbit (P₂≈540 yr) + 331 astrometric
observations spanning six decades + a nontrivial wobble factor. Two
literature-fidelity subtleties the paper should carry over correctly if it
cites this system:

- **The published astrometric wobble amplitude is not the ORBIT3 inner
  semimajor axis.** Tokovinin (2016)'s `a_astrometric = 0.110″` must be
  converted via `a_true = a_astrometric / f` (using `f = 1/F ≈ 0.2857` with
  `F≈3.50`) to get the ORBIT3 inner `a = 0.385″` actually used as the fit's
  starting point. Using `0.110″` directly is a real, easy-to-make error.
- **The raw data file uses the legacy ORBITT2 I1/I2 tag convention
  backwards from ORBIT3's own** (outer pair tagged `I1`, inner pair tagged
  `I2` — the reverse of the modern convention), and needs an explicit
  swap before fitting (`_convert_legacy_orbit3_observations` in
  `validation/benchmark_runner.py`).

**Fit quality**: PySeshat's fit *improves* on the literature solution
(χ²=694.51 vs. 699.73 at the literature starting point, Δ=5.23), dynamical
mass from the outer orbit + a 35.50±1.26 mas parallax gives
**M_dyn ≈ 3.26 ± 0.40 M☉ (≈12.4% relative)**, consistent with both the
~10.6% expected from parallax uncertainty alone and the ≈3.4 M☉ sum of
Tokovinin's approximate component masses.

**The one apparent residual anomaly, resolved**: a single inner-orbit
Y-residual outlier (~3.4-3.8σ, epoch 2009.7552) was traced to two
independently-reduced speckle-astrometry measurements at that exact epoch
(θ=191.99°/ρ=0.4057″ tagged "I" vs. θ=187.91°/ρ=0.4115″ tagged "HA", ~4°
apart) — a real measurement-to-measurement disagreement in the source
data, not a PySeshat convention or model bug. The other 8 inner-orbit
points and all 322 outer-orbit points are statistically unremarkable
(RMS≈1.0). Worth a note in the paper if ζ Aqr's residuals are shown; worth
a second look (down-weighting or excluding that one point) if a tighter
inner-orbit fit is ever needed.

**No orbit plot for this system yet** — see §4.10/§6 for why (the outer
orbit alone, without its wobble term, would be a misleading figure).

### 3.4 HIP 111805 (hierarchical triple, ORBIT3)

The most literature-complete benchmark in the project: a full 20-parameter
published solution with uncertainties (Tokovinin & Latham 2017). Notable
as the empirical justification for one of the software's real design
choices — the published solution has **f = −0.330**, proving the ORBIT3
wobble factor must not be bounded to `[0,1]` the way its naive physical
definition (`f = q_inner/(1+q_inner)`) might suggest (§4.5). Because the
`.in3` file's header already *is* the fitted optimum, PySeshat's fit
reproduces it exactly (χ²=502.10, every parameter's relative error 0.0%) —
this is a genuine, if trivial, regression check (the optimizer correctly
recognizes it's already at a stationary point), not a meaningful accuracy
demonstration on its own; ζ Aqr is the system that actually stress-tests
convergence from a non-trivial starting point.

### 3.5 HD214222, HD191854, HD130669, HD184467 (Al-Wardat SED systems)

All four come from the same source paper's Table 4 (photometry) and
Table 5/6 (component parameters, parallax). Literature values used as the
comparison target:

| System | Primary Teff/R/M | Secondary Teff/R/M | Distance |
|---|---|---|---|
| HD214222 | 6075 K / 1.140 R☉ / 1.12 M☉ | 6000 K / 1.095 R☉ / 1.09 M☉ | 70.9±1.2 pc |
| HD191854 | 5930 K / 1.585 R☉ / 1.12 M☉ | 5440 K / 0.906 R☉ / 0.89 M☉ | 56.2±4.1 pc |
| HD130669 | 5290 K / 0.956 R☉ / 0.89 M☉ | 5230 K / 0.950 R☉ / 0.88 M☉ | 46.3±0.6 pc |
| HD184467 | 5340 K / 0.832 R☉ / 0.91 M☉ | 5185 K / 0.783 R☉ / 0.85 M☉ | 17.83±0.22 pc |

Two important, general findings from validating these four systems
(details in §6-7):

- **Table 4's "measured" magnitudes are the source paper's own synthetic
  model output**, not independent catalog photometry — confirmed by direct
  inspection of the paper (it's labeled "Magnitudes and Color Indices of
  the Entire Synthetic Spectrum..."). This reframes what these four
  benchmarks actually validate: PySeshat's synthetic photometry pipeline
  reproducing *another pipeline's* synthetic photometry to within its own
  documented systematic (~0.25-0.5 mag per band, §6), not
  model-vs-real-sky-measurement agreement. Worth being explicit about in
  the paper's methodology section, since it changes the epistemic status
  of the comparison.
- **A real, physically-understood (Teff, R) degeneracy of unresolved
  photometry**, independent of how many bands are available (confirmed at
  4 bands, again at 10 bands, and again after implementing a real
  fit-flags/prior mechanism) — an unresolved combined-light SED constrains
  each component's flux only through a Teff/R combination, so no number of
  broadband photometric points alone can separate two components' radii
  without an external constraint. This is *why* the real Al-Wardat method
  combines SED fitting with evolutionary-track cross-checks rather than
  fitting radius from photometry alone — PySeshat's finding independently
  reproduces the methodological reason for that design. See §4.7 for how
  this is currently handled (a Gaussian prior tying radius to the
  literature/evolutionary value) and future-work item 1 (§7) for the real
  fix (tie radius to a mass + mass-radius relation via `evolution/binary.py`,
  not yet wired in).

**HD191854 is the clean case**: its two components have a 490 K Teff split
(5930 K vs. 5440 K) — the fit converges with no atmosphere-grid edge
pinning, radius exact to the literature value, `reduced χ²=0.107`. The
other three systems (HD214222: 6075/6000 K, 75 K split; HD130669:
5290/5230 K, 60 K split; HD184467: 5340/5185 K, 155 K split) all have one
component's fitted Teff pin exactly at the local atmosphere grid's edge —
confirmed via direct experiment (extending the grid range for HD130669 and
HD184467 down to 4250 K, well past their literature Teff) to be a genuine
near-twin-component-temperature degeneracy, not a grid-coverage gap:
widening the grid only *relocates* the pin and makes the individual-Teff
match *worse* (e.g. HD130669's secondary_temperature error grew from 4.40%
to 13.14% with the wider grid), while the combined-light chi-square barely
changes either way. **This is a genuine finding worth a sentence or two in
the paper**: unresolved photometry has very little power to separate two
components' individual temperatures when those temperatures are close to
each other, however precise the photometry — reduced chi-square staying
excellent (~0.11-0.15) throughout is itself evidence that many different
(Teff₁, Teff₂) splits fit the *combined* light almost equally well when
the two components are near-twins.

**Stromgren u is excluded from all four fits.** It's present in Table 4
but disagrees with the model by ~1.7 mag (~4.7x in flux) — far outside the
~0.25-0.5 mag systematic every other band shows. Two literature sources
were checked (the SVO Filter Profile Service's own Vega zero point, and
Gray et al. 2011's independent CALSPEC-derived AB-relative-to-Vega
offsets) and neither supports a correction anywhere near 1.7 mag; the
originally-suspected explanation (a Crawford & Barnes 1970 historical
uvbybeta calibration quirk) does not survive this scrutiny and has been
retracted in the project's own records. Given Table 4 is confirmed
synthetic output from another pipeline (above), the most likely
explanation is a difference between the two pipelines' internal
Stromgren-u zero-point/passband conventions specifically — not a physical
miscalibration in either. No corrected zero point has been applied (that
would mean reverse-fitting an "empirical" offset from the benchmark data
itself, which would be circular/fabricated); Stromgren u stays excluded
pending an authoritative source, and is retained in each system's
`excluded_measurements` field rather than silently dropped.

**No SED-fit or real-system HR-diagram figure exists for these four
systems yet** — see §4.10 and §7 (future-work items 9-10).

## 4. Design choices made throughout the software (with rationale)

The project's standing priority order, throughout: **scientific
correctness → faithful ORBIT/ORBIT3 reproduction → preserve validated
behavior → reproducibility → minimal changes → testability → performance
→ architectural elegance.** This is a validation/debugging project, not a
redesign project — discrepancies get diagnosed, not "fixed" by
refactoring, unless the refactor is itself the validated fix (as with the
fit-flags mechanism below).

**4.1 Legacy ORBITT2 and modern ORBIT3 triple formalisms are kept
architecturally separate, not unified.** They are not two syntactic forms
of the same model: legacy uses a scale parameter `B` on the resolved inner
orbit (`r_I2 = B·r_inner`, 15 parameters); ORBIT3 uses a physically
distinct wobble factor `f` (`POS2 = r_outer + f·r_inner`, 20 parameters).
Forcing a shared abstraction would risk silently changing the physics and
would block validating each historical formulation independently.

**4.2 Orbital elements are dataclasses, not bare parameter vectors** —
elements are scientific state; an optimizer's flat vector is just one
representation of that state. This is what lets literature parameters be
frozen, arbitrary subsets fitted (`fit_flags`), and fixed/free quantities
tracked cleanly throughout the whole inference stack.

**4.3 Residuals are computed in Cartesian coordinates (X=ρcosθ,
Y=ρsinθ), never polar directly** — avoids the 0°/360° position-angle
discontinuity that would otherwise corrupt residuals near that boundary.
Applies uniformly to both binary and triple residual calculations.

**4.4 The angle/sign convention matches Tokovinin/ORBIT internally, not
an independently-chosen textbook convention** (specific Thiele-Innes
element signs, θ=atan2(Y,X)). Small convention mismatches (Ω definition,
position-angle direction, F/G signs) can silently produce numerically
plausible but mirrored/wrong solutions, so every convention choice here
was checked explicitly against the original formalism rather than assumed.

**4.5 The ORBIT3 wobble factor f is deliberately left unbounded, not
restricted to (0,1).** Its naive physical definition
(`f = q_inner/(1+q_inner)`) would suggest 0<f<1, but the actual published
solution for HIP 111805 has `f=-0.330` — bounding it would make that real
system's own published solution unreachable.

**4.6 One shared angle convention serves both the visual (Thiele-Innes)
and spectroscopic (RV) halves of a combined fit** — there is no separate
primary/secondary reference-frame shift applied to RV data the way a
generic from-scratch two-body derivation might assume. This was
specifically re-derived and empirically reconfirmed (not merely assumed):
independent fits of both FIN379 and GL765, starting from their literature
`omega`/`K1`/`K2` values, converge back to those same values (not ~180°
away, not with K1/K2 swapped) — if the sign convention were actually
backwards, the optimizer would have had to move `omega` by ~180° to fit
the same data, and it doesn't need to. Locked in by
`tests/test_rv_sign_convention_validation.py`.

**4.7 The `PhotometricParameters`/`OrbitalElements` fit-flags/prior
mechanism** (implemented 2026-08-27, the most substantial design addition
during Phase 1): both element classes now support (a) genuinely holding any
subset of parameters fixed (`fit_flags`, absent from the optimizer's
vector entirely — not merely bounds-pinned to a point) and (b) an optional
Gaussian prior tying a *fitted* parameter to an external estimate
(`priors: {name: (mean, sigma)}`). The Gaussian prior is expressed through
ordinary least-squares machinery, not a separate mechanism: a
`(value-mean)/sigma` pseudo-residual is appended to the real measurement
residual vector for the optimizer to see, while a parallel prior-free
residual/statistics path is kept for reporting genuine data-model fit
quality (chi-square, RMS) uncontaminated by the prior term. This directly
replaced an earlier workaround (pinning radius to a literature value via
near-zero-width optimizer bounds, which trivially "matched" by
construction and gave no real sense of whether the photometric data
agreed or disagreed with that value).

**4.8 Cross-fitting-path statistics are normalized to one schema.** The
binary path, the ORBIT3 path, and the Al-Wardat SED path each historically
computed chi-square/reduced-chi-square/RMS under different internal key
names; `validation/benchmark_runner.py` now normalizes every path's
`"statistics"` dict to the same key set (`chi_square`, `reduced_chi_square`,
`rms`, `n_data`, `n_params`) before returning, so cross-system comparisons
(as in the results table in §2) are apples-to-apples without each
consumer needing to know which path produced a given result.

**4.9 CK04 (Castelli & Kurucz 2004, updated ATLAS9) was chosen as the
initial/only atmosphere grid** because it's directly usable and compatible
with the intended Al-Wardat-style SED workflow — a pragmatic choice, not
the product of a rigorous CK04-vs-PHOENIX-vs-other-grid comparison. Adding
another grid family later would be an extension, not a replacement.

**4.10 Plotting functions draw exactly what's passed to them, with no
implicit resampling, smoothing, or convention correction — verified, not
assumed.** As of this pass, `tests/test_orbit_plot.py`/`test_dynamics.py`
assert directly on the matplotlib objects each plotting function produces
(`ax.collections[].get_offsets()`, `ax.get_lines()[].get_xdata()/get_ydata()`)
against the exact input arrays, rather than only checking that a
`Figure`/`Axes` came back without an exception. Three literature
conventions were checked explicitly: the HR-diagram inverted-Teff axis
(already correct, already tested); the RV phase-folding zero point at
periastron/T0 (confirmed correct, now regression-tested); and the
residual-plot sign convention against the fit's own residual functions
(confirmed consistent, no mismatch). **One convention was found NOT to be
implemented and deliberately left unfixed rather than guessed at**: visual
orbit plots have no explicit North-up/East-left handling anywhere in the
codebase (`plot_visual_orbit()` plots X horizontally/Y vertically exactly
as given, with generic axis labels, no inversion). Per this project's
diagnose-don't-silently-refactor priority, this was documented (in
`plot_visual_orbit()`'s own docstring, in `docs/guide/visualization.md`,
and here) and locked in with regression tests on the *current* numerical
behavior, rather than changed without a published figure to verify
against — see §6 and §7 (future work item 8) for what's needed to close
this out. `scripts/generate_benchmark_plots.py` generates the full
available plot set (visual orbit, residuals, phase-folded RV,
separation-vs-time) for the 2 of 8 systems a flat-`OrbitalElements`
orbit plot is directly correct for (FIN379, GL765) — see §6/§7 for why the
other 6 are intentionally out of scope for now, not overlooked.

## 5. Bugs found and fixed during development (relevant to data-integrity claims)

Several real, if usually narrow-impact, bugs were found and fixed during
validation. None of the published benchmark results in §2 were affected
by any of them (each is called out below), but they're worth recording
since they establish the software's actual, tested correctness envelope
rather than an assumed one:

- **Southern-declination parsing (`io/coordinates.py`)**: `float("-00")`
  evaluates to `-0.0` in Python, and `-0.0 < 0` is `False`, so any
  declination with |degrees|<1° south (e.g. "-00:30:00") silently flipped
  to positive. Fixed by reading the sign from the string itself.
- **Inclination normalization (`orbit/orbital_elements.py`)**: an
  out-of-range inclination was being *clamped* (`np.clip(i,0,180)`)
  rather than *folded* — since the orbit model only ever uses `cos(i)`
  (even about both 0° and 180°), clamping actually changes the physical
  value (e.g. `i=-10°` clamped to `0°` gives `cos=1` instead of the
  correct `cos(10°)≈0.985`), a real model distortion whenever an
  unbounded optimizer's finite-difference Jacobian probes outside
  `[0,180]`. Fixed to reflect (`i<0 → -i`, `i>180 → 360-i`), which exactly
  preserves `cos(i)`. None of the four orbit/triple benchmark systems have
  `i` near this boundary, so no published result was affected.
- **A genuine ORBIT3 data-loss bug in the legacy fit path**
  (`triple/triple_fit.py`'s legacy `array_to_triple`): every legacy
  ORBITT2 fit silently reset `K1, K2, f, K3, K4, V0` to `0.0` on the
  returned `TripleOrbitalElements`, even when the input template carried
  real ORBIT3-relevant values — because a second, independently-maintained
  copy of the same conversion logic existed elsewhere in the codebase
  (`triple_parameters.py`) that *did* preserve them, no test caught the
  divergence. Fixed by consolidating to one canonical implementation
  (now documented in the function's own docstring, added this pass — see
  §6). Legacy-only benchmarks (FIN379, GL765) don't carry ORBIT3-specific
  fields, so this specific bug never affected a published legacy result,
  but it would have silently corrupted any future joint legacy+ORBIT3
  workflow.
- **MCMC sampler crash on an out-of-grid parameter proposal**
  (`inference/photometric_posterior.py`): an emcee walker proposing a
  temperature just outside a partially-loaded CK04 grid's range raised an
  uncaught `ValueError` that killed the whole sampler run, because the
  photometric prior's temperature bounds are generic physical limits
  (2500-50000K), not tied to whatever specific grid files a given run
  actually has loaded. Fixed by catching the interpolation error and
  treating it as `-inf` log-posterior, the same way an out-of-prior-range
  proposal already was — the sampler now steers away naturally instead of
  dying.
- **`separation_table()`'s column names don't match `separation_vs_time()`'s
  dict keys** (`analysis/dynamics.py`): not a scientific bug (both are
  correct, self-consistent outputs), but a real API-consistency trap —
  `separation_vs_time()` returns a dict keyed `"epoch"`/`"rho"`, while
  `separation_table()`'s DataFrame wraps the same values under
  `"Epoch"`/`"Separation"`. Found while writing a real test for this
  function this pass; not a defect worth "fixing" (changing either name
  now would break the other's established callers) but now documented in
  both functions' docstrings so it doesn't cost a second debugging session.
- Several other narrower fixes (a malformed-orbital-element parse now
  warns instead of silently defaulting to a divide-by-zero-prone `0.0`; a
  duplicated dead validation block in `photometry/filters.py`; degrees-of-
  freedom validation added consistently across `analysis/statistics.py`,
  `analysis/binary_consistency.py`'s luminosity fields renamed from
  "sigma" to the z-score they actually are, `evolution/interpolate.py`'s
  interpolation now rejecting non-monotonic input) — see
  `docs/project_decisions.md`'s "Whole-codebase audit" section for the
  complete list with verification detail on each.

## 6. Known, currently-open limitations

- **HD214222/HD130669/HD184467's near-twin-component-temperature
  degeneracy** (§3.5) — a genuine data/model limitation of unresolved SED
  fitting, not something the software can currently resolve without
  external data (resolved spectroscopy, or an interferometric flux
  ratio/angular-diameter measurement).
- **Stromgren u remains excluded** (§3.5) pending an authoritative,
  literature-sourced zero point for the classical Stromgren uvbybeta
  system's u band specifically.
- **The ~0.25-0.5 mag per-band systematic between PySeshat's synthetic
  photometry and the source paper's** (§3.5) is understood
  (different internal absolute-flux-calibration conventions between the
  two pipelines) but not eliminated — the corrected 0.30 mag assumed
  measurement uncertainty absorbs it statistically rather than removing
  it, which is the honest thing to do given Table 4 is itself synthetic
  output, not a real catalog value with its own independent error bar.
- **ζ Aqr's one outlier epoch** (2009.7552, §3.3) has not been
  specifically down-weighted or excluded — worth a second look if a
  tighter inner-orbit fit is needed for the paper.
- **Radius is currently freed with a literature-tied Gaussian prior**
  (10% assumed relative sigma, since the source paper gives no formal
  radius uncertainty for any of the four systems) rather than a *derived*
  prior from an independent physical relation — see §7 item 1 for the
  intended real fix.
- **The North-up/East-left visual-orbit convention has not been verified**
  (§4.10) — `plot_visual_orbit()` plots X/Y exactly as given, with no
  axis inversion and no explicit North/East commitment anywhere in the
  codebase; whether the current orientation already happens to be correct
  depends on `io/orbit_parser.py`'s theta-parsing sign convention, which
  has not been checked against a published figure. Regression tests lock
  in current behavior but do not assert it's the *correct* orientation.
  **Verify this against an actual published orbit figure before using
  `plot_visual_orbit()`'s output in the paper** — see §7 item 8.
- **No triple-aware orbit plot, no SED-fit figure, no real-system
  HR-diagram figure** (§4.10, `docs/paper_reproduction_notes.md`) — three
  of the paper's planned figures/use-cases have no supporting plot
  function or wiring yet; each is scoped as its own future-work item
  below rather than left as one vague gap.
- **165 public functions/methods across `src/pyseshat` have a docstring
  under 3 lines** (per `scripts/audit_docstrings.py`, run this pass) —
  most are simple property-style getters where this is genuinely fine (a
  1-line docstring on `n_samples`/`success`/`message` is not a gap), but
  the audit itself has not been fully triaged function-by-function; this
  pass strengthened the ~15 highest-value hits (key plotting functions,
  mass/luminosity physics functions, the ephemeris module, the
  `array_to_triple` function tied to the ORBIT3 data-loss bug above) and
  left the rest for a future targeted pass — see §7 item 11.

## 7. Future work

Ranked roughly by expected value; items marked **[EXECUTED]** were
identified as feasible during this or an earlier pass and actually
implemented — see each item's note for what was done and where.

1. **Tie the radius prior to mass + a mass-radius relation, or to
   evolutionary-track output, instead of the literature radius directly.**
   The current 10%-relative-sigma Gaussian prior (§4.7) uses the
   literature radius as its own mean — correct as a validation check
   (does the photometry-only fit disagree with the literature?), but
   circular as a genuine *independent* constraint. `evolution/binary.py`
   and `photometry/evolution_adapter.py` already connect a fitted
   photometric solution to evolutionary-track consistency checks; wiring
   evolutionary-track-predicted radius (given each component's literature
   mass) into `PhotometricParameters.priors` instead of the literature
   radius directly would make radius recovery a genuine independent test
   rather than a tautology. Not executed — requires designing
   the evolution-context contract `inference/joint_likelihood.py`
   currently leaves undefined (`JointInferenceProblem.evolution_context`
   is typed `Any` with no per-component sample/observables contract yet).
   See `docs/guide/evolutionary_checks.md`.
2. **A properly-sourced Stromgren u zero point.** Would restore the 10th
   real photometric measurement per Al-Wardat system. Requires a
   literature source this project hasn't found yet (checked: SVO Filter
   Profile Service, Gray et al. 2011 — neither supports the needed
   correction); a targeted literature search specifically for a modern
   Stromgren-system re-calibration (e.g. a Gray 1998-style paper, if one
   exists with the right band) is the concrete next step, not more
   pipeline debugging.
3. **[EXECUTED] Six runnable, narrated example scripts** covering the
   full range of the software's capabilities (visual+RV fit, MCMC
   posterior exploration, hierarchical-triple ORBIT3 fit, Al-Wardat SED
   fit, joint orbit+photometry fit, and an annotated template for a user's
   own data) — see `examples/`, all independently verified to run and
   reproduce the numbers reported in this document. This both closes
   Phase 2 of the development plan and gives the paper's methods section
   ready-made, reproducible worked examples to cite or adapt figures from.
4. **[EXECUTED] Strengthened plot-correctness tests, a literature-
   convention audit, and a scoped full-plot-set generation pass** (Phase
   4 of the development plan). Concretely: `tests/test_orbit_plot.py` and
   `tests/test_dynamics.py` now assert on actual plotted data (not just
   "runs without raising"); the HR-diagram/RV-phase/residual-sign
   conventions were checked and confirmed correct; the North-up/East-left
   convention was checked, found unimplemented, and deliberately
   documented rather than silently changed (§4.10, §6); and
   `scripts/generate_benchmark_plots.py` generates and a visual pass
   confirmed the full plot set for the 2 of 8 systems it currently
   applies to (FIN379, GL765) — see `docs/project_decisions.md`'s
   "Phase 3/4" entry (2026-08-27) for the complete writeup, and
   `docs/guide/visualization.md` for the user-facing version.
5. **[EXECUTED] A user guide and documentation pass** (Phase 3) — see
   `docs/guide/` (6 pages: data formats, orbit fitting, triple systems,
   Al-Wardat SED fitting, evolutionary-track checks, visualization),
   `docs/paper_reproduction_notes.md` (figure/section-by-section mapping
   to the target paper), `README.md`, and a targeted API-docstring
   strengthening pass (§6, §11 below) driven by
   `scripts/audit_docstrings.py`.
6. **Extend the CK04 grid, or add a second grid family, for target
   systems outside the current 3500-50000K/[M/H]=0.0 coverage** if future
   benchmark systems need it. The full 608-file [M/H]=0.0 grid is already
   downloaded (`scripts/download_ck04_grid.py`); a different metallicity
   or a different grid family (e.g. PHOENIX) would be a genuine extension
   requiring its own interpolation/loading code, not a config change.
7. **A joint-fit convenience wrapper** (`fit_joint(...)`, analogous to
   `fit_visual_orbit`/`fit_photometry`) — `inference/joint_problem.py`/
   `joint_parameters.py`/`joint_likelihood.py` are real, tested building
   blocks with no single entry point tying them together yet; Example 5
   (`examples/05_joint_orbit_photometry_fit.py`) demonstrates composing
   one by hand. Worth promoting to a real library function once a second
   real use case (beyond the synthetic demonstration) exists to validate
   the API design against.
8. **Verify (and, if needed, correct) the North-up/East-left visual-orbit
   convention against a published figure.** (§4.10, §6.) Concrete next
   step: reproduce one published orbit from `wardat_2002.pdf`,
   `wardat_2012.pdf`, or `Tokovinin_2016_ApJ_831_151.pdf` side by side
   with `plot_visual_orbit()`'s output for the same system/epoch range,
   confirm or correct the position-angle sense, then lock the *correct*
   convention in with a new regression test (not just the current-behavior
   ones already in place). This is a domain-expertise/visual-comparison
   task, not something safely automatable without that comparison.
9. **Build a SED-fit figure** (essential figure 4 per `Paper_Roadmap.pdf`;
   `docs/paper_reproduction_notes.md`) — no plotting function currently
   exists for `fit_photometry()`'s output; a fitted-model-vs-measured
   magnitude plot per passband (likely magnitude- or flux-vs-wavelength)
   would close this. `examples/04_alwardat_sed_fit.py` already computes
   everything such a plot would need (fitted parameters, per-band
   residuals); it just isn't drawn yet.
10. **Run a real evolutionary-track cross-check and HR-diagram figure for
    the four Al-Wardat systems** (essential figure 5;
    `docs/paper_reproduction_notes.md`, `docs/guide/evolutionary_checks.md`)
    — `plot_binary_hrdiagram()` itself is convention-correct and tested,
    but has never been run against real HD214222/191854/130669/184467
    data; doing so would also directly support future-work item 1 above
    (an evolutionary-track-derived radius prior).
11. **Triage the remaining ~150 thin-docstring hits from
    `scripts/audit_docstrings.py`** (§6) not addressed this pass — most
    are simple property getters that likely don't need expansion, but a
    human pass through the full list (saved by re-running the script) is
    the fast way to separate genuine gaps from adequate one-liners, per
    `docs/development_plan_phase2.md` Phase 3's own guidance not to
    rewrite everything uniformly.
12. **Down-weight or investigate ζ Aqr's 2009.7552 outlier epoch**
    specifically (§3.3, §6) if a tighter published inner-orbit solution is
    needed for the paper.
13. **Compare against other RV/orbit-fitting packages (PyAstronomy,
    RadVel, Exo-Striker) and run a Gaia-binary accuracy test**, and add
    named classical-binary case studies (e.g. Capella, Sirius) —
    `Development_Plan.pdf`'s planned Validation and Case Studies sections
    call for both; neither has been started (see
    `docs/paper_reproduction_notes.md` for the full gap list). These are
    the two largest genuinely-unstarted pieces of the paper's planned
    scope, distinct from anything the current 8-system benchmark suite
    covers.

## 8. Where to find things

- `README.md`, `examples/README.md` — quickstart and the full example
  index.
- `docs/guide/` — user guide (data formats, orbit fitting, triple
  systems, Al-Wardat SED fitting and its limitations, evolutionary-track
  consistency, visualization).
- `docs/paper_reproduction_notes.md` — maps the target paper's planned
  structure and its 6 essential figures to the exact script/benchmark
  that reproduces each, and states plainly which are not built yet.
- `docs/project_decisions.md` — the complete, dated engineering decision
  log this document is distilled from; the authoritative source if a
  detail here needs double-checking or expanding.
- `docs/development_plan_phase2.md` — the phased plan (0-4) this and the
  prior pass's work completes.
- `scripts/generate_benchmark_plots.py` — regenerates the full available
  plot set for FIN379/GL765 under `validation/results/plots/`.
- `scripts/audit_docstrings.py` — regenerates the thin-docstring audit
  referenced in §6/§7 item 11.
- `validation/expected_results/*.json` — every benchmark system's
  literature-sourced starting point, fit mask, and comparison target, and
  (for the four SED systems) full provenance notes on every assumption
  made (measurement uncertainty, radius prior sigma, excluded
  measurements).
- `validation/results/*.json` — the actual fitted output each run
  produces, regenerated by `python validation/benchmark_runner.py`.
