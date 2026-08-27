# PySeshat — Project Decisions & Validation State

Distilled from a ChatGPT project-history summary (2026-08-26) covering design
decisions, rejected approaches, and open issues on the ORBIT3/triple-star work.
Treat this as institutional memory: read it before touching `triple/`,
`orbit/`, or the validation benchmarks, and update it when a decision here is
revisited.

## Priority order

Scientific correctness → faithful ORBIT/ORBIT3 reproduction → preserve
validated behavior → reproducibility → minimal changes → testability →
performance → architectural elegance.

The project is in a **controlled validation/debugging phase**, not a
redesign phase. Discrepancies get diagnosed, not "fixed" by refactoring.

## Key decisions (with rationale)

**Legacy `ORBITT2` and `ORBIT3` triple formalisms stay separate — do not
unify them.** They're not two syntactic forms of the same model: legacy uses
a scale parameter `B` on the resolved inner orbit (`r_I2 = B * r_inner`, 15
parameters); ORBIT3 uses a physically distinct wobble factor `f`
(`POS2 = r_outer + f * r_inner`, 20 parameters: inner `P,T,e,a,Ω,ω,i,K1,K2,f`
+ outer `P2,T2,e2,a2,Ω2,ω2,i2,K3,K4,V0`). Forcing a shared abstraction risks
silently changing the physics and would block validating each historical
formulation independently.

**`TripleOrbitalElements` is a dataclass**, not a bare parameter vector —
elements are scientific state; the optimizer's vector is just a
representation of it. This is what lets literature parameters be frozen,
subsets fitted, and fixed/free quantities tracked cleanly. Already
implemented; not up for redesign.

**Residuals are Cartesian (X = ρcosθ, Y = ρsinθ), not polar (ρ,θ)
directly** — avoids the 0°/360° discontinuity in position angle. Applies to
both binary and triple residual calculations. A residual discrepancy does
not by itself imply this coordinate choice is wrong.

**Angle/sign convention is internally consistent with Tokovinin/ORBIT, not
an independently-chosen textbook convention.** Thiele-Innes:
`A = a(cosΩcosω − sinΩsinω·cosi)`, `B = a(sinΩcosω + cosΩsinω·cosi)`,
`F = a(−cosΩsinω − sinΩcosω·cosi)`, `G = a(−sinΩsinω + cosΩcosω·cosi)`,
with `θ = atan2(Y,X)`. Small convention mismatches (Ω definition, PA
direction, X/Y orientation, F/G signs, ascending-node convention) can
produce numerically plausible but mirrored/wrong solutions — check
explicitly against any external package rather than assuming conventions
match.

**ORBIT3 wobble factor `f` is NOT bounded to [0,1].** Physically
`f = q_inner / (1 + q_inner)` implies `0 < f < 1`, but the fitted/reported
ORBIT3 parameter isn't constrained that way in practice — current optimizer
bounds deliberately allow `f` unbounded (confirmed empirically: HIP 111805's
published solution has `f = −0.330`). Do not re-impose positivity.

**ζ Aqr: the astrometric wobble amplitude ≠ the true inner semimajor
axis.** Tokovinin's published `a_astrometric = 0.110″` is not the ORBIT3
inner `a`. With `F ≈ 3.50` and `f = 1/F ≈ 0.285714`:
`a_true = F × a_astrometric = 0.385″`, so `f × a_true = a_astrometric`.
Using `0.110″` directly as the ORBIT3 inner `a` is wrong — this conversion
is encoded in the ζ Aqr validation test.

**ζ Aqr (and hierarchical triples generally) must be fit simultaneously,
not sequentially.** `r_outer,obs = r_outer + f·r_inner` — the outer
observation contains the inner wobble. Fit-outer-then-fit-inner-residuals
was explicitly identified as physically incorrect.

**"CK04" means the Castelli & Kurucz (2004) *updated ATLAS9* grid, not an
alternative atmosphere family.** `load_ck04_fits()` reads STScI CK04 FITS
tables, interpolating in `(Teff, log g, [M/H])`. It was chosen as the
initial grid because it's directly usable and compatible with the intended
Al-Wardat-style SED workflow — not the product of a rigorous CK04-vs-PHOENIX
comparison. Adding another grid later is an extension, not a replacement.

## Explicitly rejected approaches (don't re-litigate)

- Refactoring architecture in response to a scientific discrepancy — the
  loop is inspect → diagnose → smallest justified change → test → interpret.
- Unifying legacy/ORBIT3 triple parameterizations.
- Bounding `f` to [0,1].
- Fitting ζ Aqr's outer orbit first, then the inner orbit from residuals.
- Replacing the Cartesian residual framework without evidence it's wrong.
- Jumping to MCMC to paper over a deterministic model/convention error —
  MCMC is part of the eventual framework (already built/tested, e.g. FIN379
  posteriors) but doesn't substitute for getting the deterministic ORBIT3
  fit right first.
- Treating a literature orbit as automatically "the correct optimum" — the
  validation standard is whether PySeshat's fit *improves* the objective
  relative to the literature starting point, not element-by-element
  equality (literature and PySeshat can legitimately differ in data
  weighting, available observations, fixed/free parameters).

## Current bottleneck & next step

**Open question:** can PySeshat's ORBIT3 implementation reproduce the
published Tokovinin triple solution and its residual structure closely
enough that any remaining gap is scientifically explainable? Not yet fully
resolved — there's a small unexplained discrepancy for ζ Aqr specifically.

**Agreed next step: residual-decomposition diagnostic**, not optimizer
replacement. Decompose by I1 vs I2, X vs Y (astrometric) and compare
Tokovinin vs PySeshat χ². This infrastructure already exists for HIP 111805
(see `tests/test_hip111805_*_diagnostic.py`) — the same approach needs to be
applied to ζ Aqr. Do NOT respond to the discrepancy by: rewriting the
optimizer, redesigning triple architecture, switching residual definitions,
adding another atmosphere model, jumping to MCMC, or changing conventions
without evidence.

## Benchmark reference values

**ζ Aquarii** (Tokovinin 2016) — inner Aa–Ab: `P=25.95 yr, T0=2006.52,
e=0.872, a_astrometric=0.110″, Ω=293.7°, ω=100.9°, i=11.8°`. Outer A–B:
`P2=540.0 yr, T02=1981.50, e2=0.419, a2=3.496″, Ω2=131.3°, ω2=269.3°,
i2=142.0°`. Masses ≈ M_Aa 1.4 M☉, M_B 1.4 M☉, M_Ab 0.6 M☉. Highly eccentric
inner orbit + very long outer orbit + extensive astrometry + nontrivial
wobble — this is why it's the key ORBIT3 stress test.

**HIP 111805** (Tokovinin & Latham 2017) — full 20-parameter solution
encoded in tests, including uncertainties. Inner: `P=1.5012 yr,
T0=1986.093, e=0.022, a=0.0385″, Ω=334.5°, ω=232.9°, i=85.80°, K1=13.13,
K2=19.21, f=−0.330`. Outer: `P2=30.127 yr, T02=2010.179, e2=0.324,
a2=0.3361″, Ω2=154.25°, ω2=84.92°, i2=88.28°, K3=6.06, K4=8.60, V0=−22.58`.
The `f=−0.330` value is the empirical justification for not constraining
`f` to [0,1].

**FIN379** — more a software/inference regression benchmark than the
central ORBIT3 physics benchmark. Starting vector (P,T0,e,a,Ω,ω,i,K1,K2,V0)
≈ `(6.690, 2008.8443, 0.4988, 0.1003, 183.9°, 191.0°, 42°, 7.43, 8.05,
−0.51)`. A known ingestion issue: RV data with an invalid component flag
(expects `1`/primary or `2`/secondary) — a data-format issue, not evidence
against the orbit equations. Used extensively for deterministic fitting,
likelihood tests, posterior identity tests, MCMC diagnostics, mass calcs.

**GL765** — model-vs-data sanity check (catalog orbit at observed epochs,
RMS residuals in X,Y). No definitive numerical RMS is established as project
knowledge yet — don't treat any specific GL765 RMS figure as settled.

**Validation hierarchy**, strongest to weakest: (1) reproducing published
orbital elements, (2) reproducing published residuals, (3) reproducing the
original IDL ORBIT/ORBIT3 numerical result bit-for-bit on the same input.
The project's goal includes (3); it isn't established yet.

## ζ Aqr residual-decomposition run (2026-08-26)

Ran the existing `tests/test_zetaaqr_residual_diagnostics.py` (the
diagnostic already existed — no new code needed) in a scratch environment
with the current `src/pyseshat`. Result: **PySeshat's fit already improves
on the literature solution** (χ² 694.51 vs 699.73, Δ=5.23) and is not
failing to converge.

Decomposing further: the inner orbit (`I1` after the file's I1/I2 swap in
the test, N=9) carries a Y-residual mean of ≈−0.42σ in *both* the literature
and fitted solutions — this looked like it could be a systematic convention
bug. Per-point inspection (not previously done) shows it isn't: 8 of the 9
`I1` points have |residuals| ≲ 1.3σ, and the −0.42 mean is entirely driven
by one point at epoch 2009.7552 (res_y ≈ −3.4 to −3.8σ). Cross-checking
against `ZetaAqr.in2`: there are **two independent measurements at that
same epoch** (θ=191.99°/ρ=0.4057″ vs θ=187.91°/ρ=0.4115″, ~4° apart in θ,
tagged `I` and `HA` — different reductions/instruments) plus an earlier,
now-commented-out (`C`-prefixed) pair of measurements at the same epoch
with tighter claimed error bars that were evidently revised upward (0.005″
→ 0.009″). One of the two active measurements is the outlier; the other is
unremarkable.

**Conclusion:** the ζ Aqr small residual discrepancy the project flagged as
open is very likely real measurement-to-measurement scatter between two
near-duplicate speckle astrometry reductions at one epoch, not a PySeshat
convention/model bug — the outer orbit (`I2` after swap, N=322) residuals
are statistically unremarkable (RMS≈1.0) and the other 8 inner-orbit points
are well-behaved. This is not yet a *closed* question (a single outlier
point deserves a second look — e.g. whether the fit is sensitive to
down-weighting or excluding it, and whether the same epoch shows a similar
split in other reductions of this system), but it reframes the open issue
from "something may be wrong with the ORBIT3 implementation" to "one
specific epoch's data has known internal disagreement" — a much narrower,
more tractable question. HIP 111805's `test_hip111805_tokovinin_residual_decomposition.py`
also passes cleanly in this environment (no run-level detail dug into yet).

## `triple/` ORBIT3 parameter-vector consolidation (2026-08-26)

Audited `triple/` against `orbit3.pro` per the explicit request to "match the
triple module workflow to the ORBIT3 original implementation without losing
the modular style." The core physics/workflow (Kepler/ephemeris formulas,
finite-difference derivative construction — including the two IDL quirks in
the `T0`/`V0` step sizes, deliberately reproduced rather than "fixed" —
residual block ordering, weighting) was already faithful. No bugs there.

**What was actually wrong: architectural duplication, and one real bug
hiding inside it.** The ORBIT3 20-parameter conversion logic and its
ordering list existed independently in three places — `triple_fit.py` (the
module actually imported by `orbit3_fit.py`, i.e. the live path),
`triple_parameters.py` (a parallel, publicly-documented module with two
identically-named functions `orbit3_fitted_parameters` /
`update_orbit3_fitted_parameters`, not wired into the live fitting path),
and `orbit3_bounds.py` (a third hardcoded copy of just the ordering list).
`TripleOrbitalElements` in `triple_elements.py` additionally hardcoded the
same 20-name list three more times internally. `CLAUDE.md` itself credited
`triple_parameters.py` as the ORBIT3 parameter-vector location, which didn't
match the code.

Comparing the two duplicate implementations line-by-line surfaced a genuine
divergence, not just harmless copy-paste: `triple_fit.py`'s legacy
`array_to_triple` (used by every legacy ORBITT2 fit iteration, and by the
final `fit_triple_orbit()` result) built the returned `TripleOrbitalElements`
without passing `K1, K2, f, K3, K4, V0` through — since those aren't dataclass
required fields, they silently reset to `0.0` on every legacy fit, even when
the template object had real ORBIT3-relevant values on it. `triple_parameters.py`'s
independent copy did carry them over correctly. Reproduced directly:
`triple_fit.array_to_triple(vec, triple)` returned `K1=K2=f=K3=K4=V0=0.0` for
a template with `K1=13.1, K2=19.2, f=-0.33, K3=6.06, K4=8.60, V0=-22.575`,
while `triple_parameters.array_to_triple` on the same input preserved them.
No existing test caught this because the legacy 15-parameter round-trip
tests only check the 15 legacy values, which aren't affected.

**Fix applied** (minimal, behavior-preserving, all existing names/signatures
kept):
- `triple_elements.py` now defines `ORBIT3_PARAMETER_ORDER` once, at module
  level, and its three internal hardcoded copies (`fitted_mask()`,
  `fitted_parameter_names()`, `parameter_names()`) all reference it.
- `triple_fit.py` imports `ORBIT3_PARAMETER_ORDER` from `triple_elements.py`
  instead of hardcoding its own copy, and re-exports it for backward
  compatibility. Its legacy `array_to_triple` now carries `K1, K2, f, K3,
  K4, V0` over from the template, fixing the data-loss bug above.
- `orbit3_bounds.py` imports `ORBIT3_PARAMETER_ORDER` from `triple_elements.py`
  instead of hardcoding a third copy.
- `triple_parameters.py` is now a thin, explicitly-documented backward-
  compatible wrapper: every function delegates to the canonical
  implementation in `triple_fit.py` (or the shared order list in
  `triple_elements.py`) rather than reimplementing it. All existing public
  names (`triple_to_orbit3_array`, `orbit3_array_to_triple`,
  `orbit3_fitted_parameters`, `update_orbit3_fitted_parameters`,
  `legacy_triple_parameter_names`, `orbit3_parameter_names`, legacy
  `triple_to_array`/`array_to_triple`) are unchanged, so nothing that
  imports from it needs to change.
- `CLAUDE.md`'s architecture section corrected to point at `triple_fit.py`
  as the canonical ORBIT3 parameter-vector location and describe
  `triple_parameters.py`'s wrapper role.

**Verification:** isolated the array-preservation divergence with a direct
before/after reproduction; confirmed all four `ORBIT3_PARAMETER_ORDER`
references (`triple_elements`, `triple_fit`, `triple_parameters`,
`orbit3_bounds`) are now the same imported object, not independent copies;
ran the full staged test suite before and after (`pytest tests/ -q`) — one
pre-existing failure, `test_hip111805_orbit3.py::test_hip111805_orbit3_residual_decomposition`
(a strict `<` on two floats that come out numerically equal,
`502.1038162393516 < 502.1038162393516`), reproduces identically against the
original, unconsolidated code and is unrelated to this change (it exercises
only the ORBIT3 path, not the legacy `array_to_triple` this fix touches) —
everything else passes both before and after (58/59 locally staged tests).
That one failure is a pre-existing loose assertion (should very likely be
`<=`, or the fixture should be perturbed off the optimum) worth a follow-up,
but out of scope for this consolidation.

**Follow-up pass over the rest of `triple/`:** checked `triple_bounds.py`,
`triple_diagnostics.py`, `triple_identifiability.py`, `triple_observation.py`,
`triple_pca.py`, `triple_statistics.py`, `triple_fit_result.py` against
`orbit3.pro` and against each other. `triple_diagnostics.py`,
`triple_identifiability.py`, `triple_pca.py`, `triple_statistics.py`,
`triple_observation.py`, and `triple_fit_result.py` are generic/independent
(chi-square, covariance PCA, correlation matrix, observation container, fit
report) with no duplicated ORBIT3/legacy-specific state — nothing to
consolidate there. `orbit3_fit.py` (the `fit_orbit3()` entry point, the
finite-difference `del`-vector construction, the custom LM Jacobian) was
re-checked line-by-line against `alleph`/`fitorb` and matches, including the
`T0` step reusing `P`'s value and `V0`'s step reusing `K3`'s value.

`triple_bounds.py` (legacy ORBITT2 bounds) turned out to hold a **fourth**
independent hardcoded copy of the legacy 15-name parameter order (as direct
`elements.fit_flags["..."]` lookups rather than a list) — the same
duplication pattern as the ORBIT3 case, for a structural reason: `triple_fit.py`
imports `build_triple_bounds` from `triple_bounds.py`, so `triple_bounds.py`
importing `LEGACY_TRIPLE_PARAMETER_ORDER` back from `triple_fit.py` would be
circular. Fixed by moving `LEGACY_TRIPLE_PARAMETER_ORDER` to
`triple_elements.py` alongside `ORBIT3_PARAMETER_ORDER` (which has no
`triple/`-internal dependencies, so both `triple_fit.py` and
`triple_bounds.py` can import it without a cycle) and having both modules
import it instead of hardcoding. `triple_bounds.py`'s `_legacy_fitted_mask`
and `triple_fit.py`'s `legacy_triple_fitted_mask` remain two separate
functions (merging them would reintroduce the same circular-import problem),
but they now both walk the one canonical order list, so the two can no
longer silently drift apart the way the ORBIT3 vectors had. Verified by
identity check (`triple_bounds.LEGACY_TRIPLE_PARAMETER_ORDER is
triple_elements.LEGACY_TRIPLE_PARAMETER_ORDER`) and by re-running the full
staged suite (`pytest tests/ -q`): same single pre-existing failure noted
above, everything else (58 tests) passes.

Separately noted, not a bug: `inference/` (including `mcmc_engine.py`) has
no imports from `triple/` at all — MCMC is wired up for binary/joint fitting
(e.g. FIN379) but there is currently no MCMC entry point for triple/ORBIT3
orbits. That's a gap/missing feature, not a duplication or correctness
issue, and wasn't touched here.

## Whole-codebase audit (2026-08-26)

Full audit per explicit request: hidden bugs, scientific-accuracy review,
test-suite consolidation, unnecessary-script cleanup, next steps from the
paper roadmap. Used three parallel reviews (orbit/io/inference;
stellar/photometry; evolution/analysis), each independently verified before
acting. Read `Paper_Roadmap.pdf`, `Development_Plan.pdf`, and
`validation_phase_plan.txt` (BinariesPaper root) for the roadmap context.

### Fixed (verified, low-risk)

- **`io/coordinates.py` `parse_dec`**: `float("-00")` is `-0.0` in Python,
  and `-0.0 < 0` is `False`, so any southern declination with |degrees| < 1
  (e.g. `"-00:30:00"`) silently flipped to positive. Fixed by reading the
  sign from the string itself rather than the parsed float. Verified:
  `parse_dec("-00:30:00") == -0.5` (was `+0.5`).
- **`orbit/orbital_elements.py` `normalize_angles`**: inclination was
  clamped with `np.clip(i, 0, 180)`, not folded. The visual-orbit model
  (`thiele_innes`) only ever uses `cos(i)`, which is even about both 0 and
  180 (`cos(-x)=cos(x)`, `cos(180+x)=cos(180-x)`), so the physically correct
  operation is a reflection (`i<0 -> -i`, `i>180 -> 360-i`), which exactly
  preserves `cos(i)`; clipping instead *changes* `cos(i)` (e.g. `i=-10`
  clipped to `0` gives `cos=1` instead of the correct `cos(10)≈0.985`) — a
  real, if usually dormant, model distortion. This affects both `orbit/`
  and `triple/` (the latter's inner/outer elements are `OrbitalElements`
  instances) whenever an unbounded `method="lm"` fit's finite-difference
  Jacobian probes `i` outside `[0,180]`; bounded methods (`trf`/`dogbox`)
  were already protected by explicit optimizer bounds. None of the four
  current benchmark systems have `i` anywhere near the boundary, so no
  currently-validated result is affected — verified by rerunning the full
  `triple`/`orbit3` test suite (same single pre-existing unrelated failure,
  everything else still passes) and by direct check: `i=-10` now folds to
  `10.0` with `cos` matching exactly; `i=190` folds to `170.0` likewise.
- **`io/orbit_parser.py`**: a malformed orbital-element value silently
  became `0.0` with no diagnostic, inconsistent with the visual-row and
  RV-row parsing blocks in the same file, which both print a warning on a
  parse failure. Added the same warning for consistency — a `P=0.0` from a
  silently-swallowed parse error would otherwise propagate into
  `M=2π(t-T0)/P` as a divide-by-zero with no indication of the real cause.
- **`photometry/filters.py`**: a throughput-validation `if`/`raise` block
  was duplicated verbatim immediately after itself (dead, unreachable
  second copy with a slightly different message). Removed the duplicate.
- **`tests/test_rv_model.py`**: rewritten. The original file had **no
  `def test_...` function and no `assert` anywhere** — it was a top-level
  script that printed values for a human to read at collection time, so it
  could never fail under `pytest` regardless of correctness. Replaced with
  a real parametrized test asserting finiteness, non-triviality (an RV
  curve with nonzero K should actually vary), and that `compute_rv_model`'s
  per-component dispatch reproduces `compute_rv_orbit` called directly —
  all verified to pass on FIN379/GL765. Deliberately does **not** assert
  which physical star the K1-vs-K2 sign maps to — see the open finding
  below.

### RESOLVED (2026-08-27): RV K1/K2 sign convention

The concern raised in the original audit — that `compute_rv_orbit`'s
K1/K2 sign pattern looked backwards relative to `to_spectroscopic_convention()`'s
documented +180° visual-to-spectroscopic omega shift — is **not a bug**.
The user recalled that the current convention had already been needed to
get consistent joint orbits on the benchmark systems in an earlier commit,
which pointed at checking the checked-in run artifacts directly instead of
re-deriving from first principles.

The root-level `FIN379_fit.json` (a real prior combined visual+RV fit, with
K1, K2, V0 all free parameters, `created_utc: 2026-08-19`) already proves
this empirically: starting from FIN379.inp's literature values (`omega=191.0,
K1=7.43, K2=8.05`), the fit — using exactly today's unshifted
`compute_rv_orbit` — converges to `omega=191.158±0.652, K1=7.42904±0.0418,
K2=8.05030±0.0415`, matching the literature to well within uncertainty on
every one of the 10 free parameters. If the sign convention were actually
backwards, the optimizer would have had to move omega by roughly 180° (or
swap K1/K2) to fit the same RV data — it didn't need to, because the
current convention is correct for how this codebase's ORBIT-format
benchmark files are actually solved: they use ONE `omega` for both the
visual (Thiele-Innes) and spectroscopic (RV) parts of a joint fit, with no
separate primary/secondary reference frames the way a generic textbook
two-body derivation assumes. That also explains why the ORBIT3/triple
module needed no such shift either (see above) — it's the same underlying
convention, not a coincidence.

Made this permanent with a new regression test,
`tests/test_rv_sign_convention_validation.py`: it independently re-fits
both FIN379 and GL765 (not reusing the cached `FIN379_fit.json` — a fresh
optimizer run against the same literature starting point and real RV data)
and asserts the fitted omega lands near the literature value (not ~180°
away) and K1/K2 land near their literature values with the same
labelling. **Both systems pass independently** — GL765 (`omega=76.7,
K1=7.82, K2=7.52` in the literature) is a second, independent confirmation,
not just a repeat of FIN379's already-known-good case. Also added a
warning to `to_spectroscopic_convention()`'s docstring: it is unused
everywhere, does not reflect this codebase's actual validated convention,
and must not be wired into `compute_rv_orbit` without re-deriving why
FIN379/GL765 would still need to converge correctly afterward.

### RESOLVED (2026-08-27): the other five flagged consistency items

All five were fixed, with regression tests added, verified against the
locally staged repo (30+ tests passed, including the full RV/orbit
regression suite — no regressions), and committed back to the device.

- **`inference/joint_likelihood.py`**: `joint_log_likelihood` now raises
  `NotImplementedError` when `has_evolution_constraints=True`, matching
  `joint_log_likelihood_from_parameters` — both paths fail loudly instead
  of one silently contributing `0.0`. Also removed the unused
  `evolution_log_likelihood` import and left a comment explaining why it
  isn't wired in yet: `JointInferenceProblem.evolution_context` is typed
  `Any` with no defined per-component sample/observables contract, so
  there's nothing well-defined to pass it — a real integration needs that
  contract designed first, not just a call to silence the error. New test:
  `test_joint_log_likelihood_raises_for_evolution_constraints` in
  `tests/test_joint_likelihood.py`.
- **`analysis/statistics.py`** (`combined_fit_summary`, both call sites):
  `dof = max(n - n_params, 1)` replaced with the same
  raise-on-nonpositive-dof pattern as `reduced_chi_square()` in the same
  module. New tests in `tests/test_dof_validation.py`.
- **`analysis/binary_consistency.py`**: `primary_luminosity_sigma` /
  `secondary_luminosity_sigma` renamed to `primary_luminosity_z_score` /
  `secondary_luminosity_z_score` (they hold `(observed-model)/σ_observed`,
  not a 1σ uncertainty). Added a docstring note on `BinaryConsistencyResult`
  spelling this out. No other callers existed anywhere in the codebase
  besides the two dataclass-construction sites and `tests/test_binary_consistency.py`
  / `tests/test_binary_consistency_integration.py`, both updated to match.
- **`evolution/interpolate.py`** (`_interp`): now raises `ValueError` if
  `xp` is not strictly increasing (`np.any(np.diff(xp) <= 0)`), before
  calling `np.interp` — catches both a fully reversed array and a
  non-monotonic one (e.g. a repeated age from a blue-loop segment). New
  tests `test_interp_unsorted_xp_raises` / `test_interp_non_monotonic_xp_raises`
  in `tests/test_interpolate.py`.
- **`inference/covariance.py`**: the `scale_by_reduced_chi2=True` branch's
  `dof = max(n-n_par, 1)` now raises `ValueError` on nonpositive dof,
  consistent with the `statistics.py` fix above (same underlying pattern,
  still currently-dead/unwired code otherwise). New tests in
  `tests/test_dof_validation.py`.

Additional finding made in the course of this fix, and since resolved:
`tests/test_statistics.py` and `tests/test_reduced_chi_square_rv.py` were
both print-only scripts with zero pytest assertions (0 items collected) —
the same defect `test_rv_model.py` had before the earlier audit fixed it.
Both rewritten in the same style, parametrized over FIN379/GL765(/ZetaAqr):
`test_statistics.py` now asserts `chi_square`/`reduced_chi_square`/
`rms_residual`/`fit_summary` are mutually consistent, that
`combined_fit_summary`'s visual+RV totals match the component residuals
exactly, and that `parameter_comparison` covers every orbital element;
`test_reduced_chi_square_rv.py` now asserts `reduced_chi_square == chi_square
/ dof` exactly and adds a dedicated raise-on-nonpositive-dof regression
case for `inference.likelihood.reduced_chi_square` (a sibling of the
`analysis.statistics`/`inference.covariance` functions fixed above — this
one already raised correctly, now it's locked in by a test). 12/12 pass
locally, no regressions.

- Minor dead code noted but not touched (no behavior change, low value for
  the edit cost): several unused imports (`inference/optimizer.py`,
  `inference/likelihood.py`, `inference/photometric_posterior.py`,
  `inference/photometric_optimizer.py`), `inference/node_disambiguation.py`'s
  `node_assignment_chi2`/`select_best_node_assignment` never called
  (`optimizer.py` reimplements the same logic inline instead),
  `photometry/parameters.py`'s `update_parameters(parameters, values)`
  never actually reads its `parameters` template argument (harmless today,
  latent trap if the dataclass grows a fixed field later),
  `evolution/isochrone.py` reaching into `evolution.age`'s private
  `_track_chi2` instead of the public `track_chi2`/`track_residuals` pair
  (two independently-maintained chi-square implementations that currently
  agree numerically but could silently diverge).

### Confirmed correct (no bugs found)

Stellar atmosphere/SED chain (`stellar/atmospheres.py`, `stellar/sed_model.py`,
all of `photometry/`): CCM89 extinction coefficients match Cardelli, Clayton
& Mathis (1989) exactly with correct attenuation sign; Pogson magnitude
convention consistent throughout; composite binary SED
`F = F1·(R1/d)² + F2·(R2/d)²` correct with no primary/secondary swaps;
CK04 grid bilinear interpolation in (Teff, log g) correct in all branches,
raises rather than silently extrapolating out of range. Mass/error-propagation
chain (`analysis/masses.py`, `analysis/error_propagation.py`): Kepler's third
law correctly cubes parallax (`M=(a/ϖ)³/P²`); the mass-uncertainty formula
`(σ_M/M)² = 9(σ_a/a)² + 4(σ_P/P)² + 9(σ_ϖ/ϖ)²` matches exactly; SB2
mass-function formula matches the standard form. `orbit3_fit.py`'s
finite-difference Jacobian construction re-verified line-by-line against
`alleph`/`fitorb` in `orbit3.pro`, including the `T0`/`V0` step-size quirks.

### Test-suite / repo hygiene — DONE (2026-08-26)

Per `validation_phase_plan.txt`'s own explicit point #10 ("I would not turn
the current 166 tests into 300 tests just for the sake of increasing test
coverage... we have crossed the point of diminishing returns"), identified
23 cleanup candidates in three categories. This session has no file-delete
capability on the user's device (read/write only), so the actual removal
was carried out via a Claude Code (VSCode) session with real shell/git
access, working from an itemized list handed off in chat. Outcome:

1. Nine literally-empty (0-byte) `tests/test_*.py` files — **all 9 removed**
   via `git rm` (tracked, staged): `test_combined_fit.py`,
   `test_conventions.py`, `test_covariance.py`, `test_kepler_solver.py`,
   `test_mcmc.py`, `test_orbit_fit.py`, `test_thiele_innes.py`,
   `test_time_normalization.py`,
   `test_hip111805_orbit3_residual_ordering_diagnostic.py` (this last one
   was untracked, so its removal leaves no git diff — it simply no longer
   exists on disk).
2. Thirteen `tests/test_hip111805_*diagnostic*.py` / `_jacobian_columns.py`
   investigation-log files — **12 of 13 removed** (untracked, no git diff).
   **One kept, correctly**: `test_hip111805_lm_backend_diagnostic.py` was
   flagged back rather than deleted, because on inspection its terminal
   assertion (`np.sum(r_numeric**2) < np.sum(r0**2)`) checks a real,
   specific claim (LM optimization actually reduced chi²), not just a
   generic success/finiteness check like the other twelve. Left in place;
   it's redundant with `test_hip111805_orbit3.py`'s stronger version of the
   same claim but not incorrect or empty, so removing it wasn't clearly
   justified by the original "no real assertion" rationale.
3. `notebooks/benchmark_validation.ipynb` (stale duplicate of
   `validation/benchmark_validation.ipynb`, the one `CLAUDE.md` actually
   documents) — **removed** via `git rm` (tracked, staged).

Net: 22 of 23 candidates removed, 1 kept with a documented reason. As of
this note the removals are staged/committed locally in the pyseshat repo
(commit message covering all three categories); nothing further needed
here.

`MinimalOrbitSolver/` and the root-level `FIN379_fit.json`/`FIN379_report.txt`/
`fin379_ephemeris.*` were explicitly documented in `CLAUDE.md` as intentional
(a standalone cross-check solver and checked-in run artifacts, respectively)
and were left untouched in this pass. **Revisited below (2026-08-27)** —
the user asked for this to be re-inspected rather than taken on faith, and
part of it genuinely was obsolete.

### Second cleanup pass (2026-08-27): `MinimalOrbitSolver/`, `validation/`,
### `examples/`, `data/`, root-level benchmark files

Re-inspected each folder/file CLAUDE.md's directory description calls out,
rather than reusing the earlier "intentional" conclusion at face value.
Live directory listings and file contents were pulled fresh from the
device (the locally staged copy used elsewhere in this session was stale
for these paths — it predated all of them). Findings:

- **`MinimalOrbitSolver/`** — **recommend removing the whole folder.**
  `AstrometricObservation.py` + `utils.py` are a standalone, non-package
  prototype orbit solver (own Kepler solver, own Thiele-Innes constants,
  own `fit_orbit_xy`/plotting, hardcoded `SCALE` array) that duplicates
  functionality `src/pyseshat` now implements far more rigorously
  (tested, integrated with the optimizer/likelihood stack). It also
  carries its own copies of `FIN379.inp`, `GL765.inp`, `ZetaAqr.in2` —
  byte-for-byte identical (same size + mtime) to the canonical copies in
  `validation/benchmark_systems/`. Zero references to `MinimalOrbitSolver`
  anywhere in `src/`, `tests/`, or `validation/` (confirmed by grep).
  CLAUDE.md calls it an intentional "cross-check," but nothing in the
  current automated pipeline invokes it — that role has been fully
  superseded by `validation/benchmark_runner.py` and the regression suite
  (`test_perfect_orbit_recovery.py`, `test_rv_sign_convention_validation.py`,
  etc.), which check the *same* three systems with actual assertions
  instead of a manually-rerun notebook. Includes `ORBIT_test.html`
  (516 KB) and `ORBIT_test.ipynb` (230 KB), both stale exploratory
  artifacts from before `src/pyseshat` existed as a package.
- **`validation/` — NOT obsolete, keep entirely.** This is the live
  validation pipeline: `benchmark_runner.py` fits `benchmark_systems/`
  against `expected_results/` and writes `results/` (`benchmark_report.json`,
  `benchmark_summary.csv`, `figures/*.png` — all regenerated within the
  last day). This *is* the mechanism behind the paper roadmap's
  "Benchmark Results v1" deliverable. `benchmark_systems/HIP111805/` is
  still exercised by the one retained diagnostic test
  (`test_hip111805_lm_backend_diagnostic.py` → `test_hip111805_orbit3.py`'s
  `load_hip111805()`), so it's in active use too, not a leftover. Minor
  fragility noted but not touched: `comparison.py`, `metrics.py`,
  `benchmark_runner.py`, and the two `validation/test_*.py` files all use
  bare same-directory imports (`from metrics import ...`) that only
  resolve when `validation/` itself is on `sys.path` — i.e. they only work
  run as scripts from inside that directory, not as part of the
  `pyseshat` package. Not a deletion candidate, just worth knowing.
- **`examples/` — NOT obsolete, keep.** Contains only `hrdiagram_demo.py`,
  which is a real, working demo of a real feature (binary H-R diagram
  construction and plotting) importing live `pyseshat.evolution`/
  `pyseshat.visualization` modules — not a duplicate of anything, not
  dead. The user's framing suggested this folder might be a cleanup
  target; it isn't.
- **`data/` — empty, undocumented in CLAUDE.md.** Nothing in it, nothing
  references it. Recommend removing (harmless either way — it holds
  nothing today).
- **Root-level single-benchmark-system files** — mixed verdict, not a
  blanket removal:
  - `FIN379_fit.json` — **keep.** This is the actual evidence artifact
    the RV sign-convention resolution above is built on, and it's cited
    both in this document and in `test_rv_sign_convention_validation.py`'s
    docstring. Removing it would leave those references dangling.
  - `FIN379_report.txt` — **recommend removing.** A human-readable
    duplicate of the same fit already preserved in `FIN379_fit.json`; no
    code references it anywhere.
  - `fin379_ephemeris.csv` / `fin379_ephemeris.txt` — **recommend
    removing both.** The same 6-row 2025–2030 ephemeris table in two
    formats, produced by neither the current validation pipeline nor
    anything else in the repo, with zero references anywhere.
- **`architecture_snapshot.py`** (root, found in the course of this audit,
  not originally asked about) — **not obsolete, keep.** CLAUDE.md
  documents `python architecture_snapshot.py` as a real dev command that
  regenerates `docs/architecture/`.

Net recommendation: remove `MinimalOrbitSolver/`, `data/`,
`FIN379_report.txt`, `fin379_ephemeris.csv`, `fin379_ephemeris.txt`. Keep
`validation/`, `examples/`, `FIN379_fit.json`, `architecture_snapshot.py`
as-is. If the removals go ahead, `CLAUDE.md`'s directory-description lines
mentioning `MinimalOrbitSolver/` and the root ephemeris files as
"intentional" need updating to match — left as a follow-up for whoever
applies the deletion, so the doc doesn't describe files that no longer
exist. This session has no file-delete capability on the user's device, so
as with the first cleanup pass, execution needs to go through a separate
session with real shell/git access, working from the itemized list above.

### Next steps (from the paper roadmap)

`validation_phase_plan.txt`'s core conclusion: the architecture is already
sufficient (117 modules/213 functions/166 tests going in) — what's missing
for the paper is not more code, it's **Benchmark Results v1**: a real
quantitative validation table (P, a, e, i, χ², M_dyn, σ_M/M for FIN379,
GL765, ζAqr) plus five figures (orbit fit, residuals vs epoch, lit-vs-PySeshat
parameter comparison, MCMC posterior/credible intervals, mass-precision vs
parallax-precision). This matches the paper roadmap's Phase II ("Reproduce
Published Orbits" — explicitly "essential for publication"). Resolving the
RV sign-convention question above is a prerequisite for any benchmark that
uses RV data (FIN379, GL765) rather than pure astrometry.
