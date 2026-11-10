# Hierarchical triple systems: legacy ORBITT2 vs. modern ORBIT3

A hierarchical triple is modeled as an outer orbit (the distant third
component around the inner pair's center of mass) plus an inner orbit
(the close pair), coupled through how the inner orbit perturbs the
outer's observed astrometric position. PySeshat supports **two
historically distinct formalisms for that coupling**, kept
architecturally separate rather than unified into one shared
abstraction: they are not two syntactic forms of the same physics, so
forcing a shared representation would risk silently changing the
model.

## Legacy ORBITT2 (15 parameters)

Couples the inner orbit to the outer astrometric measurement via a
simple scale factor `B`: `r_I2 = B * r_inner`. Parsed from a `.in2`
file (see `data_formats.md`); fit via `triple.triple_fit.array_to_triple`
and the legacy residual functions in `triple_fit.py`. No RV information
is modeled in this formalism.

## Modern ORBIT3 / Tokovinin (20 parameters)

Couples the inner orbit to the outer astrometric measurement via a
physically distinct wobble factor `f`: `POS2 = r_outer + f * r_inner`,
and additionally models both the inner pair's RV semi-amplitudes
(`K1`, `K2`) and the outer pair's (`K3`, `K4`, `V0`). Parsed from a
`.in3` file; fit via `triple.orbit3_fit.fit_orbit3`, the entry point
used for both registered triple benchmarks (ζ Aquarii, HIP 111805).

```python
from pyseshat.triple.triple_parser import parse_triple_file
from pyseshat.triple.orbit3_fit import fit_orbit3

parsed = parse_triple_file("validation/benchmark_systems/HIP111805/HIP111805.in3")
fit_result = fit_orbit3(
    parsed["elements"], parsed["observations"], parsed["rv_observations"],
)
fitted = fit_result.elements   # TripleOrbitalElements, .outer and .inner
                                # are each a real OrbitalElements
```

**`f` is deliberately left unbounded** in the optimizer, not restricted
to the `(0, 1)` its naive physical definition
(`f = q_inner / (1 + q_inner)`) might suggest — HIP 111805's own
published solution has `f = -0.330`; bounding it would make that real,
already-published system's solution unreachable.

## A worked example: ζ Aquarii

ζ Aqr (`validation/benchmark_systems/ZetaAqr/`) is the project's main
ORBIT3 stress test: a highly eccentric inner orbit (e≈0.87), a very long
outer orbit (P₂≈540 yr), and 331 astrometric observations spanning six
decades. Two things worth knowing if you cite or extend this benchmark:

1. Tokovinin (2016)'s published astrometric wobble amplitude
   (`a_astrometric = 0.110″`) is **not** the ORBIT3 inner semimajor axis
   — it must be converted via `a_true = a_astrometric / f` (using
   `f = 1/F ≈ 0.2857` from `F ≈ 3.50`) to get the actual `a = 0.385″`
   starting point. This conversion is done explicitly in
   `validation/benchmark_runner.py::_build_orbit3_elements`, not left
   implicit.
2. **A triple must be fit as a whole, simultaneously** — fitting the
   outer orbit alone, or fitting the inner orbit's residuals from a
   separate outer-only fit, is explicitly wrong for a hierarchical
   triple (the outer astrometric measurement itself already contains
   the inner orbit's wobble). `run_orbit3_benchmark()` in
   `validation/benchmark_runner.py` is the canonical example of fitting
   both simultaneously.

PySeshat's fit *improves* on the literature χ² (694.51 vs. 699.73 at the
literature starting point) and recovers a dynamical mass
M_dyn ≈ 3.26 ± 0.40 M☉ (≈12.4% relative uncertainty) consistent with
both the parallax-uncertainty-only expectation (~10.6%) and Tokovinin's
approximate component-mass sum (≈3.4 M☉). A single residual outlier at
epoch 2009.7552 was investigated and explained rather than left as an
unexplained anomaly: it traces to two disagreeing speckle-astrometry
reductions present in the source data itself, not a PySeshat fitting
bug.

## HIP 111805: the most literature-complete benchmark

`validation/benchmark_systems/HIP111805/HIP111805.in3`'s header already
*is* Tokovinin & Latham (2017)'s full published 20-parameter solution
with uncertainties — so PySeshat's fit reproduces it essentially exactly
(every parameter's relative error 0.0%). This is a genuine, if
comparatively trivial, regression check (confirming the optimizer
correctly recognizes it's already at a stationary point) rather than a
convergence-from-scratch demonstration — ζ Aqr is the system that
actually stress-tests convergence from a non-trivial starting point. See
`examples/03_hierarchical_triple_orbit3.py` for a runnable comparison of
both systems side by side.

## Orbit plots for triples

`src/pyseshat/visualization/triple_orbit_plot.py`'s `plot_triple_orbit()`
is the triple-aware orbit-plotting function. `TripleOrbitalElements.outer`
is a real `OrbitalElements`, so `plot_visual_orbit()` could in principle
be pointed at it directly — but the outer astrometric observations these
systems are fit against include the inner subsystem's astrometric wobble
(the `f` term), which a flat two-body curve doesn't model; plotting one
directly against those observations would show a real, wobble-driven
residual pattern a reader could easily mistake for a fit-quality problem.
`plot_triple_orbit()` avoids this by drawing BOTH ORBIT3 visual
observables side by side, in the same North-up/East-left convention as
`plot_visual_orbit()`: the resolved inner orbit (POS1) in the left panel,
and the wobble-corrected wide/outer orbit (POS2 = r_outer + f*r_inner) in
the right panel — so neither curve is ever plotted against observations
it does not actually model.

`scripts/generate_triple_orbit_plots.py` (wired into
`scripts/generate_benchmark_plots.py`) generates both systems' figures
using the same fitted solution `validation/benchmark_runner.py`'s
`run_orbit3_benchmark()` produces for the paper's benchmark table
(cross-checked at generation time, not just assumed to match), saved to
`validation/results/plots/ZetaAqr/triple_orbit.pdf` and
`validation/results/plots/HIP111805/triple_orbit.pdf` (vector PDF, no
grid lines, per A&A's figure requirements). ζ Aqr's epoch-2009.7552
residual anomaly (see above) is drawn with a distinct marker on the
inner-orbit panel; HIP 111805's plot shows a visually striking wobble
signature, since its short ~1.5 yr inner period makes the wobble term
appear as a braided/doubled curve in the outer panel rather than a
subtle perturbation.
