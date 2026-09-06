# Visualization

`src/pyseshat/visualization/` has seven plotting functions. This page
covers what each one draws, what's actually tested about the
correctness of that drawing, and — the harder question — which
literature plotting conventions each one does or does not currently
guarantee.

An eighth plotting function, `MCMCResult.corner_plot()`
(`src/pyseshat/inference/mcmc_result.py`), lives outside this package
(it's a method on the MCMC posterior-result class rather than a
free function operating on plain arrays/elements) but is otherwise
treated identically for the purposes of this page and the
`show_grid`/`output_format`/`save_path` parameters described below.

| Function | Module | Operates on |
|---|---|---|
| `plot_visual_orbit` | `orbit_plot.py` | flat `OrbitalElements`, visual astrometry |
| `plot_residuals` | `orbit_plot.py` | any `(t, residuals)` pair |
| `plot_rv_orbit` | `orbit_plot.py` | flat `OrbitalElements`, RV data |
| `plot_separation_vs_time` | `dynamics_plot.py` | flat `OrbitalElements`-derived separation |
| `plot_binary_hrdiagram` | `hrdiagram_plot.py` | `HRDiagramData` (evolutionary tracks) |
| `plot_triple_orbit` | `triple_orbit_plot.py` | `TripleOrbitalElements` (ORBIT3), both visual observables |
| `plot_sed_fit` | `sed_plot.py` | precomputed SED curves (composite + per-component flux vs. wavelength) and observed/excluded photometric points |

`plot_triple_orbit` (new 2026-08-30, see `docs/guide/triple_systems.md`
"Orbit plots for triples") is the hierarchical-triple counterpart of
`plot_visual_orbit` — same North-up/East-left convention, but drawn as
two side-by-side panels (POS1 = resolved inner orbit; POS2 = outer +
f*inner, the wobble-corrected wide/outer orbit) rather than one, since a
flat two-body curve is not a physically correct model for either of a
triple's two coupled observation streams on its own.

## Correctness: tested on actual plotted data, not just "it ran"

`tests/test_orbit_plot.py` asserts on the real matplotlib objects each
function produces — `ax.collections[].get_offsets()` for scatter series,
`ax.get_lines()[].get_xdata()/get_ydata()` for model curves — checked
against the exact arrays passed in, not just that a `Figure`/`Axes` came
back without an exception. This catches, for example, a
swapped-axis or transposed-data regression that a "runs without raising"
check would miss entirely. Coverage: `plot_visual_orbit`'s observation
scatter and orbit-model curve; `plot_residuals`'s X/Y residual split;
`plot_rv_orbit`'s primary/secondary RV scatter (matched per-component,
not just "some points near the right magnitude") and its phase-folded
model curve. `test_dynamics.py` does the same for
`plot_separation_vs_time`.

## Literature-convention conformity

Checked against the field's standard conventions, one at a time:

- **HR-diagram Teff axis: correct, already tested.**
  `plot_binary_hrdiagram()` plots `log(Teff)` decreasing left-to-right
  (an inverted x-axis), matching the standard HR-diagram convention.
  `tests/test_hrdiagram_plot.py::test_plot_binary_hrdiagram` asserts
  `ax.xaxis_inverted()`.
- **RV phase-folding zero point: correct, tested.** `plot_rv_orbit`'s
  `phase=True` mode zeros orbital phase at `elements.T0` (periastron),
  the standard convention. See
  `tests/test_orbit_plot.py::test_plot_rv_orbit_phase_folded_model_is_centered_at_zero_phase`.
- **Residual sign convention: consistent with the fit itself, no
  mismatch.** `plot_residuals()` draws exactly the array it's handed;
  the only producer, `inference/terms/visual.py`'s
  `visual_xy_residuals`, computes `(observed - model) / sigma`, and
  every fitting pathway passes that same convention through unchanged.
- **North-up/East-left visual-orbit convention: settled and
  implemented (2026-08-27) — safe to use `plot_visual_orbit()` output
  as-is in a paper figure.** Cross-referenced the IAU's own PA
  definition ("measured relative to the north celestial pole, turning
  positive into the direction of the right ascension"), the WDS's own
  usage, the `orbitize` package's documented convention, and a textbook
  cross-check (Tatum) — all agree: PA from North, turning positive
  toward East; a published orbit figure is North up, East left.
  Re-derived from PySeshat's own Thiele-Innes formalism
  (`X = rho cos(theta)`, `Y = rho sin(theta)`, `theta` = the `.inp` PA
  column) that this codebase's `X` is the North/Dec offset and `Y` the
  East/RA*cosDec offset — cross-checked independently by confirming
  `thiele_innes()`'s A,B,F,G formula matches Gaia DR3's own documented
  formula exactly. `plot_visual_orbit()` now plots East on the
  horizontal axis (inverted, so East increases to the left) and North
  on the vertical, with explicit "East (arcsec)"/"North (arcsec)"
  labels — data values themselves are not negated, only the axis
  direction is, matching standard practice for a plotted RA/East axis.
  `compute_visual_orbit()`'s own docstring states the settled
  convention and derivation explicitly.
  `tests/test_orbit_plot.py::test_plot_visual_orbit_is_north_up_east_left`
  regression-tests both the axis inversion and the counterclockwise PA
  sense this produces (a PA=90 deg/due-East point plots left of the
  origin, matching the IAU's own stated sense); visually confirmed
  against FIN379's, Capella's, and alpha Cen AB's regenerated plots.
- **SED-fit flux-vs-wavelength convention: sourced from the
  Al-Wardat-method literature itself, not assumed (2026-09-02).**
  Before writing `plot_sed_fit()`, no plotting function for
  `fit_photometry()`'s output existed anywhere in this package
  (confirmed by grepping the whole `visualization/` package and
  `validation/results/plots/` for anything photometry/SED-plot
  related), and this was a previously-tracked gap. Checked the
  actual source papers rather than guessing a representation:
  Al-Wardat (2002, 2012) plot flux density versus wavelength ("Figure 1
  shows the best fit between the total synthetic SED and the
  observational one" — `wardat_2012.pdf`), and Abushattal et al. 2025
  (arXiv:2512.23652, the direct source of this package's four
  Al-Wardat benchmark systems) follows the same convention in its own
  Figures 1-4: wavelength (Angstrom) on the x-axis, flux on the
  y-axis, three curves per system (total/composite, component A,
  component B), with observed photometric points overplotted as
  discrete error-barred markers. `plot_sed_fit()` matches this
  directly — composite as a solid black curve, each component as a
  fainter dashed curve, observed points as black error bars (0.30 mag
  literature uncertainty per Abushattal et al. 2025, propagated to
  flux), and the codebase's own excluded Stromgren-u measurement drawn
  distinctly (open gray markers) rather than silently dropped from the
  figure. See `sed_plot.py`'s module docstring for the full derivation
  and `docs/guide/alwardat_sed.md`'s "SED-fit figure" section for the
  per-system results. **[2026-09-02, same-day follow-up]** The first
  version's passband labels (drawn at a fixed offset) collided for any
  two passbands close in wavelength/flux — reported by the user for
  HD191854's "Stromgren b"/"Johnson B" and "Johnson V"/"Tycho VT"/
  "Stromgren y" clusters. Fixed with
  [adjustText](https://github.com/Phlya/adjustText) (new pinned
  dependency), an automatic label-collision-avoidance library that
  repels every label away from every other label and from the plotted
  points, drawing a thin leader line back to a label's point whenever
  it had to move — plus a white text halo (`matplotlib.patheffects`)
  so a relocated label stays legible over a busy stretch of the model
  curve, which `adjustText` itself has no notion of avoiding. Verified
  computationally (zero pairwise label bounding-box overlap across all
  four systems, `tests/test_sed_plot.py::test_plot_sed_fit_avoids_overlapping_labels`)
  and visually (all four regenerated figures inspected directly,
  including the three near-twin systems whose tightly clustered curves
  make this the hardest case).

## Generating the full benchmark plot set

`scripts/generate_benchmark_plots.py` fits FIN379, GL765, and the two
2026-08-27 joint-benchmark orbits (Capella, alpha Cen AB — the latter
has no RV, so it gets 3 plots instead of 4) and writes each system's
full available plot set (visual orbit, residuals, phase-folded RV where
applicable, separation-vs-time) to `validation/results/plots/<system>/`.
**[2026-08-30]** It now also generates ZetaAqr's and HIP111805's
(hierarchical triples) wobble-corrected `triple_orbit.pdf` figures (see
`triple_systems.md`, `plot_triple_orbit` above), via
`scripts/generate_triple_orbit_plots.py`, and Capella's MCMC posterior
`corner.pdf` (see "show_grid, output_format, save_path" below), via
`scripts/generate_capella_corner_plot.py`. It still does **not**
generate an orbit-fit plot for the four Al-Wardat-SED-only systems
(there is no orbital fit to plot for those at all — a per-system
HR-diagram cross-check is generated for them separately instead, along
with Capella's/AlphaCenAB's own photometry rows — see
`evolutionary_checks.md`). **[2026-09-02]** It now does generate an
SED-fit plot for those four systems: `generate_sed_fit_plots()` calls
`benchmark_runner.run_alwardat_benchmark()` for each of HD214222,
HD191854, HD130669, and HD184467 (the exact same call that produces
this package's own reported reduced chi-square values, so the plotted
curve and the cited number cannot silently diverge — see "SED-fit
flux-vs-wavelength convention" above), builds the composite and
per-component model curves via
`photometry.model.synthetic_binary_sed()`, and writes each system's
`sed_fit.pdf` to `validation/results/plots/<system>/`.
`Capella_photometry` and `AlphaCenAB_photometry` are deliberately
excluded from this step — both are entirely synthetic benchmarks
generated by this same pipeline, so an "observed vs. model" SED-fit
figure would not be a meaningful validation figure for them the way it
is for the four real (if paper-synthetic-origin) Abushattal systems.
Run it with:

```
MPLBACKEND=Agg PYTHONPATH=src:validation python scripts/generate_benchmark_plots.py
```

## show_grid, output_format, save_path

**[2026-08-30]** Every plotting function on this page (plus
`MCMCResult.corner_plot()`) accepts three additional keyword
arguments:

- **`show_grid: bool`** — toggles `Axes.grid()` on the plotted axes.
  Defaults to `True` everywhere (matching each function's original
  always-on-grid behavior, so no pre-existing call site's appearance
  changes), **except `MCMCResult.corner_plot()`, which defaults to
  `False`** — `corner.corner()` draws no grid lines by default and
  this function had never added any, so `False` is the value that
  actually preserves *its* original behavior.
- **`output_format: str`** — the file format passed to
  `Figure.savefig(..., format=output_format)` (e.g. `"png"`, `"pdf"`,
  `"svg"`). Defaults to `"png"` everywhere.
- **`save_path`** — if given, the figure is saved to this path (in
  `output_format`) before returning; if `None` (the default), the
  function behaves exactly as before (no internal save; the caller
  saves the returned `Figure` externally if it wants to).

**The paper's own figure-generation pipeline overrides both defaults.**
`scripts/generate_benchmark_plots.py`, `scripts/generate_triple_orbit_plots.py`,
and `scripts/generate_capella_corner_plot.py` all pass
`output_format="pdf"`, `show_grid=False` for every figure they write
— vector PDF without grid lines is what A&A has specifically requested
for this paper's submitted figures. This is a pipeline-level default
only: the plotting functions themselves keep the general-purpose
defaults above for interactive/exploratory use, `examples/`, and every
other test in this codebase.
