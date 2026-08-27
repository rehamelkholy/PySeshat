# Visualization

`src/pyseshat/visualization/` has exactly five plotting functions.
This page covers what each one draws, what's actually tested about the
correctness of that drawing, and — the harder question — which
literature plotting conventions each one does or does not currently
guarantee.

| Function | Module | Operates on |
|---|---|---|
| `plot_visual_orbit` | `orbit_plot.py` | flat `OrbitalElements`, visual astrometry |
| `plot_residuals` | `orbit_plot.py` | any `(t, residuals)` pair |
| `plot_rv_orbit` | `orbit_plot.py` | flat `OrbitalElements`, RV data |
| `plot_separation_vs_time` | `dynamics_plot.py` | flat `OrbitalElements`-derived separation |
| `plot_binary_hrdiagram` | `hrdiagram_plot.py` | `HRDiagramData` (evolutionary tracks) |

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
  See `docs/project_decisions.md`'s 2026-08-27 dated entry for the full
  derivation.

## Generating the full benchmark plot set

`scripts/generate_benchmark_plots.py` fits FIN379, GL765, and the two
2026-08-27 joint-benchmark orbits (Capella, alpha Cen AB — the latter
has no RV, so it gets 3 plots instead of 4) and writes each system's
full available plot set (visual orbit, residuals, phase-folded RV where
applicable, separation-vs-time) to `validation/results/plots/<system>/`.
It deliberately does **not** attempt the other 8 of 12 registered
benchmark systems — see that script's module docstring for the full
reasoning: ZetaAqr/HIP111805 (hierarchical triples) have no
triple-aware orbit-plot function yet (see `triple_systems.md`), and the
six Al-Wardat-SED-only systems (the original four plus
Capella/AlphaCenAB's own photometry rows) have no orbital fit to plot at
all, and no per-system HR-diagram cross-check has been wired up yet (see
`evolutionary_checks.md`). Run it with:

```
MPLBACKEND=Agg python scripts/generate_benchmark_plots.py
```
