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
- **North-up/East-left visual-orbit convention: NOT verified — read
  before using `plot_visual_orbit()` in a paper figure.**
  `plot_visual_orbit()` plots X horizontally and Y vertically exactly as
  passed in, with no axis inversion, and `compute_visual_orbit()`
  defines its `X`/`Y` only abstractly, without tying either explicitly
  to North or East anywhere in the codebase. Whether the plotted
  orientation already matches the standard North-up/East-left,
  position-angle-from-North-through-East convention depends on the sign
  convention baked into `io/orbit_parser.py`'s `theta` parsing, which
  has not been checked against a published figure. This was
  investigated and deliberately **not** silently "fixed" — the correct
  answer requires reproducing an actual published orbit figure side by
  side (e.g. from `wardat_2002.pdf`, `wardat_2012.pdf`, or
  `Tokovinin_2016_ApJ_831_151.pdf`) and is exactly the kind of
  presentation decision this project's priorities call for verifying
  rather than guessing at. `plot_visual_orbit()`'s own docstring in
  `src/pyseshat/visualization/orbit_plot.py` documents this in full;
  regression tests lock in the *current* numerical behavior so a future
  change would be caught, but do not assert the orientation is correct.
  **Action for the paper**: do this comparison once before using any
  `plot_visual_orbit()` figure in a publication.

## Generating the full benchmark plot set

`scripts/generate_benchmark_plots.py` fits FIN379 and GL765 and writes
their full available plot set (visual orbit, residuals, phase-folded RV,
separation-vs-time) to `validation/results/plots/<system>/`. It
deliberately does **not** attempt the other 6 of 8 registered benchmark
systems — see that script's module docstring for the full reasoning:
ZetaAqr/HIP111805 (hierarchical triples) have no triple-aware orbit-plot
function yet (see `triple_systems.md`), and the 4 Al-Wardat SED systems
have no orbital fit to plot at all, and no per-system HR-diagram
cross-check has been wired up yet (see `evolutionary_checks.md`). Run
it with:

```
MPLBACKEND=Agg python scripts/generate_benchmark_plots.py
```
