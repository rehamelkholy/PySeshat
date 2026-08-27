# Evolutionary-track consistency checks

`evolution/` provides an independent cross-check for a fitted or
literature binary solution: given each component's `(Teff, L)` (or
`(Teff, R)`, convertible via `analysis/luminosity.py`), find the
best-matching point(s) on a stellar evolutionary track/isochrone grid
and check whether both components are consistent with a single common
age (as two components of a real binary must be, if they formed
together).

```python
from pyseshat.evolution.binary import fit_binary_tracks

result = fit_binary_tracks(
    primary_track=..., secondary_track=...,       # EvolutionTrack, one per component's mass
    primary_log_teff=..., primary_log_luminosity=...,
    primary_sigma_log_teff=..., primary_sigma_log_luminosity=...,
    secondary_log_teff=..., secondary_log_luminosity=...,
    secondary_sigma_log_teff=..., secondary_sigma_log_luminosity=...,
)

result.age_difference   # 0 for a perfectly consistent common age
```

`fit_binary_tracks()` is deliberately the *first stage* — it fits each
component's track independently, without imposing a common age;
`age_difference` is the diagnostic that tells you how far apart the two
independent best-fit ages actually are. A full common-age fit
(`evolution/age.py`'s `fit_common_age`) constrains both components to
share one age directly rather than just measuring the discrepancy after
the fact.

## How this connects to Al-Wardat SED fitting

This is the module that gives the real Al-Wardat method its second half
— SED fitting alone cannot separate two unresolved components' radii
(see `alwardat_sed.md`'s degeneracy finding), so the published method
uses an evolutionary-track cross-check as the external constraint that
breaks the degeneracy. `photometry/evolution_adapter.py` and
`inference/joint_likelihood.py`'s `JointInferenceProblem` already
provide the connective tissue between a photometric fit and an
evolutionary-track check, but the specific wiring this project's SED
benchmarks would need — evolutionary-track-predicted radius (given each
component's literature mass) feeding directly into
`PhotometricParameters.priors` in place of the literature radius itself
— is **not yet done**. `JointInferenceProblem.evolution_context` is
currently typed `Any` with no defined per-component sample/observables
contract; designing that contract is the concrete blocker (future work
item 1 in `docs/paper_reference_notes.md`), not further evolutionary-track
code being missing.

## Where this is tested today

`tests/test_evolution_adapter.py`, `tests/test_interpolate.py`, and
`tests/test_hrdiagram_plot.py` cover the individual pieces
(track/isochrone interpolation, the adapter, and HR-diagram plotting —
see `visualization.md`) with synthetic tracks. None of the four
registered Al-Wardat benchmark systems currently run a real
evolutionary-track cross-check as part of `validation/benchmark_runner.py`
— this is the same gap as the missing radius-prior wiring above, and is
tracked the same way.
