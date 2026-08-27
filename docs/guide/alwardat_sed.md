# Al-Wardat unresolved-binary SED fitting

This is the method the paper this project supports is built around
(Abushattal, Georgakarakos, Al-Wardat, Algnamat, Haboubi, Dewi,
Abu-Alrob & Hussein 2025, AJ, in press; arXiv:2512.23652;
doi:10.3847/1538-3881/ae0582): given a binary's *combined*, unresolved
photometry (magnitudes in several passbands, with no way to separate
each component's individual flux directly), fit a two-component
Castelli-Kurucz (CK04) synthetic spectral model to recover each
component's effective temperature, radius, and the system's reddening
E(B-V). See `examples/04_alwardat_sed_fit.py` for the full runnable
version this page summarizes.

**Unlike the orbit-fitting pages, there is no orbital fit here at all**
— this validates stellar atmosphere modeling given a set of measured
magnitudes, using `photometry/` and `stellar/atmospheres.py`
independently of `orbit/`/`triple/`.

```python
from pyseshat.stellar.atmospheres import load_ck04_grid
from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.observations import PhotometricMeasurement
from pyseshat.photometry.parameters import PhotometricParameters
from pyseshat.inference.photometric_optimizer import fit_photometry, build_parameter_bounds

# measurements, atmosphere_grid, passbands, zero_points loaded from
# validation/expected_results/HD214222.json — see data_formats.md

parameters = PhotometricParameters(
    primary_temperature=..., secondary_temperature=...,
    primary_radius=..., secondary_radius=..., ebv=...,
    fit_flags={"primary_temperature": True, "secondary_temperature": True,
               "primary_radius": True, "secondary_radius": True, "ebv": True},
    priors={"primary_radius": (lit_R1, 0.10 * lit_R1),
            "secondary_radius": (lit_R2, 0.10 * lit_R2)},
)

fit_result = fit_photometry(
    parameters=parameters, atmosphere_grid=atmosphere_grid,
    passbands=passbands, zero_points=zero_points, measurements=measurements,
    primary_logg=..., secondary_logg=..., metallicity=..., distance=...,
    bounds=build_parameter_bounds(parameters, atmosphere_grid=atmosphere_grid),
)
```

## The fit-flags/prior mechanism

`PhotometricParameters` mirrors `OrbitalElements.fit_flags`: any subset
of the 5 photometric parameters can be held fixed at its starting value,
and any *fitted* parameter can optionally carry a Gaussian prior tying
it to an external estimate (`priors={name: (mean, sigma)}`). This
replaced an earlier workaround (pinning a parameter via near-zero-width
optimizer bounds, which trivially "matched" the target by construction
and gave no real sense of whether the photometric data agreed or
disagreed with it) — the prior is expressed as an ordinary pseudo-residual
appended to the real measurement residuals for the optimizer, while
`photometric_residuals_from_problem()` gives you back the pure
measurement residuals (without the prior's pseudo-residual mixed in) for
honest chi-square/RMS reporting. See
`docs/project_decisions.md`'s "`PhotometricParameters` fit-flags/prior
mechanism" entry for the full derivation.

## Why radius needs a prior at all: a real, physical degeneracy

All five photometric parameters are freed in every registered benchmark
— including both components' radii — but radius is freed **with** a
prior tying it to the system's literature value (10% assumed relative
sigma; the source paper gives no formal radius uncertainty for any of
the four systems). This is not an arbitrary choice: an unresolved
combined-light SED constrains each component's flux only through a
`(Teff, R)` combination, so no number of broadband photometric points
alone can separate two components' radii without an external constraint
— confirmed experimentally at 4 bands, again at 10 bands, and again
after implementing the fit-flags mechanism above (the degeneracy is not
a grid-coverage or measurement-count artifact). This is *why* the real
Al-Wardat method combines SED fitting with evolutionary-track
cross-checks (`evolutionary_checks.md`) rather than fitting radius from
photometry alone — this software's own finding independently reproduces
the methodological reason for that design.

**The prior currently used is circular as a genuine independent test**
(it uses the literature radius as its own mean, so "the fit recovers the
literature radius" is close to true by construction) — the intended real
fix is to tie the prior to an evolutionary-track-predicted radius (given
each component's literature mass) instead, via `evolution/binary.py`;
not yet wired in, tracked as future work item 1 in
`docs/paper_reference_notes.md`.

## A second, independent finding: near-twin components are hard to split

Of the four registered benchmark systems, HD191854 (a well-separated
490 K Teff split, 5930 K vs. 5440 K) converges cleanly with no
atmosphere-grid edge pinning. The other three — HD214222 (75 K split),
HD130669 (60 K split), HD184467 (155 K split) — each have one
component's fitted temperature pin exactly at the local CK04 grid's
edge. Confirmed by direct experiment (widening the grid for HD130669 and
HD184467 down to 4250 K, well past their literature temperatures) to be
a genuine near-twin-component-temperature degeneracy, not a
grid-coverage gap: widening the grid only relocates the pin and makes
the individual-temperature match *worse*, while the combined-light
chi-square barely changes either way. Reduced chi-square staying
excellent (~0.11-0.15) throughout despite this is itself evidence that
many different `(Teff1, Teff2)` splits fit the *combined* light almost
equally well when the two components are near-twins — worth keeping in
mind before over-interpreting any single system's individual-component
temperature split from unresolved photometry alone when the components
are close in temperature.

## Stromgren u: excluded, and why

Present in the source paper's Table 4 but disagreeing with PySeshat's
model by ~1.7 mag (far outside the ~0.25-0.5 mag systematic every other
band shows) in all four systems. Two literature sources were checked
(the SVO Filter Profile Service's own Vega zero point, and Gray et al.
2011's independent CALSPEC-derived offsets) and neither supports a
correction anywhere near 1.7 mag; the originally-suspected explanation
(a Crawford & Barnes 1970 historical uvbybeta calibration quirk) does
not survive this scrutiny and has been retracted in the project's own
records. Reverse-fitting an empirical correction from the benchmark data
itself was considered and rejected as fabrication. Stromgren u stays
excluded from all four fits pending an authoritative source — see each
system's `excluded_measurements` field in its `validation/expected_results/`
JSON, and `docs/project_decisions.md` for the full chain of reasoning.

## An important epistemic caveat for the paper

The source paper's Table 4 magnitudes are that paper's **own synthetic
model output**, not independent catalog photometry — confirmed by direct
inspection (Table 4 is captioned as synthetic-spectrum magnitudes, not
observed catalog values). This reframes what these four benchmarks
actually validate: PySeshat's synthetic photometry pipeline reproducing
*another pipeline's* synthetic photometry to within its own documented
systematic (~0.25-0.5 mag per band, absorbed into the assumed 0.30 mag
measurement uncertainty), not model-vs-real-sky-measurement agreement.
Worth stating explicitly in the paper's methodology section, since it
changes the epistemic status of the comparison from "matches real
observations" to "matches another group's synthetic pipeline output to
within a documented systematic."
