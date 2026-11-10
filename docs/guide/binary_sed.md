# unresolved-binary SED fitting

This is the method the paper this project supports is built around
(Abushattal, Georgakarakos, Al-Wardat, Algnamat, Haboubi, Dewi,
Abu-Alrob & Hussein 2025, AJ, 170, 268; arXiv:2512.23652;
doi:10.3847/1538-3881/ae0582): given a binary's *combined*, unresolved
photometry (magnitudes in several passbands, with no way to separate
each component's individual flux directly), fit a two-component
Castelli-Kurucz (CK04) synthetic spectral model to recover each
component's effective temperature and radius. See
`examples/04_binary_sed_fit.py` for the full runnable version this page
summarizes.

**Unlike the orbit-fitting pages, there is no orbital fit here at all**
— this validates stellar atmosphere modeling given a set of measured
magnitudes, using `photometry/` and `stellar/atmospheres.py`
independently of `orbit/`/`triple/`.

```python
from pyseshat.stellar.atmospheres import load_ck04_grid
from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.observations import MagnitudeDifference, PhotometricMeasurement
from pyseshat.photometry.parameters import PhotometricParameters
from pyseshat.photometry.calibration import vega_zero_points
from pyseshat.inference.photometric_optimizer import fit_photometry, build_parameter_bounds

# measurements, atmosphere_grid, passbands loaded from
# validation/expected_results/HD214222.json — see data_formats.md

# Standard-system zero points: the CALSPEC Vega spectrum shipped with
# PySeshat integrated through the same passbands, with Vega's
# standard-system magnitudes (Vega u = 1.432 in Stromgren uvby).
zero_points = vega_zero_points(passbands)

parameters = PhotometricParameters(
    primary_temperature=..., secondary_temperature=...,
    primary_radius=..., secondary_radius=..., ebv=0.0,
    fit_flags={"primary_temperature": True, "secondary_temperature": True,
               "primary_radius": True, "secondary_radius": True, "ebv": False},
    priors={"primary_radius": (lit_R1, 0.10 * lit_R1),
            "secondary_radius": (lit_R2, 0.10 * lit_R2)},
)

fit_result = fit_photometry(
    parameters=parameters, atmosphere_grid=atmosphere_grid,
    passbands=passbands, zero_points=zero_points, measurements=measurements,
    magnitude_differences=[MagnitudeDifference("Johnson V", 0.15, 0.05)],
    primary_logg=..., secondary_logg=..., metallicity=..., distance=...,
    bounds=build_parameter_bounds(parameters, atmosphere_grid=atmosphere_grid),
)
```

## Zero points: standard systems, not Vega = 0

PySeshat follows the synthetic-magnitude convention of Maíz Apellániz
(2006, AJ, 131, 1184; 2007, ASP Conf. Ser., 364, 227):
`m_P = -2.5 log10(F_P / F_P,Vega) + ZP_P`, with the target's and Vega's
fluxes integrated through the *same* passband P and ZP_P Vega's
magnitude in the standard system of that passband
(`pyseshat.photometry.calibration`: `VEGA_STANDARD_MAGNITUDES`, Table 4
of Maíz Apellániz 2007, e.g. Strömgren u = 1.432; the CALSPEC Vega
spectrum `alpha_lyr_stis_003` of Bohlin 2007, for which those values are
defined, ships with the package; `vega_zero_points(passbands)` returns
the zero points).

Before 0.2.0 the benchmarks used SVO Filter Profile Service "Vega
system" zero points directly. That set ZP_P = 0 (Strömgren u 1.43 mag
too bright, which was mistaken for a band-specific pipeline difference,
so Strömgren u was excluded from the fits) and used Vega fluxes computed
through SVO's passband curves, which for Johnson U and B are not the
curves in `validation/photometry/passbands/` (offsets of about +0.07
and -0.07 mag).

## Passband curves: the ones the zero points belong to

Each Vega magnitude in Maíz Apellániz (2007, Table 4) is defined for a
specific sensitivity curve. The bundled curves are therefore those
curves (`validation/photometry/passbands/README.md`): the photon-counting
Johnson UBV, Strömgren uvby and Tycho BT/VT curves of Maíz Apellániz
(2006, Tables 1-3; new U and u curves derived there, the others from
Bessell 1990, Matsushima 1969 and Bessell 2000), and the energy-integrating
Cousins R curve of Bessell (1983, PASP, 95, 480, Table AII). A
`# detector: photon` header makes `synthetic_band_flux` weight the curve
by λ (Maíz Apellániz 2006, Eq. A1).

Before 0.2.0 the Johnson UBV, Strömgren and Tycho VT files were other
curves of unrecorded origin, every curve was integrated as
energy-integrating, and the R band was a "Johnson R" curve of unknown
origin used with the Cousins R zero point. The observed solar spectrum
then gave U-B = 0.06 instead of about 0.17 (Willmer 2018), so for real
photometry of solar-type stars the U band pulled fitted temperatures
down by a few per cent (alpha Cen: 2-4 per cent).

Checks (`tests/test_synthetic_photometry_sun_calibration.py`), observed
CALSPEC solar spectrum: U, B, V = 5.645, 5.494, 4.829 (Willmer 2018:
5.61, 5.44, 4.81), U-B = 0.15, and Strömgren b-y, m1, c1 = 0.431, 0.191,
0.307 (Meléndez et al. 2010, A&A, 522, A98: 0.411, 0.212, 0.332). The
CK04 solar model gives m1 and c1 that differ from the observed Sun by
+0.065 and -0.036 mag, a model-atmosphere difference; that test uses a
0.08 mag tolerance for m1 and c1.

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
chi-square/RMS reporting.

## Why radius needs a prior at all: a real, physical degeneracy

Both temperatures and both radii are freed in every registered benchmark
(E(B-V) is held at 0; see below), but radius is freed **with** a
prior tying it to the system's literature value (10% assumed relative
sigma; the source paper gives no formal radius uncertainty for any of
the four systems). This is not an arbitrary choice: an unresolved
combined-light SED constrains each component's flux only through a
`(Teff, R)` combination, so no number of broadband photometric points
alone can separate two components' radii without an external constraint
— confirmed experimentally at 4 bands, again at 10 bands, and again
after implementing the fit-flags mechanism above (the degeneracy is not
a grid-coverage or measurement-count artifact). This is *why* the real
binary SED-fitting method combines SED fitting with evolutionary-track
cross-checks (`evolutionary_checks.md`) rather than fitting radius from
photometry alone — this software's own finding independently reproduces
the methodological reason for that design.

**The prior currently used is circular as a genuine independent test**
(it uses the literature radius as its own mean, so "the fit recovers the
literature radius" is close to true by construction) — the intended real
fix is to tie the prior to an evolutionary-track-predicted radius (given
each component's literature mass) instead, via `evolution/binary.py`;
not yet wired in, tracked as future work.

## Temperature, reddening and the component split

`scripts/sed_wide_grid.py` refits the four Abushattal et al. (2025)
systems on one common CK04 grid (solar metallicity, 3500-10000 K) from
five starting temperature pairs (`validation/results/sed_wide_grid*.json`):

- **E(B-V) free**: the combined light does not separate temperature
  from reddening; solutions as hot as ~8000 K with E(B-V) ~ 0.55 lie
  within Δχ² < 1 of the best fit.
- **E(B-V) = 0** (all four systems are within 75 pc): the temperatures
  are better defined, but the split between the components is not —
  fits from different starts exchange the components or put most of the
  light in one of them, with temperature errors up to 17%.
- **E(B-V) = 0 plus the component magnitude difference** Δm_V (as in the
  published method, which takes it from speckle interferometry or
  spectroscopy): all five starts converge to the same solution for every
  system. This is the benchmark configuration (`magnitude_differences`
  in each expected_results file, σ = 0.05 mag assumed).

The fitted temperatures are 2.7-4.9% below the published ones while the
component splits agree to within 16 K. The offset traces mostly to the
source paper itself: its Table 4 V magnitudes are 0.13-0.40 mag fainter
than its own Table 5 absolute magnitudes imply at the adopted distances,
while PySeshat reproduces those absolute magnitudes to within 0.08 mag
(`scripts/published_params_photometry.py`); rescaling the distances to
remove the mean offset gives temperatures within 0.6% of the published
values (`scripts/refit_scaled_distance.py`).

## SED-fit figure (essential figure 4)

`validation/results/plots/<system>/sed_fit.pdf` exists for all four
Abushattal et al. (2025) benchmark systems (HD214222, HD191854,
HD130669, HD184467). See
`docs/guide/visualization.md`'s "SED-fit flux-vs-wavelength
convention" section for the full visual-convention derivation from
Al-Wardat (2002, 2012) and Abushattal et al. 2025's own SED figures
(flux density vs. wavelength, composite + per-component model curves,
observed points at each passband's pivot wavelength with the assumed
0.30 mag uncertainty propagated to flux error bars; any measurement
listed under `excluded_measurements` would be drawn distinctly, but
since 0.2.0 none of the four systems excludes a band).

Each figure is generated from that system's own fitted solution —
`scripts/generate_benchmark_plots.py`'s `generate_sed_fit_plots()`
calls the same `benchmark_runner.run_sed_benchmark()` used to
produce the registered reduced-chi-square values (0.08-0.13 in 0.2.0),
so the plotted curve cannot silently drift from the reported number.

## CK04 grid coverage, and what metallicity range this package supports

**The full CK04 grid is downloaded and available** (a grid-completeness
audit confirms this):
`validation/atmosphere_models/` holds all 608 FITS files specified by
Castelli & Kurucz (2004) --

- **Effective temperature**: 3500-50000 K, 76 grid points (non-uniform
  spacing; finer near typical F/G/K temperatures).
- **Surface gravity**: log g = 0.0-5.0 (cgs), 0.5-dex steps.
- **Metallicity**: [M/H] in {-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, +0.2,
  +0.5} dex (directories `ckm25`...`ckm05`, `ckp00`, `ckp02`, `ckp05`).

`AtmosphereGrid.interpolate()` (`stellar/atmospheres.py`) bilinearly
interpolates in (Teff, log g) but selects metallicity **exactly** --
metallicity is not interpolated, so a target star's [Fe/H] must be
rounded to the nearest available grid step (0.0 is exact for a
genuinely solar star; the largest possible rounding error is 0.25 dex,
at the midpoint between two adjacent steps). `load_ck04_grid()` only
loads whichever specific files you pass it, so a per-system
`atmosphere_files` list (as used throughout `validation/expected_results/`)
must bracket BOTH the target Teff range AND include files from the
correct metallicity subdirectory for a non-solar fit.

**Package scope for general use**: this package targets
**[Fe/H] roughly -1.0 to +0.5 dex** as its documented default scope --
solar-neighborhood F/G/K dwarfs and subgiants, matching the target
population of the benchmark method (Sun-like SB2 binaries), not just the
12 benchmark systems' own literature values. The full downloaded grid
(-2.5 to +0.5 dex) already covers this target range with margin on the
metal-poor side; a target star outside -1.0 to +0.5 dex is still
fittable (the grid extends to -2.5 dex) but is outside this package's
primary documented use case. A target star's `AtmosphereModel`/`AtmosphereGrid`
docstrings (`stellar/atmospheres.py`) and `load_ck04_fits()` itself
place no artificial restriction beyond the physical grid boundaries --
`interpolate()` simply raises `ValueError` for a Teff/logg outside the
loaded files' range, or for a metallicity with no exact grid match.

**Benchmark-specific metallicity choices, and why they don't all use
literature [Fe/H] even though the full grid is now available:**

| System | Literature [Fe/H] | Nearest grid step | Used | Why |
|---|---|---|---|---|
| HD214222 | -0.05 | 0.0 (0.05 away) | 0.0 | nearest grid point |
| HD191854 | -0.05 | 0.0 (0.05 away) | 0.0 | nearest grid point |
| HD130669 | -0.03 | 0.0 (0.03 away) | 0.0 | nearest grid point |
| HD184467 | -0.25 | tie: 0.0 and -0.5 (both 0.25 away) | 0.0 | empirically tested both (see below); 0.0 gives the better reduced chi2 |
| Capella (photometry) | -0.04 | 0.0 (0.04 away) | 0.0 | nearest grid point |
| alpha Cen AB (photometry) | ~+0.23 | +0.2 (0.03 away) | +0.2 | nearest grid point |

For the four Abushattal-authored systems (HD214222/HD191854/HD130669/HD184467),
their literature [Fe/H] values are all close enough to solar (within
0.05 dex, except HD184467's exact tie) that 0.0 genuinely is, or ties
for, the nearest available grid point -- **not** a data-availability
compromise, even though it was originally documented as one back when
only 4 files were downloaded.

Since 0.2.0 the Capella and alpha Cen AB photometric benchmarks fit real
catalogue photometry (`scripts/build_real_sed_benchmarks.py`; Johnson UBV
from VizieR II/168, component magnitude differences, and the distance
from the orbital parallax of PySeshat's own real-data orbit fit), so both
use the nearest grid point. Before 0.2.0 their measurements were
synthetic, generated at metallicity 0.0, and alpha Cen AB was therefore
fitted at 0.0 (`scripts/ck04_metallicity_rerun_check.py` records that
comparison for the old synthetic data).

No previously-reported benchmark fit-quality number (reduced chi2,
worst-parameter relative error) changed as a result of this audit.

## An important epistemic caveat for the paper

The source paper's Table 4 magnitudes are that paper's **own synthetic
model output**, not independent catalog photometry — confirmed by direct
inspection (Table 4 is captioned as synthetic-spectrum magnitudes, not
observed catalog values). This reframes what these four benchmarks
actually validate: PySeshat's synthetic photometry pipeline reproducing
*another pipeline's* synthetic photometry, not model-vs-real-sky-measurement
agreement. The 0.30 mag per band is therefore a tolerance, not a
measurement error; best-fit residuals are 0.07-0.09 mag rms
(`scripts/sed_residuals_at_fit.py`).
Worth stating explicitly in the paper's methodology section, since it
changes the epistemic status of the comparison from "matches real
observations" to "matches another group's synthetic pipeline output to
within a documented systematic."
