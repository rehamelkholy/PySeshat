# Changelog

## 0.2.0

### Added
- Dataset-level weighting for combined visual + radial-velocity fits
  (`fit_visual_orbit(..., weighting="auto" | "fixed" | "dataset")`). Each
  dataset's errors are rescaled by `sqrt(chi2_k / nu_k)` and iterated to
  convergence; errors are only inflated unless `allow_deflation=True`. The
  scales, per-dataset diagnostics and iteration count are stored in the fit
  metadata (`error_scales`, `dataset_diagnostics`, `weighting`,
  `reweight_iterations`). Optional error floors: `visual_error_floor`,
  `rv_error_floor`.
- Real-data benchmarks for Capella (Torres et al. 2009) and alpha Cen AB
  (Akeson et al. 2021), a GL 765 benchmark (Balega et al. 2007), a
  PySeshat-versus-published comparison, a weighting-sensitivity study and
  orbit-coverage metrics (`scripts/`, `validation/results/`). The VizieR tables
  are not redistributed; see the README in each benchmark folder.
- Third-party licence notice for the bundled ORBIT / ORBIT3 example files
  (`third_party/ORBIT/NOTICE.md`).
- Component magnitude differences (e.g. from speckle interferometry) as an SED-fitting
  constraint: `photometry.observations.MagnitudeDifference`,
  `photometry.model.synthetic_magnitude_differences`, `fit_photometry(...,
  magnitude_differences=...)`, `PhotometricContext.magnitude_differences`, and a
  `magnitude_differences` entry in the benchmark files. Tests:
  `tests/test_magnitude_difference.py`.
- `photometry.calibration`: standard-system zero points following Maíz Apellániz
  (2006, 2007): Vega's standard magnitudes (`VEGA_STANDARD_MAGNITUDES`, Table 4
  of Maíz Apellániz 2007, ASP Conf. Ser., 364, 227), the CALSPEC Vega spectrum
  `alpha_lyr_stis_003` (Bohlin 2007; shipped as package data), and
  `vega_zero_points(passbands)`, which integrates Vega through the same passbands
  as the target. Tests: `tests/test_photometric_calibration.py`, and solar checks
  (Willmer 2018 magnitudes, Meléndez et al. 2010 Strömgren indices) with a CK04
  model and with the observed CALSPEC solar spectrum (`tests/data/`) in
  `tests/test_synthetic_photometry_sun_calibration.py`.
- Photon-counting passbands: `PhotometricPassband.photon_counting`,
  `load_passband(..., photon_counting=None)` (read from a `# detector: photon|energy`
  header line; default energy-integrating), and lambda weighting of photon-counting
  curves in `synthetic_band_flux` (Maíz Apellániz 2006, AJ, 131, 1184, Eq. A1).
  Provenance table: `validation/photometry/passbands/README.md`. Tests:
  `tests/test_photon_counting_passband.py`.
- `fit_visual_orbit_mcmc(..., parameterization="sqrt_e")` (also in
  `fit_visual_orbit_mcmc_from_problem`): samples the mean longitude and
  sqrt(e) sin/cos omega in place of (T0, e, omega), with the Jacobian term, so
  the posterior is unchanged and near-circular orbits mix far better. The
  default (`"standard"`) is unchanged; results are returned in the standard
  elements. `pyseshat.inference.reparameterization`; tests:
  `tests/test_sqrt_e_parameterization.py`.

### Fixed
- Binary MCMC: T0 is now folded to within P/2 of the least-squares T0 (using
  each sample's P). The model depends on T0 only modulo P; on near-circular
  orbits the walkers drifted across many periods, which inflated the T0
  spread and the autocorrelation times. Tests: `tests/test_mcmc_t0_folding.py`.
- Visual-only fits (no RV data) with K1, K2 or V0 flagged as fitted (the
  `OrbitalElements` default) counted these parameters in the degrees of
  freedom, reported zero uncertainties for them, and let them drift without
  bound in MCMC. They are now held fixed when there are no RV data. The
  ORBIT-format benchmark inputs were not affected (their flags are read from
  the files). Tests: `tests/test_visual_only_rv_flags.py`.
- `fit_orbit3()` did not converge. Its default (SciPy LM with the translated
  ORBIT3 forward-difference Jacobian, whose increments are 1 per cent of each
  element value and 1 degree for angles, as in `orbit3.pro`) stopped after a
  few iterations: for HIP 111805 it did not move from the published start
  (chi2 = 502.10) and did not recover from displaced starts, whereas the chi2
  minimum is 385.44; for zeta Aqr it stopped at 694.51 instead of 686.84. The
  default is now `method="trf"` (bounded) with SciPy's numerical Jacobian
  (`jac="2-point"`). The ORBIT3 Jacobian remains available as `jac="orbit3"`
  (use `method="lm", jac="orbit3"` to reproduce the previous behaviour).
  Diagnostics: `scripts/hip111805_recovery.py`,
  `scripts/orbit3_convergence_check.py`, `scripts/hip111805_paper_config.py`,
  `scripts/orbit3_paper_comparison.py`; tests: `tests/test_orbit3_convergence.py`.
  All ORBIT3 benchmark numbers change.
- Synthetic magnitudes in the benchmarks and examples were not on the standard
  photometric systems. SVO "Vega system" zero points were used directly: Vega = 0
  in every band (Strömgren u and v 1.43 and 0.18 mag too bright), and Vega fluxes
  computed through SVO's passband curves, which for Johnson U and B differ from the
  bundled passband files (offsets of about +0.07 and -0.07 mag). The u error had
  been taken for a pipeline difference and Strömgren u was excluded from the SED
  benchmarks. Zero points now come from `photometry.calibration.vega_zero_points`;
  Strömgren u is fitted again. The four Abushattal et al. (2025) SED benchmark
  results change; the synthetic Capella and alpha Cen photometry was shifted by
  the zero-point change in each band (same noise draws), so their fits are
  essentially unchanged. `examples/04_binary_sed_fit.py` now also follows the
  benchmark configuration (E(B-V) fixed, magnitude-difference constraint), which
  it did not after the SED benchmark change below.
- The bundled passband curves were not the curves the zero points are defined for,
  and were all integrated as energy-integrating. The Johnson U, B, V, Strömgren u, v,
  b, y and Tycho VT files are now the photon-counting curves of Maíz Apellániz (2006,
  Tables 1-3; Tycho BT was already the Table 1 curve), integrated with lambda
  weighting. The observed solar spectrum now gives U-B = 0.15 (Willmer 2018: 0.17;
  previously 0.06). The CK04 solar-model Strömgren test uses 0.08 mag for m1 and c1
  (model minus observed Sun: +0.065 and -0.036 mag).
- `io/fit_result_io.py` uses timezone-aware UTC timestamps.

### Changed
- The R band is "Cousins R" (was "Johnson R"), with the Bessell (1983, PASP, 95, 480,
  Table AII) curve, for which the Cousins R zero point of Maíz Apellániz (2007) is
  defined; the previous `johnson_r.dat` (origin unknown) is removed. Benchmark files,
  scripts, Example 04 and tests use the new name.
- `scripts/stromgren_standard_zp_test.py` (a pre-0.2.0 comparison) needs the old
  passband files and now exits with a message; its recorded results are kept.
- SED benchmarks (HD 130669, HD 184467, HD 191854, HD 214222): one common CK04 grid
  (3500-10000 K) instead of 3-4 nodes bracketing the published temperatures, E(B-V) held at 0
  (all within 75 pc) and the Johnson-V component magnitude difference of Abushattal et al.
  (2025) Table 4 as a constraint. With the narrow grids the temperature agreement was bounded
  by construction; with E(B-V) free the component temperatures are unconstrained
  (`scripts/sed_wide_grid.py`, `validation/results/sed_wide_grid*.json`).
- Binary SED fitting is no longer named after a specific method:
  `docs/guide/alwardat_sed.md` -> `docs/guide/binary_sed.md`,
  `examples/04_alwardat_sed_fit.py` -> `examples/04_binary_sed_fit.py`,
  `examples/templates/your_alwardat_system.json` -> `your_sed_system.json`;
  the benchmark-runner path name is now `binary_sed` (the old `alwardat_sed`
  value is still accepted as an alias).
- Benchmark runner reports ORBIT3 parameter uncertainties and has wider
  summary-table columns.
- Package title: "PySeshat: A Python Package for Binary- and Triple-Star Orbit
  Determination and Binary Spectral Energy Distribution Fitting".
- The Girardi et al. (2000) grid (VizieR J/A+AS/141/371) is now bundled
  as package data (`pyseshat/evolution/data/girardi2000/`, moved from the
  repository's `data/evolution/girardi2000/`), so
  `load_default_evolution_model()` also works after `pip install`; before,
  it looked for the files outside the installed package.
- Comments, docstrings and documentation were tidied for public release.

### Removed
- Unused dependencies `numba` and `astroquery`.
- `requirements.txt` now lists only the direct dependencies, pinned to the
  versions used for the paper results.

## 0.1.0
- Initial public release.
