# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PySeshat is a Python framework for binary/triple star orbit determination and stellar modeling: fitting visual (astrometric) and radial-velocity orbits, hierarchical triple systems, photometric SEDs, and stellar-evolution tracks, with joint likelihood inference (least-squares + MCMC) tying orbital, photometric, and evolutionary constraints together.

## Project decisions & validation state

**Read `docs/project_decisions.md` before touching `triple/`, `orbit/`, `photometric_optimizer.py`/`photometric_mcmc.py`, or the validation benchmarks.** It captures decisions already made (and why) — the legacy/ORBIT3 split, the Cartesian-residual and angle-sign conventions, why the ORBIT3 wobble factor `f` is unbounded, the ζ Aqr astrometric-vs-true-`a` distinction, what's been explicitly rejected and shouldn't be re-litigated, current benchmark reference values, and the current bottleneck/next step. The project priority is scientific correctness and faithful ORBIT/ORBIT3 reproduction over architectural changes — this is a validation/debugging phase, not a redesign phase. **Current status (see "`PhotometricParameters` fit-flags/prior mechanism, Al-Wardat measurement-uncertainty reassessment, and Stromgren u disposition (2026-08-27)" in that doc, the latest Al-Wardat SED section):** Phase 1 of the phase development plan (`docs/development_plan_phase2.md`) is fully resolved. `PhotometricParameters` now has a real fit-flags/prior mechanism mirroring `OrbitalElements.fit_flags` (see "Photometry / stellar / evolution" below), and all four registered Al-Wardat SED systems (HD214222, HD191854, HD130669, HD184467) reach a genuinely good literature match: radius errors ≤1.10%, Teff errors 0.64%–6.37%, reduced χ² 0.107–0.152. Getting there took two changes beyond the fit-flags mechanism itself: raising the assumed per-measurement photometric uncertainty from 0.02 to 0.30 mag (the source paper's Table 4 magnitudes turned out to be that paper's own *synthetic* output, not real catalog photometry, so 0.02 mag understated the true pipeline-to-pipeline systematic by over an order of magnitude), and accepting a documented, genuine near-twin-temperature degeneracy (three of the four systems have primary/secondary literature Teff within ~75–155K of each other; one component's Teff pins at its atmosphere grid's edge in each case, confirmed via grid-extension tests to be a data/model limitation, not a coverage gap or a bug). Stromgren u remains excluded — the originally-suspected Crawford & Barnes (1970) zero-point explanation did not survive literature scrutiny and was retracted; no correction of the required ~1.7 mag magnitude was found, and none was fabricated. Proceeding next to Phases 2–4: example workflows, full documentation, and plotting-vs-literature-style validation.

## Commands

Editable install (package lives under `src/`, already installed into `.venv`):

```
pip install -e .
```

Run the full test suite (pytest, no config file — invoke from repo root):

```
pytest tests/
```

Run a single test file or test:

```
pytest tests/test_orbit3_fit.py
pytest tests/test_orbit3_fit.py::test_some_case -v
```

Regenerate the architecture snapshot docs (`docs/architecture/{package_tree.txt,public_api.md,statistics.md,class_diagram.puml}`) after structural changes — it parses the AST only, no project imports:

```
python architecture_snapshot.py
```

Run the cross-system validation benchmarks (fits real systems in `validation/benchmark_systems/` and compares to `validation/expected_results/`; writes `validation/results/`):

```
python validation/benchmark_runner.py
```

## Architecture

### Layering

Data flows through the package roughly as:

```
io.orbit_parser / triple.triple_parser   (parse .inp/.in2 files → OrbitalElements/TripleOrbitalElements + observations)
        ↓
observations.*  (VisualObservationSet, RVObservationSet, SystemObservations)
        ↓
inference.problem / inference.photometric_problem / inference.joint_problem
        (frozen dataclasses bundling elements + a LikelihoodContext; expose .t/.X_obs/.rv_data/... and is_visual_only/is_combined_fit helpers)
        ↓
inference.solver → inference.solver_engine → inference.optimizer / inference.optimizer_engine
        (scipy least_squares under the hood; "solve()" is a thin pass-through to the concrete fit_visual_orbit-style function)
        ↓
inference.mcmc_engine (emcee, seeded from the least-squares solution + covariance)
        ↓
inference.fit_result / base_fit_result / mcmc_result / triple.triple_fit_result
        ↓
analysis.* (chi-square/statistics/masses/luminosity/consistency), io.fit_result_io, io.report_writer, visualization.*
```

Each inference sub-domain (orbit, photometric, joint) follows the same Problem → Likelihood/Posterior → Solver/Optimizer/MCMC → FitResult pattern; `inference/terms/` holds the raw residual functions (`visual.py`, `radial_velocity.py`, `joint.py`) that the likelihood modules wrap.

### Orbit vs. triple vs. joint

- `orbit/` — single-orbit (binary) Kepler solving and models: `orbit_model.py` (`solve_kepler`, `thiele_innes`, `compute_visual_orbit`), `rv_model.py`.
- `triple/` — hierarchical triple systems, and it deliberately carries **two parallel formalisms** that must both keep working:
  - **Legacy** (`compute_triple_orbit`, `triple_fit.py`'s `legacy_triple_*` functions, `triple_orbit.py`'s "Legacy hierarchical triple orbit" section) — reproduces the existing ORBITT2 implementation/convention.
  - **ORBIT3** (`orbit3_fit.py`, `orbit3_bounds.py`, `triple_fit.py`'s `orbit3_*` functions, `triple_orbit.py`'s `compute_orbit3_*`) — the Tokovinin ORBIT3-style formalism, fit via `fit_orbit3()`. `triple_fit.py` is the canonical, live implementation of the ORBIT3 parameter-vector conversion functions (`orbit3_to_array`, `array_to_orbit3`, `orbit3_fitted_parameters`, `update_orbit3_fitted_parameters`) — it's what `orbit3_fit.py` actually imports. `triple_parameters.py` exposes the same functionality under slightly different names (`triple_to_orbit3_array`, `orbit3_array_to_triple`, ...) purely as a **backward-compatible wrapper module**: every function in it delegates to `triple_fit.py` rather than reimplementing the logic — don't add new independent logic there. The single source of truth for the 20-name ORBIT3 parameter ordering is `triple_elements.ORBIT3_PARAMETER_ORDER`; `triple_fit.py` and `orbit3_bounds.py` both import it rather than redefining it. `orbit3_fit.py` also has the Bayesian entry point, `fit_orbit3_mcmc()` — the triple/ORBIT3 counterpart of `inference.optimizer.fit_visual_orbit_mcmc()`, built on `triple.orbit3_priors.log_prior_orbit3` and `triple.orbit3_likelihood.{log_likelihood_orbit3,log_posterior_orbit3}`, reusing the fully generic `inference.mcmc_engine.run_emcee`/`inference.mcmc_result.MCMCResult` (neither needed any changes — they always operated on a flat parameter vector and a caller-supplied log-probability function).
  Functions/tests are named accordingly (`orbit3_...` vs bare/`legacy_...`) — check which formalism a call site is using before changing shared helpers like `triple_fit.py` or `triple_residuals.py`, since both paths route through them.
- `inference/joint_*` — combines an orbital `InferenceProblem` with a `PhotometricInferenceProblem` into one `JointInferenceProblem` for simultaneous orbit+photometry fitting.

### Photometry / stellar / evolution

- `stellar/atmospheres.py` loads Castelli-Kurucz (CK04) synthetic spectral grids; `stellar/sed_model.py` builds composite binary SEDs from them.
- `photometry/` turns SEDs into synthetic magnitudes/colors (`synthetic.py`), applies extinction (`extinction.py`, CCM89), and computes photometric residuals/likelihood against `PhotometricMeasurement` observations. `inference/photometric_optimizer.py`'s `fit_photometry`/`fit_photometry_from_problem` (least-squares) and `inference/photometric_mcmc.py`'s `sample_photometry`/`sample_photometry_from_problem` (MCMC) are the Al-Wardat SED-fitting entry points — each rebuilds a trial `PhotometricInferenceProblem` at the optimizer's/sampler's current parameter vector via a small `_photometric_..._for_optimizer`/`_for_sampler` wrapper (required because scipy/emcee call the residual/log-probability function as `fn(x, *args)`, not `fn(problem)`; see `docs/project_decisions.md` for the bug this fixed). `PhotometricParameters` gained a fit-flags/prior mechanism (2026-08-27) mirroring `OrbitalElements.fit_flags`: `fit_flags` (a parameter can be held genuinely fixed) plus `priors` (a parameter can be freed with a soft Gaussian prior, e.g. radius tied to a literature value) — `fitted_parameters()`/`fitted_parameter_names()` (now an instance method) return only the currently-fitted subset, and `update_parameters()` returns a new instance since `PhotometricParameters` is still frozen. `photometric_residuals_with_priors_from_problem()`/`log_photometric_prior()` fold the Gaussian prior into the least-squares/MCMC objective respectively, while the plain measurement-only residual/statistics functions stay prior-free for reported fit-quality numbers. `validation/benchmark_runner.py::run_alwardat_benchmark()` now uses this mechanism directly (a per-system `fit_flags` dict from `expected["free_parameters"]`, plus a `priors` dict pinning each freed radius to its literature value with a 10% relative sigma) instead of the earlier near-zero-width-bounds pinning workaround. See `docs/project_decisions.md`, "`PhotometricParameters` fit-flags/prior mechanism..." for the full design and validation.
- `evolution/` loads PARSEC tracks/isochrones (`parsers/parsec.py`, `grids.py`) and fits stellar ages/masses to observables (`age.py`, `isochrone.py`, `model_fit.py`); `evolution/binary.py` and `photometry/evolution_adapter.py` connect a binary's fitted orbital/photometric solution to evolutionary-model consistency checks (see `analysis/binary_consistency.py`).

### Validation & reference data

- `validation/` runs deterministic end-to-end benchmark fits against known systems (`benchmark_systems/`) and compares to `expected_results/` via `metrics.py`; `benchmark_validation.ipynb` and `results/` hold the latest run's report/figures. All eight systems registered in `benchmark_registry.csv` are `supported: yes`: FIN379 and GL765 go through `run_benchmark()`'s plain binary path; ZetaAqr and HIP111805 (`"fitting_path": "orbit3"`) are routed to `run_orbit3_benchmark()`, fitting the real 20-parameter ORBIT3 formalism; HD214222, HD191854, HD130669, HD184467 (`"fitting_path": "alwardat_sed"`, an Al-Wardat-authored 2025 AJ paper — see `docs/project_decisions.md`) are routed to `run_alwardat_benchmark()`, which fits the unresolved-binary photometric SED model (`inference.photometric_optimizer.fit_photometry`) instead — no orbit data file at all, since this path validates stellar modelling given a known orbital solution, not orbit-fitting. Each `alwardat_sed` system's `atmosphere_files` list must bracket *both* components' literature Teff, not just one — HD130669/HD184467 originally listed only the two grid files bracketing their secondary's Teff and failed with "Initial guess is outside of provided bounds" until a third file was added for the primary (see `docs/project_decisions.md`). Each `alwardat_sed` system's `measurements` now carries 9 real Table-4 bands (Johnson U/B/V/R, Stromgren v/b/y, Tycho BT/VT; Stromgren u is sourced but excluded — see `excluded_measurements` and `docs/project_decisions.md` for the zero-point anomaly that caused this, reconfirmed 2026-08-27 with the originally-suspected Crawford & Barnes 1970 explanation retracted for lack of literature support) fit with all 5 photometric parameters freed (`free_parameters`: Teff1, Teff2, R1, R2, E(B-V)) — radius is freed WITH a Gaussian prior tying it to the system's literature value (`PhotometricParameters.priors`, 10% assumed relative sigma) rather than pinned via near-zero-width bounds, per the fit-flags/prior mechanism described above. Each system's `measurements[].uncertainty` is 0.30 mag (raised from an earlier, since-retracted 0.02 mag assumption — see `docs/project_decisions.md`'s measurement-uncertainty reassessment). `run_benchmark()` dispatches on `expected["fitting_path"]`, and `benchmark_runner.py`'s `run_all_benchmarks()` (what `python validation/benchmark_runner.py` actually runs) iterates every `supported` registry row and writes one `results/<system>.json` per system. All three fitting paths normalize their `"statistics"` dict through the same `_normalize_statistics()` helper (`chi_square`/`reduced_chi_square`/`rms`/`n_data`/`n_params`) and return a comparable top-level result schema; `HIP111805` and the four `alwardat_sed` systems have no `paper_table_row` (regression/reproduction benchmarks, not part of the paper's Benchmark Results v1 table), so they have no dynamical-mass entry.
- The four `tests/test_*.py` demo scripts that exercise FIN379 end to end (`test_ephemeris_result.py`, `test_fit_result_io.py`, `test_fit_result_roundtrip.py`, `test_report_writer.py`) have no `def test_...` functions — pytest executes their module-level code on collection but counts 0 tests from them — and write their FIN379 output into `tests/artifacts/` (gitignored), anchored via `Path(__file__).resolve().parent`, not the repo root.

### Test naming conventions

Most `tests/test_*.py` map 1:1 to a `src/pyseshat/...` module. Files named `tests/test_<system>_<topic>_diagnostic.py` (e.g. `test_hip111805_orbit3_jacobian_diagnostic.py`, `test_zetaaqr_residual_diagnostics.py`) are targeted investigations into a specific real system's fit behavior rather than unit tests of one function — read the module docstring/first test to see what specific symptom is being isolated before assuming it's redundant with the non-diagnostic test for the same area.
