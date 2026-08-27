# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PySeshat is a Python framework for binary/triple star orbit determination and stellar modeling: fitting visual (astrometric) and radial-velocity orbits, hierarchical triple systems, photometric SEDs, and stellar-evolution tracks, with joint likelihood inference (least-squares + MCMC) tying orbital, photometric, and evolutionary constraints together.

## Project decisions & validation state

**Read `docs/project_decisions.md` before touching `triple/`, `orbit/`, or the validation benchmarks.** It captures decisions already made (and why) — the legacy/ORBIT3 split, the Cartesian-residual and angle-sign conventions, why the ORBIT3 wobble factor `f` is unbounded, the ζ Aqr astrometric-vs-true-`a` distinction, what's been explicitly rejected and shouldn't be re-litigated, current benchmark reference values, and the current bottleneck/next step (Al-Wardat/SED end-to-end validation, to be followed by validating all produced plots throughout the software). The project priority is scientific correctness and faithful ORBIT/ORBIT3 reproduction over architectural changes — this is a validation/debugging phase, not a redesign phase.

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
- `photometry/` turns SEDs into synthetic magnitudes/colors (`synthetic.py`), applies extinction (`extinction.py`, CCM89), and computes photometric residuals/likelihood against `PhotometricMeasurement` observations.
- `evolution/` loads PARSEC tracks/isochrones (`parsers/parsec.py`, `grids.py`) and fits stellar ages/masses to observables (`age.py`, `isochrone.py`, `model_fit.py`); `evolution/binary.py` and `photometry/evolution_adapter.py` connect a binary's fitted orbital/photometric solution to evolutionary-model consistency checks (see `analysis/binary_consistency.py`).

### Validation & reference data

- `validation/` runs deterministic end-to-end benchmark fits against known systems (`benchmark_systems/`) and compares to `expected_results/` via `metrics.py`; `benchmark_validation.ipynb` and `results/` hold the latest run's report/figures. Four systems are registered in `benchmark_registry.csv`: FIN379 and GL765 go through `run_benchmark()`'s plain binary path; ZetaAqr and HIP111805 (both `"fitting_path": "orbit3"` in their `expected_results/*.json`) are routed to `run_orbit3_benchmark()`, which fits the real 20-parameter ORBIT3 formalism instead. `run_benchmark()` dispatches on that field, and `benchmark_runner.py`'s `run_all_benchmarks()` (what `python validation/benchmark_runner.py` actually runs) iterates every `supported` registry row and writes one `results/<system>.json` per system. Both fitting paths normalize their `"statistics"` dict through the same `_normalize_statistics()` helper (`chi_square`/`reduced_chi_square`/`rms`/`n_data`/`n_params`) and return the same top-level result schema, so results are comparable across systems without special-casing; `HIP111805` currently has no `paper_table_row` (it's a regression/reproduction benchmark for the ORBIT3 code path, not part of the paper's Benchmark Results v1 table), so it has no dynamical-mass entry.
- Root-level `FIN379_fit.json` is a checked-in output artifact, not source.

### Test naming conventions

Most `tests/test_*.py` map 1:1 to a `src/pyseshat/...` module. Files named `tests/test_<system>_<topic>_diagnostic.py` (e.g. `test_hip111805_orbit3_jacobian_diagnostic.py`, `test_zetaaqr_residual_diagnostics.py`) are targeted investigations into a specific real system's fit behavior rather than unit tests of one function — read the module docstring/first test to see what specific symptom is being isolated before assuming it's redundant with the non-diagnostic test for the same area.
