# PySeshat

PySeshat is a Python framework for binary and triple star orbit
determination and stellar modeling: fitting visual (astrometric)
and radial-velocity orbits, hierarchical triple systems,
unresolved-binary photometric SEDs, and stellar-evolution tracks,
with joint likelihood inference (least-squares + MCMC) tying
orbital, photometric, and evolutionary constraints together.

## Install

```
pip install -e .
```

Requires Python >=3.10. Dependencies (numpy, scipy, astropy,
matplotlib, pandas, emcee, corner, numba, astroquery) install
automatically via `pip install -e .`.

## Quickstart

Fit a real visual+RV binary orbit end to end -- parse a data file,
fit with least squares, plot the orbit, write a report -- using a
benchmark system already checked into this repo:

```
python examples/01_visual_rv_binary_fit.py
```

See `examples/README.md` for the full set of runnable examples,
including MCMC posterior sampling, a hierarchical-triple ORBIT3
fit, an unresolved-binary photometric SED fit, a joint
orbit+photometry fit, and templates for bringing your own data.

## What's in this repo

- **`src/pyseshat/`** -- the package itself. `orbit/` and
  `triple/` fit single- and hierarchical-triple-star orbits;
  `photometry/` and `stellar/` fit unresolved-binary SEDs against
  Castelli-Kurucz (CK04) synthetic spectra; `evolution/` fits
  stellar ages/masses to PARSEC evolutionary tracks; `inference/`
  ties all three together (least-squares optimizers, MCMC
  samplers, and a joint orbit+photometry inference path);
  `visualization/` plots orbits, residuals, RV curves, and H-R
  diagrams; `io/` parses Tokovinin ORBIT-format data files and
  writes fit reports.
- **`examples/`** -- runnable, narrated example scripts (see
  `examples/README.md`) and annotated input-file templates
  (`examples/templates/`) for bringing your own data.
- **`validation/`** -- deterministic end-to-end benchmark fits
  against real published systems (`benchmark_systems/`), compared
  to their independently-published solutions
  (`expected_results/`). Run the full suite with
  `python validation/benchmark_runner.py`.
- **`docs/`** -- `project_decisions.md` is the project's running
  decision log: what's been decided and why, current validation
  status, and known limitations -- read it before making
  architectural changes to `triple/`, `orbit/`,
  `photometric_optimizer.py`/`photometric_mcmc.py`, or the
  validation benchmarks. `development_plan_phase2.md` tracks the
  remaining phased work. `docs/architecture/` holds
  auto-generated package-structure snapshots
  (`architecture_snapshot.py`).
- **`CLAUDE.md`** -- a more detailed internal architecture/agent
  guidance document (package layering, module responsibilities,
  naming conventions) -- useful background even if you're not
  using it with an AI coding assistant.

## Running the test suite

```
pytest tests/
```

## Running the validation benchmarks

Fits eight real systems (a mix of visual+RV binaries,
hierarchical triples fit via the full ORBIT3 formalism, and
Al-Wardat-method unresolved-binary photometric SED systems)
against their published/literature solutions, and writes one
result file per system into `validation/results/`:

```
python validation/benchmark_runner.py
```

See `docs/project_decisions.md` for the current results and their
interpretation, including known, documented limitations (e.g. a
near-twin-component-temperature degeneracy affecting some SED
systems).

## Status

This project is in a validation/debugging phase: the priority is
scientific correctness and faithful reproduction of the underlying
orbit-fitting (ORBIT/ORBIT3) and Al-Wardat stellar-modeling
methods, not architectural redesign. See
`docs/project_decisions.md`'s "Current bottleneck & next step"
section for exactly what's been validated and what remains open.
