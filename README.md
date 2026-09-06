<p align="center">
  <img src="https://raw.githubusercontent.com/rehamelkholy/PySeshat/master/docs/assets/logo.png" alt="PySeshat logo" width="300">
</p>

# PySeshat

PySeshat is a Python framework for binary and triple star orbit
determination and stellar modeling: fitting visual (astrometric)
and radial-velocity orbits, hierarchical triple systems,
unresolved-binary photometric SEDs, and stellar-evolution tracks,
with joint likelihood inference (least-squares + MCMC) tying
orbital, photometric, and evolutionary constraints together, plus
Gaia DR3 non-single-star (NSS) orbit parsing.

## Install

From PyPI:

```
pip install pyseshat
```

From source (editable install):

```
pip install -e .
```

Requires Python >=3.10. Dependencies (numpy, scipy, astropy,
matplotlib, pandas, emcee, corner, numba, astroquery, adjustText)
install automatically via either install method.

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
orbit+photometry fit, a Gaia DR3 NSS orbit parse, an
evolutionary-track consistency check, and templates for bringing
your own data.

## What's in this repo

- **`src/pyseshat/`** -- the package itself. `orbit/` and
  `triple/` fit single- and hierarchical-triple-star orbits (both
  the legacy and full 20-parameter ORBIT3 formalisms);
  `photometry/` and `stellar/` fit unresolved-binary SEDs against
  Castelli-Kurucz (CK04) synthetic spectra; `evolution/` fits
  stellar ages/masses to PARSEC or Girardi (2000) evolutionary
  tracks/isochrones; `inference/` ties all three together
  (least-squares optimizers, MCMC samplers, and a joint
  orbit+photometry inference path); `visualization/` plots orbits,
  residuals, RV curves, H-R diagrams, SED fits, and MCMC corner
  plots; `io/` parses Tokovinin ORBIT-format data files, Gaia DR3
  NSS archive rows, and writes fit reports.
- **`examples/`** -- runnable, narrated example scripts (see
  `examples/README.md`) and annotated input-file templates
  (`examples/templates/`) for bringing your own data.
- **`validation/`** -- deterministic end-to-end benchmark fits
  against real published systems and named joint case studies
  (`benchmark_systems/`), compared to their independently-published
  solutions (`expected_results/`). Run the full suite with
  `python validation/benchmark_runner.py`.
- **`docs/guide/`** -- user-facing guides: data formats, orbit
  fitting, triple systems, visualization conventions, evolutionary
  consistency checks, and the Al-Wardat SED-fitting method.
  `docs/architecture/` holds auto-generated package-structure
  snapshots (`architecture_snapshot.py`).

## Running the test suite

```
pytest tests/
```

## Running the validation benchmarks

Fits real and named-synthetic systems (a mix of visual+RV
binaries, hierarchical triples fit via the full ORBIT3 formalism,
Al-Wardat-method unresolved-binary photometric SED systems, and
joint orbit+photometry case studies) against their
published/literature solutions, and writes one result file per
system into `validation/results/`:

```
python validation/benchmark_runner.py
```

## License

MIT -- see [`LICENSE`](LICENSE).
