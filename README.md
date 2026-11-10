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
matplotlib, pandas, emcee, corner, adjustText)
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
  consistency checks, and the binary SED-fitting method.
  `docs/architecture/` holds auto-generated package-structure
  snapshots (`architecture_snapshot.py`).

## Running the test suite

```
pytest tests/
```

## Running the validation benchmarks

Fits real and named-synthetic systems (a mix of visual+RV
binaries, hierarchical triples fit via the full ORBIT3 formalism,
unresolved-binary photometric SED systems, and
joint orbit+photometry case studies) against their
published/literature solutions, and writes one result file per
system into `validation/results/`:

```
python validation/benchmark_runner.py
```

## Citation

If you use PySeshat in your research or publications, please cite it using the following BibTeX entry:

```
@misc{pyseshat2026,
  author = {{El-Kholy}, R.~I. and {Hayman}, Z.~M.},
  title  = {PySeshat: A Python Package for Binary- and Triple-Star Orbit Determination and Binary Spectral Energy Distribution Fitting},
  year   = {2026},
  doi    = {10.5281/zenodo.22543909},
  url    = {https://github.com/rehamelkholy/PySeshat},
}
```

## License

MIT -- see [`LICENSE`](LICENSE).

### Third-party data

The ORBIT example files (FIN 379, GL 765, Zeta Aqr) under
`validation/benchmark_systems/` come from A. Tokovinin's ORBIT release
(GPL-2.0-only, doi:10.5281/zenodo.61119); the HIP 111805 file comes from his
ORBIT3 release (CC BY 4.0, doi:10.5281/zenodo.321854). See
[`third_party/ORBIT/NOTICE.md`](third_party/ORBIT/NOTICE.md).

### Data sources and acknowledgements

* Real-data benchmarks: Torres, Claret & Young (2009, ApJ 700, 1349; VizieR J/ApJ/700/1349) for Capella,
  with the astrometry compiled there from Hummel et al. (1994), Baldwin et al. (1996), Kraus et al. (2005)
  and speckle/imaging sources; Akeson et al. (2021, AJ 162, 14; VizieR J/AJ/162/14) for alpha Cen AB;
  Tokovinin (2016, ApJ 831, 151) and Tokovinin & Latham (2017, ApJ 838, 54) for the zeta Aqr and HIP 111805
  published solutions; Balega et al. (2007, A&A 464, 635) for GL 765. The VizieR tables are not
  redistributed; see the README.md in `validation/benchmark_systems/Capella_Torres2009/` and
  `.../AlphaCenAB_Akeson2021/`.
* "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France
  (DOI: 10.26093/cds/vizier). The original description of the VizieR service was published in
  Ochsenbein et al. (2000, A&AS 143, 23)."
* Evolutionary tracks and isochrones: Girardi et al. (2000, A&AS 141, 371; VizieR J/A+AS/141/371),
  bundled unchanged in `src/pyseshat/evolution/data/girardi2000/` (see the README there).
* SED benchmark systems: Abushattal et al. (2025, arXiv:2512.23652); the SED-fitting approach follows
  Al-Wardat (2002, 2012) and Abushattal et al. (2025).
