# PySeshat examples

Runnable, narrated example scripts, each using real benchmark data
already checked into this repo (`validation/`, `data/`) so they run
out of the box with no external downloads or network access. Run any
of them from the repo root, e.g.:

```
python examples/01_visual_rv_binary_fit.py
```

Each script writes any plots/reports it produces into
`examples/output/` (gitignored).

| Script | What it shows |
|---|---|
| `01_visual_rv_binary_fit.py` | The "hello world" workflow: parse a `.inp` file (FIN379), fit with least squares, plot the orbit and residuals, write a text report. |
| `02_mcmc_posterior_exploration.py` | Bayesian posterior sampling with emcee on top of Example 1's fit: corner plot, convergence diagnostics (acceptance fraction, autocorrelation time), posterior credible intervals vs. the least-squares point estimate. |
| `03_hierarchical_triple_orbit3.py` | A hierarchical triple (Zeta Aquarii) fit through the real 20-parameter ORBIT3 formalism, and why a plain two-body fit -- or the older legacy ORBITT2 data format ZetaAqr's own file ships in -- is wrong for a triple; converts the legacy format and shows both sides of the split. |
| `04_alwardat_sed_fit.py` | An Al-Wardat unresolved-binary photometric SED fit (HD214222): recovering component Teff/radius/E(B-V) from combined-light photometry alone, with a Gaussian radius prior. |
| `05_joint_orbit_photometry_fit.py` | Simultaneous orbit + photometry fitting via `JointInferenceProblem`/`JointParameters`, composed from lower-level pieces (no convenience wrapper exists yet) against a synthetic combined dataset. |
| `06_your_own_data_template.py` | Loads two annotated templates (`templates/your_orbit.inp`, `templates/your_alwardat_system.json`) showing exactly which fields your own data needs, without reading the parser source. |
| `07_gaia_dr3_nss_parsing.py` | Parses a Gaia DR3 non-single-star (NSS) archive row (Thiele-Innes constants) into the same `OrbitalElements` used everywhere else in this package, using the real, published Gaia BH1 system as a worked example. |
| `08_evolutionary_consistency_check.py` | Independent per-component age fitting against the real Girardi (2000) evolutionary-track grid, and the age-agreement diagnostic that tells you whether two components are consistent with a common age -- using Capella's literature Teff/radius. |
| `hrdiagram_demo.py` | Binary H-R diagram plotting from synthetic evolutionary tracks and observables. |

## Which example to start from

- Have visual astrometry and/or radial velocities for a binary? Start with **01**, then **02** for posterior uncertainties.
- Have a hierarchical triple (inner pair + outer companion)? Start with **03**.
- Have combined-light (unresolved) photometry and want component Teff/radius? Start with **04**.
- Want to combine both at once? Read **05** -- it's a real, tested capability, but a lower-level one (you compose the fit yourself from `JointInferenceProblem`/`JointParameters`, the way `05` does).
- Have your own data file and don't know the input format? Start with **06**, then adapt whichever of 01-05 matches your data.
- Have a Gaia DR3 NSS archive row (or want to check whether your target has one)? Start with **07**.
- Have literature (Teff, radius) for both components and want an independent age-consistency check? Start with **08**.

## Where these fit in the package

The pieces used across these examples fit together as
`io`/`triple.triple_parser` -> `inference.*` ->
`orbit.orbit_model`/`stellar.*` ->
`visualization.*`/`io.report_writer`/`analysis.*`. See
`docs/guide/` for the per-topic guides behind the numbers these
scripts reproduce (data formats, orbit fitting, triple systems,
visualization conventions, evolutionary consistency checks, and the
Al-Wardat SED-fitting method).
