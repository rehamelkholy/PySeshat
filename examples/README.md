# PySeshat examples

Runnable, narrated example scripts, each using real benchmark data
already checked into this repo (`validation/`) so they run out of
the box with no external downloads or network access. Run any of
them from the repo root, e.g.:

```
python examples/01_visual_rv_binary_fit.py
```

Each script writes any plots/reports it produces into
`examples/output/` (gitignored).

| Script | What it shows |
|---|---|
| `01_visual_rv_binary_fit.py` | The "hello world" workflow: parse a `.inp` file (FIN379), fit with least squares, plot the orbit and residuals, write a text report. |
| `02_mcmc_posterior_exploration.py` | Bayesian posterior sampling with emcee on top of Example 1's fit: corner plot, convergence diagnostics (acceptance fraction, autocorrelation time), posterior credible intervals vs. the least-squares point estimate. |
| `03_hierarchical_triple_orbit3.py` | A hierarchical triple (Zeta Aquarii) fit through the real 20-parameter ORBIT3 formalism, and why a plain two-body fit is wrong for a triple. |
| `04_alwardat_sed_fit.py` | An Al-Wardat unresolved-binary photometric SED fit (HD214222): recovering component Teff/radius/E(B-V) from combined-light photometry alone, with a Gaussian radius prior. |
| `05_joint_orbit_photometry_fit.py` | Simultaneous orbit + photometry fitting via `JointInferenceProblem`/`JointParameters`, composed from lower-level pieces (no convenience wrapper exists yet) against a synthetic combined dataset. |
| `06_your_own_data_template.py` | Loads two annotated templates (`templates/your_orbit.inp`, `templates/your_alwardat_system.json`) showing exactly which fields your own data needs, without reading the parser source. |
| `hrdiagram_demo.py` | Binary H-R diagram plotting from synthetic evolutionary tracks and observables. |

## Which example to start from

- Have visual astrometry and/or radial velocities for a binary? Start with **01**, then **02** for posterior uncertainties.
- Have a hierarchical triple (inner pair + outer companion)? Start with **03**.
- Have combined-light (unresolved) photometry and want component Teff/radius? Start with **04**.
- Want to combine both at once? Read **05** -- it's a real, tested capability, but a lower-level one (you compose the fit yourself from `JointInferenceProblem`/`JointParameters`, the way `05` does).
- Have your own data file and don't know the input format? Start with **06**, then adapt whichever of 01-05 matches your data.

## Where these fit in the package

See `CLAUDE.md`'s "Layering" section for how the pieces used across
these examples fit together (`io`/`triple.triple_parser` ->
`inference.*` -> `orbit.orbit_model`/`stellar.*` ->
`visualization.*`/`io.report_writer`/`analysis.*`), and
`docs/project_decisions.md` for the validation history and known
limitations (e.g. the near-twin-component-temperature degeneracy
Example 4 references) behind the numbers these scripts reproduce.
