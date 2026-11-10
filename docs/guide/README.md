# PySeshat user guide

This is prose documentation for people using PySeshat, organized by
topic. Each guide below explains what a piece of the package does, why
it's built the way it is, and points at the real validation data and
examples that back up its claims.

Pages, roughly in the order a new user would want them:

1. [`data_formats.md`](data_formats.md) — the `.inp`/`.in2`/`.in3` file
   formats, the binary SED JSON format, and Gaia DR3 NSS
   `nss_two_body_orbit` rows; what each parser returns.
2. [`orbit_fitting.md`](orbit_fitting.md) — fitting a visual+RV binary
   orbit, least-squares and MCMC.
3. [`triple_systems.md`](triple_systems.md) — hierarchical triples, and
   the legacy ORBITT2 vs. modern ORBIT3 formalism split.
4. [`binary_sed.md`](binary_sed.md) — unresolved-binary SED
   fitting, and its current, real limitations.
5. [`evolutionary_checks.md`](evolutionary_checks.md) — evolutionary-track
   consistency checks.
6. [`visualization.md`](visualization.md) — the plotting functions, what
   they draw, and the literature conventions they follow.

Each page assumes you've run `pip install -e .` from the repo root and
points at real files already checked into `validation/` and `examples/`
so every code snippet below is runnable, not illustrative pseudocode.
See `examples/README.md` for the full set of narrated, runnable example
scripts these pages draw from.
