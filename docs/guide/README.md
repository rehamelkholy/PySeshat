# PySeshat user guide

This is prose documentation for people using PySeshat, not maintaining
it — for the internal engineering decision log (why things are built the
way they are, what's been debugged and how), see `docs/project_decisions.md`.
For a reference distilling everything validated so far, organized for
writing the paper this software supports, see `docs/paper_reference_notes.md`.

Pages, roughly in the order a new user would want them:

1. [`data_formats.md`](data_formats.md) — the `.inp`/`.in2`/`.in3` file
   formats and the Al-Wardat SED JSON format; what each parser returns.
2. [`orbit_fitting.md`](orbit_fitting.md) — fitting a visual+RV binary
   orbit, least-squares and MCMC.
3. [`triple_systems.md`](triple_systems.md) — hierarchical triples, and
   the legacy ORBITT2 vs. modern ORBIT3 formalism split.
4. [`alwardat_sed.md`](alwardat_sed.md) — Al-Wardat unresolved-binary SED
   fitting, and its current, real limitations.
5. [`evolutionary_checks.md`](evolutionary_checks.md) — evolutionary-track
   consistency checks.
6. [`visualization.md`](visualization.md) — the plotting functions, what
   they draw, and what conventions they do (and don't yet) guarantee.

See also [`../paper_reproduction_notes.md`](../paper_reproduction_notes.md),
which maps the target publication's planned sections and figures to the
exact script or benchmark that reproduces each one.

Each page assumes you've run `pip install -e .` from the repo root and
points at real files already checked into `validation/` and `examples/`
so every code snippet below is runnable, not illustrative pseudocode.
See `examples/README.md` for the full set of narrated, runnable example
scripts these pages draw from.
