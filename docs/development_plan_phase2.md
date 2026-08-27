# PySeshat development plan — remaining phases (2026-08-27)

This picks up where `Development_Plan.pdf` / `Paper_Roadmap.pdf` left off,
now that the core orbit-fitting (legacy + ORBIT3), triple-star, and
Al-Wardat SED-fitting pipelines have all been run end-to-end and validated
against real benchmark systems at least once. It covers everything
requested this pass: completing the CK04 grid, closing out the remaining
Al-Wardat scientific gaps, example workflows, full documentation, and
plotting validation against literature convention — sequenced so each
phase's output is usable before the next depends on it.

Status references throughout point to `docs/project_decisions.md`, which
remains the authoritative log of what's been decided and why.

## Phase 0 — Unblock the data (in progress, mostly on the user's side now)

**What's blocking everything downstream:** only 4 of 608 CK04 atmosphere
FITS files are present locally, so 3 of the 4 Al-Wardat benchmarks can't
run, and any new target system outside the narrow 5750–6250K/[M/H]=0.0
window hits the same wall.

- [x] Full CK04 grid manifest + batch downloader delivered
      (`scripts/download_ck04_grid.py`, `scripts/ck04_manifest.csv`) —
      **action needed from the user**: run it on a machine with normal
      internet access (STScI's CDBS archive is blocked from both this
      sandbox and the device-bridge shell).
- [ ] Once downloaded: flip `HD191854`/`HD130669`/`HD184467` from
      `supported: no` to `yes` in `benchmark_registry.csv`, re-run
      `python validation/benchmark_runner.py`, confirm all 8 registered
      systems fit successfully.
- [ ] Re-check whether `secondary_temperature` for HD214222 converges
      inside the grid (rather than pinned at the current 5750K edge) once
      the grid extends below that.

This phase is nearly free (one script run) but gates real validation
numbers for 3 of 8 benchmark systems, so it should happen first.

## Phase 1 — Close the remaining Al-Wardat scientific gaps (RESOLVED 2026-08-27)

Two concrete, well-scoped items came out of an earlier pass's root-cause
investigation (`docs/project_decisions.md`, "CK04 grid completion,
root-cause of the HD214222 residual, and expanded photometry"); both are
now resolved — full writeup in `docs/project_decisions.md`,
"`PhotometricParameters` fit-flags/prior mechanism, Al-Wardat
measurement-uncertainty reassessment, and Stromgren u disposition
(2026-08-27)".

1. **A calibrated Stromgren u zero point — investigated, NOT restored.**
   The original framing ("a known, documented quirk of the classical
   Crawford & Barnes (1970) uvbybeta calibration, not a bug") did not
   survive scrutiny: no literature source (SVO Filter Profile Service,
   Gray et al. 2011) supports a correction anywhere near the required
   ~1.7 mag, and direct inspection of the source paper confirmed its
   Table 4 magnitudes are that paper's own *synthetic* model output, not
   real catalog photometry — so the anomaly is most likely a
   pipeline-internal zero-point convention difference, not a historical
   calibration quirk. Reverse-fitting an empirical correction from the
   benchmark data itself was considered and rejected as fabrication.
   Stromgren u stays excluded; this item is closed with an honest
   negative result, not a restored 10th measurement.
2. **A fit-flags/prior mechanism for `PhotometricParameters` —
   implemented.** `PhotometricParameters` now mirrors
   `OrbitalElements.fit_flags` (fit flags plus an optional Gaussian
   prior, e.g. radius tied to the system's literature value with an
   assumed 10% relative sigma), replacing the old bounds-pinning
   workaround in `benchmark_runner.py`, across exactly the scope
   originally called out: `photometric_likelihood.py`,
   `photometric_optimizer.py`, `inference/priors.py` (`photometric_problem.py`/
   `photometric_context.py`/`photometric_mcmc.py` needed no changes —
   already compatible), plus a new `tests/test_photometric_fit_flags.py`
   mirroring `test_triple_fit_flags.py`.

Combined with a third finding made while validating these two (the
Al-Wardat benchmark's assumed 0.02 mag per-measurement uncertainty
rested on the same false "real catalog photometry" premise as item 1,
and was corrected to 0.30 mag) and a fourth (a genuine, now-documented
near-twin-component-temperature degeneracy affecting three of the four
systems), all four registered Al-Wardat SED systems now reach a good
literature match: radius errors ≤1.10%, Teff errors 0.64%–6.37%, reduced
χ² 0.107–0.152.

## Phase 2 — Example workflows for users

There is currently exactly one example script (`examples/hrdiagram_demo.py`)
and an empty `README.md`. A user with a `.inp`/`.in2`/`.in3` file or a
literature table has no on-ramp. Proposed set of runnable, narrated
examples (each a script or notebook under `examples/`, using real
benchmark data already in the repo so they run out of the box):

1. **Visual+RV binary fit from scratch** — parse a `.inp` file
   (`FIN379` or `GL765`), fit with `fit_visual_orbit`, plot the orbit and
   residuals, write a report. The "hello world" workflow.
2. **MCMC posterior exploration** — take the least-squares result from
   (1), seed `run_emcee`, produce a corner plot, discuss convergence
   diagnostics (`mcmc_diagnostics.py`).
3. **Hierarchical triple (ORBIT3)** — parse a `.in2`/`.in3` file
   (`ZetaAqr` or `HIP111805`), fit with `fit_orbit3`, compare legacy vs.
   ORBIT3 formalism outputs.
4. **Al-Wardat SED fit** — the HD214222 pipeline end to end: load CK04
   grid + passbands, fit photometry, interpret `parameter_errors` and
   the pinned-vs-free parameter tradeoff, cross-check against
   evolutionary tracks (`evolution/binary.py`).
5. **Joint orbit+photometry fit** — `inference/joint_problem.py`
   combining an `InferenceProblem` and `PhotometricInferenceProblem`;
   currently has no example anywhere despite being a real, tested
   capability.
6. **From your own data** — a template script showing exactly which
   file formats/fields are needed (an annotated `.inp`, an annotated
   Al-Wardat-style JSON) so a new user can substitute their own system
   without reading the parser source.

Each example should be short (run in well under a minute), narrated with
comments explaining *why* each step happens, and reference the specific
benchmark JSON/`.inp` file it uses so it's reproducible.

## Phase 3 — Full documentation

Current state: `README.md` is empty, `docs/` has only
`project_decisions.md` (an internal decision log, not user-facing) and
auto-generated architecture snapshots (`package_tree.txt`,
`public_api.md`, `statistics.md`, `class_diagram.puml` via
`architecture_snapshot.py`). Nothing explains the package to a new user
or reader of the paper.

Proposed structure:

1. **`README.md`** — what PySeshat is, install instructions
   (`pip install -e .`), a 10-line quickstart pointing at Phase 2's
   example (1), links to the rest of the docs.
2. **User guide** (`docs/guide/`, one page per topic, prose not just
   docstrings): data formats and parsers; orbit fitting (least-squares +
   MCMC); triple systems and the legacy/ORBIT3 distinction (this
   *needs* documenting outside `CLAUDE.md`, since that file is an
   internal agent-guidance doc, not user documentation); Al-Wardat SED
   fitting and its current limitations (point at
   `docs/project_decisions.md`'s degeneracy finding rather than
   duplicating it); evolutionary-track consistency checks; visualization.
3. **API reference** — `architecture_snapshot.py`'s `public_api.md`
   already inventories every public symbol; the gap is narrative
   docstrings are inconsistent in depth across modules (some, like
   `atmospheres.py`, are excellent; others are terse). A documentation
   pass should target modules with thin docstrings specifically, not
   rewrite everything uniformly — an audit script (grep for functions
   whose docstring is under ~3 lines) is a fast way to find the actual
   gaps rather than guessing.
4. **Paper-reproduction notes** — a page mapping each figure/table in
   the target publication to the exact script/benchmark that reproduces
   it, since that traceability is the actual point of this repo per
   `Paper_Roadmap.pdf`.
5. Decide once (don't re-litigate per module): plain Markdown docs
   served from the repo, or a Sphinx/MkDocs build. Given the project's
   validation-first priority, plain Markdown under `docs/` is the lower-
   overhead choice unless the user wants a hosted docs site.

## Phase 4 — Plotting: correctness and literature-convention conformity

Five plotting functions exist (`orbit_plot.py`: `plot_visual_orbit`,
`plot_residuals`, `plot_rv_orbit`; `hrdiagram_plot.py`:
`plot_binary_hrdiagram`; `dynamics_plot.py`: `plot_separation_vs_time`).
"Testing plotting capabilities and conformity to literature style" means
two distinct checks, both currently absent:

1. **Correctness** — does the plot show what the data says? (axis
   scaling, unit labels, error bars actually reflecting the fit
   covariance, no silent truncation). Existing tests
   (`test_orbit_plot.py`, `test_hrdiagram_plot.py`, `test_corner_plot.py`)
   mostly check that a plot *runs without raising* and returns a
   `Figure`/`Axes` — not that it's drawn correctly. Strengthening these
   to assert on actual plotted data (e.g. that `plot_visual_orbit`'s
   plotted separation/PA matches the underlying `X_obs`/`Y_obs` arrays)
   is real, missing test coverage.
2. **Literature-convention conformity** — this is the harder, more
   valuable check and needs to happen against the actual target
   publication and the field's conventions, not just "does it look
   reasonable": visual-orbit plots conventionally show North up / East
   left with the correct position-angle sense (a flipped convention is a
   classic, easy-to-miss bug in this kind of code); HR diagrams
   conventionally plot Teff decreasing left-to-right on a log or reverse
   axis, not increasing; RV curves are conventionally phase-folded with
   a stated zero-phase convention; residual plots should use the same
   sign convention as the fit's own residual functions
   (`inference/terms/`). Recommended approach: pick 2–3 published
   figures from the papers already in this project (`wardat_2002.pdf`,
   `wardat_2012.pdf`, `Tokovinin_2016_ApJ_831_151.pdf`, or the target
   paper itself) and reproduce them exactly with PySeshat's plotting
   functions side by side, visually confirmed once, then locked in with
   a regression test on the specific convention (e.g. "PA increases
   counterclockwise from North") rather than the whole image.
3. Add this as a standing item in the benchmark suite: once Phase 0
   unblocks all 8 systems, generate the full plot set for each and do
   one visual pass before considering the validation phase closed.

## Suggested sequencing

Phase 0 (data unblock) has no dependencies and should happen immediately
— it's one script run by the user and gates real numbers for half the
Al-Wardat benchmark suite. Phase 1 (scientific gaps) and Phase 4
(plotting) can both start in parallel with Phase 0, since neither depends
on the full grid being present (Phase 1's fit-flags work benefits from,
but doesn't require, more benchmark systems being unblocked; Phase 4 can
start on the 5 already-runnable systems). Phases 2 and 3 (examples,
documentation) are best done *after* Phase 1 settles the fit-flags API,
since example workflows and docs written against the current
bounds-pinning workaround would need rewriting once that lands — writing
them last avoids that churn.

## Not in this plan

Two items remain genuinely open and outside PySeshat's own scope:

- The git-commit handoff for all of this session's work (both the prior
  ORBIT3 segment, still awaiting the user's paste-back, and this Al-Wardat
  segment) — this cloud session has no direct git access.
- Anything requiring data this environment cannot fetch on its own
  (the CK04 grid, a properly-sourced Stromgren u zero point) — flagged
  above with exactly what's needed and why, not silently worked around.
