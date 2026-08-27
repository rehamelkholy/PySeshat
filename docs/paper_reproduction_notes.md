# Paper-reproduction notes

Maps the target publication's planned structure and figures (per
`Paper_Roadmap.pdf` and `Development_Plan.pdf`, both in the claude.ai
"Binaries Paper" project) to the exact script, benchmark, or test in
this repo that reproduces the corresponding result — the actual point
of this repo per `Paper_Roadmap.pdf`'s Phase VI. Where a mapping is
incomplete or the underlying capability doesn't exist yet, that's
stated plainly rather than left implicit.

## The paper's planned structure (`Development_Plan.pdf`, Phase 4)

| Section | What it covers | Where in this repo |
|---|---|---|
| 1. Introduction | ORBIT's legacy/limitations, need for a modern Python replacement | `docs/project_decisions.md`'s design-choices section (legacy ORBITT2 vs. modern ORBIT3, why kept separate) |
| 2. Software Description | PySeshat architecture, data handling, orbit fitting, Al-Wardat module | `docs/guide/` (all 6 pages), `docs/architecture/public_api.md`/`package_tree.txt` (auto-generated via `architecture_snapshot.py`) |
| 3. Validation | Benchmark against Tokovinin's ORBIT solutions; compare with PyAstronomy/RadVel/Exo-Striker; Gaia-binary accuracy tests | `validation/benchmark_runner.py` + `validation/expected_results/*.json` for the Tokovinin-solution comparison (done, all 8 systems); **no comparison against PyAstronomy/RadVel/Exo-Striker or a Gaia-binary accuracy test exists in this repo** — genuinely not started, not merely undocumented |
| 4. Case Studies | Reproduce classical binary orbits (e.g. Capella, Sirius); demonstrate Al-Wardat method on a real composite system | **Not started** — the 8 registered benchmark systems (FIN379, GL765, ζ Aqr, HIP 111805, HD214222/191854/130669/184467) are the validation suite, not the named case studies (Capella, Sirius) the plan calls for; the Al-Wardat demonstration itself is well covered (`examples/04_alwardat_sed_fit.py`, all 4 SED benchmarks) |
| 5. Conclusions | Advantages, community-contribution roadmap | Draws on `docs/paper_reference_notes.md` §6-7 (known limitations, future work) |

## The 6 essential figures (`Paper_Roadmap.pdf`, "Essential Figures for the Paper")

| # | Figure | Reproduced by | Status |
|---|---|---|---|
| 1 | Orbit fit example (observed positions + model orbit) | `plot_visual_orbit()`; `scripts/generate_benchmark_plots.py` writes `validation/results/plots/{FIN379,GL765}/visual_orbit.png`; `examples/01_visual_rv_binary_fit.py` | Done for FIN379/GL765. **Read `docs/guide/visualization.md`'s North-up/East-left caveat before using either PNG as-is in the paper** — the orientation has not been verified against a published figure. |
| 2 | Residual distribution | `plot_residuals()`; same script writes `.../residuals.png` per system | Done for FIN379/GL765; not yet generated for the triple or SED systems (see `docs/guide/visualization.md` for why) |
| 3 | Posterior distributions | `MCMCResult.corner_plot()`; `examples/02_mcmc_posterior_exploration.py` | Done — runnable end to end; not pre-rendered to a saved PNG in this repo, since MCMC output depends on the run (seed the RNG, as the example does, for a reproducible figure) |
| 4 | SED fit | **No dedicated plotting function exists.** `examples/04_alwardat_sed_fit.py` fits and numerically reports the SED comparison (fitted vs. observed magnitude per band) but does not plot it | **Gap.** Building this (fitted model spectrum + measured points per passband, likely a magnitude-vs-wavelength or flux-vs-wavelength plot) is real, concrete future work — see `docs/paper_reference_notes.md`'s "Future work" |
| 5 | HR diagram placement | `plot_binary_hrdiagram()`; tested with synthetic data in `tests/test_hrdiagram_plot.py` | **Gap for the 4 real Al-Wardat systems specifically** — no evolutionary-track cross-check has been run against any of HD214222/191854/130669/184467 yet (see `docs/guide/evolutionary_checks.md`); the plotting function itself works and is convention-correct, it just has no real-system data plugged in yet |
| 6 | Pipeline diagram | Not a data-driven plot — a static architecture diagram | `docs/architecture/package_tree.txt`, `class_diagram.puml` (via `architecture_snapshot.py`) give the raw structure to draw from; no finished diagram exists yet |

## Validated numeric claims (Section 3/"Validation")

Every number in `docs/paper_reference_notes.md` §2 (the cross-system
results table) and §3 (per-system nuances) is reproducible by running
`python validation/benchmark_runner.py`, which regenerates
`validation/results/*.json` from the literature-sourced starting points
in `validation/expected_results/*.json` — nothing in that table is
hand-transcribed or hand-tuned per run. If a number in the paper needs
double-checking against the code, this is the one command to re-run.

## What this leaves genuinely open

Three concrete gaps, not four vague ones:

1. No comparison against other RV/orbit-fitting packages (PyAstronomy,
   RadVel, Exo-Striker) or a Gaia-binary accuracy test — Section 3 of
   the paper plan calls for both; neither has been started.
2. No named classical-binary case study (Capella, Sirius) — Section 4
   calls for one; the 8 registered systems substitute as the validation
   suite but are not what that section describes.
3. No SED-fit figure (essential figure 4) and no real-system HR-diagram
   figure (essential figure 5) — both are concrete, scoped, buildable
   future work, not open scientific questions.
