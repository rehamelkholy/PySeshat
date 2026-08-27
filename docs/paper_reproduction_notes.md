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
| 3. Validation | Benchmark against Tokovinin's ORBIT solutions; compare with PyAstronomy/RadVel/Exo-Striker; Gaia-binary accuracy tests | `validation/benchmark_runner.py` + `validation/expected_results/*.json` for the Tokovinin-solution comparison (done, all 12 systems). **[DONE 2026-08-27]** `scripts/compare_other_packages.py` compares against RadVel (independent RV fit, period/K/gamma recover to <=1.6%) and PyAstronomy (independent Kepler-physics cross-check, agrees to 9e-14 km/s); Exo-Striker confirmed installable but not exercised — honestly disclosed as GUI-only/no headless API, see the script's own printed output. `src/pyseshat/io/gaia_parser.py` parses Gaia DR3 NSS orbits and is validated against one real published system (Gaia BH1); a live Gaia-archive query itself was not possible (network egress blocked — see `docs/paper_reference_notes.md` §7 item 14) |
| 4. Case Studies | Reproduce classical binary orbits (e.g. Capella, Sirius); demonstrate Al-Wardat method on a real composite system | **[DONE 2026-08-27]** Capella and alpha Centauri AB (substituting for Sirius, with an explicit documented justification — Sirius B is a white dwarf, unsuited to the Al-Wardat method) are now registered benchmark systems that EACH validate both orbit-fitting AND Al-Wardat photometry for the same physical system — see `docs/paper_reference_notes.md` §3.6-3.7 and `scripts/build_joint_benchmarks.py`. The Al-Wardat demonstration on the original 4 systems remains well covered too (`examples/04_alwardat_sed_fit.py`) |
| 5. Conclusions | Advantages, community-contribution roadmap | Draws on `docs/paper_reference_notes.md` §6-7 (known limitations, future work) |

## The 6 essential figures (`Paper_Roadmap.pdf`, "Essential Figures for the Paper")

| # | Figure | Reproduced by | Status |
|---|---|---|---|
| 1 | Orbit fit example (observed positions + model orbit) | `plot_visual_orbit()`; `scripts/generate_benchmark_plots.py` writes `validation/results/plots/{FIN379,GL765,Capella_orbit,AlphaCenAB_orbit}/visual_orbit.png`; `examples/01_visual_rv_binary_fit.py` | Done for FIN379/GL765/Capella/AlphaCenAB. **[RESOLVED 2026-08-27]** North-up/East-left orientation settled against literature (IAU/WDS/`orbitize` convention) and implemented — safe to use any of these PNGs as-is; see `docs/project_decisions.md`'s dated entry. |
| 2 | Residual distribution | `plot_residuals()`; same script writes `.../residuals.png` per system | Done for FIN379/GL765/Capella/AlphaCenAB; not yet generated for the triple or SED-only systems (see `docs/guide/visualization.md` for why) |
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

As of 2026-08-27, the package-comparison and named-case-study gaps
below are closed (see the table above and
`docs/paper_reference_notes.md` §3.6-3.7, §7 items 13-14). What
remains:

1. **No live Gaia-archive query.** `io/gaia_parser.py` is validated by
   round-tripping its own inversion formula and by one real system's
   PUBLISHED elements (Gaia BH1), but this session's network access
   could not reach the Gaia TAP archive itself, so no row of Gaia's raw
   archive-level output has been fed through the parser yet — see
   `docs/paper_reference_notes.md` §7 item 14 for the concrete next
   step (a specific ADQL query) once archive access is available.
2. No SED-fit figure (essential figure 4) and no real-system HR-diagram
   figure (essential figure 5) — both are concrete, scoped, buildable
   future work, not open scientific questions.
3. Exo-Striker was not exercised (confirmed installable, but a GUI-only
   tool with no documented headless API) — disclosed as an honest
   limitation rather than forced through; see
   `scripts/compare_other_packages.py`'s own printed output.
