# Evolutionary-track consistency checks

`evolution/` provides an independent cross-check for a fitted or
literature binary solution: given each component's `(Teff, L)` (or
`(Teff, R)`, convertible via `analysis/luminosity.py`), find the
best-matching point(s) on a stellar evolutionary track/isochrone grid
and check whether both components are consistent with a single common
age (as two components of a real binary must be, if they formed
together).

```python
from pyseshat.evolution.binary import fit_binary_tracks

result = fit_binary_tracks(
    primary_track=..., secondary_track=...,       # EvolutionTrack, one per component's mass
    primary_log_teff=..., primary_log_luminosity=...,
    primary_sigma_log_teff=..., primary_sigma_log_luminosity=...,
    secondary_log_teff=..., secondary_log_luminosity=...,
    secondary_sigma_log_teff=..., secondary_sigma_log_luminosity=...,
)

result.age_difference   # 0 for a perfectly consistent common age
```

`fit_binary_tracks()` is deliberately the *first stage* — it fits each
component's track independently, without imposing a common age;
`age_difference` is the diagnostic that tells you how far apart the two
independent best-fit ages actually are. A full common-age fit
(`evolution/age.py`'s `fit_common_age`) constrains both components to
share one age directly rather than just measuring the discrepancy after
the fact.

## How this connects to Al-Wardat SED fitting

This is the module that gives the real Al-Wardat method its second half
— SED fitting alone cannot separate two unresolved components' radii
(see `alwardat_sed.md`'s degeneracy finding), so the published method
uses an evolutionary-track cross-check as the external constraint that
breaks the degeneracy. `photometry/evolution_adapter.py` and
`inference/joint_likelihood.py`'s `JointInferenceProblem` already
provide the connective tissue between a photometric fit and an
evolutionary-track check, but the specific wiring this project's SED
benchmarks would need — evolutionary-track-predicted radius (given each
component's literature mass) feeding directly into
`PhotometricParameters.priors` in place of the literature radius itself
— is **not yet done**. `JointInferenceProblem.evolution_context` is
currently typed `Any` with no defined per-component sample/observables
contract; designing that contract is the concrete blocker (future work
item 1 in `docs/paper_reference_notes.md`), not further evolutionary-track
code being missing.

## Where this is tested today

`tests/test_evolution_adapter.py`, `tests/test_interpolate.py`, and
`tests/test_hrdiagram_plot.py` cover the individual pieces
(track/isochrone interpolation, the adapter, and HR-diagram plotting —
see `visualization.md`) with synthetic tracks. **[2026-08-29]**
`tests/test_evolution_grids.py` and `tests/test_evolution_binary.py`
now additionally cover the grid-loading/lookup/fitting layer itself
(`evolution/grids.py`, `evolution/parsers/parsec.py`,
`evolution/parsers/girardi2000.py`, `evolution/search.py`,
`evolution/age.py`, `evolution/binary.py`) — previously untested
despite `docs/project_decisions.md` having once (incorrectly) recorded
that a `tests/test_parsec.py` existed and exercised it; that file was
never actually present in the repository, so this parsing/fitting
layer had zero direct test coverage until this entry. None of the six
registered systems that would use it (the four Al-Wardat SED systems
plus Capella/AlphaCenAB's photometry rows) currently run a real
evolutionary-track cross-check as part of
`validation/benchmark_runner.py` — this is the same gap as the missing
radius-prior wiring above, and is tracked the same way.

## Evolutionary-track grid: which one, and its current state (2026-08-29)

**Decision: Girardi et al. (2000), A&AS, 141, 371,
doi:10.1051/aas:2000126, at Z=0.019 (solar), is this project's default
evolutionary-track/isochrone grid**, for direct comparability with the
Al-Wardat SED-fitting literature this project benchmarks against — the
HD25811, HIP70973/HIP72479, and HIP102029 papers, and by extension
Abushattal et al. 2025 (arXiv:2512.23652), whose four SB2 systems
(HD214222, HD191854, HD130669, HD184467) are this project's own SED
benchmarks, all consistently place their fitted components on Girardi
(2000) tracks/isochrones.

**Prior state, confirmed by direct audit before this change.** No
evolutionary-track grid — Girardi, PARSEC, or otherwise — was
previously wired in with any real data, and no prior decision on grid
choice existed in this repo's own documentation. Specifically:

- `evolution/binary.py`'s functions (`fit_binary_tracks`,
  `fit_binary_evolution_from_orbit`) are, and remain, fully
  grid-agnostic — they take an `EvolutionModel` as an explicit
  parameter and never import or reference any specific grid by name.
  There was nothing to "replace" at that layer.
- `photometry/evolution_adapter.py` is, and remains, a pure
  data-format bridge (`PhotometricParameters` → `StellarObservables`)
  with no grid-selection logic of any kind.
- `evolution/grids.py`/`evolution/parsers/parsec.py` provided parsing
  *infrastructure* for exactly one grid family (PARSEC), but **zero
  PARSEC data files were ever bundled or referenced anywhere in the
  codebase** — `load_parsec()` requires the caller to supply
  `track_files`/`isochrone_files` explicitly and defaults to empty
  tuples. `docs/project_decisions.md`'s own prior audit (search for
  "Audit for other missing external files") had already found and
  recorded this: "no external track files are referenced or needed
  anywhere in the current codebase." "PARSEC" was never a considered
  literature-comparability decision — it was simply the one parser
  format that happened to be implemented, with no data behind it.
- The commonly-cited `tests/test_parsec.py` "exercising the parser
  against synthetic fixture data" did not actually exist in the
  repository (see the section above) — the parser had literally never
  been run against any file, synthetic or real, before 2026-08-29.

**What changed on 2026-08-29 (code-level integration).**
`evolution/parsers/girardi2000.py` was added, mirroring the PARSEC
parser's design exactly (a caller-supplied `column_map` from canonical
field name to actual file column name, so the parser does not need to
assume the literal CDS header spelling of a real downloaded copy of
the tables) and documenting the grid's real citation, metallicity
coverage (Z = 0.0004, 0.001, 0.004, 0.008, 0.019, 0.03; Z=0.019 is the
paper's own identified solar value), and mass coverage (0.15-7 Msun,
solar-scaled, not alpha-enhanced composition) directly from the
published paper. `evolution/grids.py` registers it as `"girardi2000"`
in `load_model()`'s supported-models table, alongside the pre-existing
`"parsec"` entry (kept, unchanged, exactly as it was — nothing was
removed), and adds `DEFAULT_EVOLUTION_MODEL_NAME = "girardi2000"` plus
a `load_default_evolution_model()` convenience function that raises a
clear, informative `FileNotFoundError` when no data has actually been
placed at the conventional `data/evolution/girardi2000/` location —
an honest failing default rather than a stub that would silently
return an empty or fabricated grid. While fixing up this parsing layer
a real, previously-latent bug was also found and fixed in *both*
parsers' shared `_load_ascii_table()` helper: `numpy.genfromtxt(...,
comments="#")` only strips the `#` character from a line, not the
whole line, so a descriptive comment header whose remaining text
happens to have a different word count than the real data header
(exactly the kind of comment line a real downloaded CDS table has)
made column-count detection fail on every data row. This had never
been caught because, per the point above, the parser had never
actually been run against any file with a comment line before. Fixed
by pre-filtering whole comment lines before handing the text to
`genfromtxt`; regression-tested in `tests/test_evolution_grids.py`.

**What did *not* change, and why: no real numeric grid data has been
sourced.** Actually obtaining the Girardi (2000) tabulated
tracks/isochrones requires reaching an external astronomy data
service, and every avenue available from this development environment
was tried and confirmed blocked on 2026-08-29:

- Direct network access (`curl`/`urllib`) from the cloud sandbox is
  blocked at the organization egress-proxy level — confirmed not just
  for astronomy-specific hosts but for a plain `google.com` request
  too (`CONNECT tunnel failed, response 403`), so this is a blanket
  policy, not a per-domain block.
- The same egress restriction was independently confirmed on the
  user's own linked device (`device_bash`) — `stev.oapd.inaf.it`,
  `vizier.cds.unistra.fr`, and `cdsarc.cds.unistra.fr` all returned
  `403` through that machine's proxy too.
- `WebFetch` (a page-fetch-and-summarize tool, not a general HTTP
  client) *does* reach some of these hosts read-only — it confirmed
  the Padova group's **legacy** CMD 2.7 interface
  (`https://stev.oapd.inaf.it/cgi-bin/cmd_2.7`) still lists "Girardi
  et al. (2000) + simplified TP-AGB (for M<=7 Msun) + Bertelli et al.
  (1994)" as a selectable track/isochrone set (the *current*
  PARSEC-only CMD form at `/cgi-bin/cmd` does not offer it at all) —
  but it cannot submit the resulting CGI form or download the
  computed output table, only view static page text.
- VizieR (`vizier.cds.unistra.fr`, catalog J/A+AS/141/371) and the CDS
  FTP archive (`cdsarc.cds.unistra.fr`) both refuse `WebFetch` as well
  — VizieR with an active bot-detection "Access Denied" page, CDS FTP
  via `robots.txt`.
- No browser-automation session was available (the Chrome extension
  bridge reported "not connected") to drive the CMD 2.7 form
  interactively instead.

This is a confirmed external blocker, the same category as the
Gaia-archive network block already documented for `io/gaia_parser.py`
(see `docs/project_decisions.md` and `docs/paper_reference_notes.md`
§7 item 14) — not something fixable from within this session, and not
something to work around by approximating or fabricating grid values:
per this project's own standing rule, no evolutionary-track numeric
data is invented. **Consequently, no evolutionary consistency check
has been run against any of the six systems that would use this grid**
(the four Al-Wardat SED systems, plus Capella and alpha Cen AB's
photometry rows) — each system's `validation/expected_results/*.json`
now carries an `evolutionary_track_source` provenance block recording
this decision and this exact blocker (`"status": "not yet run"`), so
that nobody mistakes silence in that file for a completed check. No
system's previously-reported age/mass/evolutionary-stage placement in
`docs/project_decisions.md` or `docs/paper_reference_notes.md` has
changed, because none was ever computed against a real grid to begin
with (there is nothing to have shifted from).

**To unblock this**, a real copy of the Girardi (2000) tracks and
isochrones at Z=0.019 needs to reach this repository from a network
that isn't behind the same egress restriction — e.g. downloaded by the
user directly from the CMD 2.7 form
(`https://stev.oapd.inaf.it/cgi-bin/cmd_2.7`) or from VizieR catalog
J/A+AS/141/371 on an unrestricted connection, then supplied as
attached files or placed at `data/evolution/girardi2000/`. Once real
files are in hand, `load_model("girardi2000", track_files=...,
isochrone_files=..., track_column_map=..., isochrone_column_map=...,
track_metadata=..., isochrone_metadata=...)` loads them directly (see
`evolution/parsers/girardi2000.py`'s module docstring for the expected
column semantics), and `evolution/binary.py`'s existing
`fit_binary_evolution_from_orbit()` — unchanged, since it was already
grid-agnostic — runs the actual cross-check. This is future work item
1 in `docs/paper_reference_notes.md`, updated 2026-08-29 with this
grid-choice context.

## Blocker closed (2026-08-29): real Girardi (2000) data obtained, six-system check run

The network blocker documented above is now closed. The user downloaded
the two real tables directly from VizieR (not through this session's
egress-restricted network) and supplied them as attached files:

- `mstracks.tsv` — VizieR catalog **J/A+AS/141/371**, table
  **mstracks** ("main sequence from ZAMS on"), 13,723 rows.
- `isochrones.tsv` — same catalog, table **isochron** (isochrones),
  64,381 rows.

Both files were placed at `data/evolution/girardi2000/` (repo root,
per this project's `src/`-layout convention — see
`docs/project_decisions.md`'s dated entry for why this is outside
`src/pyseshat/`, not inside it). Neither file matched the generic
`column_map`-based schema the code-level integration above had
assumed sight-unseen; the actual VizieR ASU-TSV export format,
column set, and several real-data quirks were inspected directly
before writing any loader code, and are documented in full in
`evolution/parsers/girardi2000.py`'s module docstring:

- **File format**: VizieR ASU-TSV — a metadata block of `#`-prefixed
  lines, then exactly 3 non-`#` lines (tab-separated column header,
  units row, a dashes/separator row whose first field is literally
  `-`), then tab-separated data. Located programmatically by finding
  the dashes row and taking the header 2 lines above it. This is a
  **different** parsing path from the legacy single-track
  `_load_ascii_table()`/`numpy.genfromtxt(comments="#")` loader (and
  its already-documented comment-stripping bug), so that bug does not
  recur here.
- **Overshoot column** (0/1): 1 = the grid's full/main set (complete
  0.15–7.0 Msun coverage, 34 masses at Z=0.019) = the new
  `DEFAULT_OVERSHOOT = 1`; 0 is a smaller comparison-only subset
  (1.2–8.0 Msun, 17 masses).
- **No radius column** in either table. `log_radius` is derived via
  the Stefan-Boltzmann law from `logL`/`logTeff`, using this
  codebase's own `SOLAR_EFFECTIVE_TEMPERATURE` constant
  (`analysis/luminosity.py`) for exact self-consistency with
  `compute_log_luminosity()`.
- **`Stage` column** (text, sparse, ~88% blank) cannot populate
  `EvolutionTrack.phase` (which `interpolate.py` casts via `int(...)`)
  — `phase=None` for real Girardi data; `Stage` is still looked up
  and reported separately as a descriptive label only.
- **Non-monotonic ages**: 12 of 34 mass tracks (Z=0.019,
  Overshoot=1) have a repeated/decreasing `Age` value late in the
  track (an AGB thermal-pulse artifact). Each affected track is
  truncated at its first non-monotonic transition; verified this
  loses no age range needed by any of the six benchmark systems below.
- **Duplicate isochrone masses**: 27 of 50 age-slices have duplicate
  (not decreasing) `Mini` values from blue-loop crossings, handled via
  `drop_duplicates(subset="Mini", keep="first")` (required since
  track/isochrone interpolation needs strictly increasing x-values).

`load_girardi2000_vizier_tracks()` / `load_girardi2000_vizier_isochrones()`
(new, in `evolution/parsers/girardi2000.py`) load these files directly.
`evolution/grids.py`'s `load_default_evolution_model()` — previously a
stub raising `FileNotFoundError` — now looks for `mstracks.tsv` /
`isochrones.tsv` at `data/evolution/girardi2000/` and, when present,
returns a real, loaded `EvolutionModel` (34 tracks, 50 isochrones,
Z=0.019). `DEFAULT_EVOLUTION_MODEL_NAME = "girardi2000"` now resolves
to this real data, closing out this section's "no real numeric grid
data has been sourced" limitation.

### Six-system results

The evolutionary consistency check (independent per-component
best-fit age, then a joint common-age fit) was run for all six
systems that need it — the four Al-Wardat SED systems (HD214222,
HD191854, HD130669, HD184467) plus the two joint case studies
(Capella, alpha Cen AB) — via the shared pipeline in
`evolution/benchmark_checks.py` (`scripts/run_evolutionary_checks.py`
runs it and writes `validation/results/evolutionary_checks.json`;
`scripts/generate_benchmark_plots.py`'s `generate_hrdiagrams()` uses
the exact same fitted objects to build each system's HR-diagram
figure at `validation/results/plots/<system>/hrdiagram.png`, so the
plotted points and these numbers cannot drift apart). Component
masses come directly from `literature_component_parameters` for the
four HD systems and AlphaCenAB (`AlphaCenAB_orbit.json`'s
`component_masses_Msun`, Akeson et al. 2021), and are derived from
Capella's literature SB2 orbital elements (Torres et al. 2015 Table 1)
via `analysis.masses.component_masses_from_elements` (M1=2.5688,
M2=2.4828 Msun) — no separately-stated literature mass for Capella
exists anywhere else in this repository.

All six common-age fits converged (none failed). Every component in
all six systems places as a dwarf/main-sequence star (log g ≥ 4.0)
**except Capella**, whose two components place as a giant
(log g≈2.7) and a subgiant (log g≈3.0–3.1) — physically expected for
a well-known G-giant + G-subgiant binary, not a surprise. No
component's requested mass fell outside the grid's 0.15–7 Msun
coverage. Two things were flagged as genuinely worth a second look
rather than silently accepted:

- **HD184467's secondary** (0.85 Msun) sits exactly equidistant
  (0.05 Msun) between the grid's 0.8 and 0.9 Msun tracks —
  `nearest_track()` resolves the tie by taking the first in
  ascending iteration order (0.8 Msun), giving a 5.9% mass mismatch
  and, read independently, a 15.5 Gyr best-fit age that exceeds a
  Hubble time. This is a genuine grid-spacing coincidence for this
  one component, not a bug in the search or fit code — flagged in
  `evolutionary_checks.json` and this system's
  `evolutionary_track_source.result_summary`. The joint common-age
  fit for this system still converges (7.9 Gyr), constrained
  primarily by the better-behaved primary.
- **Sparse local age-grid spacing** (local tabulated-age spacing
  exceeding 20% of the age itself at the best-fit point) is flagged
  for at least one component in nearly every system — an expected
  consequence of Girardi (2000)'s age sampling being coarser than
  this project's own fit resolution, not a fit-quality problem. It
  means the *reported* age is only as precise as the grid's own
  tabulation there, independent of how good the chi2 looks; treat
  ages carrying this flag as approximate to the nearest tabulated
  grid point rather than as precision solutions.

Full per-component numbers (requested/nearest mass, mismatch %,
independent best-fit age/chi2/log g, evolutionary state, all flags,
the joint common-age fit) are in
`validation/results/evolutionary_checks.json` and summarized in each
system's own `validation/expected_results/*.json` under
`evolutionary_track_source.result_summary`.

**No prior age/mass/evolutionary-stage number existed anywhere else
in this project's documentation for any of these six systems before
this run** (confirmed by direct search) — so there is nothing for
these results to disagree with; this is the first time such numbers
have been computed in this project. This is distinct from — and does
not include — a comparison against each source paper's *own*
separately-published age/mass values, which was not performed as
part of this run and remains open if wanted.

See `docs/project_decisions.md`'s 2026-08-29 "Girardi (2000) blocker
closed" entry for the full narrative.
