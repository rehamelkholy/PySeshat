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
