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

## How this connects to binary SED fitting

This is the module that gives the real binary SED-fitting method its second half
— SED fitting alone cannot separate two unresolved components' radii
(see `binary_sed.md`'s degeneracy finding), so the published method
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
contract; designing that contract is the concrete blocker, not further
evolutionary-track code being missing.

## Where this is tested today

`tests/test_evolution_adapter.py`, `tests/test_interpolate.py`, and
`tests/test_hrdiagram_plot.py` cover the individual pieces
(track/isochrone interpolation, the adapter, and HR-diagram plotting —
see `visualization.md`) with synthetic tracks.
`tests/test_evolution_grids.py` and `tests/test_evolution_binary.py`
additionally cover the grid-loading/lookup/fitting layer itself
(`evolution/grids.py`, `evolution/parsers/parsec.py`,
`evolution/parsers/girardi2000.py`, `evolution/search.py`,
`evolution/age.py`, `evolution/binary.py`).
`validation/benchmark_runner.py` does not itself run the evolutionary-track
cross-check for the six registered systems that use it (the four binary SED
systems plus Capella/AlphaCenAB's photometry rows); it is run separately via
`scripts/run_evolutionary_checks.py` (see below). This is the same gap as the
missing radius-prior wiring above.

## Evolutionary-track grid: which one

**Default grid: Girardi et al. (2000), A&AS, 141, 371,
doi:10.1051/aas:2000126, at Z=0.019 (solar)**, chosen for direct
comparability with the binary SED-fitting literature this package
benchmarks against — the HD25811, HIP70973/HIP72479, and HIP102029
papers, and by extension Abushattal et al. 2025 (arXiv:2512.23652),
whose four SB2 systems (HD214222, HD191854, HD130669, HD184467) are
this package's SED benchmarks, all consistently place their fitted
components on Girardi (2000) tracks/isochrones.

Grid-selection layering:

- `evolution/binary.py`'s functions (`fit_binary_tracks`,
  `fit_binary_evolution_from_orbit`) are fully grid-agnostic — they take
  an `EvolutionModel` as an explicit parameter and never import or
  reference any specific grid by name.
- `photometry/evolution_adapter.py` is a pure data-format bridge
  (`PhotometricParameters` → `StellarObservables`) with no grid-selection
  logic of any kind.
- `evolution/grids.py` registers parsers for two grid families,
  `"girardi2000"` (the default, `DEFAULT_EVOLUTION_MODEL_NAME`) and
  `"parsec"`, in `load_model()`'s supported-models table.
  `load_parsec()` requires the caller to supply
  `track_files`/`isochrone_files` explicitly; no PARSEC data files are
  bundled.
- `evolution/parsers/girardi2000.py` documents the grid's citation,
  metallicity coverage (Z = 0.0004, 0.001, 0.004, 0.008, 0.019, 0.03;
  Z=0.019 is the paper's own identified solar value), and mass coverage
  (0.15-7 Msun, solar-scaled, not alpha-enhanced composition).
  `load_default_evolution_model()` loads the copy bundled with the
  package (`pyseshat/evolution/data/girardi2000/`) and raises a clear
  `FileNotFoundError` if the files are missing from the given
  `data_dir`, rather than silently returning an empty or fabricated
  grid.
- The shared `_load_ascii_table()` helper used by the generic parsers
  pre-filters whole comment lines before handing the text to
  `numpy.genfromtxt`, because `genfromtxt(..., comments="#")` only strips
  the `#` character from a line, not the whole line, so a descriptive
  comment header with a different word count than the real data header
  would otherwise break column-count detection; this is covered in
  `tests/test_evolution_grids.py`.

## Obtaining the Girardi (2000) data

The Z = 0.019 grid used by default is bundled with the package (the
VizieR tables, unchanged; see
`src/pyseshat/evolution/data/girardi2000/README.md`), so no download is
needed. To use another copy, pass `data_dir=` to
`load_default_evolution_model()`. The tables are available from:

- VizieR catalog **J/A+AS/141/371** (tables `mstracks` and `isochron`),
  exported as ASU-TSV;
- the Padova group's legacy CMD 2.7 interface
  (`https://stev.oapd.inaf.it/cgi-bin/cmd_2.7`), which still lists
  "Girardi et al. (2000) + simplified TP-AGB (for M<=7 Msun) + Bertelli
  et al. (1994)" as a selectable track/isochrone set (the current
  PARSEC-only CMD form does not offer it).

`load_default_evolution_model()` returns a loaded `EvolutionModel`;
alternatively call
`load_model("girardi2000", track_files=..., isochrone_files=...,
track_column_map=..., isochrone_column_map=..., track_metadata=...,
isochrone_metadata=...)` directly (see
`evolution/parsers/girardi2000.py`'s module docstring for the expected
column semantics). `evolution/binary.py`'s
`fit_binary_evolution_from_orbit()` then runs the actual cross-check.
No evolutionary-track numeric data is ever invented or approximated: if
the grid is absent, the check cannot be run.

## Real Girardi (2000) data: six-system check

The VizieR export consists of two tables:

- `mstracks.tsv` — VizieR catalog **J/A+AS/141/371**, table
  **mstracks** ("main sequence from ZAMS on"), 13,723 rows.
- `isochrones.tsv` — same catalog, table **isochron** (isochrones),
  64,381 rows.

Both files live at `src/pyseshat/evolution/data/girardi2000/`. The generic
`column_map`-based schema does not apply to them; the actual VizieR
ASU-TSV export format, column set, and several real-data quirks are
documented in full in `evolution/parsers/girardi2000.py`'s module
docstring:

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
  0.15–7.0 Msun coverage, 34 masses at Z=0.019) = the
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
(in `evolution/parsers/girardi2000.py`) load these files directly.
`evolution/grids.py`'s `load_default_evolution_model()` looks for `mstracks.tsv` /
`isochrones.tsv` in the bundled `evolution/data/girardi2000/` (or a given
`data_dir`) and, when present,
returns a real, loaded `EvolutionModel` (34 tracks, 50 isochrones,
Z=0.019). `DEFAULT_EVOLUTION_MODEL_NAME = "girardi2000"` resolves
to this data.

### Six-system results

The evolutionary consistency check (independent per-component
best-fit age, then a joint common-age fit) was run for all six
systems that need it — the four binary SED systems (HD214222,
HD191854, HD130669, HD184467) plus the two joint case studies
(Capella, alpha Cen AB) — via the shared pipeline in
`evolution/benchmark_checks.py` (`scripts/run_evolutionary_checks.py`
runs it and writes `validation/results/evolutionary_checks.json`;
`scripts/generate_benchmark_plots.py`'s `generate_hrdiagrams()` uses
the exact same fitted objects to build each system's HR-diagram
figure at `validation/results/plots/<system>/hrdiagram.pdf` (vector
PDF, no grid lines, per A&A's figure requirements), so the plotted
points and these numbers cannot drift apart). Component
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

No independent age/mass/evolutionary-stage values are recorded for
these six systems, so these results are not compared against any
prior numbers. They are also distinct from — and do not include — a
comparison against each source paper's *own* separately-published
age/mass values, which has not been performed.
