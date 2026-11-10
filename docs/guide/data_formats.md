# Data formats and parsers

PySeshat reads two families of input file, one per fitting pipeline.

## `.inp` — Tokovinin ORBIT-format visual+RV binary

Plain text, one system per file (see
`validation/benchmark_systems/FIN379/FIN379.inp` for a real, fully
commented example). A header block of `key value` lines gives the
starting orbital elements and metadata:

```
Object:  FIN 379
RA:      2.4412
Dec:    -25.30
P        6.6900      # period, years
T        2008.8443   # T0, periastron epoch (decimal year)
e       0.4988
a       0.1003        # semimajor axis, arcsec
W        183.9        # Omega, deg
w        191.0        # omega, deg
i        42.0         # deg
K1       7.43         # primary RV semi-amplitude, km/s
K2       8.05         # secondary RV semi-amplitude, km/s
V0      -0.51         # systemic velocity, km/s
```

followed by one row per observation. Visual astrometry rows are
`epoch theta rho sigma <tag>` (position angle in degrees, separation in
arcsec); RV rows carry a trailing component flag (0 = primary/Va,
1 = secondary/Vb). Epochs may be given either as decimal years or as
JD-2400000 (reduced Julian date) — `io/orbit_parser.py` detects which
and normalizes via `jd_to_decimal_year()`. Lines starting with `C` are
comments (often carrying literature cross-checks, e.g. FIN379's `C
Hrt2012a ...` line citing an independent published orbit) and are
skipped by the parser but worth reading by eye — they're where a raw
file's own provenance notes live.

Parse with:

```python
from pyseshat.io.orbit_parser import parse_orbit_file

parsed = parse_orbit_file("validation/benchmark_systems/FIN379/FIN379.inp")
elements = parsed["elements"]        # OrbitalElements
visual = parsed["visual_data"]       # ndarray, columns [epoch, theta, rho, sigma]
rv = parsed["rv_data"]               # ndarray, columns [epoch, velocity, sigma, component]
```

A real, historical format wrinkle worth knowing about if you write your
own `.inp` file: some raw WDS-sourced rows have no space between a
value and the following tag column (`0.0505 0.0015I1` rather than
`0.0505 0.0015 I1`) — the parser handles this, but it's a reminder these
files are transcribed from heterogeneous historical sources, not a
clean format PySeshat itself defines.

## `.in2` / `.in3` — hierarchical triple

Two related but distinct formats for a hierarchical triple's outer +
inner orbit, both parsed by `parse_triple_file()`:

- **`.in2`** — the legacy ORBITT2 format. Outer-orbit elements
  (`P, T, e, a, W, w, i`) plus inner-orbit elements prefixed `2`
  (`P2, T2, e2, a2, W2, w2, i2`) plus a scale parameter `B`. See
  `validation/benchmark_systems/ZetaAqr/*.in2`.
- **`.in3`** — the modern ORBIT3 format. Same outer/inner element
  naming, but the inner orbit additionally carries `K1, K2, f` (RV
  semi-amplitudes and the astrometric wobble factor) and the outer orbit
  carries `K3, K4, V0`. See `validation/benchmark_systems/HIP111805/*.in3`.

**Two format-specific quirks that matter if you're fitting your own
triple system** (both cost real debugging time on the two registered
triple benchmarks):

1. A published astrometric wobble amplitude (e.g. from a paper's own
   Table) is usually the *astrometric* wobble `a_astrometric`, not the
   ORBIT3 inner semimajor axis `a` this parser/fitter needs directly —
   converting requires `a_true = a_astrometric / f`. Using the published
   astrometric value directly as `a` is a real, easy mistake (see ζ Aqr
   in `orbit_fitting.md`/`triple_systems.md`).
2. Some `.in2`-format raw files use the inner/outer astrometric tag
   convention (`I1`/`I2`) backwards relative to ORBIT3's own convention.
   `validation/benchmark_runner.py`'s
   `_convert_legacy_orbit3_observations()` handles this for the ζ Aqr
   benchmark specifically — if you bring your own `.in2` file, check
   which convention it actually uses rather than assuming.

```python
from pyseshat.triple.triple_parser import parse_triple_file

parsed = parse_triple_file("validation/benchmark_systems/HIP111805/HIP111805.in3")
elements = parsed["elements"]              # TripleOrbitalElements
observations = parsed["observations"]      # outer-orbit visual astrometry
rv_observations = parsed["rv_observations"]
```

## binary SED JSON

Unlike the two formats above, an unresolved-binary SED
benchmark is not a raw instrument-format file — it's a JSON file per
system under `validation/expected_results/<SYSTEM>.json`, since there is
no orbital fit involved at all (see `binary_sed.md`). Key fields (see
`validation/expected_results/HD214222.json` for a complete real
example):

- `measurements` — list of `{passband, magnitude, uncertainty}`, the
  actual data being fit. The benchmark runner knows the passbands
  "Johnson U/B/V", "Cousins R", "Stromgren u/v/b/y" and "Tycho BT/VT"
  (curves and sources: `validation/photometry/passbands/README.md`).

Passband files are two-column ASCII (wavelength in Angstrom,
sensitivity). A header line `# detector: photon` marks a photon-counting
curve P(λ), integrated with a λ weight (Maíz Apellániz 2006, Eq. A1);
`# detector: energy` or no such line means an energy-integrating curve.
`load_passband(..., photon_counting=True/False)` overrides the header.
- `initial_parameters` — starting `primary_temperature`,
  `secondary_temperature`, `primary_radius`, `secondary_radius`, `ebv`.
- `atmosphere_files` — which CK04 grid FITS files (under
  `validation/atmosphere_models/`) this system's fit needs loaded; must
  bracket **both** components' temperatures, not just one (a real bug
  this project hit).
- `literature_component_parameters` — the independently-published
  comparison target this benchmark's fit is checked against.
- `distance_pc`, `metallicity`, `primary_logg`, `secondary_logg` — held
  fixed during the fit (not free parameters).
- `excluded_measurements` (optional) — measurements present in the source
  data but deliberately excluded from fitting, with a `note` explaining
  why. None of the current benchmarks uses it (Strömgren u was excluded
  before 0.2.0 because of a zero-point error; see `binary_sed.md`).
- `magnitude_differences` (optional) — component magnitude differences
  (`passband`, `value` = m_secondary - m_primary, `uncertainty`) added
  to the fit as extra residuals.
- `radius_prior_note` — documents the assumed relative sigma used for
  each system's radius Gaussian prior (see `binary_sed.md`).

This JSON format is specific to this project's benchmark suite, not a
standard the source paper defines — building your own system's file by
hand from a literature Table is expected; use an existing one as a
template (`examples/06_your_own_data_template.py` includes an annotated
copy for exactly this purpose).

## Gaia DR3 NSS `nss_two_body_orbit` rows

`src/pyseshat/io/gaia_parser.py` converts a row of Gaia DR3's
Non-Single-Star two-body-orbit archive table
(`gaiadr3.nss_two_body_orbit`) into this codebase's own
`OrbitalElements`. Unlike the formats above, this is not a file this
project ships examples of — a "row" is any mapping with the archive's
own column names (a `dict`, a pandas `Series`/`DataFrame` row, or an
astropy `Table` row all work):

```python
from pyseshat.io.gaia_parser import parse_nss_orbital_row

row = {
    "nss_solution_type": "Orbital",
    "a_thiele_innes": ..., "b_thiele_innes": ..., "f_thiele_innes": ..., "g_thiele_innes": ...,
    "period": ...,          # days
    "eccentricity": ...,
    "t_periastron": ...,    # days, relative to J2016.0
    "parallax": ...,        # mas
}
result = parse_nss_orbital_row(row)
elements = result["elements"]        # OrbitalElements
distance_pc = result["distance_pc"]
```

Only `Orbital`/`OrbitalAlternative*`/`OrbitalTargetedSearch*` (pure
astrometric) and `AstroSpectroSB1` (astrometric + single-lined RV
jointly fit) solution types carry the Thiele-Innes constants this
module needs; purely spectroscopic (`SB1`/`SB2`/...) and eclipsing
solution types raise `ValueError` rather than silently returning a
physically meaningless orbit — see the module's own docstring for the
full scope, the Thiele-Innes→Campbell inversion method (Halbwachs et
al. 2023), and the ascending/descending-node ±180° ambiguity that pure
astrometric (no-RV) solutions carry.

**Optional NaN columns**: a genuine CSV/VOTable export represents an
unpopulated *optional* column (e.g. `inclination`/`arg_periastron` on
a pure-astrometric `Orbital` row, which Gaia's own pipeline never
populates for that solution type) as NaN once loaded through
pandas/astropy — not Python `None`, which only a hand-built dict can
produce. A plain `row.get(key) is not None` check would silently
accept NaN as "present" and propagate it straight into a NaN
`OrbitalElements` field, so `gaia_parser._is_present()` treats NaN and
`None` identically; regression-tested in
`tests/test_gaia_parser.py`. Worth knowing if you write your own
row-loading code against a real archive export: don't assume `is not
None` is enough for an optional numeric column.

**Validation status: validated against a real, retrieved Gaia Archive
export**, not just a round-trip. `tests/
test_gaia_parser.py` forward-transforms El-Badry et al. 2023's own
*published* Campbell elements for Gaia BH1 (source_id
`4373465352415301632`) through this codebase's own `thiele_innes()`
and checks they round-trip back out through `parse_nss_orbital_row()`
— this validates the inversion algebra and unit conversions, but was
never a check against archive-retrieved data on its own. A live TAP
query (`astroquery.gaia`) may be blocked in environments with restricted
network access (for example an HTTP 403 for `gea.esac.esa.int`); the
real export was obtained via a manual path instead: an ADQL query run
through the Gaia Archive's own web interface
(<https://gea.esac.esa.int/archive/>), exported as CSV.

The script originally targeted Gaia BH1 and Gaia BH2, but the query
for that pair returned zero rows for both — Ryu et al. 2025
(arXiv:2411.00654) explains this: both have low (~39%/~27%) modeled
probability of surviving Gaia's pipeline into the *public* catalog,
despite being real, well-characterized systems. `scripts/
validate_gaia_archive_export.py` was rewritten to target **RX Cam**
(source_id `470361114339849472`) and **HIP 36189** (source_id
`3162827836766605440`) instead — both explicitly named by source_id,
with real Gaia-fitted parameters, in Gaia Collaboration 2023's own DR3
companion paper (arXiv:2206.05595). (Whether this paper's own benchmark systems Capella/alpha Cen AB
have Gaia NSS solutions was not determined; their NSS status remains
undetermined rather than ruled out.)

Run against the real export, `scripts/validate_gaia_archive_export.py`
reports:
- **RX Cam**: archive-derived period/eccentricity/inclination match
  the Gaia team's own published reading of this exact row to
  sub-0.25%, and are consistent within the row's own error bars with
  Groenewegen (2013)'s fully independent, pre-Gaia literature values —
  a clean, direct validation of the Thiele-Innes-to-Campbell inversion
  against real archive data.
- **HIP 36189**: the naive forward-predicted position initially missed
  the independent Herald et al. (2020) occultation observation by a
  wide margin. Investigated rather than accepted or dismissed: Gaia's
  `Orbital`-type solutions fit the astrometric *photocentre's* orbit,
  not the relative (star-to-star) orbit an occultation resolves — a
  real, physically necessary distinction, already documented in this
  module's own "Scope" section above. Applying Gaia Collaboration
  2023's own mass-function relation (using the paper's own quoted
  component masses and magnitude difference) to convert the
  photocentre amplitude to a relative-orbit amplitude, and propagating
  this specific row's own quoted Thiele-Innes uncertainties (Monte
  Carlo, 4000 draws) for the position angle, brings both separation and
  position angle into agreement.

Both corrections are permanent, re-runnable parts of the script (not
one-off analysis).
