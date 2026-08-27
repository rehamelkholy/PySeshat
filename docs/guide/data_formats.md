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
triple system** (both documented in depth in `docs/project_decisions.md`
and `docs/paper_reference_notes.md`, since they cost real debugging time
on the two registered triple benchmarks):

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

## Al-Wardat SED JSON

Unlike the two formats above, an Al-Wardat unresolved-binary SED
benchmark is not a raw instrument-format file — it's a JSON file per
system under `validation/expected_results/<SYSTEM>.json`, since there is
no orbital fit involved at all (see `alwardat_sed.md`). Key fields (see
`validation/expected_results/HD214222.json` for a complete real
example):

- `measurements` — list of `{passband, magnitude, uncertainty}`, the
  actual data being fit.
- `initial_parameters` — starting `primary_temperature`,
  `secondary_temperature`, `primary_radius`, `secondary_radius`, `ebv`.
- `atmosphere_files` — which CK04 grid FITS files (under
  `validation/atmosphere_models/`) this system's fit needs loaded; must
  bracket **both** components' temperatures, not just one (a real bug
  this project hit — see `docs/project_decisions.md`).
- `literature_component_parameters` — the independently-published
  comparison target this benchmark's fit is checked against.
- `distance_pc`, `metallicity`, `primary_logg`, `secondary_logg` — held
  fixed during the fit (not free parameters).
- `excluded_measurements` — measurements present in the source data but
  deliberately excluded from fitting, with a `note` explaining why (e.g.
  Stromgren u — see `alwardat_sed.md`).
- `radius_prior_note` — documents the assumed relative sigma used for
  each system's radius Gaussian prior (see `alwardat_sed.md`).

This JSON format is specific to this project's benchmark suite, not a
standard the source paper defines — building your own system's file by
hand from a literature Table is expected; use an existing one as a
template (`examples/06_your_own_data_template.py` includes an annotated
copy for exactly this purpose).
