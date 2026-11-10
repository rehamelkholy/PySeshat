"""
Example 8: Evolutionary-track consistency check.

Given each component's (Teff, radius) -- from an SED fit, or straight
from the literature, as here -- an evolutionary-track/isochrone grid
gives an independent cross-check: find each component's best-fit age
on the grid separately, then ask whether both ages agree, as they
must for two components of a real binary that formed together.

This uses this project's default grid, Girardi et al. (2000), A&AS,
141, 371 (Z=0.019, solar), loaded from the real VizieR J/A+AS/141/371
export bundled with the package at
``pyseshat/evolution/data/girardi2000/`` -- no download needed to run this
example. See ``docs/guide/evolutionary_checks.md`` for the grid
choice, its real-data quirks, and the six-system validation results
this same pipeline produced.

Component literature values are Capella's (alpha Aurigae), Torres,
Claret, Pavlovski & Dotter 2015, ApJ 807, 26, Table 3 -- the same
system used for the joint orbit+photometry case study in Example 5,
here cross-checked against stellar evolution models instead of an
SED. Capella is a well-known evolved G-giant + G-subgiant pair, so
this is a real, physically expected case where the two components'
independent best-fit ages should land relatively close together, and
where at least one component pins as evolved (not a main-sequence
dwarf) -- unlike most of this package's other benchmark systems,
which are unevolved Sun-like dwarfs.

Run

    python examples/08_evolutionary_consistency_check.py
"""

from __future__ import annotations

import numpy as np

from pyseshat.analysis.luminosity import compute_log_luminosity
from pyseshat.evolution.binary import fit_binary_tracks
from pyseshat.evolution.grids import load_default_evolution_model
from pyseshat.evolution.search import nearest_track


# ==========================================================
# 1. Load the real Girardi (2000) grid bundled with the
#    package (pyseshat/evolution/data/girardi2000/).
# ==========================================================

model = load_default_evolution_model()

print()
print("Evolutionary-track consistency check")
print("=" * 70)
print(f"Grid: {model.name}, {len(model.tracks)} tracks, "
      f"{len(model.isochrones)} isochrones")
print()


# ==========================================================
# 2. Component literature values (Capella, Torres et al.
#    2015 Table 3) and masses (Torres et al. 2015 Table 1
#    SB2 orbital elements, via the same mass-from-elements
#    path Example 5/the joint case study uses).
# ==========================================================

CAPELLA_PRIMARY_TEFF_K = 4970.0
CAPELLA_PRIMARY_RADIUS_RSUN = 11.98
CAPELLA_PRIMARY_MASS_MSUN = 2.5688

CAPELLA_SECONDARY_TEFF_K = 5730.0
CAPELLA_SECONDARY_RADIUS_RSUN = 8.83
CAPELLA_SECONDARY_MASS_MSUN = 2.4828

METALLICITY = 0.019   # Girardi (2000)'s own solar value


# ==========================================================
# 3. Find each component's nearest available track by mass,
#    and its observed (log Teff, log L) from the literature
#    (Teff, R) via the Stefan-Boltzmann law.
# ==========================================================

primary_track = nearest_track(
    model, initial_mass=CAPELLA_PRIMARY_MASS_MSUN,
    metallicity=METALLICITY,
)

secondary_track = nearest_track(
    model, initial_mass=CAPELLA_SECONDARY_MASS_MSUN,
    metallicity=METALLICITY,
)

for role, requested, track in (
    ("primary", CAPELLA_PRIMARY_MASS_MSUN, primary_track),
    ("secondary", CAPELLA_SECONDARY_MASS_MSUN, secondary_track),
):
    mismatch_pct = (
        abs(track.initial_mass - requested) / requested * 100.0
    )
    print(f"{role}: requested M={requested:.3f} Msun -> nearest "
          f"track M={track.initial_mass:.3f} Msun "
          f"({mismatch_pct:.1f}% off)")

print()


# ==========================================================
# 4. Independent per-component fit, then a joint check of
#    how far apart the two independent best-fit ages land.
# ==========================================================

result = fit_binary_tracks(
    primary_track=primary_track,
    secondary_track=secondary_track,
    primary_log_teff=float(np.log10(CAPELLA_PRIMARY_TEFF_K)),
    primary_log_luminosity=compute_log_luminosity(
        radius_solar=CAPELLA_PRIMARY_RADIUS_RSUN,
        temperature_kelvin=CAPELLA_PRIMARY_TEFF_K,
    ),
    primary_sigma_log_teff=0.01,
    primary_sigma_log_luminosity=0.03,
    secondary_log_teff=float(np.log10(CAPELLA_SECONDARY_TEFF_K)),
    secondary_log_luminosity=compute_log_luminosity(
        radius_solar=CAPELLA_SECONDARY_RADIUS_RSUN,
        temperature_kelvin=CAPELLA_SECONDARY_TEFF_K,
    ),
    secondary_sigma_log_teff=0.01,
    secondary_sigma_log_luminosity=0.03,
)

print("Independent best-fit ages:")
print(f"  primary:   age={result.primary.sample.age:.3e} yr, "
      f"chi2={result.primary.chi2:.3f}, "
      f"log g={result.primary.sample.log_g:.2f}")
print(f"  secondary: age={result.secondary.sample.age:.3e} yr, "
      f"chi2={result.secondary.chi2:.3f}, "
      f"log g={result.secondary.sample.log_g:.2f}")
print()
print(f"age_difference: {result.age_difference:.3e} yr "
      f"({result.age_difference / max(result.primary.sample.age, result.secondary.sample.age) * 100.0:.1f}% "
      "of the older independent age)")
print()
print("A small age_difference relative to the age itself is the "
      "signature of a physically consistent binary; log g < 4.0 for "
      "both components here (an evolved giant + subgiant pair) is "
      "the physically expected result for the real Capella system, "
      "not a fit-quality problem -- see docs/guide/evolutionary_checks.md "
      "for the full six-system validation, including a genuine "
      "grid-spacing edge case flagged for one HD-benchmark component.")
print()
print("A joint common-age fit (evolution.age.fit_common_age) -- "
      "constraining both components to share exactly one age -- is "
      "the next stage beyond this independent check; see "
      "scripts/run_evolutionary_checks.py for that full pipeline "
      "across all six of this project's benchmark systems.")
