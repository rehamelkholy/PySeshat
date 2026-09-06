"""
Example 7: Parsing a Gaia DR3 non-single-star (NSS) orbital solution.

Gaia DR3's ``gaiadr3.nss_two_body_orbit`` table publishes astrometric
(and, for some solution types, spectroscopic) binary-star orbits as
Thiele-Innes constants (A, B, F, G) rather than the Campbell elements
(a, i, Omega, omega) this codebase fits and reports everywhere else.
``pyseshat.io.gaia_parser`` inverts Thiele-Innes back to Campbell
elements (Halbwachs et al. 2023, Appendix A) and returns a plain
``OrbitalElements`` instance, so a Gaia archive row slots directly
into the same plotting/reporting/comparison machinery used throughout
this package.

This example uses a real, published, citable system rather than a
synthetic one: Gaia BH1 (source_id 4373465352415301632), El-Badry
et al. 2023, MNRAS 518, 1057 ("A Sun-like star orbiting a black
hole"), Table 1 -- the ~9.6 Msun companion is a dormant black hole
contributing zero light, so this system's Gaia photocentric orbit IS
the true relative orbit (see gaia_parser's module docstring, "The
photocentric-orbit caveat", for when that equivalence does and does
not hold in general).

El-Badry et al. 2023 publish the *derived* Campbell elements rather
than Gaia's own raw archive-level Thiele-Innes constants, so -- to
keep this example runnable with no network access or archive
credentials -- the paper's own published elements are forward-
transformed through this codebase's own ``thiele_innes()`` to obtain
a stand-in A/B/F/G "as if" read from the archive, which are then fed
through ``parse_nss_orbital_row()`` exactly as a real CSV/VOTable
export's columns would be. This round-trips the paper's own published
numbers back out, demonstrating the parser end to end; it is not a
re-fit of Gaia's raw per-CCD astrometry. For pulling real rows from
the live archive yourself, see ``scripts/validate_gaia_archive_export.py``
and ``docs/guide/data_formats.md``'s Gaia section.

Run

    python examples/07_gaia_dr3_nss_parsing.py
"""

from __future__ import annotations

from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.orbit.orbit_model import thiele_innes
from pyseshat.io.gaia_parser import (
    GAIA_DR3_REF_EPOCH_JD,
    parse_nss_orbital_row,
)


# ==========================================================
# 1. Gaia BH1's published Campbell elements (El-Badry et al.
#    2023, Table 1), used here only to build a stand-in
#    archive row -- see the module docstring above.
# ==========================================================

GAIA_BH1_SOURCE_ID = 4373465352415301632

published_elements = OrbitalElements(
    P=185.77 / 365.25,        # yr; Table 1 gives days
    T0=0.0,                   # placeholder -- t_periastron is a separate archive column
    e=0.48,
    a=2.67 / 1000.0,          # arcsec; Table 1's 2.67 mas photocentre a0
    Omega=89.6,                # deg
    omega=-10.3 % 360.0,       # deg
    i=121.2,                   # deg
)

t_periastron_days = -12.0      # relative to Gaia's ref_epoch J2016.0
parallax_mas = 2.09

A, B, F, G = thiele_innes(published_elements)


# ==========================================================
# 2. Build a row shaped exactly like one from a real
#    gaiadr3.nss_two_body_orbit CSV/VOTable export.
# ==========================================================

archive_row = {
    "source_id": GAIA_BH1_SOURCE_ID,
    "nss_solution_type": "Orbital",
    "a_thiele_innes": A * 1000.0,   # arcsec -> mas, Gaia's own unit
    "b_thiele_innes": B * 1000.0,
    "f_thiele_innes": F * 1000.0,
    "g_thiele_innes": G * 1000.0,
    "period": 185.77,               # days
    "eccentricity": 0.48,
    "t_periastron": t_periastron_days,
    "parallax": parallax_mas,
}


# ==========================================================
# 3. Parse it -- this is the actual public API this example
#    demonstrates.
# ==========================================================

result = parse_nss_orbital_row(archive_row)

elements = result["elements"]


# ==========================================================
# 4. Report
# ==========================================================

print()
print("Gaia DR3 NSS orbit parse")
print("=" * 70)
print(f"source_id            {result['source_id']}")
print(f"solution_type        {result['solution_type']}")
print(f"distance (1/parallax) {result['distance_pc']:.2f} pc")
print()
print("Recovered Campbell elements:")
print(f"  P      {elements.P * 365.25:.2f} d  "
      f"({elements.P:.4f} yr)")
print(f"  e      {elements.e:.4f}")
print(f"  a      {elements.a * 1000.0:.2f} mas")
print(f"  i      {elements.i:.2f} deg")
print(f"  Omega  {elements.Omega:.2f} deg")
print(f"  omega  {elements.omega:.2f} deg")
print(f"  T0     {elements.T0:.4f} yr (decimal year, "
      f"reference epoch JD {GAIA_DR3_REF_EPOCH_JD})")
print()
print("Compare to El-Badry et al. 2023 (MNRAS 518, 1057), Table 1: "
      "P=185.77 d, e=0.48, a0=2.67 mas, i=121.2 deg, Omega=89.6 deg, "
      "omega=-10.3 deg (mod 360 = 349.7 deg).")
print()
print("Note: Omega/omega are only recoverable up to the documented "
      "+/-180 deg ascending/descending-node ambiguity for a pure "
      "'Orbital' (astrometry-only) solution type -- see "
      "gaia_parser's module docstring. 'AstroSpectroSB1' rows carry "
      "an RV-derived arg_periastron/inclination that resolves this "
      "ambiguity instead, and parse_nss_orbital_row() prefers those "
      "archive-published values over its own inversion whenever "
      "they're present.")
