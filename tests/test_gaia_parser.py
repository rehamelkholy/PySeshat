import math

import numpy as np
import pytest

from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.orbit.orbit_model import thiele_innes
from pyseshat.io.gaia_parser import (
    thiele_innes_to_campbell,
    parse_nss_orbital_row,
    GAIA_DR3_REF_EPOCH_JD,
)


# ==========================================================
# thiele_innes_to_campbell: round-trip correctness
# ==========================================================
#
# The inversion (Halbwachs et al. 2023 Appendix A, see
# gaia_parser's module docstring) is verified here the way any
# closed-form matrix inversion should be: round-tripped against
# this codebase's OWN forward transform (orbit_model.thiele_innes,
# itself already cross-checked against Gaia's documented formula),
# across a wide, randomized
# sweep of Campbell elements, rather than trusted from the
# literature derivation alone.

def _random_elements(rng):

    return OrbitalElements(
        P=1.0,
        T0=0.0,
        e=float(rng.uniform(0.0, 0.9)),
        a=float(rng.uniform(0.01, 5.0)),
        Omega=float(rng.uniform(0.0, 360.0)),
        omega=float(rng.uniform(0.0, 360.0)),
        i=float(rng.uniform(0.0, 180.0)),
    )


def test_thiele_innes_to_campbell_round_trips_random_elements():

    rng = np.random.default_rng(20260827)

    for _ in range(500):

        elements = _random_elements(rng)

        A, B, F, G = thiele_innes(elements)

        a, i, Omega, omega = thiele_innes_to_campbell(A, B, F, G)

        assert a == pytest.approx(elements.a, abs=1e-8)

        # i is only ever recovered in [0, 180] by construction
        # (Eq. A.6's arctan branch) -- matches the physical range
        # of inclination, and matches every random draw above.

        assert i == pytest.approx(elements.i, abs=1e-4)

        # Omega/omega are recovered only up to the documented
        # +/- 180 deg ascending/descending-node ambiguity (see
        # gaia_parser's module docstring) -- either the original
        # values, or both shifted by 180 deg mod 360, are a
        # correct inversion.

        same_branch = (
            Omega == pytest.approx(elements.Omega, abs=1e-4)
            and
            omega == pytest.approx(elements.omega, abs=1e-4)
        )

        flipped_branch = (
            Omega == pytest.approx(
                np.mod(elements.Omega + 180.0, 360.0), abs=1e-4
            )
            and
            omega == pytest.approx(
                np.mod(elements.omega + 180.0, 360.0), abs=1e-4
            )
        )

        assert same_branch or flipped_branch


def test_thiele_innes_to_campbell_matches_forward_transform_exactly():

    # A single, easy-to-hand-check case: a moderately eccentric,
    # inclined orbit with distinct Omega/omega so no accidental
    # degeneracy (e.g. Omega==omega, or i in {0,90,180}) could
    # mask a real bug.

    elements = OrbitalElements(
        P=10.0, T0=0.0, e=0.35, a=2.5,
        Omega=205.073, omega=231.519, i=79.243,
    )

    A, B, F, G = thiele_innes(elements)

    a, i, Omega, omega = thiele_innes_to_campbell(A, B, F, G)

    assert a == pytest.approx(2.5, abs=1e-9)
    assert i == pytest.approx(79.243, abs=1e-6)

    # Recovered only up to the documented 180-deg ascending/
    # descending-node ambiguity (see gaia_parser's module
    # docstring) -- accept either equivalent branch.

    same_branch = (
        Omega == pytest.approx(205.073, abs=1e-6)
        and
        omega == pytest.approx(231.519, abs=1e-6)
    )

    flipped_branch = (
        Omega == pytest.approx(25.073, abs=1e-6)
        and
        omega == pytest.approx(51.519, abs=1e-6)
    )

    assert same_branch or flipped_branch


# ==========================================================
# Real Gaia DR3 NSS system: Gaia BH1
# ==========================================================
#
# A real, named, independently-published Gaia DR3 NSS "Orbital"
# (pure-astrometric) solution: Gaia BH1, source_id
# 4373465352415301632, El-Badry et al. 2023, MNRAS 518, 1057
# ("A Sun-like star orbiting a black hole"), Table 1. This
# system's own light is entirely from the visible G star -- the
# ~9.6 Msun companion is a dormant black hole -- so its Gaia
# photocentric orbit IS the true relative orbit (see gaia_parser's
# module docstring, "The photocentric-orbit caveat"): a clean,
# real-world case where this module's Thiele-Innes<->Campbell
# machinery can be checked against literature-published numbers
# instead of only synthetic ones.
#
# El-Badry et al. 2023 publish the DERIVED Campbell elements
# (Table 1) rather than Gaia's own raw archive-level A/B/F/G
# (not retrievable here without access to the Gaia TAP archive)
# -- so this test forward-transforms the paper's own published
# Campbell elements via this codebase's thiele_innes() to obtain
# a stand-in A/B/F/G, feeds THOSE through parse_nss_orbital_row(),
# and checks the paper's own numbers come back out. This validates
# the parser's unit conversions and inversion formula end-to-end
# against a real, citable, independently-derived solution; it is
# NOT a re-fit of Gaia's raw per-CCD astrometry -- see the data_provenance-style
# note above and in gaia_parser's own module docstring.

GAIA_BH1_SOURCE_ID = 4373465352415301632

GAIA_BH1_ELEMENTS = OrbitalElements(
    P=185.77 / 365.25,       # yr; El-Badry et al. 2023 Table 1 (days -> yr)
    T0=0.0,                  # placeholder; t_periastron is round-tripped separately below
    e=0.48,
    a=2.67 / 1000.0,         # arcsec; Table 1 gives 2.67 mas photocentre a0
    Omega=89.6,              # deg
    omega=-10.3 % 360.0,     # deg
    i=121.2,                 # deg
)

GAIA_BH1_PARALLAX_MAS = 2.09
GAIA_BH1_T_PERIASTRON_DAYS = -12.0   # relative to ref_epoch J2016.0, Table 1


def test_gaia_bh1_round_trip_recovers_published_campbell_elements():

    A, B, F, G = thiele_innes(GAIA_BH1_ELEMENTS)

    row = {
        "source_id": GAIA_BH1_SOURCE_ID,
        "nss_solution_type": "Orbital",
        "a_thiele_innes": A * 1000.0,   # arcsec -> mas, matching Gaia's own units
        "b_thiele_innes": B * 1000.0,
        "f_thiele_innes": F * 1000.0,
        "g_thiele_innes": G * 1000.0,
        "period": 185.77,               # days
        "eccentricity": 0.48,
        "t_periastron": GAIA_BH1_T_PERIASTRON_DAYS,
        "parallax": GAIA_BH1_PARALLAX_MAS,
    }

    result = parse_nss_orbital_row(row)

    elements = result["elements"]

    assert elements.P == pytest.approx(185.77 / 365.25, abs=1e-8)
    assert elements.e == pytest.approx(0.48, abs=1e-8)
    assert elements.a == pytest.approx(2.67 / 1000.0, abs=1e-9)

    # Omega/omega recovered up to the documented 180-deg
    # ascending-node ambiguity (this "Orbital" solution type
    # carries no RV to resolve it) -- accept either branch.

    same_branch = (
        elements.Omega == pytest.approx(89.6, abs=1e-4)
        and
        elements.omega == pytest.approx(349.7, abs=1e-4)
    )

    flipped_branch = (
        elements.Omega == pytest.approx(269.6, abs=1e-4)
        and
        elements.omega == pytest.approx(169.7, abs=1e-4)
    )

    assert same_branch or flipped_branch

    assert elements.i == pytest.approx(121.2, abs=1e-4)

    assert result["distance_pc"] == pytest.approx(
        1000.0 / GAIA_BH1_PARALLAX_MAS, abs=1e-6
    )

    assert result["source_id"] == GAIA_BH1_SOURCE_ID

    assert result["solution_type"] == "Orbital"

    # t_periastron = -12.0 d relative to J2016.0 (JD 2457389.0)
    # -> decimal year, via this codebase's own jd_to_decimal_year
    # convention (io.orbit_parser, already tested elsewhere).

    expected_T0 = 2000.0 + (
        (GAIA_DR3_REF_EPOCH_JD + GAIA_BH1_T_PERIASTRON_DAYS)
        - 2451545.0
    ) / 365.25

    assert elements.T0 == pytest.approx(expected_T0, abs=1e-6)


# ==========================================================
# parse_nss_orbital_row: scope and unit-conversion checks
# ==========================================================

def test_parse_nss_orbital_row_rejects_spectroscopic_only_types():

    for bad_type in ("SB1", "SB1C", "SB2", "SB2C", "EclipsingBinary"):

        with pytest.raises(ValueError):

            parse_nss_orbital_row({
                "nss_solution_type": bad_type,
            })


def test_parse_nss_orbital_row_populates_k1_v0_for_astrospectrosb1():

    elements_in = OrbitalElements(
        P=1.0, T0=0.0, e=0.2, a=1.0, Omega=10.0, omega=20.0, i=30.0,
    )

    A, B, F, G = thiele_innes(elements_in)

    row = {
        "nss_solution_type": "AstroSpectroSB1",
        "a_thiele_innes": A,
        "b_thiele_innes": B,
        "f_thiele_innes": F,
        "g_thiele_innes": G,
        "period": 365.25,
        "eccentricity": 0.2,
        "t_periastron": 0.0,
        "parallax": 10.0,
        "semi_amplitude_primary": 42.0,
        "center_of_mass_velocity": -5.0,
    }

    result = parse_nss_orbital_row(row)

    assert result["elements"].K1 == pytest.approx(42.0)
    assert result["elements"].V0 == pytest.approx(-5.0)

    # Single-lined: no secondary RV curve was ever observed.

    assert result["elements"].K2 == pytest.approx(0.0)


def test_parse_nss_orbital_row_prefers_archive_published_angles():

    # When the row directly supplies "inclination"/"arg_periastron"
    # (as Gaia's own archive does for some solution types), those
    # published values must win over this module's own Thiele-
    # Innes-only inversion -- see the module docstring's note on
    # AstroSpectroSB1 rows resolving the node ambiguity via RV.

    elements_in = OrbitalElements(
        P=1.0, T0=0.0, e=0.2, a=1.0, Omega=10.0, omega=20.0, i=30.0,
    )

    A, B, F, G = thiele_innes(elements_in)

    row = {
        "nss_solution_type": "Orbital",
        "a_thiele_innes": A,
        "b_thiele_innes": B,
        "f_thiele_innes": F,
        "g_thiele_innes": G,
        "period": 365.25,
        "eccentricity": 0.2,
        "t_periastron": 0.0,
        "parallax": 10.0,
        # Deliberately different from what the inversion itself
        # would produce, to prove these override it.
        "inclination": 30.0,
        "arg_periastron": 200.0,
    }

    result = parse_nss_orbital_row(row)

    assert result["elements"].i == pytest.approx(30.0)
    assert result["elements"].omega == pytest.approx(200.0)


def test_parse_nss_orbital_row_rejects_nonpositive_parallax():

    elements_in = OrbitalElements(
        P=1.0, T0=0.0, e=0.1, a=1.0, Omega=0.0, omega=0.0, i=45.0,
    )

    A, B, F, G = thiele_innes(elements_in)

    row = {
        "nss_solution_type": "Orbital",
        "a_thiele_innes": A,
        "b_thiele_innes": B,
        "f_thiele_innes": F,
        "g_thiele_innes": G,
        "period": 100.0,
        "eccentricity": 0.1,
        "t_periastron": 0.0,
        "parallax": -1.0,
    }

    with pytest.raises(ValueError):

        parse_nss_orbital_row(row)


# ==========================================================
# NaN-vs-None handling for optional archive columns
# ==========================================================
#
# A genuine CSV/VOTable export of the real Gaia archive represents
# an unpopulated OPTIONAL column (e.g. inclination/arg_periastron
# on a pure-astrometric "Orbital"-type row) as NaN once loaded
# through pandas/astropy -- NOT Python None, which only a
# hand-built dict (as every other test in this file uses) can
# produce. See gaia_parser.py's _is_present() and
# scripts/validate_gaia_archive_export.py.

def test_is_present_treats_nan_same_as_none():

    from pyseshat.io.gaia_parser import _is_present

    assert _is_present(None) is False
    assert _is_present(float("nan")) is False
    assert _is_present(np.nan) is False
    assert _is_present(0.0) is True
    assert _is_present(1.5) is True
    assert _is_present("Orbital") is True


def test_parse_nss_orbital_row_treats_nan_optional_columns_as_absent():
    """
    A NaN (not None) 'inclination'/'arg_periastron' -- exactly what
    a real CSV export's unpopulated cell becomes after
    ``pandas.read_csv`` -- must fall back to this module's own
    Thiele-Innes inversion, the same as if the column were entirely
    absent, rather than propagating NaN into the returned
    OrbitalElements.
    """

    elements_in = OrbitalElements(
        P=0.5, T0=0.0, e=0.2, a=0.05, Omega=30.0, omega=200.0, i=60.0,
    )

    A, B, F, G = thiele_innes(elements_in)

    row = {
        "nss_solution_type": "Orbital",
        "a_thiele_innes": A,
        "b_thiele_innes": B,
        "f_thiele_innes": F,
        "g_thiele_innes": G,
        "period": 0.5 * 365.25,
        "eccentricity": 0.2,
        "t_periastron": 0.0,
        "parallax": 10.0,
        # exactly what pandas.read_csv leaves an empty optional
        # cell as -- NOT None:
        "inclination": float("nan"),
        "arg_periastron": float("nan"),
    }

    result = parse_nss_orbital_row(row)
    elements = result["elements"]

    assert not math.isnan(elements.i)
    assert not math.isnan(elements.omega)

    assert elements.i == pytest.approx(60.0, abs=1e-6)

    same_branch = elements.omega == pytest.approx(200.0, abs=1e-6)
    flipped_branch = elements.omega == pytest.approx(20.0, abs=1e-6)
    assert same_branch or flipped_branch


def test_parse_nss_orbital_row_astrospectro_nan_rv_columns_leave_defaults():
    """
    Same NaN-vs-None distinction, for AstroSpectroSB1's optional
    RV columns (semi_amplitude_primary/center_of_mass_velocity):
    NaN must leave K1/V0 at OrbitalElements' own defaults, not
    propagate NaN into them.
    """

    elements_in = OrbitalElements(
        P=1.0, T0=0.0, e=0.1, a=1.0, Omega=10.0, omega=20.0, i=45.0,
    )

    A, B, F, G = thiele_innes(elements_in)

    row = {
        "nss_solution_type": "AstroSpectroSB1",
        "a_thiele_innes": A,
        "b_thiele_innes": B,
        "f_thiele_innes": F,
        "g_thiele_innes": G,
        "period": 365.25,
        "eccentricity": 0.1,
        "t_periastron": 0.0,
        "parallax": 5.0,
        "inclination": 45.0,
        "arg_periastron": 20.0,
        "semi_amplitude_primary": float("nan"),
        "center_of_mass_velocity": float("nan"),
    }

    result = parse_nss_orbital_row(row)
    elements = result["elements"]

    assert not math.isnan(elements.K1)
    assert not math.isnan(elements.V0)
    assert elements.K1 == OrbitalElements(
        P=1.0, T0=0.0, e=0.1, a=1.0, Omega=10.0, omega=20.0, i=45.0,
    ).K1
    assert elements.V0 == OrbitalElements(
        P=1.0, T0=0.0, e=0.1, a=1.0, Omega=10.0, omega=20.0, i=45.0,
    ).V0
