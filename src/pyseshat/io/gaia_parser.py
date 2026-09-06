"""
Parse Gaia DR3 Non-Single-Star (NSS) two-body-orbit solutions
(the ``gaiadr3.nss_two_body_orbit`` archive table) into this
codebase's ``OrbitalElements``.

Scope (2026-08-27)
--------------------------------------------------------------
Gaia DR3's NSS orbital catalogue covers several distinct
``nss_solution_type`` values, each with a different fitted
parametrization (see
``gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/
chap_datamodel/sec_dm_non--single_stars_tables/
ssec_dm_nss_two_body_orbit.html``, the archive's own column-by-
column data model documentation, cross-checked 2026-08-27):

- ``Orbital`` / ``OrbitalAlternative`` / ``OrbitalAlternativeValidated``
  / ``OrbitalTargetedSearch`` / ``OrbitalTargetedSearchValidated``:
  pure ASTROMETRIC solutions -- the photocentre's Thiele-Innes
  constants (A, B, F, G, in mas), period, eccentricity, and
  t_periastron are fitted; NO radial-velocity information is
  part of the Gaia solution itself. THIS is the type this module
  parses (``parse_nss_orbital_row`` / ``gaia_row_to_orbital_elements``
  below).
- ``AstroSpectroSB1``: astrometric Thiele-Innes constants A, B, F, G
  PLUS a single-lined spectroscopic orbit (``semi_amplitude_primary``,
  ``center_of_mass_velocity``) fitted jointly. This module also
  handles this type -- K1/V0 are populated from the row when
  present, K2 is left at its ``OrbitalElements`` default of 0.0
  (single-lined: the companion's own RV curve was never observed).
- ``SB1``/``SB1C``/``SB2``/``SB2C``: pure SPECTROSCOPIC solutions,
  no astrometric Thiele-Innes elements at all -- there is nothing
  for this module (which builds a *visual*-orbit-capable
  ``OrbitalElements``) to parse; ``parse_nss_orbital_row`` raises
  ``ValueError`` for these rather than silently returning a
  physically meaningless all-zero visual orbit.
- ``EclipsingBinary``/``EclipsingSpectro``: photometric eclipse-
  model parametrization (fill factors, radii sums), not a
  Keplerian element set at all -- also raises ``ValueError``.

Thiele-Innes -> Campbell inversion
--------------------------------------------------------------
Gaia's own archive table stores the FITTED Thiele-Innes constants
(A, B, F, G) directly; it does NOT provide the semi-major axis or
longitude of the ascending node as their own columns (only
``inclination`` and ``arg_periastron`` are given as separately
derived convenience columns for some solution types -- see the
data-model page cited above, and El-Badry et al. 2023, MNRAS 518,
1057, Table 1, which likewise derives -- rather than reads off the
archive -- the semi-major axis and ascending node for Gaia BH1).
This module therefore implements the full A,B,F,G -> (a, i, Omega,
omega) inversion itself, following the same overall method as
Halbwachs et al. 2023 (Gaia DR3 astrometric binary star
processing, A&A 674, A9, arXiv:2206.05726), Appendix A, Eqs.
A.2-A.6 -- the paper describing Gaia's own NSS astrometric-binary
pipeline (semi-major axis via their u/v construction, Eq. A.2;
Omega+/-omega via arctan of the same A+G/B-F and A-G/B+F
combinations, Eqs. A.3-A.4). This module's own exact algebraic
form (in particular for inclination, where a self-contained
P/M-ratio derivation is used in place of the paper's d1/d2
branching -- see ``thiele_innes_to_campbell``'s inline comment)
was independently re-derived from ``orbit_model.thiele_innes()``'s
own forward formula rather than transcribed sign-for-sign from
the paper (whose exact equations this session could only access
through a summarized secondary reading of the PDF, not the
typeset original), and its CORRECTNESS is established the
authoritative way for this codebase: cross-verified (2026-08-27,
see ``tests/test_gaia_parser.py``) by round-tripping it against
``orbit.orbit_model.thiele_innes()`` (this codebase's OWN A,B,F,G
forward formula, itself independently cross-checked against Gaia's
own documented formula in Sect. 7.2.5 of the GDR3 documentation --
see ``docs/project_decisions.md``): for many random Campbell
element sets, computing A,B,F,G via ``thiele_innes()`` and then
inverting them via this module's ``thiele_innes_to_campbell()``
recovers the original (a, i, Omega, omega) to within floating-
point precision.

180-degree (Omega, omega) <-> (Omega+180, omega+180) ambiguity
--------------------------------------------------------------
A, B, F, G are UNCHANGED under the simultaneous replacement
Omega -> Omega+180, omega -> omega+180 (this is a real, well-known
property of the Thiele-Innes parametrization, not a bug in this
module -- see e.g. Heintz 1978, "Double Stars", or any standard
visual-orbit reference): astrometry (or a photocentric orbit)
alone cannot distinguish between the ascending and descending node
without independent information (typically radial velocity, which
tells you which node the star is receding from vs. approaching
through). This module resolves the ambiguity by the same
convention the Washington Double Star Catalog and this codebase's
own ``.inp``-parsed orbits implicitly use: it does not attempt to
guess -- it returns whichever of the two equivalent (Omega, omega)
solutions the inversion formula's principal branches naturally
produce (both are equally valid orbit-shape solutions; they
describe the same physical ellipse), and states this explicitly
in ``parse_nss_orbital_row``'s return so a caller with independent
RV-sign information can flip by 180 deg if needed. For
``AstroSpectroSB1`` solutions (which DO carry RV via
``semi_amplitude_primary``), Gaia's own pipeline already resolves
this ambiguity internally before publishing ``inclination`` and
``arg_periastron`` as their own columns -- this module prefers
those archive-published values over its own inversion's arg_periastron
whenever they are present in the row (see ``parse_nss_orbital_row``).

Units and conventions
--------------------------------------------------------------
Gaia's archive units (mas for A/B/F/G and the derived semi-major
axis, days for period/t_periastron) are converted here to this
codebase's own ``OrbitalElements`` conventions (arcsec, years) --
see ``orbit.orbital_elements.OrbitalElements``'s own docstring and
every existing ``.inp``-derived benchmark system for the target
convention. ``t_periastron`` is given by Gaia relative to
``gaia_source.ref_epoch`` (J2016.0 for the whole nss_two_body_orbit
table, i.e. JD 2457389.0) in the range [-period/2, +period/2] --
see the data-model page cited above; this module adds it to
J2016.0 and reuses ``io.orbit_parser.jd_to_decimal_year`` (this
codebase's own, already-tested reduced-JD-to-decimal-year
convention) for the final conversion to a decimal year ``T0``.

The photocentric-orbit caveat
--------------------------------------------------------------
For a pure ``Orbital`` solution, A/B/F/G (and hence the derived
semi-major axis) describe the PHOTOCENTRE's orbit around the
system barycentre, not necessarily the two stars' relative-
separation orbit that ``orbit.orbit_model.compute_visual_orbit``
otherwise models throughout this codebase (see every ``.inp``-
derived benchmark, where ``a`` is the resolved pair's true
angular separation scale). These coincide EXACTLY only when the
companion contributes zero light to the combined flux (so the
photocentre sits exactly on the visible star, whose reflex motion
around the barycentre IS then the full relative orbit, scaled only
by the mass ratio) -- true for a genuinely dark companion (a
compact-object companion such as Gaia BH1's black hole, or a very
faint late-type/degenerate secondary), false in general for two
comparably-luminous resolved stars, where the photocentre sits
somewhere between them and its wobble amplitude is suppressed by
the light-ratio-dependent factor ``(1 - beta)`` documented in
standard astrometric-binary references (``beta`` = the secondary's
fractional light contribution). This module does NOT attempt that
general light-fraction correction (it is not knowable from the
NSS orbit table alone -- it needs an independently known flux
ratio) and returns the semi-major axis exactly as fitted (the
photocentre's own a0), documenting this caveat rather than
silently mislabeling it as a resolved relative-orbit scale.
"""

from __future__ import annotations

import math

import numpy as np

from pyseshat.orbit.orbital_elements import OrbitalElements

from pyseshat.io.orbit_parser import jd_to_decimal_year


def _is_present(value):
    """
    True if ``value`` is a real, populated optional field -- False
    for Python ``None`` AND for NaN (``float('nan')``,
    ``numpy.nan``, pandas' own NaN) alike.

    This distinction matters because a genuine CSV/VOTable export
    of the real Gaia archive represents an unpopulated OPTIONAL
    column (e.g. ``inclination``/``arg_periastron`` on a pure-
    astrometric ``Orbital``-type row, which Gaia's own pipeline
    never fits for that solution type) as NaN when loaded through
    pandas/astropy, NOT Python ``None`` -- a hand-built dict (as
    this module's own tests originally used) can omit a key or set
    it to ``None``, but a real archive export cannot. Checking only
    ``row.get(key) is not None`` therefore silently accepts NaN as
    "present" and lets it propagate into a NaN ``OrbitalElements``
    field for every downstream calculation -- confirmed as a real,
    reproducible bug (2026-09-01) via a CSV round-trip smoke test
    performed while preparing this module's first genuine
    retrieved-archive-data validation (see
    ``scripts/validate_gaia_archive_export.py`` and
    ``docs/project_decisions.md``'s dated entry), before any real
    archive data was in hand -- i.e. caught proactively, not
    discovered by a downstream NaN making it into a paper figure.
    """

    if value is None:

        return False

    try:

        return not math.isnan(value)

    except TypeError:

        # Non-float, non-None (e.g. a string) -- treat as present;
        # the caller's own float()/int() conversion will raise a
        # clear error if it's genuinely unusable.

        return True


# ==========================================================
# Constants
# ==========================================================

# Gaia DR3's adopted reference epoch for the whole
# nss_two_body_orbit table (J2016.0), as a full Julian date --
# see the data-model page cited in this module's docstring.
GAIA_DR3_REF_EPOCH_JD = 2457389.0

MAS_PER_ARCSEC = 1000.0

# nss_solution_type values this module can build a Keplerian
# OrbitalElements from (all Thiele-Innes-parametrized).
_SUPPORTED_SOLUTION_TYPES = (
    "Orbital",
    "OrbitalAlternative",
    "OrbitalAlternativeValidated",
    "OrbitalTargetedSearch",
    "OrbitalTargetedSearchValidated",
    "AstroSpectroSB1",
)


# ==========================================================
# Thiele-Innes -> Campbell inversion
# ==========================================================

def thiele_innes_to_campbell(A, B, F, G):
    """
    Invert Thiele-Innes constants (A, B, F, G) to the Campbell
    elements (a, i, Omega, omega) they encode.

    Implements Halbwachs et al. 2023 (A&A 674, A9,
    arXiv:2206.05726), Appendix A, Eqs. A.2-A.6 -- see this
    module's docstring for the full derivation cross-check and
    the 180-degree ascending/descending-node ambiguity this
    inversion cannot resolve on its own.

    Parameters
    ----------
    A, B, F, G : float
        Thiele-Innes constants, in any consistent length unit
        (this codebase always passes mas, matching Gaia's own
        column units, but the inversion itself is unit-agnostic
        -- ``a`` is returned in the same unit as A/B/F/G).

    Returns
    -------
    a : float
        Semi-major axis, same unit as A/B/F/G.

    i : float
        Inclination, degrees, in [0, 180].

    Omega : float
        Longitude of the ascending node, degrees, in [0, 360).

    omega : float
        Argument of periastron, degrees, in [0, 360).
    """

    A = float(A)
    B = float(B)
    F = float(F)
    G = float(G)

    # ------------------------------------------------------
    # Semi-major axis (Eq. A.2)
    # ------------------------------------------------------

    u = 0.5 * (A**2 + B**2 + F**2 + G**2)

    v = A * G - B * F

    # u >= |v| always (Cauchy-Schwarz on the underlying rotation
    # matrix) -- guard against a tiny negative radicand from
    # floating-point roundoff only.

    radicand = max(u**2 - v**2, 0.0)

    a = np.sqrt(u + np.sqrt(radicand))

    # ------------------------------------------------------
    # omega +/- Omega (Eqs. A.3-A.4)
    # ------------------------------------------------------
    #
    # atan2 (not atan) is used so the full [-180, 180) branch is
    # resolved directly from the actual signs of the numerator/
    # denominator, rather than needing the paper's separate
    # "mod pi" + sign-correction bookkeeping.

    sum_angle = np.rad2deg(
        np.arctan2(B - F, A + G)
    )

    diff_angle = np.rad2deg(
        np.arctan2(B + F, A - G)
    )

    Omega = 0.5 * (sum_angle + diff_angle)

    omega = 0.5 * (sum_angle - diff_angle)

    # ------------------------------------------------------
    # Inclination.
    #
    # Derived directly from the same A,B,F,G forward formula
    # this codebase's own orbit_model.thiele_innes() implements
    # (see this module's docstring): grouping terms gives
    #
    #     A+G = a(1+cos i) cos(Omega+omega)
    #     B-F = a(1+cos i) sin(Omega+omega)
    #     A-G = a(1-cos i) cos(Omega-omega)
    #     B+F = a(1-cos i) sin(Omega-omega)
    #
    # so P := sqrt((A+G)^2 + (B-F)^2) = a(1+cos i) and
    # M := sqrt((A-G)^2 + (B+F)^2) = a(1-cos i) (both manifestly
    # >= 0, since 1 +/- cos i in [0, 2] and a > 0), giving
    # cos i = (P-M)/(P+M) directly -- algebraically equivalent to
    # Halbwachs et al. 2023 Eqs. A.5-A.6's d1/d2 formulation, but
    # this form is what's actually verified by this module's own
    # round-trip tests (tests/test_gaia_parser.py).
    # ------------------------------------------------------

    P_term = np.sqrt((A + G) ** 2 + (B - F) ** 2)

    M_term = np.sqrt((A - G) ** 2 + (B + F) ** 2)

    cos_i = (
        (P_term - M_term) / (P_term + M_term)
        if (P_term + M_term) > 0.0
        else 1.0
    )

    cos_i = np.clip(cos_i, -1.0, 1.0)

    i = np.rad2deg(np.arccos(cos_i))

    return (
        float(a),
        float(np.mod(i, 360.0)),
        float(np.mod(Omega, 360.0)),
        float(np.mod(omega, 360.0)),
    )


# ==========================================================
# nss_two_body_orbit row -> OrbitalElements
# ==========================================================

def parse_nss_orbital_row(row):
    """
    Convert one ``gaiadr3.nss_two_body_orbit`` row (any mapping
    with the archive's own column names -- a dict, a pandas
    Series, or an astropy Table row all work) into this
    codebase's ``OrbitalElements`` plus the system's distance.

    Parameters
    ----------
    row : mapping
        Must provide (at minimum) ``nss_solution_type``,
        ``a_thiele_innes``, ``b_thiele_innes``, ``f_thiele_innes``,
        ``g_thiele_innes``, ``period`` (days), ``eccentricity``,
        ``t_periastron`` (days, relative to J2016.0), and
        ``parallax`` (mas). For ``AstroSpectroSB1`` rows,
        ``semi_amplitude_primary`` (km/s) and
        ``center_of_mass_velocity`` (km/s) are also read, when
        present, into K1/V0.

    Returns
    -------
    dict with keys:
        "elements" : OrbitalElements
            P (yr), T0 (decimal year), e, a (arcsec), Omega
            (deg), omega (deg), i (deg), and K1/V0 (km/s) when
            this is an AstroSpectroSB1 row with RV columns
            present (K2 always left at the OrbitalElements
            default of 0.0 -- see this module's docstring).
        "distance_pc" : float
            1000 / parallax_mas -- the standard small-angle
            parallax-to-distance conversion, NOT a full
            Bayesian distance estimate (see Bailer-Jones 2015
            for why that matters at low parallax S/N; this
            module does not attempt that correction).
        "source_id" : the row's ``source_id``, if present, else
            None.
        "solution_type" : the row's ``nss_solution_type``.
        "ascending_node_ambiguity" : True
            Always True for these solution types -- see this
            module's docstring, "180-degree ... ambiguity".

    Raises
    ------
    ValueError
        If ``nss_solution_type`` is a purely spectroscopic
        (SB1/SB1C/SB2/SB2C) or eclipsing-photometric solution
        type, which carries no Thiele-Innes astrometric elements
        this module could parse into a visual-orbit-capable
        ``OrbitalElements`` -- see this module's docstring,
        "Scope".
    """

    solution_type = row["nss_solution_type"]

    if solution_type not in _SUPPORTED_SOLUTION_TYPES:

        raise ValueError(

            f"nss_solution_type={solution_type!r} carries no "
            "Thiele-Innes astrometric orbit for this module to "
            "parse (supported types: "
            f"{', '.join(_SUPPORTED_SOLUTION_TYPES)}). "
            "Purely spectroscopic (SB1/SB1C/SB2/SB2C) and "
            "eclipsing-photometric solution types are out of "
            "scope -- see gaia_parser's module docstring."

        )

    A = float(row["a_thiele_innes"])
    B = float(row["b_thiele_innes"])
    F = float(row["f_thiele_innes"])
    G = float(row["g_thiele_innes"])

    a_mas, i_deg, Omega_deg, omega_from_inversion = (
        thiele_innes_to_campbell(A, B, F, G)
    )

    # Prefer the archive's own directly-published arg_periastron/
    # inclination columns over this module's own inversion when
    # they are present -- for AstroSpectroSB1 rows in particular,
    # Gaia's pipeline has already used the RV sign to resolve the
    # ascending/descending-node ambiguity this module's own
    # Thiele-Innes-only inversion cannot (see this module's
    # docstring).

    omega_deg = float(
        row["arg_periastron"]
    ) if (
        _is_present(row.get("arg_periastron"))
    ) else omega_from_inversion

    if _is_present(row.get("inclination")):

        i_deg = float(row["inclination"])

    period_days = float(row["period"])

    eccentricity = float(row["eccentricity"])

    t_periastron_days = float(row["t_periastron"])

    t_periastron_jd_full = (
        GAIA_DR3_REF_EPOCH_JD
        +
        t_periastron_days
    )

    T0 = jd_to_decimal_year(
        t_periastron_jd_full - 2400000.0
    )

    P_years = period_days / 365.25

    a_arcsec = a_mas / MAS_PER_ARCSEC

    kwargs = dict(
        P=P_years,
        T0=T0,
        e=eccentricity,
        a=a_arcsec,
        Omega=Omega_deg,
        omega=omega_deg,
        i=i_deg,
    )

    if solution_type == "AstroSpectroSB1":

        if _is_present(row.get("semi_amplitude_primary")):

            kwargs["K1"] = float(
                row["semi_amplitude_primary"]
            )

        if _is_present(row.get("center_of_mass_velocity")):

            kwargs["V0"] = float(
                row["center_of_mass_velocity"]
            )

    elements = OrbitalElements(**kwargs)

    parallax_mas = float(row["parallax"])

    if parallax_mas <= 0.0:

        raise ValueError(

            "Non-positive parallax "
            f"({parallax_mas} mas) cannot be converted to a "
            "distance."

        )

    distance_pc = 1000.0 / parallax_mas

    return {

        "elements": elements,

        "distance_pc": distance_pc,

        "source_id": row.get("source_id"),

        "solution_type": solution_type,

        "ascending_node_ambiguity": True,

    }
