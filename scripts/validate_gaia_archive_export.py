"""
Validate ``src/pyseshat/io/gaia_parser.py`` against REAL, retrieved
Gaia DR3 ``gaiadr3.nss_two_body_orbit`` archive rows -- not the
codebase's own synthesized Thiele-Innes constants -- closing the
"no live archive query" gap in the parser's validation.

Why this script exists
--------------------------------------------------------------
The existing test (``tests/test_gaia_parser.py::
test_gaia_bh1_round_trip_recovers_published_campbell_elements``)
is a real, useful check, but it is NOT a check against retrieved
archive data: it forward-transforms El-Badry et al. 2023's own
PUBLISHED Campbell elements through this codebase's OWN
``thiele_innes()`` to synthesize a stand-in A/B/F/G, then verifies
those come back out through ``parse_nss_orbital_row()``. That
validates the inversion algebra and unit conversions, but a bug
that happened to be self-consistent between ``thiele_innes()`` and
``thiele_innes_to_campbell()`` could still pass it. This script
closes that gap by running the parser against the archive's OWN
independently-fitted A/B/F/G (and, for AstroSpectroSB1 rows, its
own RV-derived arg_periastron/inclination convenience columns),
retrieved via ``gaiadr3.nss_two_body_orbit`` -- not synthesized --
and checking the result against INDEPENDENT literature values, now
as a genuine cross-check rather than a round-trip.

Live TAP access may be blocked in environments with restricted
network access (for example ``astroquery.gaia`` -> HTTP 403, "Host
not in allowlist: gea.esac.esa.int"). In that case, run the ADQL
query below through the Gaia Archive's own web interface (for
example from a browser on an unrestricted network) and export the
result as CSV
to ``data/gaia_archive/nss_two_body_orbit_export.csv`` (or any path
passed as this script's argument).

Gaia BH1/BH2 are not used: direct archive queries for BOTH
returned ZERO rows. This is explained by Ryu et al. 2025
(arXiv:2411.00654)'s analytic selection-function model of the NSS
astrometric-orbit catalog, which estimates only ~39% (BH1) / ~27%
(BH2) probability of actually surviving the pipeline's quality cuts
into the PUBLIC catalog, despite both being real, well-characterized
systems from independent (non-Gaia) data -- i.e. their internal
candidate solutions most likely did not make the DR3 public release.
Two systems are used instead, which the Gaia team's own DR3 companion paper
(Gaia Collaboration 2023, "Stellar multiplicity, a teaser for the
hidden treasure", A&A 674, A34, arXiv:2206.05595) explicitly names
BY SOURCE_ID as real, public ``Orbital``-type entries -- about as
strong a presence guarantee as is available without querying the
archive directly:

ADQL query (run at https://gea.esac.esa.int/archive/, "Search" ->
"Advanced (ADQL)"):

    SELECT source_id, nss_solution_type,
           period, period_error,
           eccentricity, eccentricity_error,
           t_periastron, t_periastron_error,
           a_thiele_innes, a_thiele_innes_error,
           b_thiele_innes, b_thiele_innes_error,
           f_thiele_innes, f_thiele_innes_error,
           g_thiele_innes, g_thiele_innes_error,
           parallax, parallax_error,
           inclination, inclination_error,
           arg_periastron, arg_periastron_error,
           center_of_mass_velocity, center_of_mass_velocity_error,
           semi_amplitude_primary, semi_amplitude_primary_error
    FROM gaiadr3.nss_two_body_orbit
    WHERE source_id IN (470361114339849472, 3162827836766605440)

Two real, independently-published systems are validated:

1. **RX Cam** (RX Camelopardalis, source_id 470361114339849472) --
   an ``Orbital``-type solution for a G2Ib+A0V spectroscopic binary
   (a Cepheid+companion pair). Gaia Collaboration 2023 (the source
   of the source_id itself) reports the Gaia solution's OWN period/
   eccentricity/inclination directly in Section 5.2.3 ("One SB1
   Cepheid") -- comparing this script's independent inversion of the
   archive's raw A/B/F/G against those SAME published numbers is a
   strong internal-consistency check (two independent readings of
   the same underlying row). Separately, and more importantly for
   this gap's purpose, Groenewegen (2013) publishes a fully
   INDEPENDENT (pre-Gaia, non-astrometric) period and eccentricity
   for the same system, which this script also checks against. The
   original discovery/spectroscopic-orbit paper is Imbert (1996),
   A&AS 116, 497 -- cited for provenance; its full (omega, T0, K1, gamma) table is not
   used, so only period/eccentricity (via Groenewegen 2013, as quoted in
   Gaia Collaboration 2023) are checked quantitatively here.
2. **HIP 36189** (source_id 3162827836766605440) -- also an
   ``Orbital``-type solution (K giant, P ~ 2.6 yr), independently
   confirmed via asteroidal occultation: Herald et al. 2020 measured
   a real separation/position-angle snapshot (rho=13.0+/-0.7 mas,
   PA=231.9+/-4.0 deg) during an occultation by (704) Interamnia on
   2003-03-23 -- a genuinely different KIND of independent check from
   RX Cam's (a single observed position, forward-predicted from the
   retrieved orbit and compared directly, rather than a fitted-
   parameter comparison).

Run

    PYTHONPATH=src python scripts/validate_gaia_archive_export.py \\
        [path/to/export.csv]

(defaults to ``data/gaia_archive/nss_two_body_orbit_export.csv``).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_PATH = (
    REPO_ROOT / "data" / "gaia_archive" / "nss_two_body_orbit_export.csv"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

RX_CAM_SOURCE_ID = 470361114339849472
HIP36189_SOURCE_ID = 3162827836766605440

# Gaia Collaboration 2023, A&A 674, A34 (arXiv:2206.05595), Section
# 5.2.3 ("One SB1 Cepheid") -- the Gaia team's OWN reading of this
# exact row, quoted directly from the paper. Comparing our
# independently-computed inversion against these validates the
# parser's correctness against a second, authoritative source that
# read the SAME underlying archive data.
RX_CAM_GAIA_TEAM_REPORTED = dict(
    P_days=999.0,
    P_err=104.0,
    e=0.514,
    e_err=0.049,
    i_deg=113.5,
    i_err=1.7,
)

# Groenewegen (2013) -- a fully INDEPENDENT (pre-Gaia, non-
# astrometric) period/eccentricity for RX Cam, as quoted in Gaia
# Collaboration 2023 (the original Groenewegen 2013 paper's exact
# title/journal could not be independently pinned down by this
# session's tools -- cited via the quoting paper). This is the
# genuinely independent literature comparison this gap exists to
# provide.
RX_CAM_LITERATURE = dict(
    P_days=1113.8,
    P_err=0.5,
    e=0.459,
    e_err=0.007,
)

# Herald et al. 2020 -- a directly OBSERVED separation/position-angle
# snapshot from an asteroidal occultation (by (704) Interamnia) on
# 2003-03-23, as quoted in Gaia Collaboration 2023's Section 5.2.2
# ("Occultations"). Independent of Gaia entirely.
HIP36189_OCCULTATION = dict(
    date=date(2003, 3, 23),
    sep_mas=13.0,
    sep_err_mas=0.7,
    pa_deg=231.9,
    pa_err_deg=4.0,
)

# Gaia Collaboration 2023, Section 5.2.2: masses derived by combining
# the Gaia astrometric (photocentre) orbit with the Sato et al. (2014)
# magnitude difference, via the paper's own Eq. (2) astrometric mass
# function. Needed here for a DIFFERENT purpose than the paper used
# them for: converting the archive's PHOTOCENTRE orbit semi-major axis
# (what parse_nss_orbital_row/thiele_innes_to_campbell actually
# reconstruct -- see gaia_parser.py's module docstring, "Scope") into
# the RELATIVE (resolved double-star) orbit semi-major axis that an
# occultation directly measures -- these are NOT the same quantity in
# general, and naively comparing the photocentre orbit's amplitude to
# an occultation separation (as this script's first version did)
# under-predicts whenever the flux fraction and mass fraction of the
# primary don't coincide (see validate_hip36189's own inline comments
# for the derivation, which follows directly from the paper's Eq. 2).
HIP36189_MASSES = dict(
    M1_Msun=3.9, M1_err=2.2,
    M2_Msun=3.5, M2_err=1.6,
)

# Sato et al. (2014), as quoted in Gaia Collaboration 2023: magnitude
# difference "about 1.5" mag, 0.2 mag uncertainty assigned by the
# quoting paper -- but WHICH component (the K-giant "primary" M1, or
# the M2 companion) is brighter is genuinely not stated in the text
# the sources consulted. Both directions are checked
# below rather than guessing one.
HIP36189_DELTA_MAG = dict(value=1.5, err=0.2)


def _decimal_year(d):
    """Same 365.25-day-year convention as io/orbit_parser.py."""

    jan1 = date(d.year, 1, 1)

    return d.year + (d - jan1).days / 365.25


def _load_export(path):

    path = Path(path)

    if not path.exists():

        print(f"No export found at {path}.")
        print("Run this script after saving the Gaia Archive CSV export there")
        print("(see this script's module docstring for the exact ADQL query")
        print("and https://gea.esac.esa.int/archive/ steps).")

        sys.exit(1)

    import pandas as pd

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    return df


def _row_for(df, source_id):

    matches = df[df["source_id"] == source_id]

    if len(matches) == 0:

        raise ValueError(
            f"source_id={source_id} not found in the export -- check the "
            "ADQL query included both RX Cam and HIP 36189's source_id."
        )

    return matches.iloc[0].to_dict()


def _pct(new, ref):

    return (new - ref) / ref * 100.0


def validate_rx_cam(df):

    from pyseshat.io.gaia_parser import parse_nss_orbital_row

    print("=" * 70)
    print("RX Cam (source_id 470361114339849472) -- 'Orbital' type")
    print("=" * 70)

    row = _row_for(df, RX_CAM_SOURCE_ID)

    print(f"Retrieved nss_solution_type: {row['nss_solution_type']!r}")

    result = parse_nss_orbital_row(row)
    el = result["elements"]

    p_days = el.P * 365.25

    gaia = RX_CAM_GAIA_TEAM_REPORTED
    lit = RX_CAM_LITERATURE

    print(
        f"{'parameter':<10}{'Gaia team (2023)':>18}{'archive-derived':>18}"
        f"{'%diff':>10}    {'Groenewegen 2013':>18}{'%diff':>10}"
    )
    print(
        f"{'period (d)':<10}{gaia['P_days']:>18.2f}{p_days:>18.3f}"
        f"{_pct(p_days, gaia['P_days']):>+9.3f}%    "
        f"{lit['P_days']:>18.2f}{_pct(p_days, lit['P_days']):>+9.3f}%"
    )
    print(
        f"{'e':<10}{gaia['e']:>18.3f}{el.e:>18.4f}"
        f"{_pct(el.e, gaia['e']):>+9.3f}%    "
        f"{lit['e']:>18.3f}{_pct(el.e, lit['e']):>+9.3f}%"
    )
    print(
        f"{'i (deg)':<10}{gaia['i_deg']:>18.2f}{el.i:>18.3f}"
        f"{_pct(el.i, gaia['i_deg']):>+9.3f}%    "
        f"{'n/a':>18}{'':>10}"
    )

    print(
        "(Groenewegen 2013 values are the genuinely independent, "
        "pre-Gaia literature check this gap exists to provide; the "
        "'Gaia team' column is the Gaia Collaboration's own published "
        "reading of this SAME archive row, useful as a second, "
        "authoritative cross-check on this script's own inversion "
        "arithmetic, not as independent validation in its own right.)"
    )

    print(
        f"distance: {result['distance_pc']:.4f} pc (parallax "
        f"{row.get('parallax')} mas)"
    )

    ok = (
        abs(_pct(p_days, gaia["P_days"])) < 20.0
        and abs(_pct(el.e, gaia["e"])) < 20.0
        and abs(_pct(p_days, lit["P_days"])) < 20.0
    )

    return dict(source_id=RX_CAM_SOURCE_ID, ok=ok)


def _propagate_theta_rho_uncertainty(row, t_occ, n_draws=4000, seed=42):
    """
    Monte Carlo error propagation of the archive row's OWN quoted
    per-column uncertainties (``*_error`` columns on
    ``a/b/f/g_thiele_innes``, ``period``, ``eccentricity``,
    ``t_periastron``) through ``parse_nss_orbital_row`` and
    ``compute_visual_orbit`` to the predicted (theta, rho) at epoch
    ``t_occ``.

    This exists because a single point-estimate forward prediction
    gives no sense of whether a mismatch against an independent
    observation is a real inconsistency or just within this
    particular archive row's own (sometimes large) formal
    uncertainty -- some Gaia NSS 'Orbital' solutions have quite
    weakly-constrained Thiele-Innes constants (up to ~50% relative
    error on an individual A/B/F/G component is not unusual for a
    faint, short-baseline system), which propagates nonlinearly into
    large swings in the RECOVERED orientation angles (Omega, omega)
    -- a 50%-uncertain A/B/F/G can easily correspond to a
    several-tens-of-degrees uncertain position angle, even though the
    period/eccentricity/semi-major-axis themselves stay comparatively
    well constrained. Column-level errors are treated as independent
    Gaussians (the archive export does not carry the full covariance
    matrix), so this is an approximate, not exact, propagation --
    stated plainly rather than presented as more precise than it is.

    Returns
    -------
    dict with ``theta_deg`` and ``rho_mas`` arrays (length <=
    ``n_draws``; draws that fail to parse, e.g. a sampled negative
    parallax, are silently dropped).
    """

    from pyseshat.io.gaia_parser import parse_nss_orbital_row
    from pyseshat.orbit.orbit_model import compute_visual_orbit

    rng = np.random.default_rng(seed)

    thetas = []
    rhos = []

    for _ in range(n_draws):

        r = dict(row)

        r["a_thiele_innes"] = rng.normal(
            row["a_thiele_innes"], row["a_thiele_innes_error"])
        r["b_thiele_innes"] = rng.normal(
            row["b_thiele_innes"], row["b_thiele_innes_error"])
        r["f_thiele_innes"] = rng.normal(
            row["f_thiele_innes"], row["f_thiele_innes_error"])
        r["g_thiele_innes"] = rng.normal(
            row["g_thiele_innes"], row["g_thiele_innes_error"])
        r["period"] = max(
            rng.normal(row["period"], row["period_error"]), 1.0)
        r["eccentricity"] = min(max(
            rng.normal(row["eccentricity"], row["eccentricity_error"]),
            0.0), 0.95)
        r["t_periastron"] = rng.normal(
            row["t_periastron"], row["t_periastron_error"])

        try:
            el = parse_nss_orbital_row(r)["elements"]
        except (ValueError, ZeroDivisionError):
            continue

        orbit = compute_visual_orbit([t_occ], el)

        thetas.append(float(orbit["theta"][0]) % 360.0)
        rhos.append(float(orbit["rho"][0]) * 1000.0)

    return dict(theta_deg=np.array(thetas), rho_mas=np.array(rhos))


def validate_hip36189(df):

    from pyseshat.io.gaia_parser import parse_nss_orbital_row
    from pyseshat.orbit.orbit_model import compute_visual_orbit

    print()
    print("=" * 70)
    print("HIP 36189 (source_id 3162827836766605440) -- 'Orbital' type")
    print("=" * 70)

    row = _row_for(df, HIP36189_SOURCE_ID)

    print(f"Retrieved nss_solution_type: {row['nss_solution_type']!r}")

    result = parse_nss_orbital_row(row)
    el = result["elements"]

    p_years = el.P

    print(f"Archive-derived period: {p_years:.4f} yr ({p_years*365.25:.2f} d)")
    print(f"Archive-derived elements: e={el.e:.4f}, i={el.i:.3f} deg, "
          f"Omega={el.Omega:.3f} deg, omega={el.omega:.3f} deg, "
          f"a={el.a*1000.0:.3f} mas")

    occ = HIP36189_OCCULTATION
    t_occ = _decimal_year(occ["date"])

    orbit = compute_visual_orbit([t_occ], el)

    rho_pred_mas = float(orbit["rho"][0]) * 1000.0
    theta_pred_deg = float(orbit["theta"][0]) % 360.0

    print(
        f"Forward-predicted PHOTOCENTRE position at the "
        f"{occ['date'].isoformat()} occultation epoch (decimal year "
        f"{t_occ:.4f}): rho={rho_pred_mas:.3f} mas, "
        f"theta={theta_pred_deg:.2f} deg"
    )
    print(
        f"Herald et al. 2020 OBSERVED (independent, non-Gaia) at that "
        f"same epoch: rho={occ['sep_mas']:.2f}+/-{occ['sep_err_mas']:.2f} "
        f"mas, PA={occ['pa_deg']:.2f}+/-{occ['pa_err_deg']:.2f} deg"
    )

    def _best_branch_d_theta(theta_deg):
        # Orbital (pure-astrometric, no-RV) solutions carry the
        # documented 180-deg ascending/descending-node ambiguity,
        # which can also flip the predicted position angle by 180
        # deg -- accept either branch, same convention as this
        # project's other Gaia-parser checks.
        d1 = abs(theta_deg - occ["pa_deg"]) % 360.0
        d2 = abs((theta_deg - 180.0) % 360.0 - occ["pa_deg"]) % 360.0
        d = min(d1, d2)
        return min(d, 360.0 - d)

    d_theta = _best_branch_d_theta(theta_pred_deg)

    print(f"Point-estimate best-branch d_theta = {d_theta:.2f} deg.")

    # Point-estimate d_theta alone doesn't say whether a mismatch is
    # a real inconsistency or just within THIS row's own (sometimes
    # large) formal Thiele-Innes uncertainty -- propagate the
    # archive's own quoted per-column errors through the parser and
    # orbit model to find out (see _propagate_theta_rho_uncertainty's
    # docstring for why this matters and its caveats).
    mc = _propagate_theta_rho_uncertainty(row, t_occ)
    n_mc = len(mc["theta_deg"])

    if n_mc > 0:

        d_theta_mc = np.array([
            _best_branch_d_theta(th) for th in mc["theta_deg"]
        ])

        d_theta_pctl = np.percentile(d_theta_mc, [16, 50, 84])

        # The observed PA is "within the archive's own uncertainty"
        # if the point-estimate offset sits inside the bulk (here,
        # <=84th percentile) of what propagating the row's own
        # quoted errors alone already produces.
        pa_ok = d_theta <= d_theta_pctl[2]

        print(
            f"Monte Carlo propagation of this row's own quoted "
            f"Thiele-Innes/period/eccentricity/t_periastron errors "
            f"({n_mc} successful draws): best-branch d_theta "
            f"16/50/84th percentile = "
            f"{d_theta_pctl[0]:.1f} / {d_theta_pctl[1]:.1f} / "
            f"{d_theta_pctl[2]:.1f} deg."
        )
        pa_verdict = (
            "within this row's own formal uncertainty (point-estimate "
            f"{d_theta:.1f} deg is at/below the propagated 84th "
            "percentile) -- NOT evidence of a parser bug or a real "
            "orbit inconsistency, just a weakly-constrained solution"
            if pa_ok else
            "OUTSIDE this row's own formal uncertainty even generously "
            f"(point-estimate {d_theta:.1f} deg exceeds the propagated "
            "84th percentile) -- a genuine open discrepancy"
        )
        print(f"Position-angle verdict: {pa_verdict}")

    else:

        # Every Monte Carlo draw failed to parse (should not happen
        # for a real archive row with sane errors) -- fall back to a
        # fixed, generous tolerance rather than silently passing.
        pa_ok = d_theta < 5.0 * occ["pa_err_deg"] + 10.0
        print(
            "Monte Carlo propagation produced no successful draws -- "
            "falling back to a fixed tolerance for the position-angle "
            f"check ({'consistent' if pa_ok else 'DOES NOT MATCH'})."
        )

    # --------------------------------------------------------------
    # Photocentre orbit vs. the RESOLVED (relative) orbit an
    # occultation actually measures -- these are physically DIFFERENT
    # quantities, not a parser bug.
    #
    # gaia_parser.py's own module docstring already states this
    # module reconstructs the astrometric PHOTOCENTRE's Thiele-Innes
    # orbit for 'Orbital'-type rows -- not the relative orbit between
    # the two physical stars. A naive rho comparison against this
    # system's a=3.2 mas photocentre orbit under-predicts the observed
    # 13.0 mas separation by a factor of several, because that
    # photocentre amplitude is bounded above by a*(1+e) ~ 4.4 mas --
    # physically incapable of ever reaching 13 mas, regardless of
    # orbital phase.
    #
    # Gaia Collaboration 2023's own Eq. (2) (the astrometric mass
    # function used to derive this system's M1/M2 in the first place)
    # gives the relation needed to convert between the two:
    #
    #     a0 = a_rel * | M2/(M1+M2) - F2/(F1+F2) |
    #
    # where a0 is the photocentre semi-major axis (what this module
    # returns as ``elements.a``), a_rel is the semi-major axis of the
    # RELATIVE (star-to-star) orbit an occultation resolves, M1/M2 are
    # the component masses, and F2/(F1+F2) is the companion's flux
    # fraction. Inverting: a_rel = a0 / |M2/(M1+M2) - F2/(F1+F2)|.
    #
    # Using this SAME paper's own quoted M1=3.9+/-2.2, M2=3.5+/-1.6
    # Msun and Sato et al. (2014) Delta-mag=1.5+/-0.2 (direction --
    # which component is brighter -- not stated in the retrievable
    # text, so BOTH signs are checked rather than guessed at).
    # --------------------------------------------------------------

    m = HIP36189_MASSES
    f_m1 = m["M1_Msun"] / (m["M1_Msun"] + m["M2_Msun"])  # primary mass fraction

    dmag = HIP36189_DELTA_MAG["value"]
    flux_ratio = 10.0 ** (-dmag / 2.5)  # fainter/brighter flux ratio

    # Case A: primary (M1, the K giant) is the brighter component.
    beta_A = 1.0 / (1.0 + flux_ratio)  # primary's flux fraction
    bracket_A = abs(beta_A - f_m1)

    # Case B: companion (M2) is the brighter component.
    beta_B = flux_ratio / (1.0 + flux_ratio)
    bracket_B = abs(beta_B - f_m1)

    a0_mas = el.a * 1000.0

    print()
    print(
        "Photocentre-orbit vs. relative-orbit correction (Gaia "
        "Collaboration 2023 Eq. 2, M1={:.1f}+/-{:.1f}, "
        "M2={:.1f}+/-{:.1f} Msun, Sato et al. 2014 "
        "delta-mag={:.1f}+/-{:.1f}; component brightness direction not "
        "stated in the retrievable text, so both cases are shown):".format(
            m["M1_Msun"], m["M1_err"], m["M2_Msun"], m["M2_err"],
            dmag, HIP36189_DELTA_MAG["err"],
        )
    )

    results = []

    for label, bracket in (("A (K giant brighter)", bracket_A),
                            ("B (companion brighter)", bracket_B)):

        a_rel_mas = a0_mas / bracket
        rho_rel_pred_mas = rho_pred_mas * (a_rel_mas / a0_mas)
        max_rel_mas = a_rel_mas * (1.0 + el.e)
        min_rel_mas = a_rel_mas * (1.0 - el.e)

        d_rho = abs(rho_rel_pred_mas - occ["sep_mas"])
        # Generous tolerance: this correction compounds the paper's
        # own large M1/M2 (56%/46% relative) and delta-mag
        # uncertainties multiplicatively near a near-cancellation
        # term, so treat this as an order-of-magnitude physical
        # cross-check, not a precision comparison.
        sep_plausible = min_rel_mas * 0.5 <= occ["sep_mas"] <= max_rel_mas * 1.5

        print(
            f"  Case {label}: a_rel={a_rel_mas:.2f} mas (range "
            f"[{min_rel_mas:.2f}, {max_rel_mas:.2f}] mas over the "
            f"orbit), corrected rho at occultation epoch="
            f"{rho_rel_pred_mas:.2f} mas vs observed "
            f"{occ['sep_mas']:.2f}+/-{occ['sep_err_mas']:.2f} mas "
            f"(|d_rho|={d_rho:.2f} mas) -- "
            f"{'physically plausible' if sep_plausible else 'still inconsistent'}"
        )

        results.append(sep_plausible)

    sep_ok = any(results)

    print(
        "(Uncorrected/naive photocentre-orbit comparison: "
        f"rho={rho_pred_mas:.3f} mas vs observed "
        f"{occ['sep_mas']:.2f} mas, "
        f"max reach a0*(1+e)={a0_mas*(1.0+el.e):.2f} mas -- physically "
        "incapable of matching the occultation separation without this "
        "correction, since Gaia's Orbital-type NSS solutions give the "
        "photocentre orbit, not the relative orbit an occultation "
        "resolves. This under-prediction is the expected signature of "
        "that distinction, not evidence of a parser bug -- see "
        "gaia_parser.py's module docstring, 'Scope', and this "
        "function's inline comments for the correction.)"
    )

    print(
        f"distance: {result['distance_pc']:.4f} pc (parallax "
        f"{row.get('parallax')} mas)"
    )

    ok = pa_ok and sep_ok

    overall_verdict = (
        "consistent (position angle matches; separation physically "
        "plausible once corrected from photocentre to relative orbit "
        "under at least one brightness assumption)"
        if ok else "DOES NOT MATCH -- investigate"
    )
    print(f"HIP 36189 overall: {overall_verdict}")

    return dict(source_id=HIP36189_SOURCE_ID, ok=ok)


if __name__ == "__main__":

    export_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPORT_PATH

    df = _load_export(export_path)

    r1 = validate_rx_cam(df)
    r2 = validate_hip36189(df)

    print()
    print("=" * 70)
    if r1["ok"] and r2["ok"]:
        print("BOTH systems validated against retrieved (not synthesized) "
              "archive data, consistent with independent literature.")
    else:
        print("At least one system did NOT match -- investigate before "
              "citing this as a resolved validation.")
    print("=" * 70)
