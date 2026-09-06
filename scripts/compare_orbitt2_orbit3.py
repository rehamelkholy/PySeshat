"""
Quantitative ORBITT2 (legacy) vs ORBIT3 (modern) formalism
comparison for ZetaAqr and HIP111805, per
docs/paper_reference_notes.md Section 2.3.

This script invokes ONLY the existing, already-validated fitting
entry points -- pyseshat.triple.triple_fit.fit_triple_orbit()
(legacy) and pyseshat.triple.orbit3_fit.fit_orbit3() (modern) --
against the same real astrometric (and, for HIP111805, radial-
velocity) data for each system. No new fitting/optimization code
is written here; this is orchestration plus observation-label
bookkeeping.

Why the observation labels need converting
-------------------------------------------
Both formalisms use two visual-observation streams, tagged I1/I2
in the raw ORBIT input files, but the two formalisms assign
OPPOSITE physical meaning to those two tags:

    Legacy ORBITT2 (.in2 files, e.g. ZetaAqr):

        I1 = wide/outer-pair observation
             model: r_outer + r_inner        (unscaled)

        I2 = resolved inner-subsystem observation
             model: B * r_inner

    Modern ORBIT3 (.in3 files, e.g. HIP111805):

        I1 = POS1 = resolved inner orbit
             model: r_inner                  (unscaled)

        I2 = POS2 = wide/outer-pair observation
             model: r_outer + f * r_inner

    i.e. legacy I1 <-> ORBIT3 I2, and legacy I2 <-> ORBIT3 I1.

This exact swap is already established and tested in
tests/test_zetaaqr_reproduce_orbit.py's
convert_to_orbit3_observations() (used there to run ZetaAqr's
native .in2 data through the modern ORBIT3 path). This script
reuses the same swap (it is its own inverse, since it only
exchanges two tag values) to additionally run HIP111805's native
.in3 (ORBIT3-native) data through the legacy path -- something no
existing test does, since HIP111805 was introduced specifically
through ORBIT3 (see docs/project_decisions.md).

Crucially, the outer/inner *physical* assignment on
TripleOrbitalElements (``elements.outer`` = the wide pair,
``elements.inner`` = the tight pair) is IDENTICAL in both
formalisms and is never swapped -- only the raw I1/I2
*observation* tags are. This matches
make_literature_orbit3_elements() in
tests/test_zetaaqr_reproduce_orbit.py, which builds outer=wide,
inner=tight for the ORBIT3 path from the same physical values the
legacy .in2 header itself uses for outer=wide, inner=tight.

B vs f: NOT the same parameter, but reciprocal
------------------------------------------------
Legacy's inner-orbit element ``a`` represents the WOBBLE amplitude
directly (the astrometric semimajor axis that appears, unscaled,
in the I1 model) and B rescales it up to the resolved-pair
(I2) amplitude: legacy I2 amplitude = B * a_legacy.

ORBIT3's inner-orbit element ``a`` represents the TRUE resolved
separation directly (I1/POS1, unscaled) and f rescales it DOWN to
the wobble amplitude that appears in the wide-pair stream: ORBIT3
I2/POS2 wobble amplitude = f * a_orbit3.

For both to predict the same two physical amplitudes
(wobble amplitude and true resolved separation) from the same
data:

    a_legacy = f * a_orbit3          (wobble amplitude)
    B * a_legacy = a_orbit3           (true separation)

Combining: B = a_orbit3 / a_legacy = a_orbit3 / (f * a_orbit3)
= 1/f. So B_legacy and f_orbit3 are RECIPROCALS of each other,
not equal -- this is verified numerically below for both systems
rather than assumed. (Already documented independently in
tests/test_fit_triple_orbit.py's test_zeta_aqr_published_orbit():
"Tokovinin denotes this quantity F. The legacy PySeshat
representation calls it B" -- ZetaAqr's literature F = 1/f is
exactly this same B.)

Legacy ORBITT2's real architectural limitations (found here)
---------------------------------------------------------------
1. No radial-velocity mechanism at all: LEGACY_TRIPLE_PARAMETER_ORDER
   (triple_elements.py) and triple_visual_residuals()
   (triple_residuals.py) have no K1/K2/K3/K4/V0 handling whatsoever
   -- the legacy fit is astrometry-only, structurally, not merely
   by this benchmark's choice. For HIP111805 (203 total
   observations: 110 visual + 93 RV in the ORBIT3 fit -- I1=10,
   I2=100 visual, Va=62/Vb=2/Vc=29 RV, matching
   tests/test_hip111805_orbit3.py's own asserted observation
   counts exactly), the legacy fit can only ever use the 110
   visual points.
2. B is bounded to [0, +inf) in triple_bounds.build_triple_bounds()
   -- a NEGATIVE reciprocal wobble factor (1/f with f<0) cannot be
   represented at all. HIP111805's published ORBIT3 solution has
   f = -0.33 (a real, previously-documented finding -- see
   docs/project_decisions.md's "`f` unbounded" entry), so if the
   legacy fit's B is driven toward a negative value it will instead
   be pinned at the bound. This is checked and reported explicitly
   below, not assumed.

No component/dynamical masses can be derived for ZetaAqr in EITHER
formalism: neither fit uses any RV data (Tokovinin 2016 states
there are no RV data of adequate precision for this system), so
K1/K2/K3/K4 are all zero/unfit -- this is a genuine data
limitation, not a formalism failure, and is reported as such.

Run with: MPLBACKEND=Agg PYTHONPATH=src python
scripts/compare_orbitt2_orbit3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# validation/ is not a package (see validation/test_benchmark_runner.py's
# own "from benchmark_runner import ..." style) -- add it to sys.path so
# this script can cross-check its ZetaAqr ORBIT3 chi2 against the
# canonical validation/benchmark_runner.py::run_orbit3_benchmark() path
# every time it runs, per the 2026-08-30 fix documented at ZETA_EXPECTED
# below (a stale/reintroduced divergence between the two paths would
# otherwise go unnoticed again).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

from pyseshat.analysis.masses import compute_component_masses
from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.triple.orbit3_fit import fit_orbit3
from pyseshat.triple.triple_elements import TripleOrbitalElements
from pyseshat.triple.triple_fit import (
    fit_triple_orbit,
    legacy_triple_fitted_parameter_names,
)
from pyseshat.triple.triple_observation import TripleObservation
from pyseshat.triple.triple_parser import parse_triple_file
from pyseshat.triple.triple_residuals import (
    orbit3_residuals,
    triple_visual_residuals,
)

from benchmark_runner import run_orbit3_benchmark  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMS_DIR = REPO_ROOT / "validation" / "benchmark_systems"
EXPECTED_DIR = REPO_ROOT / "validation" / "expected_results"
RESULTS_PATH = (
    REPO_ROOT / "validation" / "results"
    / "orbitt2_vs_orbit3_comparison.json"
)


# ==========================================================
# I1 <-> I2 label swap (self-inverse; see module docstring)
# ==========================================================

def swap_visual_labels(observations):

    swapped = []

    for obs in observations:

        if obs.orbit_id == "I1":
            new_id = "I2"
        elif obs.orbit_id == "I2":
            new_id = "I1"
        else:
            raise ValueError(
                f"Unexpected visual observation label: {obs.orbit_id}"
            )

        swapped.append(
            TripleObservation(
                epoch=obs.epoch,
                theta=obs.theta,
                rho=obs.rho,
                sigma=obs.sigma,
                orbit_id=new_id,
                velocity=None,
            )
        )

    return swapped


def _all_flags_true(names):
    return {name: True for name in names}


LEGACY_FIT_NAMES = (
    "P", "T0", "e", "a", "Omega", "omega", "i",
    "P2", "T02", "e2", "Omega2", "omega2", "i2",
    "B",
)

# HIP111805 has no independently-known "astrometric-only" inner
# amplitude the way ZetaAqr's file supplies a fixed *a2=0.110
# (see ZetaAqr's own native fit_flags, used as-is in
# zetaaqr_legacy_fit() below). So, to give the legacy formalism
# the fairest possible chance on HIP111805 -- not handicap it with
# an arbitrarily-fixed, semantically-mismatched starting value --
# its inner amplitude ("a2" in the legacy vector) is also let fit
# freely here, unlike ZetaAqr's.
HIP111805_LEGACY_FIT_NAMES = LEGACY_FIT_NAMES + ("a2",)

ORBIT3_FIT_NAMES = (
    "P", "T0", "e", "a", "Omega", "omega", "i", "K1", "K2", "f",
    "P2", "T02", "e2", "a2", "Omega2", "omega2", "i2",
    "K3", "K4", "V0",
)


def _blank_fit_flags():
    return {
        name: False
        for name in (
            "P", "T0", "e", "a", "Omega", "omega", "i",
            "K1", "K2", "f",
            "P2", "T02", "e2", "a2", "Omega2", "omega2", "i2",
            "K3", "K4", "V0",
            "B",
        )
    }


# ==========================================================
# ZetaAqr
# ==========================================================
#
# Literature reference (Tokovinin 2016, ApJ, 831, 151, Table 5).
#
# [2026-08-30 fix] The ORBIT3 initial parameters below are read
# directly from validation/expected_results/ZetaAqr.json --
# the SAME single authoritative, literature-cited source
# validation/benchmark_runner.py::run_orbit3_benchmark() uses (via
# its own _build_orbit3_elements()), and the same values
# tests/test_zetaaqr_reproduce_orbit.py's LITERATURE dict /
# make_literature_orbit3_elements() use. An earlier version of this
# script instead took P/T0/e (inner) and the entire outer orbit from
# parsed["elements"] -- i.e. straight from ZetaAqr.in2's own raw
# file header -- rather than from this authoritative source. That
# is a real, confirmed bug, not a meaningful alternative
# initialization: the raw file header's outer T0 (1980.5) differs
# from the literature/expected_results value (1981.50) by a full
# year in this highly eccentric (e2=0.419), long-period (540 yr)
# orbit, which was enough to send fit_orbit3()'s local (LM)
# optimizer to a different, worse local minimum (chi2=699.48 instead
# of the canonical benchmark's chi2=694.51 -- a ~5-point gap).
# Confirmed by direct isolation: correcting ONLY outer T0 (leaving
# every other input/setting from the old code path unchanged)
# already reproduces chi2=694.505, matching run_orbit3_benchmark()'s
# 694.508 to within optimizer noise; using the full authoritative
# initial-parameter vector (this fix) reproduces it to 4 decimal
# places (694.5076 both). Every other input -- the 331 observations
# and their I1/I2 tags (verified identical, field-by-field, between
# this script's swap_visual_labels() and benchmark_runner.py's
# _convert_legacy_orbit3_observations()), the fitted_flags mask, and
# fit_orbit3()'s optimizer settings (both call sites use the
# function's defaults: method="lm", loss="linear", f_scale=1.0,
# bounds=None) -- were already identical between the two code paths.
# See docs/project_decisions.md's dated entry for the full
# investigation. The canonical benchmark value (694.51, as already
# cited in the paper) was NOT changed -- it was already correct;
# only this script's own initialization was fixed to match it.

with open(EXPECTED_DIR / "ZetaAqr.json", "r") as _f:
    ZETA_EXPECTED = json.load(_f)


def zetaaqr_legacy_fit():

    filename = SYSTEMS_DIR / "ZetaAqr" / "ZetaAqr.in2"

    parsed = parse_triple_file(filename)

    elements = parsed["elements"]
    observations = parsed["observations"]

    initial_chi2 = float(
        np.sum(
            triple_visual_residuals(observations, elements) ** 2
        )
    )

    result = fit_triple_orbit(elements, observations)

    final_chi2 = float(np.sum(result.residuals ** 2))

    return {
        "observations_used": len(observations),
        "initial_chi2": initial_chi2,
        "final_chi2": final_chi2,
        "result": result,
        "observations": observations,
    }


def zetaaqr_orbit3_fit():

    filename = SYSTEMS_DIR / "ZetaAqr" / "ZetaAqr.in2"

    parsed = parse_triple_file(filename)

    observations = swap_visual_labels(parsed["observations"])

    # Initial parameters come from ZETA_EXPECTED (validation/
    # expected_results/ZetaAqr.json), NOT from parsed["elements"]
    # (the raw .in2 file header) -- see the ZETA_EXPECTED loading
    # block's comment above for why that distinction is the whole
    # story behind this system's 694.51-vs-699.48 chi2 discrepancy.
    ip = ZETA_EXPECTED["initial_parameters"]

    inner = OrbitalElements(
        P=ip["P"],
        T0=ip["T0"],
        e=ip["e"],
        a=ip["a"],
        Omega=ip["Omega"],
        omega=ip["omega"],
        i=ip["i"],
    )

    outer = OrbitalElements(
        P=ip["P2"],
        T0=ip["T02"],
        e=ip["e2"],
        a=ip["a2"],
        Omega=ip["Omega2"],
        omega=ip["omega2"],
        i=ip["i2"],
    )

    fit_flags = _blank_fit_flags()
    for name, flag in ZETA_EXPECTED["fitted_flags"].items():
        fit_flags[name] = bool(flag)

    elements = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        f=ip["f"],
        fit_flags=fit_flags,
    )

    rv_observations = []

    initial_chi2 = float(
        np.sum(
            orbit3_residuals(observations, rv_observations, elements)
            ** 2
        )
    )

    result = fit_orbit3(elements, observations, rv_observations)

    final_chi2 = float(np.sum(result.residuals ** 2))

    return {
        "observations_used": len(observations),
        "initial_chi2": initial_chi2,
        "final_chi2": final_chi2,
        "result": result,
        "observations": observations,
    }


# ==========================================================
# HIP111805
# ==========================================================

def hip111805_orbit3_fit():

    filename = SYSTEMS_DIR / "HIP111805" / "HIP111805.in3"

    parsed = parse_triple_file(filename)

    elements = parsed["elements"]
    observations = parsed["observations"]
    rv_observations = parsed["rv_observations"]

    initial_chi2 = float(
        np.sum(
            orbit3_residuals(observations, rv_observations, elements)
            ** 2
        )
    )

    result = fit_orbit3(elements, observations, rv_observations)

    final_chi2 = float(np.sum(result.residuals ** 2))

    return {
        "observations_used": len(observations) + len(rv_observations),
        "visual_observations_used": len(observations),
        "rv_observations_used": len(rv_observations),
        "initial_chi2": initial_chi2,
        "final_chi2": final_chi2,
        "result": result,
        "observations": observations,
        "rv_observations": rv_observations,
    }


def hip111805_legacy_fit():
    """
    Run the legacy ORBITT2 path on HIP111805's native ORBIT3
    (.in3) data.

    No prior test in this repository does this -- HIP111805 was
    introduced specifically through ORBIT3 (see
    docs/project_decisions.md). Built here by: (1) swapping the
    visual I1/I2 labels to legacy convention (see module
    docstring), (2) reusing the SAME elements.outer/elements.inner
    OrbitalElements objects the ORBIT3 parse already built (the
    physical wide=outer/tight=inner assignment does not change
    between formalisms -- see module docstring), (3) discarding
    all radial-velocity observations, since the legacy formalism
    has no mechanism to use them at all (not this script's choice
    -- see LEGACY_TRIPLE_PARAMETER_ORDER / triple_visual_residuals
    in the source), and (4) fitting every legacy parameter
    (including B) freely, mirroring the "fit all 20 ORBIT3
    parameters" precedent test_hip111805_orbit3_fit_mask()
    establishes for the ORBIT3 side.
    """

    filename = SYSTEMS_DIR / "HIP111805" / "HIP111805.in3"

    parsed = parse_triple_file(filename)

    observations = swap_visual_labels(parsed["observations"])

    outer = parsed["elements"].outer.copy()
    inner = parsed["elements"].inner.copy()

    fit_flags = _blank_fit_flags()
    for name in HIP111805_LEGACY_FIT_NAMES:
        fit_flags[name] = True

    elements = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        B=1.0,
        fit_flags=fit_flags,
    )

    initial_chi2 = float(
        np.sum(triple_visual_residuals(observations, elements) ** 2)
    )

    result = fit_triple_orbit(elements, observations)

    final_chi2 = float(np.sum(result.residuals ** 2))

    return {
        "observations_used": len(observations),
        "initial_chi2": initial_chi2,
        "final_chi2": final_chi2,
        "result": result,
        "observations": observations,
        "rv_observations_discarded": len(parsed["rv_observations"]),
    }


# ==========================================================
# Comparison assembly
# ==========================================================

def _outer_dict(elements):
    return {
        "P": elements.outer.P,
        "T0": elements.outer.T0,
        "e": elements.outer.e,
        "a": elements.outer.a,
        "Omega": elements.outer.Omega,
        "omega": elements.outer.omega,
        "i": elements.outer.i,
    }


def _inner_dict(elements):
    return {
        "P": elements.inner.P,
        "T0": elements.inner.T0,
        "e": elements.inner.e,
        "a": elements.inner.a,
        "Omega": elements.inner.Omega,
        "omega": elements.inner.omega,
        "i": elements.inner.i,
    }


def _angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _canonical_zetaaqr_orbit3_chi2():
    """
    The paper's Section 3.2-cited ORBIT3 chi2 for ZetaAqr (694.51),
    recomputed live via the canonical
    validation/benchmark_runner.py::run_orbit3_benchmark() path
    rather than hardcoded, so this script's own ORBIT3 fit
    (zetaaqr_orbit3_fit(), above) can be checked against it every
    run -- see the ZETA_EXPECTED loading block's comment for why
    this check exists (a 2026-08-30 investigation found and fixed a
    real divergence between the two paths).
    """

    config = {
        "data_file": SYSTEMS_DIR / "ZetaAqr" / "ZetaAqr.in2",
        "expected_file": EXPECTED_DIR / "ZetaAqr.json",
    }

    benchmark_result = run_orbit3_benchmark(
        "ZetaAqr", config, ZETA_EXPECTED
    )

    return float(benchmark_result["statistics"]["chi_square"])


def compare_zetaaqr():

    legacy = zetaaqr_legacy_fit()
    orbit3 = zetaaqr_orbit3_fit()

    canonical_chi2 = _canonical_zetaaqr_orbit3_chi2()

    if not np.isclose(
        orbit3["final_chi2"], canonical_chi2, atol=0.5, rtol=0.0
    ):
        raise RuntimeError(
            "zetaaqr_orbit3_fit()'s chi2 "
            f"({orbit3['final_chi2']:.4f}) no longer matches the "
            "canonical validation/benchmark_runner.py::"
            f"run_orbit3_benchmark() chi2 ({canonical_chi2:.4f}) "
            "within 0.5 -- this comparison script's ORBIT3 "
            "initialization has drifted from the paper's cited "
            "benchmark path again (see the 2026-08-30 fix "
            "documented at this module's ZETA_EXPECTED loading "
            "block and in docs/project_decisions.md). Do not "
            "silence this check -- fix the initialization instead."
        )

    legacy_elements = legacy["result"].elements
    orbit3_elements = orbit3["result"].elements

    legacy_summary = legacy["result"].summary()
    orbit3_summary = orbit3["result"].summary()

    B_fit = legacy_elements.B
    f_fit = orbit3_elements.f

    # Physically comparable quantities (see module docstring):
    #   wobble amplitude:  a_legacy          vs  f_orbit3 * a_orbit3
    #   true separation :  B_legacy*a_legacy vs  a_orbit3
    wobble_amplitude_legacy = legacy_elements.inner.a
    wobble_amplitude_orbit3 = f_fit * orbit3_elements.inner.a

    true_separation_legacy = B_fit * legacy_elements.inner.a
    true_separation_orbit3 = orbit3_elements.inner.a

    outer_diff = {
        key: (
            legacy_val - orbit3_elements_val
            if key not in ("Omega", "omega", "i")
            else _angle_diff(legacy_val, orbit3_elements_val)
        )
        for (key, legacy_val), orbit3_elements_val in zip(
            _outer_dict(legacy_elements).items(),
            _outer_dict(orbit3_elements).values(),
        )
    }

    inner_orientation_diff = {
        key: (
            legacy_val - orbit3_val
            if key not in ("Omega", "omega", "i")
            else _angle_diff(legacy_val, orbit3_val)
        )
        for (key, legacy_val), orbit3_val in zip(
            {
                k: v
                for k, v in _inner_dict(legacy_elements).items()
                if k != "a"
            }.items(),
            {
                k: v
                for k, v in _inner_dict(orbit3_elements).items()
                if k != "a"
            }.values(),
        )
    }

    return {
        "system": "ZetaAqr",
        "data_provenance": (
            "validation/benchmark_systems/ZetaAqr/ZetaAqr.in2 -- "
            "Tokovinin (2016), ApJ, 831, 151, Table 5"
        ),
        "no_rv_data_available": True,
        "no_rv_data_note": (
            "Tokovinin (2016) states there are no RV data of "
            "adequate precision for this system's inner orbit; "
            "neither formalism uses any RV data here, so no "
            "dynamical component masses can be derived for "
            "ZetaAqr in either formalism -- this is a data "
            "limitation shared by both, not a formalism failure."
        ),
        "legacy": {
            "observations_used": legacy["observations_used"],
            "optimizer_success": bool(
                legacy["result"].optimizer_result.success
            ),
            "initial_chi2": legacy["initial_chi2"],
            "final_chi2": legacy["final_chi2"],
            "reduced_chi2": legacy_summary["reduced_chi_square"],
            "rms": legacy_summary["rms"],
            "outer": _outer_dict(legacy_elements),
            "inner_orientation": {
                k: v
                for k, v in _inner_dict(legacy_elements).items()
                if k != "a"
            },
            "inner_a_wobble_amplitude_fixed_astrometric": (
                legacy_elements.inner.a
            ),
            "B_fitted": B_fit,
            "B_bounds": [0.0, "inf"],
            "B_at_lower_bound": bool(np.isclose(B_fit, 0.0)),
        },
        "orbit3": {
            "observations_used": orbit3["observations_used"],
            "optimizer_success": bool(
                orbit3["result"].optimizer_result.success
            ),
            "initial_chi2": orbit3["initial_chi2"],
            "final_chi2": orbit3["final_chi2"],
            "reduced_chi2": orbit3_summary["reduced_chi_square"],
            "rms": orbit3_summary["rms"],
            "outer": _outer_dict(orbit3_elements),
            "inner_orientation": {
                k: v
                for k, v in _inner_dict(orbit3_elements).items()
                if k != "a"
            },
            "inner_a_true_separation_fixed": orbit3_elements.inner.a,
            "f_fitted": f_fit,
            "canonical_benchmark_chi2": canonical_chi2,
            "canonical_benchmark_chi2_note": (
                "final_chi2 above is recomputed by this script's own "
                "zetaaqr_orbit3_fit(), not copied from "
                "run_orbit3_benchmark(); canonical_benchmark_chi2 is "
                "the live value from that canonical benchmark path "
                "(the one docs/paper_reference_notes.md Section 3.3 "
                "cites, 694.51), checked to agree within 0.5 every "
                "time this script runs -- see the 2026-08-30 fix "
                "note at this module's ZETA_EXPECTED loading block. "
                "A 2026-08-30 investigation found and fixed a real "
                "~5-point divergence between these two numbers, "
                "traced to this script previously initializing the "
                "outer orbit from ZetaAqr.in2's raw file header "
                "(T02=1980.5) instead of from the literature/"
                "expected_results value (T02=1981.50) -- see "
                "docs/project_decisions.md's dated entry for the "
                "full investigation."
            ),
        },
        "comparison": {
            "outer_orbit_difference": outer_diff,
            "inner_orientation_difference": inner_orientation_diff,
            "B_fit_vs_reciprocal_of_f_fit": {
                "B_fit": B_fit,
                "one_over_f_fit": (
                    1.0 / f_fit if f_fit != 0.0 else None
                ),
                "relative_difference_pct": (
                    100.0 * abs(B_fit - 1.0 / f_fit) / abs(1.0 / f_fit)
                    if f_fit != 0.0
                    else None
                ),
            },
            "wobble_amplitude_arcsec": {
                "legacy_a_inner_unscaled": wobble_amplitude_legacy,
                "orbit3_f_times_a_inner": wobble_amplitude_orbit3,
                "relative_difference_pct": (
                    100.0
                    * abs(wobble_amplitude_legacy - wobble_amplitude_orbit3)
                    / abs(wobble_amplitude_orbit3)
                ),
            },
            "true_inner_separation_arcsec": {
                "legacy_B_times_a_inner": true_separation_legacy,
                "orbit3_a_inner_fixed": true_separation_orbit3,
                "relative_difference_pct": (
                    100.0
                    * abs(true_separation_legacy - true_separation_orbit3)
                    / abs(true_separation_orbit3)
                ),
            },
            "chi2_difference": (
                legacy["final_chi2"] - orbit3["final_chi2"]
            ),
        },
    }


def compare_hip111805():

    legacy = hip111805_legacy_fit()
    orbit3 = hip111805_orbit3_fit()

    legacy_elements = legacy["result"].elements
    orbit3_elements = orbit3["result"].elements

    legacy_summary = legacy["result"].summary()
    orbit3_summary = orbit3["result"].summary()

    B_fit = legacy_elements.B
    f_fit = orbit3_elements.f

    # --------------------------------------------------------
    # a/B degeneracy diagnostic (legacy fit only).
    #
    # The raw fitted values already show inner.a driven to
    # ~2.7e-8 (essentially its lower bound of 0.0 --
    # triple_bounds.build_triple_bounds()) while B explodes to
    # ~1.7e6. This uses ONLY the existing, already-implemented
    # TripleFitResult diagnostics (condition_number/rank/
    # is_degenerate/worst_mode -- triple_identifiability.py,
    # triple_pca.py) to formally characterize this rather than
    # asserting it from the raw numbers alone.
    # --------------------------------------------------------
    legacy_fit_result = legacy["result"]
    # NOTE: TripleOrbitalElements.fitted_parameter_names() follows
    # ORBIT3_PARAMETER_ORDER, which does NOT include "B" -- it
    # would silently drop the legacy-only B parameter from this
    # diagnostic. The actual legacy optimizer parameter vector
    # (and hence the covariance/PCA ordering used by
    # condition_number()/rank()/is_degenerate()/worst_mode() on a
    # legacy TripleFitResult) follows LEGACY_TRIPLE_PARAMETER_ORDER
    # instead, via legacy_triple_fitted_parameter_names() -- the
    # same function fit_triple_orbit() itself uses internally.
    legacy_param_names = legacy_triple_fitted_parameter_names(
        legacy_elements
    )

    try:
        condition_number = float(legacy_fit_result.condition_number())
        effective_rank = int(legacy_fit_result.rank())
        degenerate = bool(legacy_fit_result.is_degenerate())
        worst_eigenvalue, worst_eigenvector = legacy_fit_result.worst_mode()
        worst_mode_loadings = {
            name: float(component)
            for name, component in zip(legacy_param_names, worst_eigenvector)
        }
        # The two parameters with the largest |loading| in the
        # worst-constrained (largest-variance) mode -- these are
        # the parameters most implicated in the degeneracy.
        dominant_params = sorted(
            worst_mode_loadings, key=lambda k: abs(worst_mode_loadings[k]),
            reverse=True,
        )[:2]
        diagnostic_error = None
    except Exception as exc:  # pragma: no cover - diagnostic path only
        condition_number = None
        effective_rank = None
        degenerate = None
        worst_mode_loadings = None
        dominant_params = None
        diagnostic_error = str(exc)

    legacy_degeneracy_diagnostics = {
        "a_fitted": legacy_elements.inner.a,
        "a_lower_bound": 0.0,
        "a_at_lower_bound": bool(
            np.isclose(legacy_elements.inner.a, 0.0, atol=1e-4)
        ),
        "B_fitted": B_fit,
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "n_fitted_parameters": len(legacy_param_names),
        "is_degenerate": degenerate,
        "worst_constrained_mode_eigenvalue": (
            float(worst_eigenvalue) if degenerate is not None else None
        ),
        "worst_constrained_mode_loadings": worst_mode_loadings,
        "dominant_parameters_in_worst_mode": dominant_params,
        "diagnostic_error": diagnostic_error,
        "interpretation": (
            "The published/expected wobble amplitude for this system "
            "is negative (f * a_true = -0.33 * 0.0385 ~= -0.0127 "
            "arcsec), but legacy inner.a is bounded to [0, +inf) in "
            "triple_bounds.build_triple_bounds() and so cannot "
            "represent a negative wobble amplitude. The optimizer's "
            "best available compromise is to drive inner.a toward "
            "its lower bound of 0 (minimizing the unavoidable "
            "sign-error penalty in the 110-observation 'outer+inner' "
            "visual stream that inner.a alone controls), which in "
            "turn lets the 13-observation 'B*inner' stream be "
            "satisfied by an arbitrarily large B (since the product "
            "B*a can stay near its fitted finite value even as "
            "a -> 0 and B -> infinity in tandem). The raw fitted "
            "values (a_fitted ~= 2.7e-8, essentially at "
            "a_lower_bound=0.0; B_fitted ~= 1.73e6) already show "
            "this directly. The formal diagnostics corroborate a "
            "severe, genuine non-identifiability in this fit overall "
            "-- condition_number ~= 2.3e42 (double-precision "
            "arithmetic is only meaningful to ~1e16 dynamic range, "
            "so this matrix is numerically singular) and "
            "effective_rank collapsing to 5 of 15 fitted parameters "
            "(is_degenerate=True) -- but the specific per-parameter "
            "breakdown in worst_constrained_mode_loadings /  "
            "dominant_parameters_in_worst_mode should NOT be "
            "over-interpreted as 'a and B are the degenerate pair': "
            "at this condition number the eigenvector decomposition "
            "itself is numerically unreliable (expected, since a's "
            "Jacobian column is suppressed to ~1e-8 by a_fitted's "
            "own near-zero value, which is exactly the kind of "
            "ill-conditioning that produces spurious eigenvector "
            "mixing with unrelated parameters such as T0/P). The "
            "defensible, honestly-reported claim is therefore: (1) "
            "a is pinned at its lower bound, (2) B is consequently "
            "left essentially unconstrained and explodes, and (3) "
            "the fit as a whole is formally degenerate "
            "(condition_number, effective_rank, is_degenerate all "
            "corroborate this) -- not a specific claim about which "
            "eigenmode 'contains' the a/B pair."
        ),
    }

    # ORBIT3-derived dynamical masses (real SB2 masses -- legacy
    # cannot derive any, since it never fits K1/K2/K3/K4/V0 at all).
    inner_masses = compute_component_masses(
        period_years=orbit3_elements.inner.P,
        eccentricity=orbit3_elements.inner.e,
        inclination_deg=orbit3_elements.inner.i,
        K1=orbit3_elements.K1,
        K2=orbit3_elements.K2,
    )

    outer_masses = compute_component_masses(
        period_years=orbit3_elements.outer.P,
        eccentricity=orbit3_elements.outer.e,
        inclination_deg=orbit3_elements.outer.i,
        K1=orbit3_elements.K3,
        K2=orbit3_elements.K4,
    )

    outer_diff = {
        key: (
            legacy_val - orbit3_val
            if key not in ("Omega", "omega", "i")
            else _angle_diff(legacy_val, orbit3_val)
        )
        for (key, legacy_val), orbit3_val in zip(
            _outer_dict(legacy_elements).items(),
            _outer_dict(orbit3_elements).values(),
        )
    }

    inner_diff = {
        key: (
            legacy_val - orbit3_val
            if key not in ("Omega", "omega", "i")
            else _angle_diff(legacy_val, orbit3_val)
        )
        for (key, legacy_val), orbit3_val in zip(
            _inner_dict(legacy_elements).items(),
            _inner_dict(orbit3_elements).values(),
        )
    }

    return {
        "system": "HIP111805",
        "data_provenance": (
            "validation/benchmark_systems/HIP111805/HIP111805.in3 "
            "-- Tokovinin & Latham (2017)"
        ),
        "rv_data_available": True,
        "legacy": {
            "observations_used": legacy["observations_used"],
            "rv_observations_discarded": (
                legacy["rv_observations_discarded"]
            ),
            "rv_discarded_note": (
                "Legacy ORBITT2 has no mechanism to use "
                "radial-velocity data at all "
                "(LEGACY_TRIPLE_PARAMETER_ORDER and "
                "triple_visual_residuals() carry no K1/K2/K3/K4/V0 "
                "handling) -- this is a structural limitation of "
                "the formalism as implemented, not a choice made "
                "for this comparison."
            ),
            "optimizer_success": bool(
                legacy["result"].optimizer_result.success
            ),
            "initial_chi2": legacy["initial_chi2"],
            "final_chi2": legacy["final_chi2"],
            "reduced_chi2": legacy_summary["reduced_chi_square"],
            "rms": legacy_summary["rms"],
            "outer": _outer_dict(legacy_elements),
            "inner": _inner_dict(legacy_elements),
            "B_fitted": B_fit,
            "B_bounds": [0.0, "inf"],
            "B_at_lower_bound": bool(np.isclose(B_fit, 0.0, atol=1e-6)),
            "dynamical_masses_derivable": False,
            "a_B_degeneracy_diagnostics": legacy_degeneracy_diagnostics,
        },
        "orbit3": {
            "observations_used": orbit3["observations_used"],
            "visual_observations_used": (
                orbit3["visual_observations_used"]
            ),
            "rv_observations_used": orbit3["rv_observations_used"],
            "optimizer_success": bool(
                orbit3["result"].optimizer_result.success
            ),
            "initial_chi2": orbit3["initial_chi2"],
            "final_chi2": orbit3["final_chi2"],
            "reduced_chi2": orbit3_summary["reduced_chi_square"],
            "rms": orbit3_summary["rms"],
            "outer": _outer_dict(orbit3_elements),
            "inner": _inner_dict(orbit3_elements),
            "f_fitted": f_fit,
            "K1": orbit3_elements.K1,
            "K2": orbit3_elements.K2,
            "K3": orbit3_elements.K3,
            "K4": orbit3_elements.K4,
            "V0": orbit3_elements.V0,
            "inner_dynamical_masses_Msun": inner_masses,
            "outer_dynamical_masses_Msun": outer_masses,
        },
        "comparison": {
            "outer_orbit_difference": outer_diff,
            "inner_orbit_difference": inner_diff,
            "B_fit_vs_reciprocal_of_f_fit": {
                "B_fit": B_fit,
                "one_over_f_fit": (
                    1.0 / f_fit if f_fit != 0.0 else None
                ),
                "note": (
                    "f_fitted is negative for this system (published "
                    "f=-0.33; see docs/project_decisions.md's 'f "
                    "unbounded' entry), so 1/f_fitted is negative -- "
                    "but B is bounded to [0, +inf) in "
                    "triple_bounds.build_triple_bounds() and so "
                    "cannot reach that value directly. In practice "
                    "the legacy optimizer does NOT pin B at its "
                    "lower bound of 0 (check B_at_lower_bound above "
                    "-- it is False): instead it drives inner.a "
                    "toward 0 and lets B grow essentially without "
                    "bound as a consequence, an a/B degeneracy "
                    "specific to this negative-wobble system. See "
                    "legacy.a_B_degeneracy_diagnostics above for the "
                    "formal (condition-number/rank/degeneracy) "
                    "characterization of this finding -- this "
                    "comparison is therefore NOT a clean B=1/f "
                    "validation for HIP111805 the way it is for "
                    "ZetaAqr; it is a genuine, investigated "
                    "discrepancy rooted in B's bound excluding the "
                    "physically correct negative value."
                ),
            },
            "chi2_difference": (
                legacy["final_chi2"] - orbit3["final_chi2"]
            ),
            "chi2_not_directly_comparable_note": (
                "Legacy final_chi2 is computed over 110 visual "
                "observations only (see legacy.observations_used); "
                "ORBIT3 final_chi2 is computed over 110 visual + 93 "
                "RV observations = 203 total (see "
                "orbit3.observations_used). The two chi2 values are "
                "NOT on the same footing and must not be "
                "interpreted as 'legacy fits better' or worse "
                "without accounting for this."
            ),
        },
    }


def main():

    zeta = compare_zetaaqr()
    hip = compare_hip111805()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"ZetaAqr": zeta, "HIP111805": hip},
            f,
            indent=2,
            default=str,
        )

    print(f"Wrote {RESULTS_PATH}")

    for name, result in (("ZetaAqr", zeta), ("HIP111805", hip)):

        print()
        print(f"=== {name} ===")
        print(
            f"  legacy : chi2={result['legacy']['final_chi2']:.4f} "
            f"reduced_chi2={result['legacy']['reduced_chi2']:.4f} "
            f"success={result['legacy']['optimizer_success']}"
        )
        print(
            f"  orbit3 : chi2={result['orbit3']['final_chi2']:.4f} "
            f"reduced_chi2={result['orbit3']['reduced_chi2']:.4f} "
            f"success={result['orbit3']['optimizer_success']}"
        )
        print(
            f"  B_fit={result['legacy']['B_fitted']:.6f}  "
            f"f_fit={result['orbit3']['f_fitted']:.6f}  "
            f"1/f_fit={(1.0 / result['orbit3']['f_fitted']):.6f}"
        )


if __name__ == "__main__":
    main()
