"""
Regression guard for the RV K1/K2 sign convention in
`orbit.rv_model.compute_rv_orbit`.

Background
----------
An earlier audit flagged this convention as
possibly backwards, reasoning from `orbital_elements.py`'s own
`to_spectroscopic_convention()` docstring (which claims omega needs
a +180 degree shift between the visual and spectroscopic
references). That reasoning turned out to be wrong for this
codebase: the checked-in FIN379_fit.json (a real prior joint
visual+RV fit, with K1/K2/V0 all free) already recovers the
literature FIN379 solution -- including omega and K1/K2 -- to
within fit uncertainty, using exactly the current (unshifted)
convention. `to_spectroscopic_convention()` is unused everywhere
and does not reflect how this codebase's ORBIT-format benchmark
files are actually solved.

This test makes that empirical resolution permanent: it fits the
literature starting point (omega taken directly from the .inp
file, un-shifted) against the real combined visual+RV data for
FIN379 and GL765, and asserts the optimizer converges close to the
published omega/K1/K2 rather than needing to shift omega by ~180
degrees or flip K1/K2 to compensate. If someone "fixes" the sign
convention or wires in `to_spectroscopic_convention()` without
re-validating against these systems, this test will fail.
"""

from pathlib import Path

import numpy as np

import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)


ROOT = Path("validation/benchmark_systems")

# (file, literature K1, literature K2, literature omega [deg])
# taken directly from each .inp file's header.
BENCHMARKS = [

    (ROOT / "FIN379" / "FIN379.inp", 7.43, 8.05, 191.0),

    (ROOT / "GL765" / "GL765.inp", 7.82, 7.52, 76.7),
]


@pytest.mark.parametrize(
    "filepath,lit_K1,lit_K2,lit_omega",
    BENCHMARKS,
    ids=[b[0].name for b in BENCHMARKS]
)
def test_rv_fit_recovers_literature_omega_and_amplitudes(
    filepath,
    lit_K1,
    lit_K2,
    lit_omega
):

    parsed = parse_orbit_file(
        str(filepath)
    )

    elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    assert len(rv_data) > 0, (
        f"{filepath.name} is expected to carry RV data "
        "for this test to be meaningful."
    )

    t = visual_data[:, 0]

    theta = np.deg2rad(visual_data[:, 1])

    rho = visual_data[:, 2]

    sigma = visual_data[:, 3]

    X = rho * np.cos(theta)

    Y = rho * np.sin(theta)

    for name in ("K1", "K2", "V0"):

        elements.fit_flags[name] = True

    fit_result = fit_visual_orbit(
        elements,
        t,
        X,
        Y,
        sigma,
        sigma,
        rv_data=rv_data
    )

    assert fit_result.optimizer_result.success

    fitted = fit_result.elements

    # ------------------------------------------------------
    # omega must land close to the literature value, NOT
    # ~180 degrees away from it. A wrap-aware difference is
    # used since omega is a circular quantity.
    # ------------------------------------------------------

    omega_diff = (
        (fitted.omega - lit_omega + 180.0) % 360.0 - 180.0
    )

    assert abs(omega_diff) < 10.0, (
        f"{filepath.name}: fitted omega={fitted.omega:.2f} deg "
        f"deviates from literature omega={lit_omega:.2f} deg "
        f"by {omega_diff:.2f} deg -- if this drifts toward "
        "~180 deg, the RV sign convention has likely changed "
        "and needs re-validation, not a silent 'fix'."
    )

    # ------------------------------------------------------
    # K1/K2 must land close to their literature values with
    # the SAME sign/labelling (not swapped).
    # ------------------------------------------------------

    assert fitted.K1 == pytest.approx(lit_K1, rel=0.05)

    assert fitted.K2 == pytest.approx(lit_K2, rel=0.05)
