"""
Regression tests for `inference.likelihood.reduced_chi_square`
against combined visual+RV benchmark data.

This file used to be a print-only script (no `assert`s, 0 items
collected by pytest) that computed a reduced chi-square for FIN379
and printed diagnostics for manual inspection. Rewritten to assert
the actual invariants it was manually eyeballing.
"""

from pathlib import Path

import numpy as np

import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.inference.likelihood import (
    chi_square,
    reduced_chi_square,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


ROOT = Path("validation/benchmark_systems")

BENCHMARKS = [
    ROOT / "FIN379" / "FIN379.inp",
    ROOT / "GL765" / "GL765.inp",
]


@pytest.mark.parametrize(
    "filepath",
    BENCHMARKS,
    ids=[f.name for f in BENCHMARKS],
)
def test_reduced_chi_square_matches_chi_square_over_dof(filepath):

    parsed = parse_orbit_file(str(filepath))

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    t = visual[:, 0]

    theta = np.deg2rad(visual[:, 1])

    rho = visual[:, 2]

    sigma = visual[:, 3]

    X_obs = rho * np.cos(theta)

    Y_obs = rho * np.sin(theta)

    fit = fit_visual_orbit(
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    params = fit.elements.fitted_parameters()

    rchisq = reduced_chi_square(
        params,
        fit.elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    chi2 = chi_square(
        params,
        fit.elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    n_visual = 2 * len(t)

    n_rv = len(rv_data)

    n_fit = np.sum(fit.elements.fitted_mask())

    expected_dof = n_visual + n_rv - n_fit

    assert expected_dof > 0

    # The exact algebraic relationship the two functions must
    # maintain -- catches either one drifting independently.
    assert rchisq == pytest.approx(chi2 / expected_dof)

    assert np.isfinite(rchisq)

    assert rchisq > 0.0

    # A converged fit to real data with real uncertainties should
    # land in a sane range, not blow up or collapse to ~0 (either
    # would indicate a broken residual/uncertainty computation).
    assert 0.01 < rchisq < 100.0


def test_reduced_chi_square_raises_on_nonpositive_dof():
    """
    Regression guard: with more free parameters than data points,
    dof <= 0 and this must raise rather than silently returning a
    meaningless value.
    """

    elements = OrbitalElements(
        P=1.0,
        T0=0.0,
        e=0.1,
        a=0.1,
        Omega=10.0,
        omega=20.0,
        i=30.0,
    )

    t = np.array([0.1])

    X_obs = np.array([0.05])

    Y_obs = np.array([0.05])

    sigma = np.array([0.01])

    with pytest.raises(ValueError):

        reduced_chi_square(
            elements.fitted_parameters(),
            elements,
            t,
            X_obs,
            Y_obs,
            sigma,
            sigma,
            rv_data=None,
        )
