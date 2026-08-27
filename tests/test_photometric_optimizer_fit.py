"""
Regression tests for the unresolved-binary photometric
least-squares entry point (inference.photometric_optimizer).

Before this fix, fit_photometry_from_problem() (and therefore
fit_photometry(), the "canonical"/"legacy wrapper" Al-Wardat
SED-fitting entry points) passed
photometric_likelihood.photometric_residuals_from_problem
directly to inference.optimizer_engine.run_least_squares() as
its residual_function. scipy.optimize.least_squares always
calls residual_function(x, *args) with the optimizer's current
TRIAL parameter vector as the first argument -- but
photometric_residuals_from_problem(problem) takes a single,
already-built PhotometricInferenceProblem, not a raw parameter
vector. Every call therefore raised

    TypeError: photometric_residuals_from_problem() takes 1
    positional argument but 2 were given

immediately, before a single residual was ever evaluated. This
was never caught because nothing exercised
fit_photometry_from_problem()/fit_photometry() end-to-end
before the Al-Wardat validation work -- every existing
photometric test calls photometric_residuals_from_problem()
directly with one fixed problem (a valid call, since it never
goes through the optimizer), and residuals()/chi_square() are
unit-tested independently of the optimizer entirely. See
docs/project_decisions.md for the full write-up.

The fix adds _photometric_residuals_for_optimizer(values,
problem), which rebuilds a trial PhotometricInferenceProblem at
the optimizer's current parameter vector on every call -- the
same convention every other residual function in this package
already follows (e.g. inference.likelihood.visual_xy_residuals
takes (parameters, elements, ...)).
"""

from pathlib import Path

import numpy as np
import pytest

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)

from pyseshat.photometry.filters import (
    load_passband
)

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)

from pyseshat.photometry.parameters import (
    PhotometricParameters
)

from pyseshat.photometry.model import (
    synthetic_binary_magnitudes
)

from pyseshat.inference.photometric_optimizer import (
    _photometric_residuals_for_optimizer,
    fit_photometry,
    fit_photometry_from_problem,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem
)


# ==========================================================
# Shared fixtures
# ==========================================================

ROOT = (
    Path(__file__).resolve().parents[1]
)

ATMOSPHERE_DIR = (
    ROOT / "validation" / "atmosphere_models" / "ckp00"
)

PASSBAND_DIR = (
    ROOT / "validation" / "photometry" / "passbands"
)

# SVO Vega zero points (erg s^-1 cm^-2 Angstrom^-1), matching
# tests/test_tycho_synthetic_photometry.py and
# tests/test_ck04_photometric_chi_square.py.

ZERO_POINTS = {
    "Johnson B": 6.72553e-9,
    "Johnson V": 3.5833e-9,
}


def _grid():

    return load_ck04_grid([

        ATMOSPHERE_DIR / "ckp00_5750.fits",
        ATMOSPHERE_DIR / "ckp00_6000.fits",
        ATMOSPHERE_DIR / "ckp00_6250.fits",

    ])


def _passbands():

    return {

        "Johnson B": load_passband(
            filename=PASSBAND_DIR / "johnson_b.dat",
            name="Johnson B",
        ),

        "Johnson V": load_passband(
            filename=PASSBAND_DIR / "johnson_v.dat",
            name="Johnson V",
        ),

    }


# ==========================================================
# _photometric_residuals_for_optimizer
# ==========================================================

def test_photometric_residuals_for_optimizer_accepts_scipy_call_signature():
    """
    The residual function handed to scipy.optimize.least_squares
    must accept (values, problem) -- the exact signature scipy
    calls it with -- and return a finite residual vector one
    entry per measurement.
    """

    grid = _grid()

    passbands = _passbands()

    measurements = [
        PhotometricMeasurement(
            passband="Johnson B", magnitude=6.20, uncertainty=0.02
        ),
        PhotometricMeasurement(
            passband="Johnson V", magnitude=5.80, uncertainty=0.02
        ),
    ]

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5900.0,
        primary_radius=1.0,
        secondary_radius=0.9,
        ebv=0.0,
    )

    problem = PhotometricInferenceProblem(
        parameters=parameters,
        context=PhotometricContext(
            atmosphere_grid=grid,
            passbands=passbands,
            zero_points=ZERO_POINTS,
            measurements=measurements,
            primary_logg=4.4,
            secondary_logg=4.5,
            metallicity=0.0,
            distance=20.0,
        ),
    )

    residuals = _photometric_residuals_for_optimizer(
        parameters.fitted_parameters(),
        problem,
    )

    assert residuals.shape == (2,)

    assert np.all(
        np.isfinite(residuals)
    )


# ==========================================================
# fit_photometry_from_problem / fit_photometry
# ==========================================================

def test_fit_photometry_from_problem_runs_without_raising():
    """
    Regression guard for the TypeError described in this
    module's docstring: calling the canonical
    fit_photometry_from_problem() entry point must not raise,
    and must return a result whose optimizer actually ran
    (evaluated the model more than zero times).
    """

    grid = _grid()

    passbands = _passbands()

    measurements = [
        PhotometricMeasurement(
            passband="Johnson B", magnitude=6.20, uncertainty=0.02
        ),
        PhotometricMeasurement(
            passband="Johnson V", magnitude=5.80, uncertainty=0.02
        ),
    ]

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5900.0,
        primary_radius=1.0,
        secondary_radius=0.9,
        ebv=0.0,
    )

    problem = PhotometricInferenceProblem(
        parameters=parameters,
        context=PhotometricContext(
            atmosphere_grid=grid,
            passbands=passbands,
            zero_points=ZERO_POINTS,
            measurements=measurements,
            primary_logg=4.4,
            secondary_logg=4.5,
            metallicity=0.0,
            distance=20.0,
        ),
    )

    result = fit_photometry_from_problem(
        problem,
        bounds=(
            np.array([5750.0, 5750.0, 0.01, 0.01, 0.0]),
            np.array([6250.0, 6250.0, 100.0, 100.0, 10.0]),
        ),
    )

    assert result.optimizer_result.nfev > 0

    assert np.all(
        np.isfinite(result.residuals)
    )


def test_fit_photometry_converges_to_zero_residual_for_self_consistent_data():
    """
    End-to-end convergence test: synthesize noiseless magnitudes
    from known (Teff1, Teff2, R1, R2, ebv), perturb the initial
    guess, and confirm fit_photometry() actually searches and
    converges to a (near-)zero-residual solution. This is the
    first time this entry point has ever been exercised against
    a case with a known ground truth (see this module's
    docstring) -- a positive result here confirms the full chain
    (atmosphere interpolation -> composite SED -> extinction ->
    synthetic magnitudes -> scipy least_squares) is wired
    correctly, not just that it runs without raising.

    This test deliberately does NOT assert that the fitted
    temperatures equal the injected ones. With radius/ebv pinned
    (via a bounds range tight enough to hold them fixed -- see
    the "photometric model is under-determined with 4 passbands"
    note in docs/project_decisions.md for why only 2 free
    parameters are used at all here) and only two photometric
    bands (Johnson B, Johnson V) feeding two free temperatures,
    the problem is nominally "just constrained" (2 data, 2
    unknowns) -- but was found, while writing this test, to have
    a second, non-injected exact (chi-square ~ 0) solution
    elsewhere in (Teff1, Teff2) space (e.g. injected
    (6100 K, 5900 K) vs. a genuine alternate root near
    (5909 K, 6133 K) for the same synthetic data). This is a
    real degeneracy of two-band unresolved photometry -- distinct
    component temperature pairs can produce indistinguishable
    total-system colors -- not a code defect, so the correct
    regression guard is "did the optimizer find *a* consistent
    solution", not "did it find *the* injected one".
    """

    grid = _grid()

    passbands = _passbands()

    true_primary_teff = 6100.0

    true_secondary_teff = 5900.0

    true_primary_radius = 1.05

    true_secondary_radius = 0.95

    true_ebv = 0.05

    distance = 30.0

    primary_logg = 4.4

    secondary_logg = 4.5

    synthetic_magnitudes = synthetic_binary_magnitudes(
        atmosphere_grid=grid,
        passbands=passbands,
        zero_points=ZERO_POINTS,
        primary_temperature=true_primary_teff,
        secondary_temperature=true_secondary_teff,
        primary_logg=primary_logg,
        secondary_logg=secondary_logg,
        metallicity=0.0,
        primary_radius=true_primary_radius,
        secondary_radius=true_secondary_radius,
        distance=distance,
        ebv=true_ebv,
    )

    measurements = [
        PhotometricMeasurement(
            passband=name,
            magnitude=magnitude,
            uncertainty=0.01,
        )
        for name, magnitude in synthetic_magnitudes.items()
    ]

    # Deliberately perturbed initial guess (not the truth, and
    # not swapped between components -- a swapped start lands
    # near a competing, non-global local minimum since primary
    # and secondary have different fixed logg values).

    initial_parameters = PhotometricParameters(
        primary_temperature=true_primary_teff - 150.0,
        secondary_temperature=true_secondary_teff + 150.0,
        primary_radius=true_primary_radius,
        secondary_radius=true_secondary_radius,
        ebv=true_ebv,
    )

    epsilon = 1.0e-6

    result = fit_photometry(
        parameters=initial_parameters,
        atmosphere_grid=grid,
        passbands=passbands,
        zero_points=ZERO_POINTS,
        measurements=measurements,
        primary_logg=primary_logg,
        secondary_logg=secondary_logg,
        metallicity=0.0,
        distance=distance,
        bounds=(
            np.array([
                5750.0, 5750.0,
                true_primary_radius - epsilon,
                true_secondary_radius - epsilon,
                true_ebv - epsilon,
            ]),
            np.array([
                6250.0, 6250.0,
                true_primary_radius + epsilon,
                true_secondary_radius + epsilon,
                true_ebv + epsilon,
            ]),
        ),
    )

    assert result.optimizer_result.success

    assert np.sum(
        result.residuals ** 2
    ) < 1.0e-6

    # Radius/ebv were pinned to (effectively) their injected
    # values -- confirm they actually stayed there rather than
    # the bounds silently being ignored.

    fitted = result.elements.parameters

    assert fitted.primary_radius == pytest.approx(
        true_primary_radius, abs=1.0e-4
    )

    assert fitted.secondary_radius == pytest.approx(
        true_secondary_radius, abs=1.0e-4
    )

    assert fitted.ebv == pytest.approx(
        true_ebv, abs=1.0e-4
    )
