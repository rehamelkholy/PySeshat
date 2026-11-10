"""
Regression tests for the unresolved-binary photometric Bayesian
(MCMC) entry point (inference.photometric_mcmc).

Same defect class as tests/test_photometric_optimizer_fit.py's
target, in the MCMC sibling: sample_photometry_from_problem()
(and therefore sample_photometry(), the binary photometric
posterior-sampling entry point) passed
photometric_posterior.photometric_log_posterior_from_problem
directly to inference.mcmc_engine.run_emcee() as its
log_probability_function. emcee's EnsembleSampler always calls
log_probability_function(x, *args) with each walker's current
parameter vector as the first argument -- but
photometric_log_posterior_from_problem(problem) takes a single,
already-built PhotometricInferenceProblem, not a raw parameter
vector. Every walker evaluation therefore raised

    TypeError: photometric_log_posterior_from_problem() takes 1
    positional argument but 2 were given

before a single MCMC step was ever taken. See
tests/test_photometric_optimizer_fit.py (the optimizer
and MCMC entry points share the same bug pattern and the same
fix pattern: a small wrapper that rebuilds a trial
PhotometricInferenceProblem at the sampler's/optimizer's current
parameter vector on every call).
"""

from pathlib import Path

import numpy as np

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

from pyseshat.inference.photometric_context import (
    PhotometricContext
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem
)

from pyseshat.inference.photometric_mcmc import (
    _photometric_log_posterior_for_sampler,
    sample_photometry_from_problem,
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)


ROOT = (
    Path(__file__).resolve().parents[1]
)

ATMOSPHERE_DIR = (
    ROOT / "validation" / "atmosphere_models" / "ckp00"
)

PASSBAND_DIR = (
    ROOT / "validation" / "photometry" / "passbands"
)

ZERO_POINTS = {
    "Johnson B": 6.72553e-9,
    "Johnson V": 3.5833e-9,
}


def _problem():

    grid = load_ck04_grid([

        ATMOSPHERE_DIR / "ckp00_5750.fits",
        ATMOSPHERE_DIR / "ckp00_6000.fits",
        ATMOSPHERE_DIR / "ckp00_6250.fits",

    ])

    passbands = {

        "Johnson B": load_passband(
            filename=PASSBAND_DIR / "johnson_b.dat",
            name="Johnson B",
        ),

        "Johnson V": load_passband(
            filename=PASSBAND_DIR / "johnson_v.dat",
            name="Johnson V",
        ),

    }

    measurements = [
        PhotometricMeasurement(
            passband="Johnson B", magnitude=7.08, uncertainty=0.02
        ),
        PhotometricMeasurement(
            passband="Johnson V", magnitude=6.37, uncertainty=0.02
        ),
    ]

    parameters = PhotometricParameters(
        primary_temperature=6100.0,
        secondary_temperature=5900.0,
        primary_radius=1.05,
        secondary_radius=0.95,
        ebv=0.05,
    )

    return PhotometricInferenceProblem(
        parameters=parameters,
        context=PhotometricContext(
            atmosphere_grid=grid,
            passbands=passbands,
            zero_points=ZERO_POINTS,
            measurements=measurements,
            primary_logg=4.4,
            secondary_logg=4.5,
            metallicity=0.0,
            distance=30.0,
        ),
    )


def test_photometric_log_posterior_for_sampler_accepts_emcee_call_signature():
    """
    The log-probability function handed to emcee must accept
    (values, problem) -- the exact signature emcee calls it
    with -- and return a finite scalar.
    """

    problem = _problem()

    log_posterior = _photometric_log_posterior_for_sampler(
        problem.parameters.fitted_parameters(),
        problem,
    )

    assert np.isfinite(log_posterior)


def test_log_posterior_returns_neg_inf_outside_loaded_grid_range():
    """
    Regression guard for a real crash found while running MCMC
    exploration during SED benchmark validation: a walker proposing a
    temperature inside the generic physical prior (2500-50000K) but
    outside the specific CK04 FITS files this problem's atmosphere_grid
    has loaded (here, 5750-6250K) used to raise ValueError from
    AtmosphereGrid.interpolate(), propagating up through emcee and
    killing the whole sampler run instead of being treated as an
    unreachable (zero-probability) region.
    """

    problem = _problem()

    out_of_grid_values = np.array([
        7000.0,  # primary_temperature: above the loaded 6250K edge
        5900.0,  # secondary_temperature: unchanged, inside range
        1.05,    # primary_radius: unchanged
        0.95,    # secondary_radius: unchanged
        0.05,    # ebv: unchanged
    ])

    log_posterior = _photometric_log_posterior_for_sampler(
        out_of_grid_values,
        problem,
    )

    assert log_posterior == -np.inf


def test_sample_photometry_from_problem_runs_without_raising():
    """
    Regression guard for the TypeError described in this
    module's docstring: calling the canonical
    sample_photometry_from_problem() entry point must not raise,
    and must return a self-consistent MCMCResult (small walker/
    step counts -- this is a wiring smoke test, not a
    convergence test).
    """

    problem = _problem()

    n_walkers = 12

    n_steps = 15

    burn_in = 3

    result = sample_photometry_from_problem(
        problem,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=1,
        rng=np.random.default_rng(12345),
    )

    assert isinstance(result, MCMCResult)

    assert result.samples.shape == (
        n_walkers * (n_steps - burn_in),
        5,
    )

    assert np.all(
        np.isfinite(result.samples)
    )

    assert np.all(
        np.isfinite(result.log_probability)
    )

    assert 0.0 < result.acceptance_fraction <= 1.0
