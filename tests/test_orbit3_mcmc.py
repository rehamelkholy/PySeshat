"""
Tests for the triple/ORBIT3 Bayesian (MCMC) entry point.

Before this module existed, no MCMC/posterior-inference path
was wired up for triple/ORBIT3 orbits at all --
``inference.mcmc_engine`` was only ever exercised through the
binary/single-orbit path (``inference.optimizer.
fit_visual_orbit_mcmc``). These tests cover the new
ORBIT3-specific prior (``triple.orbit3_priors.
log_prior_orbit3``), likelihood/posterior
(``triple.orbit3_likelihood``), and the
``triple.orbit3_fit.fit_orbit3_mcmc`` entry point itself,
using the same ZetaAqr and HIP111805 benchmark fixtures as
``tests/test_zetaaqr_reproduce_orbit.py`` and
``tests/test_hip111805_orbit3.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_observation import (
    TripleObservation
)

from pyseshat.triple.triple_fit import (
    orbit3_to_array
)

from pyseshat.triple.triple_residuals import (
    orbit3_residuals
)

from pyseshat.triple.orbit3_priors import (
    log_prior_orbit3
)

from pyseshat.triple.orbit3_likelihood import (
    log_likelihood_orbit3,
    log_posterior_orbit3
)

from pyseshat.triple.orbit3_fit import (
    fit_orbit3,
    fit_orbit3_mcmc
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)


# ==========================================================
# Fixtures
# ==========================================================

ROOT = (
    Path(__file__).resolve().parents[1]
)

ZETA_FILE = (
    ROOT
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)


def _convert_legacy_orbit3_observations(observations):
    """
    Local copy of the ZetaAqr I1/I2 legacy-tag swap (see
    tests/test_zetaaqr_reproduce_orbit.py::
    convert_to_orbit3_observations and
    validation/benchmark_runner.py::
    _convert_legacy_orbit3_observations for the full
    rationale): ZetaAqr.in2's I1/I2 tags are opposite to the
    ORBIT3 convention used by orbit3_residuals().
    """

    converted = []

    for obs in observations:

        if obs.orbit_id == "I1":
            orbit_id = "I2"
        elif obs.orbit_id == "I2":
            orbit_id = "I1"
        else:
            raise ValueError(obs.orbit_id)

        converted.append(
            TripleObservation(
                epoch=obs.epoch,
                theta=obs.theta,
                rho=obs.rho,
                sigma=obs.sigma,
                orbit_id=orbit_id,
                velocity=obs.velocity
            )
        )

    return converted


def zetaaqr_elements():
    """
    ORBIT3 elements for ZetaAqr, matching
    tests/test_zetaaqr_reproduce_orbit.py::
    make_literature_orbit3_elements(): the inner semimajor
    axis is fixed (not fitted), and there is no usable RV
    data.
    """

    inner = OrbitalElements(
        P=25.95, T0=2006.52, e=0.872, a=0.385,
        Omega=293.7, omega=100.9, i=11.8
    )

    outer = OrbitalElements(
        P=540.0, T0=1981.50, e=0.419, a=3.496,
        Omega=131.3, omega=269.3, i=142.0
    )

    elements = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        f=1.0 / 3.50
    )

    for name in (
        "P", "T0", "e", "Omega", "omega", "i", "f",
        "P2", "T02", "e2", "a2", "Omega2", "omega2", "i2",
    ):
        elements.fit_flags[name] = True

    return elements


def zetaaqr_data():

    parsed = parse_triple_file(ZETA_FILE)

    observations = _convert_legacy_orbit3_observations(
        parsed["observations"]
    )

    return observations, []


# ==========================================================
# log_prior_orbit3
# ==========================================================

def test_log_prior_orbit3_finite_for_valid_elements():

    elements = zetaaqr_elements()

    assert np.isfinite(
        log_prior_orbit3(elements)
    )


def test_log_prior_orbit3_rejects_eccentricity_out_of_range():

    elements = zetaaqr_elements()

    elements.inner.e = 1.5

    assert log_prior_orbit3(elements) == -np.inf


def test_log_prior_orbit3_rejects_negative_semi_amplitude_when_fitted():

    elements = zetaaqr_elements()

    elements.fit_flags["K1"] = True

    elements.K1 = -1.0

    assert log_prior_orbit3(elements) == -np.inf


def test_log_prior_orbit3_ignores_unfitted_negative_semi_amplitude():
    """
    K1/K2/K3/K4 only constrain the prior when they are fitted
    parameters -- fixed (unfitted) values, including the
    all-zero defaults used by RV-less systems like ZetaAqr,
    must not make the prior -inf.
    """

    elements = zetaaqr_elements()

    assert elements.fit_flags["K1"] is False

    assert np.isfinite(
        log_prior_orbit3(elements)
    )


def test_log_prior_orbit3_allows_negative_wobble_factor():
    """
    f must remain unrestricted: the published HIP111805
    solution has f = -0.330 (see orbit3_bounds.
    orbit3_full_bounds() for the identical reasoning on the
    deterministic-optimizer side).
    """

    elements = zetaaqr_elements()

    elements.f = -0.330

    assert np.isfinite(
        log_prior_orbit3(elements)
    )


# ==========================================================
# log_likelihood_orbit3 / log_posterior_orbit3
# ==========================================================

def test_log_likelihood_orbit3_matches_orbit3_residuals():

    elements = zetaaqr_elements()

    observations, rv_observations = zetaaqr_data()

    fitted_values = orbit3_to_array(
        elements
    )[
        elements.fitted_mask()
    ]

    expected = -0.5 * np.sum(
        orbit3_residuals(
            observations,
            rv_observations,
            elements
        ) ** 2
    )

    actual = log_likelihood_orbit3(
        fitted_values,
        elements,
        observations,
        rv_observations
    )

    assert actual == pytest.approx(expected)


def test_log_posterior_orbit3_short_circuits_outside_prior_support():

    elements = zetaaqr_elements()

    observations, rv_observations = zetaaqr_data()

    fitted_values = orbit3_to_array(
        elements
    )[
        elements.fitted_mask()
    ]

    # The "e" (inner eccentricity) parameter is fitted and
    # is the 3rd entry of the fitted vector for this fit mask
    # (P, T0, e, ...) -- push it out of [0, 0.999999).

    fitted_values = fitted_values.copy()

    e_index = elements.fitted_parameter_names().index("e")

    fitted_values[e_index] = 5.0

    assert log_posterior_orbit3(
        fitted_values,
        elements,
        observations,
        rv_observations
    ) == -np.inf


def test_log_posterior_orbit3_equals_prior_plus_likelihood_inside_support():

    elements = zetaaqr_elements()

    observations, rv_observations = zetaaqr_data()

    fitted_values = orbit3_to_array(
        elements
    )[
        elements.fitted_mask()
    ]

    expected = (
        log_prior_orbit3(elements)
        +
        log_likelihood_orbit3(
            fitted_values,
            elements,
            observations,
            rv_observations
        )
    )

    actual = log_posterior_orbit3(
        fitted_values,
        elements,
        observations,
        rv_observations
    )

    assert actual == pytest.approx(expected)


# ==========================================================
# fit_orbit3_mcmc
# ==========================================================

def test_fit_orbit3_mcmc_returns_consistent_mcmc_result():
    """
    End-to-end smoke test: fit_orbit3_mcmc() must run to
    completion on a real hierarchical-triple benchmark system
    (ZetaAqr) and return a self-consistent MCMCResult -- this
    does not attempt to assert MCMC convergence (the walker
    count and step count below are deliberately small, purely
    for test runtime), only that the ORBIT3/MCMC wiring itself
    is correct.
    """

    elements = zetaaqr_elements()

    observations, rv_observations = zetaaqr_data()

    n_fitted = len(
        elements.fitted_parameter_names()
    )

    n_walkers = 2 * n_fitted + 4

    result = fit_orbit3_mcmc(
        elements,
        observations,
        rv_observations,
        n_walkers=n_walkers,
        n_steps=25,
        burn_in=5,
        thin=1,
        rng=np.random.default_rng(12345)
    )

    assert isinstance(
        result,
        MCMCResult
    )

    assert result.n_parameters == n_fitted

    assert result.samples.shape == (
        n_walkers * (25 - 5),
        n_fitted
    )

    assert result.parameter_names == (
        elements.fitted_parameter_names()
    )

    assert np.all(
        np.isfinite(
            result.samples
        )
    )

    assert np.all(
        np.isfinite(
            result.log_probability
        )
    )

    assert 0.0 < result.acceptance_fraction <= 1.0

    # The posterior median must land in the same physical
    # neighborhood as the least-squares solution used to seed
    # the sampler -- not an exact match (25 steps is far too
    # short to converge), just a sanity bound against a wired-
    # up-wrong likelihood/prior sending the chain somewhere
    # unphysical.

    deterministic = fit_orbit3(
        elements,
        observations,
        rv_observations
    )

    deterministic_values = orbit3_to_array(
        deterministic.elements
    )[
        deterministic.elements.fitted_mask()
    ]

    assert np.allclose(
        result.posterior_median,
        deterministic_values,
        rtol=0.2,
        atol=1.0
    )


def test_log_posterior_orbit3_finite_far_from_optimum_at_negative_f():
    """
    Regression guard for the wobble factor's unrestricted
    prior/posterior support: unlike an out-of-range
    eccentricity (which must make the posterior exactly
    -inf, per test_log_posterior_orbit3_short_circuits_
    outside_prior_support), a large negative f is a poor fit
    to ZetaAqr's data (whose true f is positive) but is not
    outside ORBIT3's parameter support -- the published
    HIP111805 solution has f = -0.330, and there is no
    physical upper/lower bound on f anywhere in this
    formalism (see orbit3_bounds.orbit3_full_bounds()). The
    posterior at such a point must therefore be finite (just
    low), never -inf.
    """

    elements = zetaaqr_elements()

    observations, rv_observations = zetaaqr_data()

    fitted_values = orbit3_to_array(
        elements
    )[
        elements.fitted_mask()
    ]

    f_index = elements.fitted_parameter_names().index("f")

    fitted_values = fitted_values.copy()

    fitted_values[f_index] = -5.0

    logp = log_posterior_orbit3(
        fitted_values,
        elements,
        observations,
        rv_observations
    )

    assert np.isfinite(logp)

    # It should also be a substantially worse (lower) posterior
    # than the literature solution, confirming the forward
    # model actually responds to f rather than ignoring it.

    logp_literature = log_posterior_orbit3(
        orbit3_to_array(elements)[elements.fitted_mask()],
        elements,
        observations,
        rv_observations
    )

    assert logp < logp_literature
