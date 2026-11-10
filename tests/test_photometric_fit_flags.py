"""
Regression tests for PhotometricParameters' fit-flags/Gaussian-
prior mechanism, mirroring
OrbitalElements.fit_flags -- see photometry/parameters.py's
class docstring. This
mechanism replaces a near-zero-width-bounds "pinning" workaround: a parameter can now
be genuinely held fixed (fit_flags[name]=False, simply absent
from the optimizer's parameter vector) or freed with a soft
Gaussian prior tying it to an independent literature estimate,
instead of always being either fully free or bounds-pinned to a
single point.
"""

import numpy as np

import pytest

from pyseshat.photometry.parameters import (
    PhotometricParameters,
    update_parameters,
)

from pyseshat.inference.priors import (
    log_photometric_prior,
    log_gaussian_prior,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
)

from pyseshat.inference.photometric_likelihood import (
    photometric_prior_residuals_from_problem,
    photometric_residuals_with_priors_from_problem,
)

from pyseshat.inference.photometric_optimizer import (
    build_parameter_bounds,
)


def _partially_fixed_parameters():

    return PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
        fit_flags={
            "primary_temperature": True,
            "secondary_temperature": True,
            "primary_radius": False,
            "secondary_radius": False,
            "ebv": True,
        },
    )


# ==========================================================
# Masks
# ==========================================================

def test_fitted_mask_and_fixed_mask():

    parameters = _partially_fixed_parameters()

    np.testing.assert_array_equal(
        parameters.fitted_mask(),
        np.array([True, True, False, False, True]),
    )

    np.testing.assert_array_equal(
        parameters.fixed_mask(),
        np.array([False, False, True, True, False]),
    )


def test_default_fit_flags_are_all_true():

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
    )

    assert np.all(
        parameters.fitted_mask()
    )


# ==========================================================
# fitted_parameters() / fitted_parameter_names()
# ==========================================================

def test_fitted_parameters_returns_only_fitted_subset():

    parameters = _partially_fixed_parameters()

    np.testing.assert_allclose(
        parameters.fitted_parameters(),
        np.array([6000.0, 5000.0, 0.1]),
    )


def test_fitted_parameter_names_respects_fit_flags():

    parameters = _partially_fixed_parameters()

    assert parameters.fitted_parameter_names() == [
        "primary_temperature",
        "secondary_temperature",
        "ebv",
    ]


# ==========================================================
# update_parameters() with a partial (fitted-only) vector
# ==========================================================

def test_update_parameters_partial_vector_preserves_fixed_values():

    parameters = _partially_fixed_parameters()

    updated = update_parameters(
        parameters,
        np.array([6200.0, 5100.0, 0.15]),
    )

    assert updated.primary_temperature == 6200.0
    assert updated.secondary_temperature == 5100.0
    assert updated.ebv == 0.15

    # Fixed parameters carried over UNCHANGED from the template,
    # even though they were not present in the update vector at
    # all.

    assert updated.primary_radius == 1.2
    assert updated.secondary_radius == 0.9

    assert updated.fit_flags == parameters.fit_flags


def test_update_parameters_wrong_length_for_partial_fit_flags():

    parameters = _partially_fixed_parameters()

    with pytest.raises(
        ValueError,
        match="exactly 3 values",
    ):

        update_parameters(
            parameters,
            np.ones(5),
        )


def test_update_parameters_all_fixed_accepts_empty_vector():

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
        fit_flags={name: False for name in PhotometricParameters.PARAMETER_ORDER},
    )

    updated = update_parameters(
        parameters,
        np.array([]),
    )

    assert updated.to_array().tolist() == parameters.to_array().tolist()


# ==========================================================
# log_photometric_prior: fixed parameters skip their bound-prior
# ==========================================================

def test_log_photometric_prior_skips_fixed_parameter_bounds():

    # A physically-impossible NEGATIVE radius would normally send
    # the Jeffreys bound-prior to -inf -- but only if that
    # parameter is actually being explored. Holding it fixed
    # means it isn't part of the fit, so its (nonsensical, never
    # reached by the optimizer/sampler) value shouldn't zero out
    # the whole posterior.

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=-1.0,
        secondary_radius=0.9,
        ebv=0.1,
        fit_flags={
            "primary_temperature": True,
            "secondary_temperature": True,
            "primary_radius": False,
            "secondary_radius": True,
            "ebv": True,
        },
    )

    logp = log_photometric_prior(parameters)

    assert np.isfinite(logp)


def test_log_photometric_prior_backward_compatible_with_no_fit_flags_attribute():

    # A bare object without fit_flags/priors (lacking the
    # fit-flags mechanism) must still be handled: every bound-
    # prior applies, matching the original unconditional
    # behavior.

    class LegacyParameters:
        primary_temperature = 6000.0
        secondary_temperature = 5000.0
        primary_radius = 1.2
        secondary_radius = 0.9
        ebv = 0.1

    logp = log_photometric_prior(LegacyParameters())

    assert np.isfinite(logp)


# ==========================================================
# Gaussian prior
# ==========================================================

def test_log_gaussian_prior_peaks_at_mean():

    at_mean = log_gaussian_prior(5.0, mean=5.0, sigma=0.5)

    off_mean = log_gaussian_prior(6.0, mean=5.0, sigma=0.5)

    assert at_mean > off_mean


def test_log_gaussian_prior_rejects_nonpositive_sigma():

    with pytest.raises(ValueError):

        log_gaussian_prior(5.0, mean=5.0, sigma=0.0)


def test_log_photometric_prior_includes_gaussian_prior_contribution():

    base = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
    )

    with_prior = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
        priors={"primary_radius": (1.2, 0.05)},
    )

    off_prior = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.5,
        secondary_radius=0.9,
        ebv=0.1,
        priors={"primary_radius": (1.2, 0.05)},
    )

    # Evaluated exactly at the prior mean, the Gaussian prior
    # contributes its peak (least negative) value -- so the
    # "with_prior" object's log-prior should exceed the
    # equivalent object with no prior term at all by exactly the
    # Gaussian's peak log-density, and should exceed the
    # off-mean object by a large margin (0.3 away at sigma=0.05
    # is 6 sigma).

    logp_base = log_photometric_prior(base)
    logp_with_prior = log_photometric_prior(with_prior)
    logp_off_prior = log_photometric_prior(off_prior)

    assert logp_with_prior > logp_off_prior

    assert logp_with_prior == pytest.approx(
        logp_base + log_gaussian_prior(1.2, 1.2, 0.05)
    )


# ==========================================================
# Prior pseudo-residuals (least-squares path)
# ==========================================================

def _problem_with_priors(priors):

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5000.0,
        primary_radius=1.2,
        secondary_radius=0.9,
        ebv=0.1,
        priors=priors,
    )

    context = PhotometricContext(
        atmosphere_grid=None,
        passbands={},
        zero_points={},
        measurements=[],
        primary_logg=4.5,
        secondary_logg=4.5,
        metallicity=0.0,
        distance=10.0,
    )

    return PhotometricInferenceProblem(
        parameters=parameters,
        context=context,
    )


def test_prior_residuals_empty_when_no_priors():

    problem = _problem_with_priors({})

    residuals = photometric_prior_residuals_from_problem(problem)

    assert residuals.shape == (0,)


def test_prior_residuals_zero_at_the_mean():

    problem = _problem_with_priors(
        {"primary_radius": (1.2, 0.05)}
    )

    residuals = photometric_prior_residuals_from_problem(problem)

    np.testing.assert_allclose(residuals, [0.0])


def test_prior_residuals_standardized_by_sigma():

    problem = _problem_with_priors(
        {"primary_radius": (1.0, 0.1)}
    )

    residuals = photometric_prior_residuals_from_problem(problem)

    # value=1.2, mean=1.0, sigma=0.1 -> (1.2-1.0)/0.1 = 2.0

    np.testing.assert_allclose(residuals, [2.0])


def test_residuals_with_priors_appends_to_measurement_residuals(monkeypatch):

    monkeypatch.setattr(
        "pyseshat.inference.photometric_likelihood."
        "photometric_residuals_from_problem",
        lambda problem: np.array([0.5, -0.5]),
    )

    problem = _problem_with_priors(
        {"primary_radius": (1.0, 0.1)}
    )

    combined = photometric_residuals_with_priors_from_problem(
        problem
    )

    np.testing.assert_allclose(combined, [0.5, -0.5, 2.0])


# ==========================================================
# build_parameter_bounds(): sized to the fitted subset
# ==========================================================

def test_build_parameter_bounds_sized_to_fitted_subset():

    parameters = _partially_fixed_parameters()

    lower, upper = build_parameter_bounds(parameters)

    assert lower.shape == (3,)
    assert upper.shape == (3,)

    # Order matches fitted_parameter_names():
    # [primary_temperature, secondary_temperature, ebv]

    np.testing.assert_allclose(
        lower,
        [2500.0, 2500.0, 0.0],
    )

    np.testing.assert_allclose(
        upper,
        [50000.0, 50000.0, 10.0],
    )


def test_build_parameter_bounds_uses_grid_teff_range():

    class FakeGrid:
        temperatures = [5750.0, 6000.0, 6250.0]

    parameters = PhotometricParameters(
        primary_temperature=6000.0,
        secondary_temperature=5900.0,
        primary_radius=1.0,
        secondary_radius=0.9,
        ebv=0.0,
    )

    lower, upper = build_parameter_bounds(
        parameters,
        atmosphere_grid=FakeGrid(),
    )

    np.testing.assert_allclose(
        lower,
        [5750.0, 5750.0, 0.01, 0.01, 0.0],
    )

    np.testing.assert_allclose(
        upper,
        [6250.0, 6250.0, 100.0, 100.0, 10.0],
    )
