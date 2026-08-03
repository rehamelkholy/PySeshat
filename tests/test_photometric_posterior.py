import numpy as np

import pytest

from pyseshat.photometry.parameters import (
    PhotometricParameters
)

from pyseshat.inference.photometric_posterior import (
    log_posterior
)


# ==========================================================
# Prior + likelihood
# ==========================================================

def test_log_posterior(monkeypatch):

    def fake_prior(parameters):

        return -2.5

    def fake_likelihood(*args, **kwargs):

        return -10.0

    monkeypatch.setattr(

        "pyseshat.inference.photometric_posterior."
        "log_photometric_prior",

        fake_prior

    )

    monkeypatch.setattr(
        "pyseshat.inference.photometric_posterior."
        "photometric_log_likelihood_from_problem",
        lambda problem: -10.0
    )

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.2,

        secondary_radius=0.9,

        ebv=0.1

    )

    value = log_posterior(

        fitted_values=np.array(

            [

                6000.0,

                5000.0,

                1.2,

                0.9,

                0.1

            ]

        ),

        parameters=parameters,

        atmosphere_grid=None,

        passbands=None,

        zero_points=None,

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0

    )

    assert value == pytest.approx(

        -12.5

    )


# ==========================================================
# Reject impossible parameter vector
# ==========================================================

def test_log_posterior_prior_rejection(

    monkeypatch

):

    called = False

    def fake_prior(parameters):

        return -np.inf

    def fake_likelihood(*args, **kwargs):

        nonlocal called

        called = True

        return -5.0

    monkeypatch.setattr(

        "pyseshat.inference.photometric_posterior."
        "log_photometric_prior",

        fake_prior

    )

    monkeypatch.setattr(

        "pyseshat.inference.photometric_posterior."
        "log_likelihood",

        fake_likelihood

    )

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.2,

        secondary_radius=0.9,

        ebv=0.1

    )

    value = log_posterior(

        fitted_values=np.array(

            [

                6000.0,

                5000.0,

                1.2,

                0.9,

                0.1

            ]

        ),

        parameters=parameters,

        atmosphere_grid=None,

        passbands=None,

        zero_points=None,

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0

    )

    assert value == -np.inf

    assert called is False


# ==========================================================
# Parameter update
# ==========================================================

def test_log_posterior_updates_parameters(

    monkeypatch

):

    captured = {}

    def fake_prior(parameters):

        captured["parameters"] = parameters

        return 0.0

    monkeypatch.setattr(

        "pyseshat.inference.photometric_posterior."
        "log_photometric_prior",

        fake_prior

    )

    monkeypatch.setattr(
        "pyseshat.inference.photometric_posterior."
        "photometric_log_likelihood_from_problem",
        lambda problem: 0.0
    )

    parameters = PhotometricParameters(

        primary_temperature=5000.0,

        secondary_temperature=4500.0,

        primary_radius=1.0,

        secondary_radius=0.8,

        ebv=0.0

    )

    log_posterior(

        fitted_values=np.array(

            [

                6500.0,

                5200.0,

                1.4,

                0.95,

                0.2

            ]

        ),

        parameters=parameters,

        atmosphere_grid=None,

        passbands=None,

        zero_points=None,

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0

    )

    updated = captured["parameters"]

    assert updated.primary_temperature == 6500.0
    assert updated.secondary_temperature == 5200.0
    assert updated.primary_radius == 1.4
    assert updated.secondary_radius == 0.95
    assert updated.ebv == 0.2

def fake_likelihood(problem):

    assert problem.parameters.primary_temperature == 6500.0
    assert problem.parameters.secondary_temperature == 5200.0
    assert problem.parameters.primary_radius == 1.4
    assert problem.parameters.secondary_radius == 0.95
    assert problem.parameters.ebv == 0.2

    return 0.0