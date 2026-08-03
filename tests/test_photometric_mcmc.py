import numpy as np

from pyseshat.photometry.parameters import (
    PhotometricParameters
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)

from pyseshat.inference.photometric_mcmc import (
    sample_photometry
)


# ==========================================================
# Wrapper around run_emcee()
# ==========================================================

def test_sample_photometry(monkeypatch):

    captured = {}

    samples = np.array(

        [

            [6000.0, 5000.0, 1.20, 0.90, 0.10],

            [6200.0, 5100.0, 1.30, 0.95, 0.12],

            [6100.0, 5050.0, 1.25, 0.92, 0.11]

        ]

    )

    log_probability = np.array(

        [

            -15.0,

            -10.0,

            -12.0

        ]

    )

    def fake_run_emcee(**kwargs):

        captured.update(kwargs)

        return {

            "sampler": object(),

            "samples": samples,

            "log_probability": log_probability,

            "acceptance_fraction": 0.37,

            "autocorrelation_time": np.array(

                [

                    20.0,

                    22.0,

                    18.0,

                    21.0,

                    19.0

                ]

            )

        }

    monkeypatch.setattr(

        "pyseshat.inference.photometric_mcmc.run_emcee",

        fake_run_emcee

    )

    parameters = PhotometricParameters(

        primary_temperature=5900.0,

        secondary_temperature=4900.0,

        primary_radius=1.1,

        secondary_radius=0.8,

        ebv=0.05

    )

    result = sample_photometry(

        parameters=parameters,

        atmosphere_grid=None,

        passbands={},

        zero_points={},

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0,

        n_walkers=16,

        n_steps=200,

        burn_in=50,

        thin=2

    )

    assert isinstance(

        result,

        MCMCResult

    )

    np.testing.assert_allclose(

        captured["initial_parameters"],

        parameters.fitted_parameters()

    )

    assert (

        captured["log_probability_function"].__name__

        ==

        "photometric_log_posterior_from_problem"

    )

    assert result.n_walkers == 16

    assert result.n_steps == 200

    assert result.burn_in == 50

    assert result.thin == 2

    assert result.acceptance_fraction == 0.37

    np.testing.assert_allclose(

        result.samples,

        samples

    )

    np.testing.assert_allclose(

        result.log_probability,

        log_probability

    )

    #
    # MAP sample is the second row
    #

    assert result.elements.primary_temperature == 6200.0

    assert result.elements.secondary_temperature == 5100.0

    assert result.elements.primary_radius == 1.30

    assert result.elements.secondary_radius == 0.95

    assert result.elements.ebv == 0.12


# ==========================================================
# Covariance forwarding
# ==========================================================

def test_sample_photometry_covariance(

    monkeypatch

):

    captured = {}

    covariance = np.eye(5)

    def fake_run_emcee(**kwargs):

        captured.update(kwargs)

        return {

            "sampler": object(),

            "samples": np.zeros((1, 5)),

            "log_probability": np.zeros(1),

            "acceptance_fraction": 0.5,

            "autocorrelation_time": None

        }

    monkeypatch.setattr(

        "pyseshat.inference.photometric_mcmc.run_emcee",

        fake_run_emcee

    )

    parameters = PhotometricParameters(

        6000,

        5000,

        1.2,

        0.9,

        0.1

    )

    sample_photometry(

        parameters=parameters,

        atmosphere_grid=None,

        passbands={},

        zero_points={},

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0,

        covariance=covariance

    )

    assert captured["covariance"] is covariance