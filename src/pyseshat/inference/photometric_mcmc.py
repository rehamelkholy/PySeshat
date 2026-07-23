import numpy as np

from pyseshat.inference.mcmc_engine import (
    run_emcee
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)

from pyseshat.inference.photometric_posterior import (
    log_posterior
)

from pyseshat.photometry.parameters import (
    update_parameters
)


# ==========================================================
# Bayesian photometric inference
# ==========================================================

def sample_photometry(
    parameters,
    atmosphere_grid,
    passbands,
    zero_points,
    measurements,
    primary_logg,
    secondary_logg,
    metallicity,
    distance,
    rv=3.1,
    covariance=None,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Sample the posterior distribution of the unresolved-binary
    photometric model using emcee.

    Parameters
    ----------
    parameters : PhotometricParameters

    atmosphere_grid

    passbands

    zero_points

    measurements

    primary_logg : float

    secondary_logg : float

    metallicity : float

    distance : float

    rv : float, default=3.1

    covariance : ndarray, optional
        Covariance matrix used to initialize walkers.

    n_walkers : int

    n_steps : int

    burn_in : int

    thin : int

    random_scale : float

    rng : numpy.random.Generator, optional

    Returns
    -------
    MCMCResult
    """

    initial_parameters = parameters.fitted_parameters()

    result = run_emcee(

        log_probability_function=log_posterior,

        log_probability_args=(

            parameters,

            atmosphere_grid,

            passbands,

            zero_points,

            measurements,

            primary_logg,

            secondary_logg,

            metallicity,

            distance,

            rv

        ),

        initial_parameters=initial_parameters,

        covariance=covariance,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng

    )

    map_index = np.argmax(

        result["log_probability"]

    )

    map_parameters = update_parameters(

        parameters,

        result["samples"][map_index]

    )

    return MCMCResult(

        elements=map_parameters,

        optimizer_result=None,

        residuals=None,

        covariance=covariance,

        uncertainties={},

        samples=result["samples"],

        log_probability=result["log_probability"],

        parameter_names=parameters.fitted_parameter_names(),

        acceptance_fraction=result["acceptance_fraction"],

        autocorrelation_time=result["autocorrelation_time"],

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin

    )