import numpy as np

from pyseshat.inference.mcmc_engine import (
    run_emcee,
)

from pyseshat.inference.mcmc_result import (
    MCMCResult,
)

from pyseshat.inference.photometric_posterior import (
    photometric_log_posterior_from_problem,
)

from pyseshat.photometry.parameters import (
    update_parameters,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
)


# ==========================================================
# emcee-compatible log-probability wrapper
# ==========================================================
#
# Same defect as inference.photometric_optimizer's
# _photometric_residuals_for_optimizer (see that module's
# comment for the full explanation): emcee's EnsembleSampler
# always calls log_probability_function(x, *args), where x is
# the CURRENT walker parameter vector -- but
# photometric_log_posterior_from_problem(problem) takes a
# single, already-built PhotometricInferenceProblem, not a raw
# parameter vector. Passing it straight through as
# log_probability_function (as this module previously did)
# means every walker evaluation raised
#
#   TypeError: photometric_log_posterior_from_problem() takes
#   1 positional argument but 2 were given
#
# before a single step was ever taken. This wrapper rebuilds a
# trial PhotometricInferenceProblem at the sampler's current
# walker position on every call, mirroring
# _photometric_residuals_for_optimizer.

def _photometric_log_posterior_for_sampler(
    values,
    problem,
):
    """
    emcee-compatible log-probability function for photometric
    MCMC sampling: rebuild the trial PhotometricInferenceProblem
    at ``values`` and delegate to
    photometric_log_posterior_from_problem().
    """

    trial_problem = PhotometricInferenceProblem(

        parameters=update_parameters(
            problem.parameters,
            values,
        ),

        context=problem.context,

    )

    return photometric_log_posterior_from_problem(
        trial_problem
    )


# ==========================================================
# Canonical Bayesian photometric inference
# ==========================================================

def sample_photometry_from_problem(
    problem,
    *,
    covariance=None,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None,
):
    """
    Canonical Bayesian interface for unresolved-binary
    photometric inference.
    """

    parameters = problem.parameters

    initial_parameters = parameters.fitted_parameters()

    result = run_emcee(

        log_probability_function=(
            _photometric_log_posterior_for_sampler
        ),

        log_probability_args=(
            problem,
        ),

        initial_parameters=initial_parameters,

        covariance=covariance,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng,

    )

    map_index = np.argmax(

        result["log_probability"]

    )

    map_parameters = update_parameters(

        parameters,

        result["samples"][map_index],

    )

    return MCMCResult(

        elements=map_parameters,

        optimizer_result=None,

        residuals=None,

        covariance=covariance,

        uncertainties={},

        samples=result["samples"],

        log_probability=result["log_probability"],

        parameter_names=(
            parameters.fitted_parameter_names()
        ),

        acceptance_fraction=(
            result["acceptance_fraction"]
        ),

        autocorrelation_time=(
            result["autocorrelation_time"]
        ),

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

    )


# ==========================================================
# Legacy wrapper
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
    rng=None,
):
    """
    Backward-compatible wrapper around the canonical
    photometric Bayesian inference interface.
    """

    problem = PhotometricInferenceProblem(

        parameters=parameters,

        context=PhotometricContext(

            atmosphere_grid=atmosphere_grid,

            passbands=passbands,

            zero_points=zero_points,

            measurements=measurements,

            primary_logg=primary_logg,

            secondary_logg=secondary_logg,

            metallicity=metallicity,

            distance=distance,

            rv=rv,

        ),

    )

    return sample_photometry_from_problem(

        problem,

        covariance=covariance,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng,

    )