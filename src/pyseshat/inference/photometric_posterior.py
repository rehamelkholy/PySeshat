"""
Posterior probability for Bayesian photometric inference.
"""

import numpy as np

from pyseshat.photometry.parameters import (
    update_parameters
)

from pyseshat.inference.photometric_likelihood import (
    log_likelihood,
    photometric_log_likelihood_from_problem,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.inference.priors import (
    log_photometric_prior
)

def photometric_log_posterior_from_problem(
    problem: PhotometricInferenceProblem,
):
    """
    Evaluate the Bayesian photometric posterior for a
    canonical photometric inference problem.
    """

    lp = log_photometric_prior(
        problem.parameters
    )

    if not np.isfinite(lp):

        return -np.inf

    ll = photometric_log_likelihood_from_problem(
        problem
    )

    return lp + ll

# ==========================================================
# Log-posterior
# ==========================================================

def log_posterior(
    fitted_values,
    parameters,
    atmosphere_grid,
    passbands,
    zero_points,
    measurements,
    primary_logg,
    secondary_logg,
    metallicity,
    distance,
    rv=3.1
):
    """
    Gaussian log-posterior for unresolved-binary photometric
    inference.

    Parameters
    ----------
    fitted_values : ndarray

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

    Returns
    -------
    float
    """

    current = update_parameters(

        parameters,

        fitted_values

    )

    context = PhotometricContext(

        atmosphere_grid=atmosphere_grid,

        passbands=passbands,

        zero_points=zero_points,

        measurements=measurements,

        primary_logg=primary_logg,

        secondary_logg=secondary_logg,

        metallicity=metallicity,

        distance=distance,

        rv=rv

    )

    problem = PhotometricInferenceProblem(

        parameters=current,

        context=context

    )

    return photometric_log_posterior_from_problem(

        problem

    )