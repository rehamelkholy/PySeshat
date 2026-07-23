"""
Posterior probability for Bayesian photometric inference.
"""

import numpy as np

from pyseshat.photometry.parameters import (
    update_parameters
)

from pyseshat.inference.photometric_likelihood import (
    log_likelihood
)

from pyseshat.inference.priors import (
    log_photometric_prior
)


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

    lp = log_photometric_prior(

        current

    )

    if not np.isfinite(

        lp

    ):

        return -np.inf

    ll = log_likelihood(

        fitted_values,

        current,

        atmosphere_grid,

        passbands,

        zero_points,

        measurements,

        primary_logg,

        secondary_logg,

        metallicity,

        distance,

        rv

    )

    return lp + ll