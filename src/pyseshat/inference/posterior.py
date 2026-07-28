"""
Posterior probability for Bayesian orbit fitting.
"""

import numpy as np

from pyseshat.inference.likelihood import (
    log_likelihood,
)

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.inference.priors import (
    log_prior
)


# ==========================================================
# Log-posterior
# ==========================================================

def log_posterior(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None
):
    """
    Gaussian log-posterior.

    Parameters
    ----------
    fitted_values : ndarray

    elements : OrbitalElements

    t, X_obs, Y_obs, sx, sy

    rv_data : ndarray or None

    Returns
    -------
    float
    """

    current = update_elements(
        elements,
        fitted_values
    )

    lp = log_prior(
        current
    )

    if not np.isfinite(lp):

        return -np.inf

    ll = log_likelihood(
        fitted_values,
        current,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data=rv_data
    )

    return lp + ll