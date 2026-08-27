"""
Likelihood and posterior probability for Bayesian ORBIT3
hierarchical-triple fitting.

Mirrors ``pyseshat.inference.likelihood.log_likelihood`` /
``pyseshat.inference.posterior.log_posterior`` (the binary
single-orbit versions), built on top of
``triple.triple_residuals.orbit3_residuals`` and
``triple.triple_fit.update_orbit3_fitted_parameters`` instead
of their binary equivalents.
"""

import numpy as np

from pyseshat.triple.triple_fit import (
    update_orbit3_fitted_parameters
)

from pyseshat.triple.triple_residuals import (
    orbit3_residuals
)

from pyseshat.triple.orbit3_priors import (
    log_prior_orbit3
)


# ==========================================================
# ORBIT3 log-likelihood
# ==========================================================

def log_likelihood_orbit3(
    fitted_values,
    elements,
    observations,
    rv_observations
):
    """
    Gaussian log-likelihood for an ORBIT3 hierarchical-triple
    solution.

    Parameters
    ----------
    fitted_values : ndarray
        Values of the currently-fitted ORBIT3 parameters, in
        ``elements.fitted_parameter_names()`` order.

    elements : TripleOrbitalElements
        Template element set (only its fit mask and fixed
        parameter values are used; the fitted parameters are
        overwritten by ``fitted_values``).

    observations : list[TripleObservation]
        Astrometric (I1/I2) observations.

    rv_observations : list[TripleObservation]
        Radial-velocity (Va/Vb/Vc) observations.

    Returns
    -------
    float

    Notes
    -----
    As in the binary path's ``log_likelihood``, this is
    ``-0.5 * sum(weighted_residuals**2)`` -- the constant
    Gaussian normalization term
    (``-0.5 * sum(log(2*pi*sigma**2))``) is omitted, since it
    does not depend on the fitted parameters and cancels in
    any posterior ratio or MCMC acceptance step.
    """

    updated = update_orbit3_fitted_parameters(
        elements,
        fitted_values
    )

    residuals = orbit3_residuals(
        observations,
        rv_observations,
        updated
    )

    return -0.5 * np.sum(
        residuals ** 2
    )


# ==========================================================
# ORBIT3 log-posterior
# ==========================================================

def log_posterior_orbit3(
    fitted_values,
    elements,
    observations,
    rv_observations
):
    """
    Gaussian log-posterior for an ORBIT3 hierarchical-triple
    solution: ``log_prior_orbit3 + log_likelihood_orbit3``.

    Returns ``-inf`` without evaluating the (comparatively
    expensive) ORBIT3 forward model at all when the trial
    point already lies outside the prior support -- the same
    short-circuit the binary path's ``log_posterior`` uses.

    Parameters
    ----------
    fitted_values : ndarray

    elements : TripleOrbitalElements

    observations : list[TripleObservation]

    rv_observations : list[TripleObservation]

    Returns
    -------
    float
    """

    current = update_orbit3_fitted_parameters(
        elements,
        fitted_values
    )

    lp = log_prior_orbit3(
        current
    )

    if not np.isfinite(lp):

        return -np.inf

    ll = log_likelihood_orbit3(
        fitted_values,
        current,
        observations,
        rv_observations
    )

    return lp + ll
