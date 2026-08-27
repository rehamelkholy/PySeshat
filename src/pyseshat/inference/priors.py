"""
Prior probability distributions for Bayesian orbit fitting.

All prior functions return the natural logarithm of the prior
probability density.

A return value of

    -np.inf

indicates that the parameter lies outside the support of the
prior.
"""

import numpy as np


# ==========================================================
# Uniform prior
# ==========================================================

def log_uniform_prior(
    value,
    lower,
    upper
):
    """
    Uniform prior.

    Parameters
    ----------
    value : float

    lower : float

    upper : float

    Returns
    -------
    float
    """

    if value < lower or value > upper:

        return -np.inf

    return -np.log(
        upper - lower
    )


# ==========================================================
# Jeffreys (log-uniform) prior
# ==========================================================

def log_jeffreys_prior(
    value,
    lower,
    upper
):
    """
    Jeffreys (log-uniform) prior for a positive scale
    parameter.

    Parameters
    ----------
    value : float

    lower : float

    upper : float

    Returns
    -------
    float
    """

    if value <= 0:

        return -np.inf

    if value < lower or value > upper:

        return -np.inf

    normalization = np.log(
        np.log(
            upper / lower
        )
    )

    return -np.log(
        value
    ) - normalization


# ==========================================================
# Isotropic inclination prior
# ==========================================================

def log_isotropic_prior(
    inclination_deg
):
    """
    Prior corresponding to isotropically distributed orbital
    planes.

    Parameters
    ----------
    inclination_deg : float

    Returns
    -------
    float
    """

    if inclination_deg < 0.0:

        return -np.inf

    if inclination_deg > 180.0:

        return -np.inf

    inclination = np.deg2rad(
        inclination_deg
    )

    s = np.sin(
        inclination
    )

    # Guard against floating-point roundoff at exactly
    # 0 degrees and 180 degrees.

    if np.isclose(
        s,
        0.0,
        atol=1e-15
    ):

        return -np.inf

    return np.log(
        s
    )


# ==========================================================
# Default Keplerian prior
# ==========================================================

def log_prior(
    elements
):
    """
    Default Bayesian prior for an elliptic Keplerian orbit.

    Parameters
    ----------
    elements : OrbitalElements

    Returns
    -------
    float
        Sum of the log-priors for the orbital parameters
        participating in the current inference problem.

    Notes
    -----
    The eccentricity support is restricted to

        0 <= e <= 0.99

    to remain consistent with the physical bounds used by
    the deterministic orbit optimizer and with the elliptic
    Keplerian orbit model implemented by PySeshat.

    Radial-velocity semi-amplitudes K1 and K2 are subject to
    positive support only when they are fitted parameters.

    Fixed K1 and K2 values do not contribute to the prior.
    This is necessary for visual-only inference, where the
    radial-velocity parameters are not part of the fitted
    parameter vector.

    The systemic velocity V0 is a signed location parameter
    and is therefore unrestricted.
    """

    logp = 0.0

    # ------------------------------------------------------
    # Period
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        elements.P,
        1e-6,
        1e8
    )

    # ------------------------------------------------------
    # Epoch of periastron
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.T0,
        -1e9,
        1e9
    )

    # ------------------------------------------------------
    # Eccentricity
    #
    # The orbit model is elliptic and the optimizer currently
    # uses an upper bound of 0.99.
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.e,
        0.0,
        0.99
    )

    # ------------------------------------------------------
    # Semi-major axis
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        elements.a,
        1e-9,
        1e6
    )

    # ------------------------------------------------------
    # Inclination
    # ------------------------------------------------------

    logp += log_isotropic_prior(
        elements.i
    )

    # ------------------------------------------------------
    # Longitude of ascending node
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.Omega,
        0.0,
        360.0
    )

    # ------------------------------------------------------
    # Argument of periastron
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.omega,
        0.0,
        360.0
    )

    # ------------------------------------------------------
    # Radial-velocity semi-amplitudes
    #
    # K1 and K2 are positive parameters when they participate
    # in the current fit.
    #
    # Fixed values are deliberately ignored so that visual-
    # only inference is unaffected by the default RV values.
    # ------------------------------------------------------

    if elements.fit_flags.get(
        "K1",
        False
    ):

        if elements.K1 <= 0.0:

            return -np.inf

    if elements.fit_flags.get(
        "K2",
        False
    ):

        if elements.K2 <= 0.0:

            return -np.inf

    # ------------------------------------------------------
    # Systemic velocity
    #
    # V0 is a signed location parameter and remains
    # unrestricted.
    # ------------------------------------------------------

    return logp


# ==========================================================
# Default photometric prior
# ==========================================================

def log_photometric_prior(
    parameters
):
    """
    Default Bayesian prior for unresolved-binary photometric
    inference.

    Parameters
    ----------
    parameters : PhotometricParameters

    Returns
    -------
    float
        Sum of the log-priors for all photometric parameters.
    """

    logp = 0.0

    # ------------------------------------------------------
    # Primary effective temperature
    # ------------------------------------------------------

    logp += log_uniform_prior(
        parameters.primary_temperature,
        2500.0,
        50000.0
    )

    # ------------------------------------------------------
    # Secondary effective temperature
    # ------------------------------------------------------

    logp += log_uniform_prior(
        parameters.secondary_temperature,
        2500.0,
        50000.0
    )

    # ------------------------------------------------------
    # Primary radius
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        parameters.primary_radius,
        0.01,
        100.0
    )

    # ------------------------------------------------------
    # Secondary radius
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        parameters.secondary_radius,
        0.01,
        100.0
    )

    # ------------------------------------------------------
    # Reddening
    # ------------------------------------------------------

    logp += log_uniform_prior(
        parameters.ebv,
        0.0,
        10.0
    )

    return logp