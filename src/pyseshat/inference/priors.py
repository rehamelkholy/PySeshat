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
# Gaussian prior
# ==========================================================

def log_gaussian_prior(
    value,
    mean,
    sigma
):
    """
    Gaussian (normal) prior.

    Parameters
    ----------
    value : float

    mean : float

    sigma : float
        Standard deviation. Must be positive.

    Returns
    -------
    float
    """

    if sigma <= 0.0:

        raise ValueError(

            f"Gaussian prior sigma must be positive "
            f"(got {sigma})."

        )

    z = (value - mean) / sigma

    return (

        -0.5 * z * z

        -
        np.log(
            sigma * np.sqrt(2.0 * np.pi)
        )

    )


# ==========================================================
# Default photometric prior
# ==========================================================

_PHOTOMETRIC_BOUND_PRIORS = {
    # name: (kind, lower, upper) -- "uniform" or "jeffreys"
    "primary_temperature": ("uniform", 2500.0, 50000.0),
    "secondary_temperature": ("uniform", 2500.0, 50000.0),
    "primary_radius": ("jeffreys", 0.01, 100.0),
    "secondary_radius": ("jeffreys", 0.01, 100.0),
    "ebv": ("uniform", 0.0, 10.0),
}


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
        Sum of the log-priors for all CURRENTLY-FITTED
        photometric parameters' physical bounds, plus any
        Gaussian priors in ``parameters.priors`` (which may
        apply to fitted or fixed parameters alike).

    Notes
    -----
    A fixed parameter's bound-prior is skipped, mirroring
    ``log_prior``'s treatment of K1/K2 above ("Fixed values are
    deliberately ignored so that ... inference is unaffected by
    the default ... values") -- a fixed parameter isn't being
    explored, so its bound-prior would only ever contribute a
    constant offset to every point in parameter space, which
    doesn't affect the posterior shape and is dropped for
    simplicity. ``PhotometricParameters`` objects lacking the
    fit-flags mechanism have no ``fit_flags``/
    ``priors`` attributes at all; ``getattr(..., None)`` below
    treats that the same as "everything fitted, no priors" so
    this function stays backward compatible with any such
    caller.
    """

    logp = 0.0

    fit_flags = getattr(
        parameters,
        "fit_flags",
        None
    )

    for name, (kind, lower, upper) in (
        _PHOTOMETRIC_BOUND_PRIORS.items()
    ):

        if (
            fit_flags is not None
            and not fit_flags.get(name, True)
        ):

            continue

        value = getattr(
            parameters,
            name
        )

        if kind == "jeffreys":

            logp += log_jeffreys_prior(
                value,
                lower,
                upper
            )

        else:

            logp += log_uniform_prior(
                value,
                lower,
                upper
            )

    # ------------------------------------------------------
    # Gaussian priors (e.g. radius tied to an independently
    # published literature value)
    # ------------------------------------------------------

    priors = getattr(
        parameters,
        "priors",
        None
    ) or {}

    for name, (mean, sigma) in priors.items():

        value = getattr(
            parameters,
            name
        )

        logp += log_gaussian_prior(
            value,
            mean,
            sigma
        )

    return logp