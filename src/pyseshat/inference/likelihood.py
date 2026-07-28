import numpy as np

from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.rv_model import (
    compute_rv_model
)
from pyseshat.orbit.orbital_elements import OrbitalElements

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.inference.terms import (
    visual_xy_residuals,
)

# ==========================================================
# RV residuals
# ==========================================================

def rv_residuals(
    fitted_values,
    elements: OrbitalElements,
    rv_data
):
    """
    Compute weighted RV residuals.

    Parameters
    ----------
    fitted_values : ndarray

    elements : OrbitalElements

    rv_data : ndarray

        Columns:
            0 : epoch
            1 : observed RV
            2 : uncertainty
            3 : component flag

    Returns
    -------
    residual vector
    """

    # ------------------------------------------------------
    # No RV data
    # ------------------------------------------------------

    if (
        rv_data is None
        or len(rv_data) == 0
    ):

        return np.array([])

    # ------------------------------------------------------
    # Update fitted parameters
    # ------------------------------------------------------

    updated_elements = update_elements(
        elements,
        fitted_values
    )

    # ------------------------------------------------------
    # Observed RVs
    # ------------------------------------------------------

    rv_obs = rv_data[:,1]

    sigma_rv = rv_data[:,2]

    # ------------------------------------------------------
    # Model RVs
    # ------------------------------------------------------

    rv_model = compute_rv_model(
        rv_data,
        updated_elements
    )

    # ------------------------------------------------------
    # Weighted residuals
    # ------------------------------------------------------

    residuals = (
        rv_obs - rv_model
    ) / sigma_rv

    return residuals

# ==========================================================
# Combined visual + RV residuals
# ==========================================================

def combined_residuals(
    fitted_values,
    elements: OrbitalElements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None
):
    """
    Combined residual vector for:
        - visual astrometry
        - radial velocities

    Parameters
    ----------
    rv_data : ndarray or None

    Returns
    -------
    residual vector
    """

    # ------------------------------------------------------
    # Visual residuals
    # ------------------------------------------------------

    res_visual = visual_xy_residuals(
        fitted_values,
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy
    )

    # ------------------------------------------------------
    # No RV data
    # ------------------------------------------------------

    if (
        rv_data is None
        or len(rv_data) == 0
    ):

        return res_visual

    # ------------------------------------------------------
    # RV residuals
    # ------------------------------------------------------

    res_rv = rv_residuals(
        fitted_values,
        elements,
        rv_data
    )

    # ------------------------------------------------------
    # Combined residual vector
    # ------------------------------------------------------

    return np.concatenate([
        res_visual,
        res_rv
    ])

# ==========================================================
# Chi-square
# ==========================================================

def chi_square(
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
    Compute chi-square statistic.
    """

    residuals = combined_residuals(
        fitted_values=fitted_values,
        elements=elements,
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data
    )

    return np.sum(
        residuals ** 2
    )


# ==========================================================
# Reduced chi-square
# ==========================================================

def reduced_chi_square(
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
    Compute reduced chi-square.
    """

    chi2 = chi_square(
        fitted_values,
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data=rv_data
    )

    # ------------------------------------------------------
    # Number of data points
    # ------------------------------------------------------

    n_visual = 2 * len(t)

    n_rv = 0

    if (
        rv_data is not None
        and len(rv_data) > 0
    ):

        n_rv = len(rv_data)

    n_obs = n_visual + n_rv

    # ------------------------------------------------------
    # Number of fitted parameters
    # ------------------------------------------------------

    n_fit = np.sum(
        elements.fitted_mask()
    )

    dof = n_obs - n_fit

    if dof <= 0:
        raise ValueError(
            f"Degrees of freedom must be positive. "
            f"Got {dof} (n_obs={n_obs}, n_fit={n_fit})."
        )

    return chi2 / dof

# ==========================================================
# Log-likelihood
# ==========================================================

def log_likelihood(
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
    Gaussian log-likelihood.

    Parameters
    ----------
    fitted_values : ndarray
        Current fitted parameter vector.

    elements : OrbitalElements

    t, X_obs, Y_obs, sx, sy

    rv_data : ndarray or None

    Returns
    -------
    float
        Log-likelihood (up to an additive constant).
    """

    residuals = combined_residuals(
        fitted_values=fitted_values,
        elements=elements,
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data
    )

    chi2 = np.sum(
        residuals ** 2
    )

    return -0.5 * chi2