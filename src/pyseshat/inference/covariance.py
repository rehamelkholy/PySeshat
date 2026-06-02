import numpy as np


# ==========================================================
# Covariance estimation
# ==========================================================

def estimate_covariance(
    optimizer_result,
    residuals
):
    """
    Estimate covariance matrix from
    least-squares Jacobian.

    Parameters
    ----------
    optimizer_result : scipy OptimizeResult

    residuals : ndarray

    Returns
    -------
    covariance : ndarray
    """

    J = optimizer_result.jac

    n_res = len(residuals)

    n_par = J.shape[1]

    dof = max(
        n_res - n_par,
        1
    )

    # ------------------------------------------------------
    # Residual variance estimate
    # ------------------------------------------------------

    sigma2 = np.sum(
        residuals**2
    ) / dof

    # ------------------------------------------------------
    # Covariance matrix
    # ------------------------------------------------------

    JTJ = J.T @ J

    covariance = (
        np.linalg.pinv(JTJ)
        * sigma2
    )

    return covariance


# ==========================================================
# Parameter uncertainties
# ==========================================================

def parameter_uncertainties(
    covariance,
    fitted_parameter_names
):
    """
    Convert covariance matrix into
    parameter uncertainties.
    """

    sigma = np.sqrt(
        np.diag(covariance)
    )

    return {
        name: err
        for name, err in zip(
            fitted_parameter_names,
            sigma
        )
    }