import numpy as np


# ==========================================================
# Covariance estimation
# ==========================================================

def estimate_covariance(
    optimizer_result,
    residuals,
    scale_by_reduced_chi2=False
):
    """
    Estimate the local parameter covariance matrix from
    the least-squares Jacobian.

    The residuals used by PySeshat are normalized by the
    observational uncertainties. Therefore, when those
    uncertainties are regarded as known, the local Gaussian
    covariance is

        C = (J^T J)^(-1)

    where J is the Jacobian of the normalized residual vector.

    An optional reduced-chi-square scaling can be requested
    for applications in which the residual variance is
    intentionally estimated from the fit:

        C = chi2_nu * (J^T J)^(-1)

    Parameters
    ----------
    optimizer_result : scipy.optimize.OptimizeResult
        Result returned by scipy.optimize.least_squares.

    residuals : ndarray
        Final normalized residual vector.

    scale_by_reduced_chi2 : bool, optional
        If True, scale the covariance matrix by the reduced
        chi-square of the fit.

        Default is False because PySeshat's residuals already
        incorporate the supplied observational uncertainties.

    Returns
    -------
    covariance : ndarray
        Estimated parameter covariance matrix.

    Raises
    ------
    ValueError
        If the Jacobian is not two-dimensional or has no
        fitted parameters.

    Notes
    -----
    The covariance estimate is a local linear approximation
    based on the optimizer Jacobian. It is appropriate for
    ordinary least-squares inference with a linear loss.

    For robust least-squares losses, such as ``soft_l1`` or
    ``huber``, this covariance should not be interpreted as a
    fully rigorous robust-loss covariance estimator.
    """

    J = np.asarray(
        optimizer_result.jac,
        dtype=float
    )

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    if J.ndim != 2:

        raise ValueError(
            "Optimizer Jacobian must be a two-dimensional array."
        )

    n_res, n_par = J.shape

    if n_par == 0:

        raise ValueError(
            "Cannot estimate covariance with zero fitted parameters."
        )

    # ------------------------------------------------------
    # Normal matrix
    # ------------------------------------------------------

    JTJ = J.T @ J

    # ------------------------------------------------------
    # Local covariance
    #
    # For normalized residuals with known observational
    # uncertainties:
    #
    #     C = (J^T J)^(-1)
    #
    # The pseudo-inverse is used so that the covariance
    # calculation remains numerically defined for singular
    # or poorly conditioned parameter combinations.
    # ------------------------------------------------------

    covariance = np.linalg.pinv(
        JTJ
    )

    # ------------------------------------------------------
    # Optional reduced-chi-square scaling
    # ------------------------------------------------------

    if scale_by_reduced_chi2:

        dof = n_res - n_par

        if dof <= 0:
            raise ValueError(
                "Degrees of freedom must be positive to scale "
                "by reduced chi-square. "
                f"Got {dof} (n_res={n_res}, n_par={n_par})."
            )

        chi2 = np.sum(
            residuals ** 2
        )

        reduced_chi2 = (
            chi2 / dof
        )

        covariance = (
            covariance
            *
            reduced_chi2
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
    Convert a parameter covariance matrix into
    one-sigma parameter uncertainties.

    Parameters
    ----------
    covariance : ndarray
        Parameter covariance matrix.

    fitted_parameter_names : sequence of str
        Names corresponding to the covariance-matrix
        parameters.

    Returns
    -------
    dict
        Mapping from parameter name to one-sigma
        uncertainty.

    Raises
    ------
    ValueError
        If the covariance matrix is not square or its
        dimension does not match the number of parameter
        names.
    """

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    fitted_parameter_names = list(
        fitted_parameter_names
    )

    if covariance.ndim != 2:

        raise ValueError(
            "Covariance matrix must be two-dimensional."
        )

    n_rows, n_cols = covariance.shape

    if n_rows != n_cols:

        raise ValueError(
            "Covariance matrix must be square."
        )

    if n_rows != len(
        fitted_parameter_names
    ):

        raise ValueError(
            "Covariance matrix dimension must match "
            "the number of fitted parameter names."
        )

    # ------------------------------------------------------
    # Parameter variances are the diagonal elements.
    # ------------------------------------------------------

    variances = np.diag(
        covariance
    )

    # Numerical roundoff can produce tiny negative values
    # for theoretically non-negative variances. Clip only
    # those values to zero.
    variances = np.maximum(
        variances,
        0.0
    )

    sigma = np.sqrt(
        variances
    )

    return {
        name: uncertainty
        for name, uncertainty in zip(
            fitted_parameter_names,
            sigma
        )
    }