import numpy as np


# ==========================================================
# Chi-square
# ==========================================================

def chi_square(
    residuals
):
    """
    Compute chi-square.

    Parameters
    ----------
    residuals : ndarray

    Returns
    -------
    float
    """

    residuals = np.asarray(
        residuals
    )

    return np.sum(
        residuals ** 2
    )


# ==========================================================
# Reduced chi-square
# ==========================================================

def reduced_chi_square(
    residuals,
    n_parameters
):
    """
    Compute reduced chi-square.

    Parameters
    ----------
    residuals : ndarray

    n_parameters : int

    Returns
    -------
    float
    """

    residuals = np.asarray(
        residuals
    )

    dof = (
        len(residuals)
        -
        n_parameters
    )

    if dof <= 0:

        raise ValueError(
            "Degrees of freedom must be positive."
        )

    return (
        chi_square(
            residuals
        )
        /
        dof
    )


# ==========================================================
# RMS residual
# ==========================================================

def rms_residual(
    residuals
):
    """
    Root-mean-square residual.

    Parameters
    ----------
    residuals : ndarray

    Returns
    -------
    float
    """

    residuals = np.asarray(
        residuals
    )

    return np.sqrt(
        np.mean(
            residuals ** 2
        )
    )


# ==========================================================
# Standard fit summary
# ==========================================================

def fit_summary(
    residuals,
    n_parameters
):
    """
    Standard diagnostics shared by all fitters.

    Parameters
    ----------
    residuals : ndarray

    n_parameters : int

    Returns
    -------
    dict
    """

    return {

        "chi_square":
            chi_square(
                residuals
            ),

        "reduced_chi_square":
            reduced_chi_square(
                residuals,
                n_parameters
            ),

        "rms":
            rms_residual(
                residuals
            )
    }