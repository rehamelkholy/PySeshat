import numpy as np


# ==========================================================
# Chi-square
# ==========================================================

def chi_square(
    residuals
):
    """
    Compute chi-square.
    """

    residuals = np.asarray(
        residuals
    )

    return np.sum(
        residuals**2
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
    """

    residuals = np.asarray(
        residuals
    )

    n_obs = len(
        residuals
    )

    dof = (
        n_obs
        -
        n_parameters
    )

    if dof <= 0:

        raise ValueError(
            "Degrees of freedom "
            "must be positive."
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
    Compute RMS.
    """

    residuals = np.asarray(
        residuals
    )

    return np.sqrt(
        np.mean(
            residuals**2
        )
    )


# ==========================================================
# Summary dictionary
# ==========================================================

def triple_fit_summary(
    residuals,
    n_parameters
):
    """
    Standard diagnostics.
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