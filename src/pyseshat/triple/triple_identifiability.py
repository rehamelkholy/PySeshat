import numpy as np


# ==========================================================
# Condition number
# ==========================================================

def covariance_condition_number(
    covariance
):
    """
    Compute covariance matrix condition number.

    Parameters
    ----------
    covariance : ndarray

    Returns
    -------
    float
    """

    covariance = np.asarray(
        covariance
    )

    return np.linalg.cond(
        covariance
    )


# ==========================================================
# Eigenvalues
# ==========================================================

def covariance_eigenvalues(
    covariance
):
    """
    Compute covariance eigenvalues.

    Parameters
    ----------
    covariance : ndarray

    Returns
    -------
    ndarray
    """

    covariance = np.asarray(
        covariance
    )

    return np.linalg.eigvalsh(
        covariance
    )


# ==========================================================
# Effective rank
# ==========================================================

def covariance_rank(
    covariance,
    tolerance=1e-12
):
    """
    Numerical rank of covariance matrix.

    Parameters
    ----------
    covariance : ndarray

    tolerance : float

    Returns
    -------
    int
    """

    eigenvalues = (
        covariance_eigenvalues(
            covariance
        )
    )

    return int(
        np.sum(
            eigenvalues > tolerance
        )
    )


# ==========================================================
# Degeneracy test
# ==========================================================

def is_degenerate(
    covariance,
    threshold=1e12
):
    """
    Check whether covariance matrix
    is effectively singular.

    Parameters
    ----------
    covariance : ndarray

    threshold : float

    Returns
    -------
    bool
    """

    condition_number = (
        covariance_condition_number(
            covariance
        )
    )

    return (
        condition_number > threshold
    )