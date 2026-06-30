import numpy as np


# ==========================================================
# Covariance PCA
# ==========================================================

def covariance_pca(
    covariance
):
    """
    Principal component analysis
    of covariance matrix.

    Parameters
    ----------
    covariance : ndarray

    Returns
    -------
    eigenvalues : ndarray

    eigenvectors : ndarray
    """

    covariance = np.asarray(
        covariance
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    return (
        eigenvalues,
        eigenvectors
    )


# ==========================================================
# Best constrained mode
# ==========================================================

def best_constrained_mode(
    covariance
):
    """
    Smallest variance mode.
    """

    eigenvalues, eigenvectors = (
        covariance_pca(
            covariance
        )
    )

    index = np.argmin(
        eigenvalues
    )

    return (
        eigenvalues[index],
        eigenvectors[:, index]
    )


# ==========================================================
# Worst constrained mode
# ==========================================================

def worst_constrained_mode(
    covariance
):
    """
    Largest variance mode.
    """

    eigenvalues, eigenvectors = (
        covariance_pca(
            covariance
        )
    )

    index = np.argmax(
        eigenvalues
    )

    return (
        eigenvalues[index],
        eigenvectors[:, index]
    )


# ==========================================================
# Variance ratio
# ==========================================================

def variance_ratio(
    covariance
):
    """
    Largest variance divided
    by smallest variance.
    """

    eigenvalues, _ = (
        covariance_pca(
            covariance
        )
    )

    return (
        np.max(
            eigenvalues
        )
        /
        np.min(
            eigenvalues
        )
    )