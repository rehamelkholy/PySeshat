"""
Gaussian likelihood for stellar-evolution constraints.
"""

from __future__ import annotations

from pyseshat.evolution.age import (
    track_residuals,
)

from pyseshat.inference.photometric_likelihood import (
    chi_square,
)


def evolution_log_likelihood(
    sample,
    observables,
):
    """
    Evaluate the Gaussian log-likelihood for a stellar
    evolution model.

    Parameters
    ----------
    sample : EvolutionSample
        Interpolated stellar evolution model.

    observables : StellarObservables
        Observed stellar quantities.

    Returns
    -------
    float
        Gaussian log-likelihood up to an additive
        normalization constant.
    """

    residuals = track_residuals(

        sample,

        observables,

    )

    return log_likelihood(

        residuals,

    )


def log_likelihood(
    residuals,
):
    """
    Gaussian log-likelihood.

    Parameters
    ----------
    residuals : array_like
        Standardized residual vector.

    Returns
    -------
    float
        Gaussian log likelihood up to an additive
        normalization constant.
    """

    return (

        -0.5

        *

        chi_square(

            residuals,

        )

    )