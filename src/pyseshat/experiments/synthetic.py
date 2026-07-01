import numpy as np


# ==========================================================
# Gaussian perturbation
# ==========================================================

def perturb_values(
    values,
    sigma,
    rng=None
):
    """
    Perturb values with independent Gaussian noise.

    Parameters
    ----------
    values : array_like

    sigma : array_like or float

    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray
    """

    values = np.asarray(values)

    sigma = np.asarray(sigma)

    if rng is None:
        rng = np.random.default_rng()

    return values + rng.normal(
        loc=0.0,
        scale=sigma,
        size=values.shape
    )


# ==========================================================
# Perturb X/Y observations
# ==========================================================

def perturb_xy(
    observations,
    rng=None
):
    """
    Perturb a VisualObservationSet.

    Parameters
    ----------
    observations : VisualObservationSet

    rng : numpy.random.Generator, optional

    Returns
    -------
    VisualObservationSet
    """

    from pyseshat.experiments.observation_set import (
        VisualObservationSet
    )

    x = perturb_values(
        observations.x,
        observations.sigma_x,
        rng=rng
    )

    y = perturb_values(
        observations.y,
        observations.sigma_y,
        rng=rng
    )

    return VisualObservationSet(

        epoch=observations.epoch,

        x=x,

        y=y,

        sigma_x=observations.sigma_x,

        sigma_y=observations.sigma_y
    )