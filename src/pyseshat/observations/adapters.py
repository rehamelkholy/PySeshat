from __future__ import annotations

from .visual import VisualObservationSet


def visual_arrays(
    observations: VisualObservationSet,
):
    """
    Return the NumPy arrays expected by the current orbit
    inference routines.

    Returns
    -------
    tuple
        (epoch, x, y, sigma_x, sigma_y)
    """

    return (
        observations.epoch,
        observations.x,
        observations.y,
        observations.sigma_x,
        observations.sigma_y,
    )