from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RVObservationSet:
    """
    Radial velocity observations.

    Parameters
    ----------
    epoch
        Observation epochs.

    velocity
        Measured radial velocities.

    sigma
        Measurement uncertainties.

    component
        Component identifier
        (0 = primary, 1 = secondary).
    """

    epoch: np.ndarray

    velocity: np.ndarray

    sigma: np.ndarray

    component: np.ndarray