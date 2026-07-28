from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VisualObservationSet:
    """
    Cartesian visual observations used by experiment
    workflows.

    Parameters
    ----------
    epoch : ndarray

    x : ndarray

    y : ndarray

    sigma_x : ndarray

    sigma_y : ndarray
    """

    epoch: np.ndarray

    x: np.ndarray

    y: np.ndarray

    sigma_x: np.ndarray

    sigma_y: np.ndarray