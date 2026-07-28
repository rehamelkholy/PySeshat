from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LikelihoodContext:
    """
    Container for all observational data used by the likelihood.
    """

    t: np.ndarray
    X_obs: np.ndarray
    Y_obs: np.ndarray
    sx: np.ndarray
    sy: np.ndarray

    rv_data: np.ndarray | None = None