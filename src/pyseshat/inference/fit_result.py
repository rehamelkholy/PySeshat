from dataclasses import dataclass
import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


@dataclass
class FitResult:
    """
    Container for orbit-fitting results.
    """

    # ------------------------------------------------------
    # Physical orbit
    # ------------------------------------------------------

    elements: OrbitalElements

    # ------------------------------------------------------
    # Optimizer backend result
    # ------------------------------------------------------

    optimizer_result: object

    # ------------------------------------------------------
    # Residual vectors
    # ------------------------------------------------------

    residuals: np.ndarray

    # ------------------------------------------------------
    # Covariance matrix
    # ------------------------------------------------------

    covariance: np.ndarray | None = None

    # ------------------------------------------------------
    # Parameter uncertainties
    # ------------------------------------------------------

    uncertainties: dict | None = None

    # ------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------

    statistics: dict | None = None