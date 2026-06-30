from dataclasses import dataclass
import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)


@dataclass
class FitResult(BaseFitResult):
    """
    Container for orbit-fitting results.
    """

    # ------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------

    statistics: dict | None = None

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------
    metadata: dict | None = None