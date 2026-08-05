from dataclasses import dataclass
from typing import Any

from pyseshat.inference.problem import (
    InferenceProblem,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
)


@dataclass(frozen=True)
class JointInferenceProblem:
    """
    Canonical inference problem for joint orbital and
    photometric binary-star modelling.

    Parameters
    ----------
    orbit_problem : InferenceProblem
        Canonical orbital inference problem.

    photometric_problem : PhotometricInferenceProblem
        Canonical photometric inference problem.
    """

    orbit_problem: InferenceProblem

    photometric_problem: PhotometricInferenceProblem

    evolution_context: Any | None = None

    @property
    def orbit(self) -> InferenceProblem:
        """
        Canonical orbital inference problem.
        """
        return self.orbit_problem

    @property
    def photometry(self) -> PhotometricInferenceProblem:
        """
        Canonical photometric inference problem.
        """
        return self.photometric_problem

    @property
    def has_visual_data(self) -> bool:
        """
        True if visual observations are available.
        """
        return self.orbit_problem.has_visual_data

    @property
    def has_rv_data(self) -> bool:
        """
        True if radial-velocity observations are available.
        """
        return self.orbit_problem.has_rv_data

    @property
    def is_visual_only(self) -> bool:
        """
        True for a purely visual binary.
        """
        return self.orbit_problem.is_visual_only

    @property
    def is_combined_fit(self) -> bool:
        """
        True if both visual and radial-velocity
        observations are available.
        """
        return self.orbit_problem.is_combined_fit

    @property
    def has_evolution_constraints(self) -> bool:
        """
        True if stellar-evolution constraints are available.
        """
        return self.evolution_context is not None