from __future__ import annotations

from dataclasses import dataclass


from pyseshat.evolution.interpolate import (
    EvolutionSample,
)


@dataclass(frozen=True)
class BinaryEvolutionSolution:
    """
    Best-fitting evolutionary solution for a binary system.

    This object represents the result of fitting both
    stellar components to a common evolutionary model.
    """

    #
    # Best-fitting stellar models.
    #
    primary: EvolutionSample
    secondary: EvolutionSample

    #
    # Shared evolutionary parameters.
    #
    age: float
    metallicity: float

    #
    # Goodness of fit.
    #
    chi2: float

    #
    # Number of observables contributing to chi².
    #
    degrees_of_freedom: int | None = None

    @property
    def reduced_chi2(self) -> float:
        """
        Reduced chi-square of the binary evolutionary fit.
        """

        if self.degrees_of_freedom is None:

            raise ValueError(
                "Reduced chi-square is unavailable because "
                "degrees_of_freedom has not been set."
            )

        if self.degrees_of_freedom <= 0:

            raise ValueError(
                "degrees_of_freedom must be positive."
            )

        return self.chi2 / self.degrees_of_freedom