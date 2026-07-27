from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.interpolate import EvolutionSample


@dataclass(frozen=True)
class BinaryEvolutionSolution:
    """
    Best-fitting evolutionary solution for a binary system.
    """

    primary: EvolutionSample
    secondary: EvolutionSample

    age: float
    metallicity: float

    chi2: float