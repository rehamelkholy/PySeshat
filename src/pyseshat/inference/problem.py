from dataclasses import dataclass

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


@dataclass(frozen=True)
class InferenceProblem:
    """
    Defines a complete orbit inference problem.

    It combines

    - orbital parameterization
    - observational data

    Future versions will also include priors,
    stellar constraints, Gaia data,
    photometry, etc.
    """

    elements: OrbitalElements
    context: LikelihoodContext