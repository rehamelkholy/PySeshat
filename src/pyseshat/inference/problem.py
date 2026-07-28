from dataclasses import dataclass, field

from pyseshat.inference.context import LikelihoodContext
from pyseshat.orbit.orbital_elements import OrbitalElements


@dataclass(frozen=True)
class InferenceProblem:
    """
    Defines a complete orbit inference problem.
    """

    elements: OrbitalElements

    context: LikelihoodContext

    method: str = "trf"

    loss: str = "linear"

    f_scale: float = 1.0

    bounds: tuple | None = None

    node_disambiguation: bool = True

    node_rtol: float = 0.01