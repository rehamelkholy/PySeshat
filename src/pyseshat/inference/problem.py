from dataclasses import dataclass

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

    @property
    def t(self):
        return self.context.t

    @property
    def X_obs(self):
        return self.context.X_obs

    @property
    def Y_obs(self):
        return self.context.Y_obs

    @property
    def sx(self):
        return self.context.sx

    @property
    def sy(self):
        return self.context.sy

    @property
    def rv_data(self):
        return self.context.rv_data

    @property
    def has_visual_data(self) -> bool:
        """
        True if visual observations are available.
        """
        return len(self.t) > 0

    @property
    def has_rv_data(self) -> bool:
        """
        True if radial-velocity observations are available.
        """
        return (
            self.rv_data is not None
            and len(self.rv_data) > 0
        )

    @property
    def is_visual_only(self) -> bool:
        """
        True for a purely visual binary.
        """
        return (
            self.has_visual_data
            and not self.has_rv_data
        )

    @property
    def is_combined_fit(self) -> bool:
        """
        True if both visual and RV observations exist.
        """
        return (
            self.has_visual_data
            and self.has_rv_data
        )