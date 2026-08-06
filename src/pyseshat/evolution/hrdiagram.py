from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.models import (
    EvolutionTrack,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)


@dataclass(frozen=True)
class HRDiagramData:
    """
    Scientific data required to construct a binary
    Hertzsprung-Russell diagram.

    This class is intentionally a data-transfer object (DTO)
    separating the scientific computation layer from the
    visualization layer.
    """

    #
    # Evolutionary tracks.
    #
    primary_track: EvolutionTrack

    secondary_track: EvolutionTrack

    #
    # Observed stellar parameters.
    #
    primary_observables: StellarObservables

    secondary_observables: StellarObservables

    #
    # Best-fitting evolutionary solution.
    #
    solution: BinaryEvolutionSolution


def build_binary_hrdiagram_data(
    *,
    primary_track: EvolutionTrack,
    secondary_track: EvolutionTrack,
    primary_observables: StellarObservables,
    secondary_observables: StellarObservables,
    solution: BinaryEvolutionSolution,
) -> HRDiagramData:
    """
    Construct the scientific data required to produce a
    binary Hertzsprung-Russell diagram.

    Parameters
    ----------
    primary_track
        Evolutionary track of the primary component.

    secondary_track
        Evolutionary track of the secondary component.

    primary_observables
        Observed parameters of the primary component.

    secondary_observables
        Observed parameters of the secondary component.

    solution
        Best-fitting binary evolutionary solution.

    Returns
    -------
    HRDiagramData
        Scientific data required for visualization.
    """

    return HRDiagramData(
        primary_track=primary_track,
        secondary_track=secondary_track,
        primary_observables=primary_observables,
        secondary_observables=secondary_observables,
        solution=solution,
    )