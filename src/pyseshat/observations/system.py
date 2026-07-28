from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.observables import StellarObservables

from .visual import VisualObservationSet
from .radial_velocity import RVObservationSet


@dataclass(frozen=True, slots=True)
class SystemObservations:
    """
    Collection of all observational constraints available for a
    stellar system.

    Each observation type is optional, allowing the same object
    to represent systems with incomplete datasets.
    """

    visual: VisualObservationSet | None = None

    radial_velocity: RVObservationSet | None = None

    stellar: StellarObservables | None = None