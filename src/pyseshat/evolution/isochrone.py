from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyseshat.evolution.age import _track_chi2
from pyseshat.evolution.interpolate import (
    EvolutionSample,
    sample_isochrone,
)
from pyseshat.evolution.models import Isochrone
from pyseshat.evolution.observables import StellarObservables

@dataclass(frozen=True)
class IsochroneFitResult:
    """
    Best-fitting point on an isochrone.
    """

    sample: EvolutionSample
    chi2: float
    index: int

def fit_isochrone(
    isochrone: Isochrone,
    *,
    observables: StellarObservables,
) -> IsochroneFitResult:
    """
    Find the best-fitting point along an isochrone.
    """

    chi2 = np.empty(len(isochrone.mass))

    for i, mass in enumerate(isochrone.mass):

        sample = sample_isochrone(
            isochrone,
            float(mass),
        )

        chi2[i] = _track_chi2(
            sample,
            observables,
        )

    best = int(np.argmin(chi2))

    return IsochroneFitResult(
        sample=sample_isochrone(
            isochrone,
            float(isochrone.mass[best]),
        ),
        chi2=float(chi2[best]),
        index=best,
    )