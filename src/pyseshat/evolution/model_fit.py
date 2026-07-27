from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.isochrone import (
    IsochroneFitResult,
    fit_isochrone,
)
from pyseshat.evolution.models import EvolutionModel
from pyseshat.evolution.observables import StellarObservables


@dataclass(frozen=True)
class EvolutionModelFitResult:
    """
    Best-fitting isochrone within an evolutionary model.
    """

    result: IsochroneFitResult
    age: float
    metallicity: float
    chi2: float


def fit_evolution_model(
    model: EvolutionModel,
    *,
    observables: StellarObservables,
) -> EvolutionModelFitResult:
    """
    Search all isochrones in an evolutionary model and return the
    best-fitting solution.
    """

    if len(model.isochrones) == 0:
        raise ValueError(
            "Evolution model contains no isochrones."
        )

    best_result = None

    for isochrone in model.isochrones:

        result = fit_isochrone(
            isochrone,
            observables=observables,
        )

        if (
            best_result is None
            or result.chi2 < best_result.chi2
        ):
            best_result = EvolutionModelFitResult(
                result=result,
                age=isochrone.age,
                metallicity=isochrone.metallicity,
                chi2=result.chi2,
            )

    return best_result