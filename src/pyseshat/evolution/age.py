from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyseshat.evolution.interpolate import (
    EvolutionSample,
    contains_age,
    sample_track,
)
from pyseshat.evolution.models import EvolutionTrack

from pyseshat.evolution.observables import StellarObservables

from pyseshat.evolution.residuals import compute_residuals

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

@dataclass(frozen=True)
class TrackFitResult:
    """
    Best-fitting point along an evolutionary track.
    """

    sample: EvolutionSample
    chi2: float
    index: int


def fit_track(
    track: EvolutionTrack,
    *,
    observables: StellarObservables,
) -> TrackFitResult:
    """
    Fit a stellar evolution track to observed stellar parameters.

    Parameters
    ----------
    track
        Evolutionary track.

    log_teff
        Observed log10 effective temperature.

    log_luminosity
        Observed log10 luminosity.

    sigma_log_teff
        Uncertainty in log10 effective temperature.

    sigma_log_luminosity
        Uncertainty in log10 luminosity.

    Returns
    -------
    TrackFitResult
    """

    chi2 = np.empty(len(track.age))

    for i, age in enumerate(track.age):

        sample = sample_track(track, float(age))

        chi2[i] = track_chi2(
            sample,
            observables,
        )

    index = int(np.argmin(chi2))

    sample = sample_track(
        track,
        float(track.age[index]),
    )

    return TrackFitResult(
        sample=sample,
        chi2=float(chi2[index]),
        index=index,
    )

def track_residuals(
    sample: EvolutionSample,
    observables: StellarObservables,
) -> np.ndarray:
    """
    Compute the standardized residual vector between an
    interpolated stellar model and a set of observed stellar
    parameters.

    Only observables with both a measured value and an
    associated uncertainty are included.

    Parameters
    ----------
    sample
        Interpolated stellar evolution model.

    observables
        Observed stellar parameters.

    Returns
    -------
    numpy.ndarray
        Standardized residual vector.
    """

    residuals = compute_residuals(
        sample,
        observables,
    )

    standardized = []

    comparisons = (
        (
            residuals.log_teff,
            observables.sigma_log_teff,
        ),
        (
            residuals.log_luminosity,
            observables.sigma_log_luminosity,
        ),
        (
            residuals.log_radius,
            observables.sigma_log_radius,
        ),
        (
            residuals.log_g,
            observables.sigma_log_g,
        ),
    )

    for residual, sigma in comparisons:

        if residual is None:
            continue

        if sigma is None:

            raise ValueError(

                "Missing uncertainty for an observed quantity."

            )

        standardized.append(

            residual / sigma

        )

    if len(standardized) == 0:

        raise ValueError(

            "At least one observable must be supplied."

        )

    return np.asarray(

        standardized,

        dtype=float,

    )

def _track_chi2(
    sample: EvolutionSample,
    observables: StellarObservables,
) -> float:
    """
    Compute the χ² between an interpolated stellar model and a set
    of observed stellar parameters.

    Only observables with both a value and an uncertainty are included
    in the χ² calculation.
    """

    residuals = compute_residuals(
        sample,
        observables,
    )

    chi2 = 0.0
    n_terms = 0

    comparisons = (
        (
            residuals.log_teff,
            observables.sigma_log_teff,
        ),
        (
            residuals.log_luminosity,
            observables.sigma_log_luminosity,
        ),
        (
            residuals.log_radius,
            observables.sigma_log_radius,
        ),
        (
            residuals.log_g,
            observables.sigma_log_g,
        ),
    )

    for residual, sigma in comparisons:

        if residual is None:
            continue

        if sigma is None:
            raise ValueError(
                "Missing uncertainty for an observed quantity."
            )

        chi2 += (residual / sigma) ** 2
        n_terms += 1

    if n_terms == 0:
        raise ValueError(
            "At least one observable must be supplied."
        )

    return chi2

def track_chi2(
    sample: EvolutionSample,
    observables: StellarObservables,
) -> float:
    """
    Compute the χ² statistic between an interpolated stellar
    model and a set of observed stellar parameters.

    This is the public statistical interface used by both
    deterministic fitting routines and Bayesian inference.

    Parameters
    ----------
    sample
        Interpolated stellar evolution model.

    observables
        Observed stellar parameters.

    Returns
    -------
    float
        χ² statistic.
    """

    return float(

        np.sum(

            track_residuals(

                sample,

                observables,

            ) ** 2

        )

    )

@dataclass(frozen=True)
class CommonAgeFitResult:
    """
    Best common-age solution for a binary system.
    """

    age: float

    primary: EvolutionSample

    secondary: EvolutionSample

    chi2: float

def fit_common_age(
    *,
    primary_track: EvolutionTrack,
    secondary_track: EvolutionTrack,
    primary_observables: StellarObservables,
    secondary_observables: StellarObservables,
) -> CommonAgeFitResult:
    """
    Fit both components of a binary assuming a common age.
    """

    best_result = None
    best_chi2 = np.inf

    ages = _common_age_grid(
        primary_track,
        secondary_track,
    )

    if len(ages) == 0:
        raise ValueError(
            "The two tracks have no overlapping age range."
        )

    for age in ages:

        if not contains_age(secondary_track, age):
            continue

        primary = sample_track(
            primary_track,
            float(age),
        )

        secondary = sample_track(
            secondary_track,
            float(age),
        )

        chi2 = (
            track_chi2(
                primary,
                primary_observables,
            )
            + track_chi2(
                secondary,
                secondary_observables,
            )
        )

        if chi2 < best_chi2:

            best_chi2 = chi2

            best_result = CommonAgeFitResult(
                age=float(age),
                primary=primary,
                secondary=secondary,
                chi2=float(chi2),
            )

    if best_result is None:
        raise ValueError(
            "The two tracks have no overlapping age range."
        )

    return best_result

def _common_age_grid(
    primary_track: EvolutionTrack,
    secondary_track: EvolutionTrack,
) -> np.ndarray:
    """
    Return the sorted union of all tabulated ages that lie
    within the overlapping age range of both tracks.
    """

    lower = max(
        primary_track.age[0],
        secondary_track.age[0],
    )

    upper = min(
        primary_track.age[-1],
        secondary_track.age[-1],
    )

    if lower > upper:
        return np.array([], dtype=float)

    ages = np.concatenate(
        (
            primary_track.age,
            secondary_track.age,
        )
    )

    ages = np.unique(ages)

    return ages[
        (ages >= lower)
        & (ages <= upper)
    ]