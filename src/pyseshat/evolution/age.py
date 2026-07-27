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

        chi2[i] = _track_chi2(
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

    chi2 = 0.0
    n_terms = 0

    def accumulate(observed, sigma, predicted):
        nonlocal chi2, n_terms

        if observed is None or sigma is None:
            return

        chi2 += ((predicted - observed) / sigma) ** 2
        n_terms += 1

    accumulate(
        observables.log_teff,
        observables.sigma_log_teff,
        sample.log_teff,
    )

    accumulate(
        observables.log_luminosity,
        observables.sigma_log_luminosity,
        sample.log_luminosity,
    )

    accumulate(
        observables.log_radius,
        observables.sigma_log_radius,
        sample.log_radius,
    )

    accumulate(
        observables.log_g,
        observables.sigma_log_g,
        sample.log_g,
    )

    if n_terms == 0:
        raise ValueError(
            "At least one observable with an uncertainty "
            "must be provided."
        )

    return float(chi2)

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
            _track_chi2(
                primary,
                primary_observables,
            )
            + _track_chi2(
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