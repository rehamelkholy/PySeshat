from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.age import (
    TrackFitResult,
    fit_track,
)
from pyseshat.evolution.models import EvolutionTrack


@dataclass(frozen=True)
class BinaryTrackFitResult:
    """
    Best-fitting evolutionary solution for a binary system.
    """

    primary: TrackFitResult
    secondary: TrackFitResult

    age_difference: float


def fit_binary_tracks(
    *,
    primary_track: EvolutionTrack,
    secondary_track: EvolutionTrack,
    primary_log_teff: float,
    primary_log_luminosity: float,
    primary_sigma_log_teff: float,
    primary_sigma_log_luminosity: float,
    secondary_log_teff: float,
    secondary_log_luminosity: float,
    secondary_sigma_log_teff: float,
    secondary_sigma_log_luminosity: float,
) -> BinaryTrackFitResult:
    """
    Fit both components of a binary independently.

    This is the first stage of the binary evolutionary fitting.
    Later versions will enforce a common age.
    """

    primary = fit_track(
        primary_track,
        log_teff=primary_log_teff,
        log_luminosity=primary_log_luminosity,
        sigma_log_teff=primary_sigma_log_teff,
        sigma_log_luminosity=primary_sigma_log_luminosity,
    )

    secondary = fit_track(
        secondary_track,
        log_teff=secondary_log_teff,
        log_luminosity=secondary_log_luminosity,
        sigma_log_teff=secondary_sigma_log_teff,
        sigma_log_luminosity=secondary_sigma_log_luminosity,
    )

    return BinaryTrackFitResult(
        primary=primary,
        secondary=secondary,
        age_difference=abs(
            primary.sample.age
            - secondary.sample.age
        ),
    )