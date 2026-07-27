from __future__ import annotations

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
    Isochrone,
)


def nearest_track(
    model: EvolutionModel,
    *,
    initial_mass: float,
    metallicity: float,
) -> EvolutionTrack:
    """
    Return the track whose initial mass and metallicity are
    closest to the requested values.
    """

    if not model.tracks:
        raise ValueError("Evolution model contains no tracks.")

    return min(
        model.tracks,
        key=lambda track: (
            abs(track.initial_mass - initial_mass),
            abs(track.metallicity - metallicity),
        ),
    )


def nearest_isochrone(
    model: EvolutionModel,
    *,
    age: float,
    metallicity: float,
) -> Isochrone:
    """
    Return the isochrone whose age and metallicity are
    closest to the requested values.
    """

    if not model.isochrones:
        raise ValueError("Evolution model contains no isochrones.")

    return min(
        model.isochrones,
        key=lambda iso: (
            abs(iso.age - age),
            abs(iso.metallicity - metallicity),
        ),
    )