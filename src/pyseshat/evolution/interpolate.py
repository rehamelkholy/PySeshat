from dataclasses import dataclass

import numpy as np

from pyseshat.evolution.models import EvolutionTrack, Isochrone


@dataclass(frozen=True)
class EvolutionSample:
    """
    Stellar parameters interpolated from an evolutionary model.
    """

    age: float
    metallicity: float

    mass: float | None = None
    initial_mass: float | None = None

    log_teff: float | None = None
    log_luminosity: float | None = None
    log_radius: float | None = None
    log_g: float | None = None

    phase: int | None = None

    @property
    def stellar_mass(self) -> float:
        """
        Return the best available stellar mass.

        Preference order:

        1. current mass
        2. initial mass

        Raises
        ------
        ValueError
            If neither quantity is available.
        """

        if self.mass is not None:
            return self.mass

        if self.initial_mass is not None:
            return self.initial_mass

        raise ValueError(
            "EvolutionSample contains no stellar mass."
        )

def _interp(x, xp, fp):
    """
    One-dimensional linear interpolation.

    Parameters
    ----------
    x : float
    xp : array_like
    fp : array_like

    Returns
    -------
    float
    """

    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.ndim != 1:
        raise ValueError("xp must be one-dimensional.")

    if fp.ndim != 1:
        raise ValueError("fp must be one-dimensional.")

    if xp.size != fp.size:
        raise ValueError("xp and fp must have the same length.")

    if xp.size > 1 and np.any(np.diff(xp) <= 0):
        raise ValueError(
            "xp must be sorted in strictly increasing order. "
            "np.interp silently returns wrong values for an "
            "unsorted or non-monotonic table (e.g. a track with "
            "a post-AGB or blue-loop segment where age or mass "
            "is not monotonic)."
        )

    if x < xp[0] or x > xp[-1]:
        raise ValueError(
            f"Interpolation point {x} is outside "
            f"the valid range [{xp[0]}, {xp[-1]}]."
        )

    return float(np.interp(x, xp, fp))

def contains_age(
    track: EvolutionTrack,
    age: float,
) -> bool:
    """
    Return whether an age lies within the interpolation
    range of an evolutionary track.

    Parameters
    ----------
    track : EvolutionTrack

    age : float

    Returns
    -------
    bool
    """

    return (
        track.age[0]
        <= age
        <= track.age[-1]
    )

def contains_mass(
    isochrone: Isochrone,
    mass: float,
) -> bool:
    """
    Return whether a stellar mass lies within the
    interpolation range of an isochrone.

    Parameters
    ----------
    isochrone : Isochrone

    mass : float

    Returns
    -------
    bool
    """

    return (
        isochrone.mass[0]
        <= mass
        <= isochrone.mass[-1]
    )

def sample_track(track: EvolutionTrack, age: float) -> EvolutionSample:
    """
    Interpolate a stellar evolution track at a given age.
    """

    phase = None

    if track.phase is not None:
        index = np.abs(track.age - age).argmin()
        phase = int(track.phase[index])

    return EvolutionSample(
        age=age,
        initial_mass=track.initial_mass,
        metallicity=track.metallicity,
        log_teff=_interp(age, track.age, track.log_teff),
        log_luminosity=_interp(age, track.age, track.log_luminosity),
        log_radius=_interp(age, track.age, track.log_radius),
        log_g=_interp(age, track.age, track.log_g),
        phase=phase,
    )


def sample_isochrone(
    isochrone: Isochrone,
    mass: float,
) -> EvolutionSample:
    """
    Interpolate an isochrone at a given stellar mass.
    """

    phase = None

    if isochrone.phase is not None:
        index = np.abs(isochrone.mass - mass).argmin()
        phase = int(isochrone.phase[index])

    return EvolutionSample(
        age=isochrone.age,
        mass=mass,
        metallicity=isochrone.metallicity,
        log_teff=_interp(
            mass,
            isochrone.mass,
            isochrone.log_teff,
        ),
        log_luminosity=_interp(
            mass,
            isochrone.mass,
            isochrone.log_luminosity,
        ),
        log_radius=_interp(
            mass,
            isochrone.mass,
            isochrone.log_radius,
        ),
        log_g=_interp(
            mass,
            isochrone.mass,
            isochrone.log_g,
        ),
        phase=phase,
    )