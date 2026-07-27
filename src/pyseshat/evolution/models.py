from dataclasses import dataclass

import numpy as np


# ==========================================================
# Evolution track
# ==========================================================

@dataclass(frozen=True)
class EvolutionTrack:
    """
    Stellar evolution track for a single initial mass.

    Each array must have the same length and represent
    successive evolutionary stages.

    Parameters
    ----------
    initial_mass : float
        Initial stellar mass (Msun).

    metallicity : float
        Metallicity [M/H] or equivalent grid value.

    age : ndarray
        Stellar age (yr).

    log_teff : ndarray
        log10(Teff / K).

    log_luminosity : ndarray
        log10(L / Lsun).

    log_radius : ndarray
        log10(R / Rsun).

    log_g : ndarray
        log10(surface gravity).

    phase : ndarray, optional
        Evolutionary phase identifiers.
    """

    initial_mass: float

    metallicity: float

    age: np.ndarray

    log_teff: np.ndarray

    log_luminosity: np.ndarray

    log_radius: np.ndarray

    log_g: np.ndarray

    phase: np.ndarray | None = None

    def __post_init__(self):

        arrays = [

            self.age,

            self.log_teff,

            self.log_luminosity,

            self.log_radius,

            self.log_g

        ]

        lengths = {

            len(a)

            for a in arrays

        }

        if self.phase is not None:

            lengths.add(

                len(self.phase)

            )

        if len(lengths) != 1:

            raise ValueError(

                "All track arrays must have the same length."

            )


# ==========================================================
# Isochrone
# ==========================================================

@dataclass(frozen=True)
class Isochrone:
    """
    Stellar isochrone for a fixed age.

    Parameters
    ----------
    age : float
        Age (yr).

    metallicity : float

    mass : ndarray
        Initial stellar masses.

    log_teff : ndarray

    log_luminosity : ndarray

    log_radius : ndarray

    log_g : ndarray

    phase : ndarray, optional
    """

    age: float

    metallicity: float

    mass: np.ndarray

    log_teff: np.ndarray

    log_luminosity: np.ndarray

    log_radius: np.ndarray

    log_g: np.ndarray

    phase: np.ndarray | None = None

    def __post_init__(self):

        arrays = [

            self.mass,

            self.log_teff,

            self.log_luminosity,

            self.log_radius,

            self.log_g

        ]

        lengths = {

            len(a)

            for a in arrays

        }

        if self.phase is not None:

            lengths.add(

                len(self.phase)

            )

        if len(lengths) != 1:

            raise ValueError(

                "All isochrone arrays must have the same length."

            )


# ==========================================================
# Evolution model
# ==========================================================

@dataclass(frozen=True)
class EvolutionModel:
    """
    Collection of stellar evolution grids.

    Parameters
    ----------
    name : str
        Model family name (e.g. PARSEC, MIST).

    tracks : tuple[EvolutionTrack]

    isochrones : tuple[Isochrone]
    """

    name: str

    tracks: tuple[EvolutionTrack, ...]

    isochrones: tuple[Isochrone, ...]

    @property
    def available_metallicities(self):

        values = {

            track.metallicity

            for track in self.tracks

        }

        values.update(

            iso.metallicity

            for iso in self.isochrones

        )

        return tuple(

            sorted(values)

        )

    @property
    def available_track_masses(self):

        return tuple(

            sorted(

                track.initial_mass

                for track in self.tracks

            )

        )

    @property
    def available_ages(self):

        return tuple(

            sorted(

                iso.age

                for iso in self.isochrones

            )

        )