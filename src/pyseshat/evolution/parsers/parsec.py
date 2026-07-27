from pathlib import Path

import numpy as np

from pyseshat.evolution.models import (
    EvolutionTrack,
    Isochrone,
)


# ==========================================================
# Utilities
# ==========================================================

def _load_ascii_table(filename):
    """
    Load a whitespace-delimited ASCII table.

    Comment lines beginning with '#' are ignored.

    Returns
    -------
    ndarray
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    return np.genfromtxt(
        filename,
        comments="#",
        names=True,
        dtype=float,
        autostrip=True,
    )


# ==========================================================
# PARSEC Track
# ==========================================================

def load_parsec_track(
    filename,
    *,
    column_map,
    initial_mass,
    metallicity,
):
    """
    Load one PARSEC evolutionary track.

    Parameters
    ----------
    filename : path-like

    column_map : dict
        Mapping from canonical names to
        PARSEC column names.

    initial_mass : float

    metallicity : float

    Returns
    -------
    EvolutionTrack
    """

    table = _load_ascii_table(filename)

    phase = None

    if "phase" in column_map:

        phase = table[
            column_map["phase"]
        ]

    return EvolutionTrack(

        initial_mass=initial_mass,

        metallicity=metallicity,

        age=table[
            column_map["age"]
        ],

        log_teff=table[
            column_map["log_teff"]
        ],

        log_luminosity=table[
            column_map["log_luminosity"]
        ],

        log_radius=table[
            column_map["log_radius"]
        ],

        log_g=table[
            column_map["log_g"]
        ],

        phase=phase,
    )


# ==========================================================
# PARSEC Isochrone
# ==========================================================

def load_parsec_isochrone(
    filename,
    *,
    column_map,
    age,
    metallicity,
):
    """
    Load one PARSEC isochrone.

    Parameters
    ----------
    filename : path-like

    column_map : dict

    age : float

    metallicity : float

    Returns
    -------
    Isochrone
    """

    table = _load_ascii_table(filename)

    phase = None

    if "phase" in column_map:

        phase = table[
            column_map["phase"]
        ]

    return Isochrone(

        age=age,

        metallicity=metallicity,

        mass=table[
            column_map["mass"]
        ],

        log_teff=table[
            column_map["log_teff"]
        ],

        log_luminosity=table[
            column_map["log_luminosity"]
        ],

        log_radius=table[
            column_map["log_radius"]
        ],

        log_g=table[
            column_map["log_g"]
        ],

        phase=phase,
    )