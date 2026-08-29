"""
Parser for Girardi et al. (2000) evolutionary tracks and isochrones.

Reference
---------
Girardi, L., Bressan, A., Bertelli, G., & Chiosi, C. 2000, A&AS, 141,
371, "Evolutionary tracks and isochrones for low- and intermediate-mass
stars: From 0.15 to 7 Msun, and from Z=0.0004 to 0.03",
doi:10.1051/aas:2000126.

This grid is the evolutionary-track/isochrone source consistently used
by the Al-Wardat SED-fitting literature this project benchmarks against
(the HD25811, HIP70973/HIP72479, and HIP102029 papers, and by extension
Abushattal et al. 2025, arXiv:2512.23652, whose four SB2 systems --
HD214222, HD191854, HD130669, HD184467 -- are this project's own SED
benchmarks). It is registered here as the ``"girardi2000"`` model in
:mod:`pyseshat.evolution.grids` alongside the pre-existing PARSEC
parser, so that evolutionary consistency checks can be run on the same
grid the literature uses, for direct comparability.

Metallicity coverage (from the paper itself, Sect. 2 and Table 1):
Z = 0.0004, 0.001, 0.004, 0.008, 0.019, 0.03, with Z = 0.019 (Y =
0.273) identified by the authors as their solar-composition value
("almost identical to the Z = 0.01886 favoured by Anders & Grevesse
1989"). Composition is solar-scaled (Grevesse & Noels 1993 mixture),
not alpha-enhanced. Mass coverage: 0.15-7 Msun. This covers every
component mass relevant to this project's benchmark systems (all are
Sun-like or subgiant/giant components below ~3 Msun) -- Z=0.019 is the
appropriate metallicity choice for all of them.

Data source
-----------
The tabulated tracks/isochrones themselves are external data, not
bundled with this repository (the same "bring your own files" pattern
already used by the pre-existing PARSEC parser --
:mod:`pyseshat.evolution.parsers.parsec` -- see that module and
``docs/guide/evolutionary_checks.md`` for why no track data ships in
the repo). They are available either via the Padova group's legacy CMD
2.7 web interface (https://stev.oapd.inaf.it/cgi-bin/cmd_2.7, which
still lists "Girardi et al. (2000) + simplified TP-AGB (for M<=7 Msun)
+ Bertelli et al. 1994" as a selectable track/isochrone set -- verified
present in that form as of 2026-08-29) or via the original VizieR
catalog, J/A+AS/141/371, at the CDS archive.

Like :mod:`pyseshat.evolution.parsers.parsec`, this module makes no
assumption about the exact column header spelling used in a
particular downloaded copy of the tables -- callers supply an explicit
``column_map`` from canonical field name to the actual column name
present in their file, exactly mirroring the PARSEC parser's own
design. This also means the parser's correctness does not depend on
knowing the literal CDS column headers in advance.
"""

from io import StringIO
from pathlib import Path

import numpy as np

from pyseshat.evolution.models import (
    EvolutionTrack,
    Isochrone,
)


# ==========================================================
# Citation metadata
# ==========================================================

CITATION = (
    "Girardi, L., Bressan, A., Bertelli, G., & Chiosi, C. 2000, "
    "A&AS, 141, 371, doi:10.1051/aas:2000126"
)

#: Metallicities tabulated by the grid (Sect. 2 of the paper).
AVAILABLE_METALLICITIES = (
    0.0004,
    0.001,
    0.004,
    0.008,
    0.019,
    0.03,
)

#: The paper's own identified solar-composition value
#: (Y = 0.273, "almost identical to ... Anders & Grevesse 1989").
SOLAR_METALLICITY = 0.019

#: Initial-mass coverage of the grid, in Msun.
MASS_RANGE_MSUN = (0.15, 7.0)


# ==========================================================
# Utilities
# ==========================================================

def _load_ascii_table(filename):
    """
    Load a whitespace-delimited ASCII table.

    Comment lines beginning with '#' are ignored. This mirrors
    :func:`pyseshat.evolution.parsers.parsec._load_ascii_table`
    exactly, since Girardi (2000) tables are distributed in the
    same general whitespace-delimited, commented-header ASCII
    format as PARSEC's -- including the fix (2026-08-29) for
    ``numpy.genfromtxt(..., comments="#")`` only stripping the
    "#" character rather than the whole comment line, which
    otherwise misdetects the column count whenever a comment
    line's remaining text has a different word count than the
    real header row. See the PARSEC parser's copy of this
    function for the full explanation.

    Returns
    -------
    ndarray
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    lines = [
        line
        for line in filename.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not lines:
        raise ValueError(
            f"No data found in {filename} "
            "(file is empty or contains only comments)."
        )

    return np.genfromtxt(
        StringIO("\n".join(lines)),
        names=True,
        dtype=float,
        autostrip=True,
    )


# ==========================================================
# Girardi (2000) track
# ==========================================================

def load_girardi2000_track(
    filename,
    *,
    column_map,
    initial_mass,
    metallicity,
):
    """
    Load one Girardi et al. (2000) evolutionary track.

    Parameters
    ----------
    filename : path-like

    column_map : dict
        Mapping from canonical names ("age", "log_teff",
        "log_luminosity", "log_radius", "log_g", and
        optionally "phase") to the column names actually
        present in the downloaded file. The archived Girardi
        (2000) track tables (CDS catalog J/A+AS/141/371)
        provide 21 columns including age, log(L/Lsun),
        log(Teff), surface gravity, central conditions, and an
        evolutionary-stage flag; only the five/six canonical
        quantities above are used here.

    initial_mass : float

    metallicity : float
        Should be one of AVAILABLE_METALLICITIES for an
        unmodified download of the grid; SOLAR_METALLICITY
        (0.019) is the appropriate choice for this project's
        Sun-like benchmark components.

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
# Girardi (2000) isochrone
# ==========================================================

def load_girardi2000_isochrone(
    filename,
    *,
    column_map,
    age,
    metallicity,
):
    """
    Load one Girardi et al. (2000) isochrone.

    Parameters
    ----------
    filename : path-like

    column_map : dict
        Mapping from canonical names ("mass", "log_teff",
        "log_luminosity", "log_radius", "log_g", and
        optionally "phase") to the column names actually
        present in the downloaded file. The archived Girardi
        (2000) isochrone tables provide 16 columns including
        age, initial and actual mass, log(L/Lsun), log(Teff),
        log(g), and broadband absolute magnitudes; only the
        canonical quantities above are used here.

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
