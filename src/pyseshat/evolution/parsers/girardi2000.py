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
knowing the literal CDS column headers in advance. This describes
``load_girardi2000_track()``/``load_girardi2000_isochrone()`` below,
for a file that already holds exactly one track or one isochrone.

Real VizieR exports: ``load_girardi2000_vizier_tracks()`` /
``load_girardi2000_vizier_isochrones()``
------------------------------------------------------------------
As of 2026-08-29, real data was obtained: VizieR catalog
J/A+AS/141/371's two ASU-TSV tables, ``mstracks`` (13,723 rows, "main
sequence [tracks], from ZAMS on") and ``isochron`` (64,381 rows,
"Isochrones") -- see ``docs/project_decisions.md``'s matching dated
entry for the exact provenance and query. Inspecting these directly
(rather than assuming they matched the schema above) found several
real-data quirks the single-track ``column_map`` functions above do
not handle, so two dedicated functions were added instead of forcing
these files through them:

- Each file is **one VizieR ASU-TSV export covering all six
  metallicities and both tracks stacked into one table** (one row per
  (mass, age) point for ``mstracks``, one row per (age, mass) point
  for ``isochron``), not a single already-selected track/isochrone --
  the opposite of what ``column_map`` above assumes. A real column
  ``Z`` (exact float metallicity) and ``Overshoot`` (0/1 flag) let a
  caller select a slice directly, rather than needing one file per
  metallicity.
- **Both tables include an ``Overshoot`` column: 0 (no core
  convective overshooting) or 1 (with overshooting).** The
  ``Overshoot=1`` set is the grid's full, main set -- e.g. for
  Z=0.019 it spans the complete published 0.15-7.0 Msun range across
  34 initial masses. ``Overshoot=0`` is a much smaller supplementary
  set covering only 1.2-8.0 Msun (17 masses) -- a comparison subset,
  not a complete alternative grid. ``Overshoot=1`` (the paper's own
  standard treatment for intermediate/high-mass stars) is therefore
  the correct default and is what both loaders below select unless
  told otherwise.
- **Neither table has a radius column at all.** ``log_radius`` is
  derived from ``logL`` and ``logTeff`` via the Stefan-Boltzmann
  relation, using this codebase's own
  :data:`pyseshat.analysis.luminosity.SOLAR_EFFECTIVE_TEMPERATURE`
  (5772 K) so the derivation is exactly self-consistent with
  ``compute_log_luminosity()``'s own inverse relation used elsewhere
  in this codebase, rather than picking an independent solar Teff.
- **``mstracks``' ``Stage`` column is a sparse text label** (e.g.
  "ZAMS", "F", "Tip", "1tp" -- 15 distinct labels, ~88% blank),
  matching the paper's own description ("Label indicating particular
  evolutionary stages ... see the paper for details"). This does not
  fit :attr:`pyseshat.evolution.models.EvolutionTrack.phase`'s
  existing numeric contract (``sample_track()`` does
  ``int(track.phase[index])``, which a string or a mostly-missing
  column cannot satisfy) -- so it is intentionally **not** loaded
  into ``phase`` here (left ``None``), rather than forcing a bad
  cast or silently mis-mapping it to something numeric-looking.
  ``isochron`` has no stage/phase column at all.
- **A minority of tracks have a non-monotonic tail in ``Age``**
  (repeated or decreasing age values late in the track -- 12 of 34
  mass tracks at Z=0.019/Overshoot=1). This is a real, known feature
  of these tracks during late thermal-pulse AGB evolution (multiple
  near-identical model time-steps), exactly the situation
  :func:`pyseshat.evolution.interpolate._interp`'s own docstring
  already anticipated ("a track with a post-AGB or blue-loop segment
  where age ... is not monotonic"), not a data-quality problem to
  paper over. Each track is truncated at its first non-monotonic
  point rather than silently mis-sorted or force-fit. Checked
  directly against this project's six benchmark systems' component
  masses (~0.8-2.6 Msun): for masses <=1.0 Msun the truncation point
  is 12-78 Gyr (irrelevant -- older than the universe); for the
  higher-mass tracks relevant to Capella (~2.5 Msun) there is no
  non-monotonicity at all up to that track's natural endpoint. No
  benchmark system loses any physically relevant age coverage from
  this truncation.
- **A minority of isochrone age-slices have duplicate (not
  decreasing) initial-mass values** (27 of 50 age-slices at
  Z=0.019/Overshoot=1, from blue-loop mass-grid crossings) which
  :func:`pyseshat.evolution.interpolate._interp` also rejects (it
  requires strictly increasing, not just non-decreasing, x-values).
  Duplicate ``Mini`` rows are dropped (keeping the first occurrence)
  per age-slice before constructing each :class:`Isochrone`.

Both real-data quirks above were found and handled by direct
inspection of the actual downloaded files (row counts, per-track
monotonicity, per-age-slice duplicate checks) before writing the
loaders, not assumed from the paper text -- see
``docs/project_decisions.md``'s 2026-08-29 entry for the full
diagnostic trail.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from pyseshat.analysis.luminosity import (
    SOLAR_EFFECTIVE_TEMPERATURE,
)

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


# ==========================================================
# Real VizieR exports (J/A+AS/141/371, tables mstracks/isochron)
# ==========================================================

#: Column present in both real VizieR tables selecting the
#: overshooting treatment: 0 = no core convective overshooting,
#: 1 = with overshooting. 1 is the grid's full, main set (see this
#: module's docstring) and is the default used below.
DEFAULT_OVERSHOOT = 1


def _log_radius_from_l_teff(log_luminosity, log_teff):
    """
    Derive log10(R/Rsun) from log10(L/Lsun) and log10(Teff) via
    the Stefan-Boltzmann law: L/Lsun = (R/Rsun)^2 (Teff/Teffsun)^4,
    i.e. log(R/Rsun) = 0.5*log(L/Lsun) - 2*log(Teff) +
    2*log(Teffsun).

    Uses this codebase's own SOLAR_EFFECTIVE_TEMPERATURE
    (pyseshat.analysis.luminosity, 5772 K) rather than an
    independently-chosen solar Teff, so that a radius derived here
    and fed back through compute_log_luminosity() elsewhere in this
    codebase is exactly self-consistent (round-trips to the same
    log_luminosity, not just approximately).

    Parameters
    ----------
    log_luminosity : array_like
        log10(L / Lsun).
    log_teff : array_like
        log10(Teff / K).

    Returns
    -------
    ndarray
        log10(R / Rsun).
    """

    return (
        0.5 * np.asarray(log_luminosity, dtype=float)
        - 2.0 * np.asarray(log_teff, dtype=float)
        + 2.0 * np.log10(SOLAR_EFFECTIVE_TEMPERATURE)
    )


def _read_vizier_asu_tsv(filename):
    """
    Parse one VizieR ASU-TSV export (as produced by
    ``https://vizier.cds.unistra.fr/viz-bin/asu-tsv?...``) into a
    pandas DataFrame.

    The format is: an arbitrary number of ``#``-prefixed metadata
    lines (server info, column descriptions, etc., possibly
    interspersed with blank lines), then exactly three lines with no
    leading ``#`` -- the real tab-separated column-name header, a
    units row, and a dashes/field-width row (whose first field is a
    literal ``-``) -- followed by the actual tab-separated data.
    This is what both ``mstracks`` and ``isochron`` (VizieR catalog
    J/A+AS/141/371) look like as downloaded 2026-08-29; the dashes
    row's distinctive first field is used to locate the header
    reliably rather than assuming a fixed line count, since the
    metadata block's length depends on the query's requested
    columns.

    Parameters
    ----------
    filename : path-like

    Returns
    -------
    pandas.DataFrame
        One row per line of real data, columns named exactly as in
        the file's own header row (e.g. "Mass", "logTeff", "Stage"
        for ``mstracks``; "Mini", "age" for ``isochron`` -- note the
        real, inconsistent capitalization of "Age"/"age" between the
        two tables, preserved here rather than normalized, so a
        caller must match the real column name).
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    with open(filename) as f:
        lines = f.readlines()

    dashes_index = None

    for i, line in enumerate(lines):

        if line.split("\t", 1)[0].strip() == "-":
            dashes_index = i
            break

    if dashes_index is None or dashes_index < 2:

        raise ValueError(
            f"Could not find the VizieR column-header/units/"
            f"dashes block in {filename} -- expected a line "
            "whose first tab-separated field is a literal '-' "
            "preceded by a column-name header line two rows "
            "above it. This file may not be a VizieR ASU-TSV "
            "export in the expected format."
        )

    header_index = dashes_index - 2

    columns = [
        c.strip()
        for c in lines[header_index].rstrip("\n").split("\t")
    ]

    return pd.read_csv(
        filename,
        sep="\t",
        header=None,
        names=columns,
        skiprows=dashes_index + 1,
        skip_blank_lines=True,
        na_values=[""],
    )


def load_girardi2000_vizier_tracks(
    filename,
    *,
    metallicity=SOLAR_METALLICITY,
    overshoot=DEFAULT_OVERSHOOT,
):
    """
    Load evolutionary tracks from a real VizieR ``mstracks``
    export (catalog J/A+AS/141/371, table
    J/A+AS/141/371/mstracks -- "Evolutionary tracks main sequence
    from ZAMS on"), one :class:`EvolutionTrack` per distinct
    initial mass at the requested metallicity/overshoot.

    See this module's docstring for the real-data handling this
    performs (radius derivation, Stage/phase, non-monotonic-age
    truncation) and why.

    Parameters
    ----------
    filename : path-like
        Path to the downloaded ``mstracks.tsv`` file.

    metallicity : float, default SOLAR_METALLICITY (0.019)
        Which of the six tabulated Z values to select. Compared
        with a small floating-point tolerance rather than exact
        equality, in case a different downloaded copy carries
        extra decimal noise.

    overshoot : int, default DEFAULT_OVERSHOOT (1)
        Which overshooting treatment to select -- 0 or 1. See
        this module's docstring for why 1 is the appropriate
        default.

    Returns
    -------
    tuple[EvolutionTrack, ...]
        One track per distinct initial mass present at the
        requested (metallicity, overshoot), sorted by age within
        each track and truncated before any non-monotonic age
        tail. Tracks left with fewer than two usable points after
        truncation are omitted (nothing to interpolate against).

    Raises
    ------
    ValueError
        If the file is missing an expected column, or no rows
        match the requested metallicity/overshoot (the error lists
        the metallicities and overshoot values actually present).
    """

    table = _read_vizier_asu_tsv(filename)

    required_columns = (
        "Overshoot", "Mass", "Z", "Age",
        "logL", "logTeff", "logg",
    )

    missing = [c for c in required_columns if c not in table.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing expected column(s) "
            f"{missing}; found columns {list(table.columns)}."
        )

    subset = table[
        (table["Overshoot"] == overshoot)
        & np.isclose(table["Z"], metallicity, rtol=1e-6, atol=1e-8)
    ]

    if subset.empty:
        raise ValueError(
            f"No rows in {filename} match metallicity="
            f"{metallicity}, overshoot={overshoot}. Metallicities "
            f"present: {sorted(table['Z'].unique())}. Overshoot "
            f"values present: {sorted(table['Overshoot'].unique())}."
        )

    tracks = []

    for mass, group in subset.groupby("Mass"):

        group = group.sort_values("Age")

        age = group["Age"].to_numpy(dtype=float)

        diffs = np.diff(age)

        non_monotonic = np.where(diffs <= 0)[0]

        if len(non_monotonic):
            cutoff = int(non_monotonic[0]) + 1
            group = group.iloc[:cutoff]
            age = age[:cutoff]

        if len(age) < 2:
            continue

        log_teff = group["logTeff"].to_numpy(dtype=float)
        log_luminosity = group["logL"].to_numpy(dtype=float)
        log_g = group["logg"].to_numpy(dtype=float)

        tracks.append(
            EvolutionTrack(
                initial_mass=float(mass),
                metallicity=float(metallicity),
                age=age,
                log_teff=log_teff,
                log_luminosity=log_luminosity,
                log_radius=_log_radius_from_l_teff(
                    log_luminosity, log_teff
                ),
                log_g=log_g,
                phase=None,
            )
        )

    return tuple(tracks)


def load_girardi2000_vizier_isochrones(
    filename,
    *,
    metallicity=SOLAR_METALLICITY,
    overshoot=DEFAULT_OVERSHOOT,
):
    """
    Load isochrones from a real VizieR ``isochron`` export
    (catalog J/A+AS/141/371, table J/A+AS/141/371/isochron --
    "Isochrones"), one :class:`Isochrone` per distinct age at the
    requested metallicity/overshoot.

    See this module's docstring for the real-data handling this
    performs (radius derivation, duplicate-mass deduplication) and
    why.

    Parameters
    ----------
    filename : path-like
        Path to the downloaded ``isochrones.tsv`` file.

    metallicity : float, default SOLAR_METALLICITY (0.019)

    overshoot : int, default DEFAULT_OVERSHOOT (1)

    Returns
    -------
    tuple[Isochrone, ...]
        One isochrone per distinct age present at the requested
        (metallicity, overshoot), sorted by (initial) mass with
        duplicate mass values dropped. Age-slices left with fewer
        than two usable points after deduplication are omitted.

    Raises
    ------
    ValueError
        If the file is missing an expected column, or no rows
        match the requested metallicity/overshoot.
    """

    table = _read_vizier_asu_tsv(filename)

    required_columns = (
        "Overshoot", "Z", "age", "Mini",
        "logL", "logTeff", "logg",
    )

    missing = [c for c in required_columns if c not in table.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing expected column(s) "
            f"{missing}; found columns {list(table.columns)}."
        )

    subset = table[
        (table["Overshoot"] == overshoot)
        & np.isclose(table["Z"], metallicity, rtol=1e-6, atol=1e-8)
    ]

    if subset.empty:
        raise ValueError(
            f"No rows in {filename} match metallicity="
            f"{metallicity}, overshoot={overshoot}. Metallicities "
            f"present: {sorted(table['Z'].unique())}. Overshoot "
            f"values present: {sorted(table['Overshoot'].unique())}."
        )

    isochrones = []

    for age, group in subset.groupby("age"):

        group = group.sort_values("Mini")

        group = group.drop_duplicates(
            subset="Mini", keep="first"
        )

        mass = group["Mini"].to_numpy(dtype=float)

        if len(mass) < 2:
            continue

        log_teff = group["logTeff"].to_numpy(dtype=float)
        log_luminosity = group["logL"].to_numpy(dtype=float)
        log_g = group["logg"].to_numpy(dtype=float)

        isochrones.append(
            Isochrone(
                age=float(age),
                metallicity=float(metallicity),
                mass=mass,
                log_teff=log_teff,
                log_luminosity=log_luminosity,
                log_radius=_log_radius_from_l_teff(
                    log_luminosity, log_teff
                ),
                log_g=log_g,
                phase=None,
            )
        )

    return tuple(isochrones)
