from pathlib import Path

from pyseshat.evolution.models import (
    EvolutionModel,
)

from pyseshat.evolution.parsers import (
    load_parsec_track,
    load_parsec_isochrone,
    load_girardi2000_track,
    load_girardi2000_isochrone,
)

from pyseshat.evolution.parsers.girardi2000 import (
    DEFAULT_OVERSHOOT,
    SOLAR_METALLICITY,
    load_girardi2000_vizier_isochrones,
    load_girardi2000_vizier_tracks,
)

#: Conventional filenames for the real VizieR J/A+AS/141/371 export,
#: as downloaded 2026-08-29 (see docs/project_decisions.md's dated
#: entry for the exact query/provenance). load_default_evolution_model()
#: looks for exactly these two names at its data directory.
VIZIER_TRACKS_FILENAME = "mstracks.tsv"
VIZIER_ISOCHRONES_FILENAME = "isochrones.tsv"


# ==========================================================
# Public API
# ==========================================================

_SUPPORTED_MODELS = {
    "parsec": "load_parsec",
    "girardi2000": "load_girardi2000",
}

#: Default evolutionary-track/isochrone grid for this project's own
#: evolutionary consistency checks, chosen for direct comparability
#: with the Al-Wardat SED-fitting literature this project benchmarks
#: against (see docs/guide/evolutionary_checks.md and
#: docs/project_decisions.md's 2026-08-29 dated entry for the full
#: rationale). PARSEC remains fully supported as an alternate model
#: -- this only changes which grid callers get if they don't specify
#: one, via load_default_evolution_model() below; load_model() itself
#: is unchanged and still requires an explicit model name.
DEFAULT_EVOLUTION_MODEL_NAME = "girardi2000"


def load_model(
    model,
    **kwargs,
):
    """
    Load a stellar evolution model.
    """

    model = model.lower()

    if model not in _SUPPORTED_MODELS:

        supported = ", ".join(
            sorted(_SUPPORTED_MODELS)
        )

        raise ValueError(

            f"Unsupported evolution model "
            f"'{model}'. "
            f"Supported models: {supported}"

        )

    return globals()[

        _SUPPORTED_MODELS[model]

    ](**kwargs)


def _default_data_dir():

    return (
        Path(__file__).resolve().parents[3]
        / "data"
        / "evolution"
        / "girardi2000"
    )


def load_girardi2000_from_vizier(
    track_path,
    isochrone_path,
    *,
    metallicity=SOLAR_METALLICITY,
    overshoot=DEFAULT_OVERSHOOT,
):
    """
    Build a full Girardi (2000) EvolutionModel from the real
    VizieR J/A+AS/141/371 exports (tables ``mstracks`` and
    ``isochron``).

    This is the entry point ``load_default_evolution_model()``
    uses once real data files are present (see below) -- exposed
    directly too, since a caller may want a metallicity/overshoot
    other than this project's own default, or to load the two
    files from a non-conventional location.

    Parameters
    ----------
    track_path : path-like
        Path to the downloaded ``mstracks.tsv`` file.

    isochrone_path : path-like
        Path to the downloaded ``isochrones.tsv`` file.

    metallicity : float, default SOLAR_METALLICITY (0.019)

    overshoot : int, default DEFAULT_OVERSHOOT (1)

    Returns
    -------
    EvolutionModel
        ``name="Girardi2000"``, with tracks/isochrones from the
        two files at the requested (metallicity, overshoot). See
        ``pyseshat.evolution.parsers.girardi2000``'s module
        docstring for the real-data handling this performs
        (radius derivation, Stage/phase, non-monotonic-age
        truncation, duplicate-mass deduplication).
    """

    tracks = load_girardi2000_vizier_tracks(
        track_path,
        metallicity=metallicity,
        overshoot=overshoot,
    )

    isochrones = load_girardi2000_vizier_isochrones(
        isochrone_path,
        metallicity=metallicity,
        overshoot=overshoot,
    )

    return EvolutionModel(

        name="Girardi2000",

        tracks=tracks,

        isochrones=isochrones,

    )


def load_default_evolution_model(
    *,
    data_dir=None,
    metallicity=SOLAR_METALLICITY,
    overshoot=DEFAULT_OVERSHOOT,
):
    """
    Load this project's default evolutionary-track/isochrone
    grid: Girardi et al. (2000), A&AS, 141, 371, at solar
    metallicity (Z=0.019) with core convective overshooting
    (the grid's full, main set -- see
    ``pyseshat.evolution.parsers.girardi2000``'s module
    docstring).

    **[2026-08-29] Now backed by real data.** Looks for
    ``mstracks.tsv`` and ``isochrones.tsv`` (the real VizieR
    J/A+AS/141/371 exports -- see
    ``docs/project_decisions.md``'s dated entry for the exact
    provenance) at ``data_dir`` and, if both are present, loads
    them via :func:`load_girardi2000_from_vizier`. This was a
    stub raising ``FileNotFoundError``/``NotImplementedError``
    before real data was obtained; that failure mode is preserved
    below for the case where the files genuinely are not there
    yet, so a caller in an environment without this data still
    gets an honest, informative error rather than a silent empty
    model.

    Parameters
    ----------
    data_dir : path-like, optional
        Directory expected to contain ``mstracks.tsv`` and
        ``isochrones.tsv``. Defaults to
        ``<repo>/data/evolution/girardi2000/``.

    metallicity : float, default SOLAR_METALLICITY (0.019)

    overshoot : int, default DEFAULT_OVERSHOOT (1)

    Returns
    -------
    EvolutionModel

    Raises
    ------
    FileNotFoundError
        If the data directory or either expected file is
        missing. The error message states exactly where to put
        them and points to the sourcing options described in
        ``pyseshat.evolution.parsers.girardi2000``'s module
        docstring and ``docs/guide/evolutionary_checks.md``.
    """

    if data_dir is None:
        data_dir = _default_data_dir()
    else:
        data_dir = Path(data_dir)

    track_path = data_dir / VIZIER_TRACKS_FILENAME
    isochrone_path = data_dir / VIZIER_ISOCHRONES_FILENAME

    if not track_path.exists() or not isochrone_path.exists():

        missing = [
            str(p)
            for p in (track_path, isochrone_path)
            if not p.exists()
        ]

        raise FileNotFoundError(

            "Girardi (2000) evolutionary-track/isochrone data "
            f"not found: missing {missing}. This grid is "
            "external data and is not bundled with the "
            "repository by default (see "
            "pyseshat.evolution.parsers.girardi2000's module "
            "docstring and docs/guide/evolutionary_checks.md). "
            "Download the VizieR J/A+AS/141/371 'mstracks' and "
            "'isochron' tables (ASU-TSV format) from "
            "https://vizier.cds.unistra.fr/viz-bin/VizieR?"
            "-source=J/A+AS/141/371, or the tracks/isochrones "
            "at Z=0.019 from the Padova group's legacy CMD 2.7 "
            "interface (https://stev.oapd.inaf.it/cgi-bin/cmd_2.7, "
            "select 'Girardi et al. (2000) + simplified TP-AGB "
            "(for M<=7 Msun) + Bertelli et al. 1994'), and place "
            f"them at '{data_dir}' named 'mstracks.tsv' and "
            "'isochrones.tsv' respectively -- or call "
            "load_girardi2000_from_vizier(track_path, "
            "isochrone_path, ...) directly with wherever they "
            "actually are."

        )

    return load_girardi2000_from_vizier(
        track_path,
        isochrone_path,
        metallicity=metallicity,
        overshoot=overshoot,
    )


# ==========================================================
# PARSEC
# ==========================================================

def load_parsec(
    *,
    track_files=(),
    isochrone_files=(),
    track_column_map,
    isochrone_column_map,
    track_metadata,
    isochrone_metadata,
):
    """
    Build an EvolutionModel from PARSEC files.

    Parameters
    ----------
    track_files : iterable[path-like]

    isochrone_files : iterable[path-like]

    track_column_map : dict

    isochrone_column_map : dict

    track_metadata : iterable[dict]

    isochrone_metadata : iterable[dict]
    """

    if len(track_files) != len(track_metadata):
        raise ValueError(
            "track_files and track_metadata "
            "must have the same length."
        )

    if len(isochrone_files) != len(isochrone_metadata):
        raise ValueError(
            "isochrone_files and "
            "isochrone_metadata "
            "must have the same length."
        )

    tracks = []

    for filename, meta in zip(
        track_files,
        track_metadata,
    ):

        tracks.append(

            load_parsec_track(

                filename,

                column_map=track_column_map,

                initial_mass=meta["initial_mass"],

                metallicity=meta["metallicity"],

            )

        )

    isochrones = []

    for filename, meta in zip(
        isochrone_files,
        isochrone_metadata,
    ):

        isochrones.append(

            load_parsec_isochrone(

                filename,

                column_map=isochrone_column_map,

                age=meta["age"],

                metallicity=meta["metallicity"],

            )

        )

    return EvolutionModel(

        name="PARSEC",

        tracks=tuple(tracks),

        isochrones=tuple(isochrones),

    )


# ==========================================================
# Girardi (2000)
# ==========================================================

def load_girardi2000(
    *,
    track_files=(),
    isochrone_files=(),
    track_column_map,
    isochrone_column_map,
    track_metadata,
    isochrone_metadata,
):
    """
    Build an EvolutionModel from Girardi et al. (2000) files.

    See pyseshat.evolution.parsers.girardi2000's module
    docstring for the full citation, metallicity coverage, and
    data-sourcing details. Z=0.019 is this grid's solar
    composition value and the appropriate metallicity for this
    project's Sun-like benchmark components.

    Parameters
    ----------
    track_files : iterable[path-like]

    isochrone_files : iterable[path-like]

    track_column_map : dict

    isochrone_column_map : dict

    track_metadata : iterable[dict]

    isochrone_metadata : iterable[dict]
    """

    if len(track_files) != len(track_metadata):
        raise ValueError(
            "track_files and track_metadata "
            "must have the same length."
        )

    if len(isochrone_files) != len(isochrone_metadata):
        raise ValueError(
            "isochrone_files and "
            "isochrone_metadata "
            "must have the same length."
        )

    tracks = []

    for filename, meta in zip(
        track_files,
        track_metadata,
    ):

        tracks.append(

            load_girardi2000_track(

                filename,

                column_map=track_column_map,

                initial_mass=meta["initial_mass"],

                metallicity=meta["metallicity"],

            )

        )

    isochrones = []

    for filename, meta in zip(
        isochrone_files,
        isochrone_metadata,
    ):

        isochrones.append(

            load_girardi2000_isochrone(

                filename,

                column_map=isochrone_column_map,

                age=meta["age"],

                metallicity=meta["metallicity"],

            )

        )

    return EvolutionModel(

        name="Girardi2000",

        tracks=tuple(tracks),

        isochrones=tuple(isochrones),

    )