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


def load_default_evolution_model(
    *,
    data_dir=None,
):
    """
    Load this project's default evolutionary-track/isochrone
    grid: Girardi et al. (2000), A&AS, 141, 371, at solar
    metallicity (Z=0.019).

    This is a thin convenience wrapper, not a new data source:
    it looks for already-downloaded Girardi (2000) track and
    isochrone files under a conventional directory and, if
    present, loads them via load_model("girardi2000", ...)
    exactly as a caller would by hand. **No Girardi (2000) data
    files are bundled with this repository** (the same
    "external data, not shipped" situation that has always
    applied to the pre-existing PARSEC parser -- see
    docs/guide/evolutionary_checks.md) -- this function exists
    so that once a user has downloaded the grid (from the
    Padova group's legacy CMD 2.7 interface or from VizieR
    catalog J/A+AS/141/371) and placed it at the conventional
    path below, every part of this codebase that wants "the
    default grid" can call one function rather than
    special-casing file discovery per script.

    Parameters
    ----------
    data_dir : path-like, optional
        Directory containing the downloaded Girardi (2000)
        ASCII track/isochrone files, plus a
        ``column_map.json`` (or equivalent) describing their
        column layout. Defaults to
        ``<repo>/data/evolution/girardi2000/``.

    Raises
    ------
    FileNotFoundError
        If no data has been placed at the expected location.
        The error message states exactly where to put it and
        points to the two sourcing options described in this
        module's own docstring and in
        docs/guide/evolutionary_checks.md.
    """

    if data_dir is None:

        data_dir = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "evolution"
            / "girardi2000"
        )

    else:

        data_dir = Path(data_dir)

    if not data_dir.exists() or not any(
        data_dir.glob("*")
    ):

        raise FileNotFoundError(

            "No Girardi (2000) evolutionary-track/isochrone "
            f"data found at '{data_dir}'. This grid is "
            "external data and is not bundled with the "
            "repository (see "
            "pyseshat.evolution.parsers.girardi2000's module "
            "docstring and docs/guide/evolutionary_checks.md). "
            "Download the tracks/isochrones at Z=0.019 (solar) "
            "from the Padova group's legacy CMD 2.7 interface "
            "(https://stev.oapd.inaf.it/cgi-bin/cmd_2.7, select "
            "'Girardi et al. (2000) + simplified TP-AGB (for "
            "M<=7 Msun) + Bertelli et al. 1994') or from VizieR "
            "catalog J/A+AS/141/371, place the files at this "
            "path, and call load_model('girardi2000', ...) "
            "directly with the appropriate column_map (or "
            "extend this function to do so automatically once "
            "a fixed file layout is agreed on)."

        )

    raise NotImplementedError(

        "Girardi (2000) data files were found at "
        f"'{data_dir}', but load_default_evolution_model() "
        "does not yet know this project's agreed-on file "
        "naming/column-map convention for them (none has been "
        "fixed yet, since no real data has been loaded through "
        "this path so far -- see "
        "docs/guide/evolutionary_checks.md). Call "
        "load_model('girardi2000', track_files=..., "
        "isochrone_files=..., track_column_map=..., "
        "isochrone_column_map=..., track_metadata=..., "
        "isochrone_metadata=...) directly instead, or extend "
        "this function once that convention is fixed."

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