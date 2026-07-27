from pathlib import Path

from pyseshat.evolution.models import (
    EvolutionModel,
)

from pyseshat.evolution.parsers import (
    load_parsec_track,
    load_parsec_isochrone,
)


# ==========================================================
# Public API
# ==========================================================

_SUPPORTED_MODELS = {
    "parsec": "load_parsec",
}


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