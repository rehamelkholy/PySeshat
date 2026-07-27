import pytest

from pyseshat.evolution.grids import (
    load_model,
    load_parsec,
)


# ==========================================================
# Unsupported models
# ==========================================================

def test_unknown_model():

    with pytest.raises(ValueError):

        load_model("unknown")


# ==========================================================
# Empty PARSEC model
# ==========================================================

def test_empty_parsec_model():

    model = load_parsec(

        track_files=[],

        isochrone_files=[],

        track_column_map={},

        isochrone_column_map={},

        track_metadata=[],

        isochrone_metadata=[]

    )

    assert model.name == "PARSEC"

    assert model.tracks == ()

    assert model.isochrones == ()


# ==========================================================
# Public dispatcher
# ==========================================================

def test_load_model_dispatch():

    model = load_model(

        "parsec",

        track_files=[],

        isochrone_files=[],

        track_column_map={},

        isochrone_column_map={},

        track_metadata=[],

        isochrone_metadata=[]

    )

    assert model.name == "PARSEC"


# ==========================================================
# Metadata consistency
# ==========================================================

def test_track_metadata_length():

    with pytest.raises(ValueError):

        load_parsec(

            track_files=["a.dat"],

            isochrone_files=[],

            track_column_map={},

            isochrone_column_map={},

            track_metadata=[],

            isochrone_metadata=[]

        )


def test_iso_metadata_length():

    with pytest.raises(ValueError):

        load_parsec(

            track_files=[],

            isochrone_files=["a.dat"],

            track_column_map={},

            isochrone_column_map={},

            track_metadata=[],

            isochrone_metadata=[]

        )