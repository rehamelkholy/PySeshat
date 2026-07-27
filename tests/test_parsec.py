import numpy as np

import pytest

from pyseshat.evolution.models import (
    EvolutionTrack,
    Isochrone,
)

from pyseshat.evolution.parsers.parsec import (
    _load_ascii_table,
    load_parsec_track,
    load_parsec_isochrone,
)


# ==========================================================
# Helpers
# ==========================================================

TRACK_CONTENT = """\
age logTe logL logR logg label
1.0e6 3.80 0.10 0.00 4.50 1
2.0e6 3.79 0.12 0.01 4.49 1
3.0e6 3.78 0.15 0.02 4.48 2
"""


ISOCHRONE_CONTENT = """\
Mini logTe logL logR logg label
0.8 3.74 -0.30 -0.05 4.60 1
1.0 3.76 0.00 0.00 4.50 2
1.2 3.80 0.30 0.10 4.35 3
"""


def make_track_file(tmp_path):

    filename = tmp_path / "track.dat"

    filename.write_text(TRACK_CONTENT)

    return filename


def make_iso_file(tmp_path):

    filename = tmp_path / "iso.dat"

    filename.write_text(ISOCHRONE_CONTENT)

    return filename


TRACK_COLUMNS = {

    "age": "age",

    "log_teff": "logTe",

    "log_luminosity": "logL",

    "log_radius": "logR",

    "log_g": "logg",

    "phase": "label",

}


ISO_COLUMNS = {

    "mass": "Mini",

    "log_teff": "logTe",

    "log_luminosity": "logL",

    "log_radius": "logR",

    "log_g": "logg",

    "phase": "label",

}


# ==========================================================
# ASCII loader
# ==========================================================

def test_load_ascii_table(tmp_path):

    filename = make_track_file(tmp_path)

    table = _load_ascii_table(filename)

    assert len(table) == 3

    assert "age" in table.dtype.names

    assert "logTe" in table.dtype.names


def test_load_ascii_missing_file():

    with pytest.raises(FileNotFoundError):

        _load_ascii_table("does_not_exist.dat")


# ==========================================================
# Evolution track
# ==========================================================

def test_load_parsec_track(tmp_path):

    filename = make_track_file(tmp_path)

    track = load_parsec_track(

        filename,

        column_map=TRACK_COLUMNS,

        initial_mass=1.5,

        metallicity=-0.2,

    )

    assert isinstance(track, EvolutionTrack)

    assert track.initial_mass == 1.5

    assert track.metallicity == -0.2

    assert len(track.age) == 3

    assert np.allclose(

        track.log_teff,

        [3.80, 3.79, 3.78]

    )

    assert np.array_equal(

        track.phase,

        [1, 1, 2]

    )


def test_load_parsec_track_without_phase(tmp_path):

    filename = make_track_file(tmp_path)

    columns = TRACK_COLUMNS.copy()

    del columns["phase"]

    track = load_parsec_track(

        filename,

        column_map=columns,

        initial_mass=1.0,

        metallicity=0.0,

    )

    assert track.phase is None


# ==========================================================
# Isochrone
# ==========================================================

def test_load_parsec_isochrone(tmp_path):

    filename = make_iso_file(tmp_path)

    iso = load_parsec_isochrone(

        filename,

        column_map=ISO_COLUMNS,

        age=5.0e9,

        metallicity=0.0,

    )

    assert isinstance(iso, Isochrone)

    assert iso.age == 5.0e9

    assert len(iso.mass) == 3

    assert np.allclose(

        iso.mass,

        [0.8, 1.0, 1.2]

    )

    assert np.array_equal(

        iso.phase,

        [1, 2, 3]

    )


def test_load_parsec_isochrone_without_phase(tmp_path):

    filename = make_iso_file(tmp_path)

    columns = ISO_COLUMNS.copy()

    del columns["phase"]

    iso = load_parsec_isochrone(

        filename,

        column_map=columns,

        age=1.0e9,

        metallicity=0.0,

    )

    assert iso.phase is None