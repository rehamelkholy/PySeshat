"""
Tests for the evolutionary-track/isochrone grid-loading layer:
pyseshat.evolution.grids, pyseshat.evolution.parsers.parsec, and
pyseshat.evolution.parsers.girardi2000.

Before this file was added, none of this loading/lookup layer had
direct test coverage (see docs/guide/evolutionary_checks.md's
2026-08-29 entry and docs/project_decisions.md for the audit that
found this gap -- and found that a previously-documented
``tests/test_parsec.py`` did not actually exist).
"""

import numpy as np

import pytest

from pyseshat.evolution.grids import (
    DEFAULT_EVOLUTION_MODEL_NAME,
    load_default_evolution_model,
    load_girardi2000,
    load_model,
    load_parsec,
)

from pyseshat.evolution.models import (
    EvolutionModel,
)

from pyseshat.evolution.parsers.girardi2000 import (
    AVAILABLE_METALLICITIES,
    CITATION,
    MASS_RANGE_MSUN,
    SOLAR_METALLICITY,
    load_girardi2000_isochrone,
    load_girardi2000_track,
)

from pyseshat.evolution.parsers.parsec import (
    load_parsec_isochrone,
    load_parsec_track,
)


# ==========================================================
# Synthetic fixture files
# ==========================================================

_TRACK_COLUMN_MAP = {
    "age": "AGE",
    "log_teff": "LOG_TE",
    "log_luminosity": "LOG_L",
    "log_radius": "LOG_R",
    "log_g": "LOG_G",
    "phase": "STAGE",
}

_ISOCHRONE_COLUMN_MAP = {
    "mass": "MASS",
    "log_teff": "LOG_TE",
    "log_luminosity": "LOG_L",
    "log_radius": "LOG_R",
    "log_g": "LOG_G",
    "phase": "STAGE",
}


def _write_track_file(path):

    path.write_text(
        "# synthetic evolutionary track fixture\n"
        "AGE LOG_TE LOG_L LOG_R LOG_G STAGE\n"
        "1.0e8 3.80 0.10 0.05 4.50 1\n"
        "5.0e8 3.78 0.20 0.10 4.45 1\n"
        "1.0e9 3.75 0.30 0.15 4.40 2\n"
    )

    return path


def _write_isochrone_file(path):

    path.write_text(
        "# synthetic isochrone fixture\n"
        "MASS LOG_TE LOG_L LOG_R LOG_G STAGE\n"
        "0.8  3.70 -0.20 -0.10 4.60 1\n"
        "1.0  3.76  0.00  0.00 4.50 1\n"
        "1.2  3.82  0.20  0.10 4.40 2\n"
    )

    return path


# ==========================================================
# PARSEC parser (pre-existing, previously untested)
# ==========================================================

def test_load_parsec_track(tmp_path):

    filename = _write_track_file(tmp_path / "track.dat")

    track = load_parsec_track(
        filename,
        column_map=_TRACK_COLUMN_MAP,
        initial_mass=1.0,
        metallicity=0.019,
    )

    assert track.initial_mass == 1.0
    assert track.metallicity == 0.019
    assert np.allclose(track.age, [1.0e8, 5.0e8, 1.0e9])
    assert np.allclose(track.log_teff, [3.80, 3.78, 3.75])
    assert track.phase is not None
    assert list(track.phase) == [1, 1, 2]


def test_load_parsec_isochrone(tmp_path):

    filename = _write_isochrone_file(
        tmp_path / "isochrone.dat"
    )

    iso = load_parsec_isochrone(
        filename,
        column_map=_ISOCHRONE_COLUMN_MAP,
        age=1.0e9,
        metallicity=0.019,
    )

    assert iso.age == 1.0e9
    assert np.allclose(iso.mass, [0.8, 1.0, 1.2])
    assert np.allclose(iso.log_luminosity, [-0.20, 0.0, 0.20])


def test_load_parsec_track_missing_file_raises(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_parsec_track(
            tmp_path / "does_not_exist.dat",
            column_map=_TRACK_COLUMN_MAP,
            initial_mass=1.0,
            metallicity=0.019,
        )


# ==========================================================
# Girardi (2000) parser (new)
# ==========================================================

def test_load_girardi2000_track(tmp_path):

    filename = _write_track_file(
        tmp_path / "girardi_track.dat"
    )

    track = load_girardi2000_track(
        filename,
        column_map=_TRACK_COLUMN_MAP,
        initial_mass=1.0,
        metallicity=SOLAR_METALLICITY,
    )

    assert track.initial_mass == 1.0
    assert track.metallicity == SOLAR_METALLICITY
    assert np.allclose(track.age, [1.0e8, 5.0e8, 1.0e9])
    assert np.allclose(
        track.log_luminosity, [0.10, 0.20, 0.30]
    )
    assert list(track.phase) == [1, 1, 2]


def test_load_girardi2000_isochrone(tmp_path):

    filename = _write_isochrone_file(
        tmp_path / "girardi_isochrone.dat"
    )

    iso = load_girardi2000_isochrone(
        filename,
        column_map=_ISOCHRONE_COLUMN_MAP,
        age=1.0e9,
        metallicity=SOLAR_METALLICITY,
    )

    assert iso.metallicity == SOLAR_METALLICITY
    assert np.allclose(iso.log_teff, [3.70, 3.76, 3.82])


def test_load_girardi2000_track_missing_file_raises(tmp_path):

    with pytest.raises(FileNotFoundError):
        load_girardi2000_track(
            tmp_path / "does_not_exist.dat",
            column_map=_TRACK_COLUMN_MAP,
            initial_mass=1.0,
            metallicity=SOLAR_METALLICITY,
        )


def test_girardi2000_metadata_matches_the_published_paper():
    """
    Regression guard on the citation/coverage constants quoted
    in docs and in the paper report -- these are transcribed
    directly from Girardi et al. 2000, A&AS, 141, 371, Sect. 2
    and Table 1 (fetched and cross-checked 2026-08-29), not
    independently re-derived, so this test exists to catch a
    future accidental edit rather than to validate the paper
    itself.
    """

    assert SOLAR_METALLICITY == 0.019
    assert SOLAR_METALLICITY in AVAILABLE_METALLICITIES
    assert AVAILABLE_METALLICITIES == (
        0.0004, 0.001, 0.004, 0.008, 0.019, 0.03,
    )
    assert MASS_RANGE_MSUN == (0.15, 7.0)
    assert "2000" in CITATION
    assert "141" in CITATION
    assert "371" in CITATION
    assert "10.1051/aas:2000126" in CITATION


# ==========================================================
# grids.load_model() dispatch
# ==========================================================

def test_load_model_parsec_dispatch(tmp_path):

    track_file = _write_track_file(
        tmp_path / "track.dat"
    )
    iso_file = _write_isochrone_file(
        tmp_path / "isochrone.dat"
    )

    model = load_model(
        "parsec",
        track_files=[track_file],
        isochrone_files=[iso_file],
        track_column_map=_TRACK_COLUMN_MAP,
        isochrone_column_map=_ISOCHRONE_COLUMN_MAP,
        track_metadata=[
            {"initial_mass": 1.0, "metallicity": 0.019}
        ],
        isochrone_metadata=[
            {"age": 1.0e9, "metallicity": 0.019}
        ],
    )

    assert isinstance(model, EvolutionModel)
    assert model.name == "PARSEC"
    assert len(model.tracks) == 1
    assert len(model.isochrones) == 1


def test_load_model_girardi2000_dispatch(tmp_path):

    track_file = _write_track_file(
        tmp_path / "track.dat"
    )
    iso_file = _write_isochrone_file(
        tmp_path / "isochrone.dat"
    )

    model = load_model(
        "girardi2000",
        track_files=[track_file],
        isochrone_files=[iso_file],
        track_column_map=_TRACK_COLUMN_MAP,
        isochrone_column_map=_ISOCHRONE_COLUMN_MAP,
        track_metadata=[
            {
                "initial_mass": 1.0,
                "metallicity": SOLAR_METALLICITY,
            }
        ],
        isochrone_metadata=[
            {"age": 1.0e9, "metallicity": SOLAR_METALLICITY}
        ],
    )

    assert isinstance(model, EvolutionModel)
    assert model.name == "Girardi2000"
    assert model.available_metallicities == (
        SOLAR_METALLICITY,
    )


def test_load_model_case_insensitive(tmp_path):

    track_file = _write_track_file(
        tmp_path / "track.dat"
    )

    model = load_model(
        "GIRARDI2000",
        track_files=[track_file],
        isochrone_files=[],
        track_column_map=_TRACK_COLUMN_MAP,
        isochrone_column_map=_ISOCHRONE_COLUMN_MAP,
        track_metadata=[
            {
                "initial_mass": 1.0,
                "metallicity": SOLAR_METALLICITY,
            }
        ],
        isochrone_metadata=[],
    )

    assert model.name == "Girardi2000"


def test_load_model_unsupported_model_raises():

    with pytest.raises(ValueError, match="Unsupported evolution model"):
        load_model("mist")


def test_load_girardi2000_mismatched_metadata_length_raises(tmp_path):

    track_file = _write_track_file(
        tmp_path / "track.dat"
    )

    with pytest.raises(ValueError):
        load_girardi2000(
            track_files=[track_file],
            isochrone_files=[],
            track_column_map=_TRACK_COLUMN_MAP,
            isochrone_column_map=_ISOCHRONE_COLUMN_MAP,
            track_metadata=[],
            isochrone_metadata=[],
        )


def test_load_parsec_mismatched_metadata_length_raises(tmp_path):

    track_file = _write_track_file(
        tmp_path / "track.dat"
    )

    with pytest.raises(ValueError):
        load_parsec(
            track_files=[track_file],
            isochrone_files=[],
            track_column_map=_TRACK_COLUMN_MAP,
            isochrone_column_map=_ISOCHRONE_COLUMN_MAP,
            track_metadata=[],
            isochrone_metadata=[],
        )


# ==========================================================
# Default-grid selection
# ==========================================================

def test_default_evolution_model_name_is_girardi2000():
    """
    This is the concrete, testable expression of the
    2026-08-29 decision (docs/project_decisions.md) to
    standardize this project's own evolutionary consistency
    checks on the same grid the Al-Wardat SED-fitting
    literature uses. PARSEC remains fully supported (see the
    dispatch tests above) -- this only changes which grid is
    named "default".
    """

    assert DEFAULT_EVOLUTION_MODEL_NAME == "girardi2000"


def test_load_default_evolution_model_raises_when_no_data_present(
    tmp_path,
):
    """
    No Girardi (2000) data ships with this repository (see
    this module's docstring and
    pyseshat.evolution.parsers.girardi2000's module docstring
    for why, and docs/guide/evolutionary_checks.md's
    2026-08-29 entry for the confirmed network blocker on
    sourcing it from within this development environment).
    load_default_evolution_model() must fail loudly and
    informatively rather than silently returning an empty
    model, so that a caller cannot mistake "no grid found" for
    "grid found, zero tracks in it".
    """

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Girardi"):
        load_default_evolution_model(data_dir=empty_dir)


def test_load_default_evolution_model_repo_state_today():
    """
    As of this change, the repository itself has no
    data/evolution/girardi2000/ directory populated with real
    tracks -- calling this with no arguments (the real
    production code path) must therefore raise, not silently
    succeed. This is intentionally an honest failing default,
    not a stub returning fabricated data.
    """

    with pytest.raises(FileNotFoundError):
        load_default_evolution_model()
