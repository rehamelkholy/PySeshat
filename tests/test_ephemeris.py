from pathlib import Path

import numpy as np

from pyseshat.ephemeris.ephemeris import (
    combined_ephemeris_table,
    generate_rv_ephemeris,
    generate_visual_ephemeris,
    rv_ephemeris_table,
    visual_ephemeris_table,
)
from pyseshat.ephemeris.ephemeris_result import EphemerisResult
from pyseshat.io.orbit_parser import parse_orbit_file


def load_elements():

    filepath = Path(
        "validation/benchmark_systems/FIN379/FIN379.inp"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    return parsed["elements"]


def make_epochs():

    return np.linspace(
        2010,
        2020,
        10,
    )


def test_generate_visual_ephemeris():

    result = generate_visual_ephemeris(
        load_elements(),
        make_epochs(),
    )

    assert isinstance(
        result,
        EphemerisResult,
    )

    assert len(result.epoch) == 10
    assert len(result.theta) == 10
    assert len(result.rho) == 10
    assert len(result.X) == 10
    assert len(result.Y) == 10


def test_visual_ephemeris_table():

    table = visual_ephemeris_table(
        load_elements(),
        make_epochs(),
    )

    assert list(table.columns) == [
        "Epoch",
        "Theta",
        "Rho",
        "X",
        "Y",
    ]

    assert len(table) == 10


def test_generate_rv_ephemeris():

    result = generate_rv_ephemeris(
        load_elements(),
        make_epochs(),
    )

    assert isinstance(
        result,
        EphemerisResult,
    )

    assert len(result.epoch) == 10
    assert len(result.rv1) == 10
    assert len(result.rv2) == 10


def test_rv_ephemeris_table():

    table = rv_ephemeris_table(
        load_elements(),
        make_epochs(),
    )

    assert list(table.columns) == [
        "Epoch",
        "RV1",
        "RV2",
    ]

    assert len(table) == 10


def test_combined_ephemeris_table():

    table = combined_ephemeris_table(
        load_elements(),
        make_epochs(),
    )

    assert list(table.columns) == [
        "Epoch",
        "Theta",
        "Rho",
        "X",
        "Y",
        "RV1",
        "RV2",
    ]

    assert len(table) == 10