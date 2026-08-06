import math

import pytest

from pyseshat.analysis.binary_consistency import (
    BinaryConsistencyResult,
    compute_mass_consistency,
    evolutionary_total_mass,
    normalized_difference,
)

from pyseshat.evolution.interpolate import EvolutionSample
from pyseshat.evolution.solution import BinaryEvolutionSolution


def test_binary_consistency_storage():

    result = BinaryConsistencyResult(
        mass_difference=0.02,
        age_difference=0.10,
        luminosity_difference=0.05,
    )

    assert result.mass_difference == pytest.approx(0.02)
    assert result.age_difference == pytest.approx(0.10)
    assert result.luminosity_difference == pytest.approx(0.05)


def test_mass_consistency_zero():

    assert compute_mass_consistency(
        2.0,
        2.0,
    ) == pytest.approx(0.0)


def test_mass_consistency_fraction():

    assert compute_mass_consistency(
        2.0,
        1.8,
    ) == pytest.approx(0.1)


def test_mass_consistency_invalid():

    with pytest.raises(ValueError):

        compute_mass_consistency(
            0.0,
            1.0,
        )


def test_normalized_difference_zero():

    assert normalized_difference(
        1.0,
        0.1,
        1.0,
        0.2,
    ) == pytest.approx(0.0)


def test_normalized_difference():

    expected = (
        2.0 - 1.8
    ) / math.sqrt(
        0.1**2 + 0.1**2
    )

    assert normalized_difference(
        2.0,
        0.1,
        1.8,
        0.1,
    ) == pytest.approx(expected)


def test_normalized_difference_zero_variance():

    with pytest.raises(ValueError):

        normalized_difference(
            1.0,
            0.0,
            1.0,
            0.0,
        )


def test_evolutionary_total_mass_from_tracks():

    solution = BinaryEvolutionSolution(

        primary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=1.2,
        ),

        secondary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=0.9,
        ),

        age=5.0,
        metallicity=0.0,
        chi2=1.0,
    )

    assert evolutionary_total_mass(
        solution
    ) == pytest.approx(2.1)


def test_evolutionary_total_mass_from_isochrone():

    solution = BinaryEvolutionSolution(

        primary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            mass=1.2,
        ),

        secondary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            mass=0.9,
        ),

        age=5.0,
        metallicity=0.0,
        chi2=1.0,
    )

    assert evolutionary_total_mass(
        solution
    ) == pytest.approx(2.1)


def test_evolutionary_total_mass_missing():

    solution = BinaryEvolutionSolution(

        primary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
        ),

        secondary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=0.9,
        ),

        age=5.0,
        metallicity=0.0,
        chi2=1.0,
    )

    with pytest.raises(ValueError):

        evolutionary_total_mass(
            solution
        )