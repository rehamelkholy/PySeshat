import math

import pytest

from pyseshat.analysis.binary_consistency import (
    BinaryConsistencyResult,
    build_binary_consistency_result,
    compute_mass_consistency,
    evolutionary_total_mass,
    normalized_difference,
)

from pyseshat.evolution.interpolate import (
    EvolutionSample,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)


def make_solution():

    return BinaryEvolutionSolution(

        primary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=1.2,
            log_luminosity=0.10,
        ),

        secondary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=0.9,
            log_luminosity=-0.10,
        ),

        age=5.0,

        metallicity=0.0,

        chi2=1.0,
    )


def make_primary_observables():

    return StellarObservables(

        log_teff=3.80,

        sigma_log_teff=0.01,

        log_luminosity=0.10,

        sigma_log_luminosity=0.02,
    )


def make_secondary_observables():

    return StellarObservables(

        log_teff=3.70,

        sigma_log_teff=0.01,

        log_luminosity=-0.10,

        sigma_log_luminosity=0.03,
    )


def test_binary_consistency_storage():

    result = BinaryConsistencyResult(

        orbital_mass=2.10,

        evolutionary_mass=2.05,

        fractional_mass_difference=0.02381,

        primary_luminosity_z_score=0.42,

        secondary_luminosity_z_score=-0.18,

        orbital_mass_sigma=0.08,
    )

    assert result.orbital_mass == pytest.approx(
        2.10
    )

    assert result.evolutionary_mass == pytest.approx(
        2.05
    )

    assert (
        result.fractional_mass_difference
        == pytest.approx(0.02381)
    )

    assert (
        result.primary_luminosity_z_score
        == pytest.approx(0.42)
    )

    assert (
        result.secondary_luminosity_z_score
        == pytest.approx(-0.18)
    )

    assert (
        result.orbital_mass_sigma
        == pytest.approx(0.08)
    )


def test_binary_consistency_storage_without_mass_uncertainty():

    result = BinaryConsistencyResult(

        orbital_mass=2.10,

        evolutionary_mass=2.05,

        fractional_mass_difference=0.02381,

        primary_luminosity_z_score=0.42,

        secondary_luminosity_z_score=-0.18,
    )

    assert result.orbital_mass_sigma is None


def test_mass_consistency_zero():

    assert compute_mass_consistency(
        2.0,
        2.0,
    ) == pytest.approx(
        0.0
    )


def test_mass_consistency_fraction():

    assert compute_mass_consistency(
        2.0,
        1.8,
    ) == pytest.approx(
        0.1
    )


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
    ) == pytest.approx(
        0.0
    )


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
    ) == pytest.approx(
        expected
    )


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
    ) == pytest.approx(
        2.1
    )


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
    ) == pytest.approx(
        2.1
    )


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


def test_build_binary_consistency_result_with_mass_uncertainty():

    result = build_binary_consistency_result(

        orbital_mass=2.10,

        orbital_mass_sigma=0.08,

        evolution_solution=make_solution(),

        primary_observables=(
            make_primary_observables()
        ),

        secondary_observables=(
            make_secondary_observables()
        ),
    )

    assert result.orbital_mass == pytest.approx(
        2.10
    )

    assert result.evolutionary_mass == pytest.approx(
        2.10
    )

    assert result.fractional_mass_difference == pytest.approx(
        0.0
    )

    assert result.orbital_mass_sigma == pytest.approx(
        0.08
    )


def test_build_binary_consistency_result_rejects_negative_mass_uncertainty():

    with pytest.raises(ValueError):

        build_binary_consistency_result(

            orbital_mass=2.10,

            orbital_mass_sigma=-0.08,

            evolution_solution=make_solution(),

            primary_observables=(
                make_primary_observables()
            ),

            secondary_observables=(
                make_secondary_observables()
            ),
        )