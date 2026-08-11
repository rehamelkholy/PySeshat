import pytest

from pyseshat.analysis.binary_consistency import (
    BinaryConsistencyResult,
    build_binary_consistency_from_elements,
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

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def make_orbital_elements():

    return OrbitalElements(

        P=10.0,

        T0=2000.0,

        e=0.0,

        a=0.5,

        Omega=0.0,

        omega=0.0,

        i=90.0,
    )


def make_evolution_solution():

    return BinaryEvolutionSolution(

        primary=EvolutionSample(
            age=5.0,
            metallicity=0.0,
            initial_mass=1.2,
            log_luminosity=0.20,
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

        log_luminosity=0.20,

        sigma_log_luminosity=0.05,
    )


def make_secondary_observables():

    return StellarObservables(

        log_teff=3.70,

        sigma_log_teff=0.01,

        log_luminosity=-0.10,

        sigma_log_luminosity=0.05,
    )


def test_binary_consistency_integration():

    result = build_binary_consistency_from_elements(

        orbital_elements=make_orbital_elements(),

        parallax_mas=50.0,

        evolution_solution=make_evolution_solution(),

        primary_observables=make_primary_observables(),

        secondary_observables=make_secondary_observables(),
    )

    assert isinstance(
        result,
        BinaryConsistencyResult,
    )

    #
    # Orbital mass:
    #
    # parallax = 50 mas = 0.05 arcsec
    #
    # a = 0.5 / 0.05 = 10 AU
    #
    # M_dyn = 10^3 / 10^2 = 10 M_sun
    #
    assert result.orbital_mass == pytest.approx(
        10.0
    )

    #
    # Evolutionary mass:
    #
    # M_evol = 1.2 + 0.9 = 2.1 M_sun
    #
    assert result.evolutionary_mass == pytest.approx(
        2.1
    )

    #
    # Fractional mass discrepancy:
    #
    # |10.0 - 2.1| / 10.0 = 0.79
    #
    assert result.fractional_mass_difference == pytest.approx(
        0.79
    )

    #
    # Primary luminosity.
    #
    assert result.primary_observed_log_luminosity == pytest.approx(
        0.20
    )

    assert result.primary_evolutionary_log_luminosity == pytest.approx(
        0.20
    )

    assert result.primary_luminosity_sigma == pytest.approx(
        0.0
    )

    #
    # Secondary luminosity.
    #
    assert result.secondary_observed_log_luminosity == pytest.approx(
        -0.10
    )

    assert result.secondary_evolutionary_log_luminosity == pytest.approx(
        -0.10
    )

    assert result.secondary_luminosity_sigma == pytest.approx(
        0.0
    )