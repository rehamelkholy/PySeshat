import numpy as np

import pytest

from pyseshat.analysis.binary_consistency import (
    BinaryConsistencyResult,
    build_binary_consistency_from_elements,
)

from pyseshat.evolution.binary import (
    fit_binary_evolution_from_orbit,
)

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def make_evolution_model():

    return EvolutionModel(

        name="Phase V Test Model",

        tracks=(

            EvolutionTrack(
                initial_mass=1.0,
                metallicity=0.0,
                age=np.array(
                    [1.0, 2.0, 3.0]
                ),
                log_teff=np.array(
                    [3.90, 3.80, 3.70]
                ),
                log_luminosity=np.array(
                    [-0.20, 0.00, 0.20]
                ),
                log_radius=np.array(
                    [0.00, 0.10, 0.20]
                ),
                log_g=np.array(
                    [4.60, 4.50, 4.40]
                ),
            ),

            EvolutionTrack(
                initial_mass=0.8,
                metallicity=0.0,
                age=np.array(
                    [1.0, 2.0, 3.0]
                ),
                log_teff=np.array(
                    [3.85, 3.75, 3.65]
                ),
                log_luminosity=np.array(
                    [-0.30, -0.10, 0.10]
                ),
                log_radius=np.array(
                    [-0.05, 0.05, 0.15]
                ),
                log_g=np.array(
                    [4.70, 4.60, 4.50]
                ),
            ),

        ),

        isochrones=(),
    )


def make_sb2_orbital_elements():

    #
    # The radial-velocity amplitudes correspond to
    #
    #     M1 = 1.0 Msun
    #     M2 = 0.8 Msun
    #
    # for P = 1 yr, e = 0, and i = 90 deg.
    #
    # The visual-orbit semimajor axis is chosen
    # independently so that the visual dynamical
    # total mass is also:
    #
    #     M_total = 1.8 Msun.
    #
    # For parallax = 100 mas = 0.1 arcsec,
    #
    #     a_AU = a_arcsec / 0.1
    #
    # and therefore
    #
    #     a_arcsec = 0.1 * (1.8)^(1/3).
    #

    semimajor_axis_arcsec = (
        0.1 * 1.8 ** (1.0 / 3.0)
    )

    return OrbitalElements(

        P=1.0,

        T0=0.0,

        e=0.0,

        a=semimajor_axis_arcsec,

        Omega=0.0,

        omega=0.0,

        i=90.0,

        K1=16.10290279498156,

        K2=20.12862849372695,
    )


def make_primary_observables():

    return StellarObservables(

        log_teff=3.80,

        sigma_log_teff=0.01,

        log_luminosity=0.00,

        sigma_log_luminosity=0.01,
    )


def make_secondary_observables():

    return StellarObservables(

        log_teff=3.75,

        sigma_log_teff=0.01,

        log_luminosity=-0.10,

        sigma_log_luminosity=0.01,
    )


def test_phase5_orbit_to_evolution_to_consistency():

    orbital_elements = (
        make_sb2_orbital_elements()
    )

    primary_observables = (
        make_primary_observables()
    )

    secondary_observables = (
        make_secondary_observables()
    )

    evolution_model = (
        make_evolution_model()
    )

    #
    # --------------------------------------------------
    # Step 1: SB2 orbital solution -> component masses
    # -> evolutionary solution.
    # --------------------------------------------------
    #

    evolution_solution = (
        fit_binary_evolution_from_orbit(

            orbital_elements=(
                orbital_elements
            ),

            evolution_model=(
                evolution_model
            ),

            metallicity=0.0,

            primary_observables=(
                primary_observables
            ),

            secondary_observables=(
                secondary_observables
            ),
        )
    )

    #
    # The selected evolutionary tracks must correspond
    # to the component masses encoded by K1 and K2.
    #

    assert evolution_solution.primary.initial_mass == pytest.approx(
        1.0
    )

    assert evolution_solution.secondary.initial_mass == pytest.approx(
        0.8
    )

    #
    # Both components must have the same fitted age.
    #

    assert evolution_solution.primary.age == pytest.approx(
        evolution_solution.secondary.age
    )

    assert evolution_solution.age == pytest.approx(
        2.0
    )

    #
    # The observables were constructed to coincide with
    # the two tracks at age = 2 yr, so the evolutionary
    # fit should have zero chi-square.
    #

    assert evolution_solution.chi2 == pytest.approx(
        0.0
    )

    #
    # --------------------------------------------------
    # Step 2: orbital solution + evolutionary solution
    # -> binary consistency.
    # --------------------------------------------------
    #

    result = (
        build_binary_consistency_from_elements(

            orbital_elements=(
                orbital_elements
            ),

            parallax_mas=100.0,

            evolution_solution=(
                evolution_solution
            ),

            primary_observables=(
                primary_observables
            ),

            secondary_observables=(
                secondary_observables
            ),
        )
    )

    assert isinstance(
        result,
        BinaryConsistencyResult,
    )

    #
    # The visual orbit was constructed to give
    #
    #     M_dyn = 1.8 Msun.
    #

    assert result.orbital_mass == pytest.approx(
        1.8
    )

    #
    # The evolutionary component masses are
    #
    #     1.0 + 0.8 = 1.8 Msun.
    #

    assert result.evolutionary_mass == pytest.approx(
        1.8
    )

    #
    # Therefore the mass consistency diagnostic
    # must be exactly zero.
    #

    assert result.fractional_mass_difference == pytest.approx(
        0.0
    )

    #
    # The photometric/evolutionary luminosities were
    # deliberately chosen to agree exactly.
    #

    assert result.primary_observed_log_luminosity == pytest.approx(
        0.0
    )

    assert result.primary_evolutionary_log_luminosity == pytest.approx(
        0.0
    )

    # Was `primary_luminosity_sigma` before 2026-08-27:
    # BinaryConsistencyResult has no such field; the real field
    # is `primary_luminosity_z_score`, the dimensionless
    # (observed - evolutionary) / sigma_observed significance --
    # see the same fix in test_phase5_fit_results.py and
    # docs/project_decisions.md.

    assert result.primary_luminosity_z_score == pytest.approx(
        0.0
    )

    assert result.secondary_observed_log_luminosity == pytest.approx(
        -0.10
    )

    assert result.secondary_evolutionary_log_luminosity == pytest.approx(
        -0.10
    )

    # Was `secondary_luminosity_sigma` -- see the comment on
    # the primary field above.

    assert result.secondary_luminosity_z_score == pytest.approx(
        0.0
    )