import numpy as np

import pytest

from pyseshat.analysis.binary_consistency import (
    BinaryConsistencyResult,
    build_binary_consistency_from_fit_results,
)

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult,
)

from pyseshat.inference.fit_result import (
    FitResult,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


class DummyOptimizerResult:

    success = True

    message = "test"


def make_evolution_model():

    return EvolutionModel(

        name="Phase V Fit-Result Test Model",

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


def make_orbital_fit_result():

    #
    # P = 1 yr
    # e = 0
    # i = 90 deg
    #
    # K1 and K2 correspond approximately to
    #
    # M1 = 1.0 Msun
    # M2 = 0.8 Msun.
    #
    # The visual semimajor axis is selected so that
    # parallax = 100 mas gives M_total = 1.8 Msun.
    #

    semimajor_axis_arcsec = (
        0.1 * 1.8 ** (1.0 / 3.0)
    )

    elements = OrbitalElements(

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

    return FitResult(

        elements=elements,

        optimizer_result=(
            DummyOptimizerResult()
        ),

        residuals=np.zeros(2),

        covariance=np.eye(
            len(
                elements.fitted_parameters()
            )
        ),

        uncertainties={},

    )


def make_photometric_fit_result():

    parameters = PhotometricParameters(

        primary_temperature=10.0 ** 3.80,

        secondary_temperature=10.0 ** 3.75,

        primary_radius=(
            10.0 ** (
                0.5 * (
                    0.00
                    - 4.0 * (
                        np.log10(
                            10.0 ** 3.80
                            / 5772.0
                        )
                    )
                )
            )
        ),

        secondary_radius=(
            10.0 ** (
                0.5 * (
                    -0.10
                    - 4.0 * (
                        np.log10(
                            10.0 ** 3.75
                            / 5772.0
                        )
                    )
                )
            )
        ),

        ebv=0.0,
    )

    uncertainties = {

        "primary_temperature": 10.0,

        "secondary_temperature": 10.0,

        "primary_radius": 0.01,

        "secondary_radius": 0.01,

        "ebv": 0.01,
    }

    return BaseFitResult(

        elements=parameters,

        optimizer_result=(
            DummyOptimizerResult()
        ),

        residuals=np.zeros(2),

        covariance=np.eye(5),

        uncertainties=uncertainties,

    )


def test_build_binary_consistency_from_fit_results():

    result = (
        build_binary_consistency_from_fit_results(

            orbital_fit_result=(
                make_orbital_fit_result()
            ),

            photometric_fit_result=(
                make_photometric_fit_result()
            ),

            parallax_mas=100.0,

            evolution_model=(
                make_evolution_model()
            ),

            metallicity=0.0,
        )
    )

    assert isinstance(
        result,
        BinaryConsistencyResult,
    )

    #
    # Dynamical mass from the visual orbit.
    #

    assert result.orbital_mass == pytest.approx(
        1.8
    )

    #
    # Evolutionary mass from the two selected
    # tracks.
    #

    assert result.evolutionary_mass == pytest.approx(
        1.8
    )

    #
    # The complete pipeline was constructed to
    # be internally consistent.
    #

    assert result.fractional_mass_difference == pytest.approx(
        0.0,
        abs=1e-10,
    )

    #
    # The photometric parameters were constructed
    # so that their luminosities match the evolutionary
    # luminosities at the selected common age.
    #

    assert result.primary_observed_log_luminosity == pytest.approx(
        0.00,
        abs=1e-10,
    )

    assert result.primary_evolutionary_log_luminosity == pytest.approx(
        0.00,
        abs=1e-10,
    )

    # `BinaryConsistencyResult` has no `primary_luminosity_sigma`
    # field (see its docstring). The actual field
    # is `primary_luminosity_z_score`, a dimensionless
    # (observed - evolutionary) / sigma_observed significance,
    # not a raw sigma; for this deliberately self-consistent
    # fixture (observed == evolutionary exactly) it evaluates
    # to 0.0 either way, which is what this assertion checks.

    assert result.primary_luminosity_z_score == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert result.secondary_observed_log_luminosity == pytest.approx(
        -0.10,
        abs=1e-10,
    )

    assert result.secondary_evolutionary_log_luminosity == pytest.approx(
        -0.10,
        abs=1e-10,
    )

    # Was `secondary_luminosity_sigma` -- see the comment on
    # the primary field above.

    assert result.secondary_luminosity_z_score == pytest.approx(
        0.0,
        abs=1e-10,
    )