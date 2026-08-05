import numpy as np

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)

from pyseshat.photometry.evolution_adapter import (
    stellar_observables_from_photometric_parameters,
)

from pyseshat.analysis.luminosity import (
    compute_log_luminosity,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)


def test_stellar_observables_from_photometric_parameters():

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.20,

        secondary_radius=0.85,

        ebv=0.05,

    )

    primary, secondary = (

        stellar_observables_from_photometric_parameters(

            parameters

        )

    )

    assert isinstance(

        primary,

        StellarObservables,

    )

    assert isinstance(

        secondary,

        StellarObservables,

    )

    # ==================================================
    # Primary
    # ==================================================

    assert np.isclose(

        primary.log_teff,

        np.log10(

            parameters.primary_temperature

        ),

    )

    assert np.isclose(

        primary.log_radius,

        np.log10(

            parameters.primary_radius

        ),

    )

    assert np.isclose(

        primary.log_luminosity,

        compute_log_luminosity(

            radius_solar=parameters.primary_radius,

            temperature_kelvin=(
                parameters.primary_temperature
            ),

        ),

    )

    assert primary.log_g is None

    # ==================================================
    # Secondary
    # ==================================================

    assert np.isclose(

        secondary.log_teff,

        np.log10(

            parameters.secondary_temperature

        ),

    )

    assert np.isclose(

        secondary.log_radius,

        np.log10(

            parameters.secondary_radius

        ),

    )

    assert np.isclose(

        secondary.log_luminosity,

        compute_log_luminosity(

            radius_solar=parameters.secondary_radius,

            temperature_kelvin=(
                parameters.secondary_temperature
            ),

        ),

    )

    assert secondary.log_g is None