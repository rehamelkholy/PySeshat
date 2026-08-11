"""
Bridge between photometric inference and stellar
evolution fitting.
"""

from __future__ import annotations

import numpy as np

from pyseshat.analysis.error_propagation import (
    sigma_log_luminosity,
    sigma_log_radius,
    sigma_log_temperature,
)

from pyseshat.analysis.luminosity import (
    compute_log_luminosity,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


def stellar_observables_from_photometric_parameters(
    parameters: PhotometricParameters,
):
    """
    Construct StellarObservables without propagated
    uncertainties.

    This low-level helper is primarily intended for
    deterministic workflows.
    """

    primary = StellarObservables(

        log_teff=np.log10(
            parameters.primary_temperature
        ),

        log_luminosity=compute_log_luminosity(

            radius_solar=parameters.primary_radius,

            temperature_kelvin=(
                parameters.primary_temperature
            ),

        ),

        log_radius=np.log10(
            parameters.primary_radius
        ),

        log_g=None,

    )

    secondary = StellarObservables(

        log_teff=np.log10(
            parameters.secondary_temperature
        ),

        log_luminosity=compute_log_luminosity(

            radius_solar=parameters.secondary_radius,

            temperature_kelvin=(
                parameters.secondary_temperature
            ),

        ),

        log_radius=np.log10(
            parameters.secondary_radius
        ),

        log_g=None,

    )

    return primary, secondary


def stellar_observables_from_solution(
    parameters: PhotometricParameters,
    uncertainties: dict[str, float],
):
    """
    Construct StellarObservables with propagated
    uncertainties.

    Parameters
    ----------
    parameters
        Best-fitting photometric parameters.

    uncertainties
        One-sigma uncertainties returned by the
        inference engine.

    Returns
    -------
    tuple
        Primary and secondary StellarObservables.
    """

    return (

        StellarObservables(

            log_teff=np.log10(
                parameters.primary_temperature
            ),

            sigma_log_teff=sigma_log_temperature(

                parameters.primary_temperature,

                uncertainties[
                    "primary_temperature"
                ],

            ),

            log_luminosity=compute_log_luminosity(

                radius_solar=parameters.primary_radius,

                temperature_kelvin=(
                    parameters.primary_temperature
                ),

            ),

            sigma_log_luminosity=sigma_log_luminosity(

                temperature=(
                    parameters.primary_temperature
                ),

                sigma_temperature=(
                    uncertainties[
                        "primary_temperature"
                    ]
                ),

                radius=(
                    parameters.primary_radius
                ),

                sigma_radius=(
                    uncertainties[
                        "primary_radius"
                    ]
                ),

            ),

            log_radius=np.log10(
                parameters.primary_radius
            ),

            sigma_log_radius=sigma_log_radius(

                parameters.primary_radius,

                uncertainties[
                    "primary_radius"
                ],

            ),

            log_g=None,

            sigma_log_g=None,

        ),

        StellarObservables(

            log_teff=np.log10(
                parameters.secondary_temperature
            ),

            sigma_log_teff=sigma_log_temperature(

                parameters.secondary_temperature,

                uncertainties[
                    "secondary_temperature"
                ],

            ),

            log_luminosity=compute_log_luminosity(

                radius_solar=(
                    parameters.secondary_radius
                ),

                temperature_kelvin=(
                    parameters.secondary_temperature
                ),

            ),

            sigma_log_luminosity=sigma_log_luminosity(

                temperature=(
                    parameters.secondary_temperature
                ),

                sigma_temperature=(
                    uncertainties[
                        "secondary_temperature"
                    ]
                ),

                radius=(
                    parameters.secondary_radius
                ),

                sigma_radius=(
                    uncertainties[
                        "secondary_radius"
                    ]
                ),

            ),

            log_radius=np.log10(
                parameters.secondary_radius
            ),

            sigma_log_radius=sigma_log_radius(

                parameters.secondary_radius,

                uncertainties[
                    "secondary_radius"
                ],

            ),

            log_g=None,

            sigma_log_g=None,

        ),
    )


def stellar_observables_from_fit_result(
    fit_result,
):
    """
    Construct StellarObservables directly from a
    photometric fit result.

    The fit result is expected to expose:

    - ``elements`` containing the fitted
      PhotometricParameters;
    - ``uncertainties`` containing the one-sigma
      parameter uncertainties.

    This function is the direct bridge between the
    photometric inference layer and the stellar
    evolution layer.

    Parameters
    ----------
    fit_result
        Photometric fit result returned by the
        inference engine.

    Returns
    -------
    tuple
        Primary and secondary StellarObservables.
    """

    return stellar_observables_from_solution(

        parameters=fit_result.elements,

        uncertainties=fit_result.uncertainties,

    )