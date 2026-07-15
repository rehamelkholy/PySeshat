from pathlib import Path

import numpy as np

from pyseshat.photometry.extinction import (
    apply_extinction,
    ccm89_optical_extinction
)

from pyseshat.photometry.filters import (
    load_passband
)

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)

from pyseshat.photometry.residuals import (
    photometric_chi_square,
    photometric_residual
)

from pyseshat.photometry.synthetic import (
    pogson_magnitude,
    synthetic_band_flux
)

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)

from pyseshat.stellar.sed_model import (
    composite_binary_sed
)


# ==========================================================
# SVO Johnson Vega zero points
# ==========================================================

JOHNSON_B_VEGA_ZERO_POINT = 6.72553e-9

JOHNSON_V_VEGA_ZERO_POINT = 3.5833e-9


# ==========================================================
# Extinction parameters
# ==========================================================

EBV = 0.20

RV = 3.10


# ==========================================================
# Repository validation-data directories
# ==========================================================

atmosphere_directory = (

    Path(
        "validation"
    )

    /

    "atmosphere_models"

    /

    "ckp00"

)


passband_directory = (

    Path(
        "validation"
    )

    /

    "photometry"

    /

    "passbands"

)


# ==========================================================
# Load atmosphere grid and passbands
# ==========================================================

grid = load_ck04_grid([

    atmosphere_directory

    /

    "ckp00_5750.fits",

    atmosphere_directory

    /

    "ckp00_6000.fits",

    atmosphere_directory

    /

    "ckp00_6250.fits"

])


johnson_b = load_passband(

    filename=(

        passband_directory

        /

        "johnson_b.dat"

    ),

    name="Johnson B"

)


johnson_v = load_passband(

    filename=(

        passband_directory

        /

        "johnson_v.dat"

    ),

    name="Johnson V"

)


# ==========================================================
# Define reusable model calculation
# ==========================================================

def calculate_model_magnitudes(
    primary_temperature,
    secondary_temperature
):
    """
    Calculate extincted Johnson B and V magnitudes
    for the validation binary.
    """

    # ======================================================
    # Interpolate component atmospheres
    # ======================================================

    primary = grid.interpolate(

        teff=primary_temperature,

        logg=4.25,

        metallicity=0.0

    )


    secondary = grid.interpolate(

        teff=secondary_temperature,

        logg=4.50,

        metallicity=0.0

    )


    # ======================================================
    # Construct intrinsic composite SED
    # ======================================================

    wavelength, intrinsic_flux = (

        composite_binary_sed(

            primary=primary,

            secondary=secondary,

            primary_radius=1.20,

            secondary_radius=0.90,

            distance=25.0

        )

    )


    # ======================================================
    # Restrict to the optical CCM89 domain
    # ======================================================

    inverse_wavelength = (

        10000.0

        /

        wavelength

    )


    optical = (

        (

            inverse_wavelength

            >=

            1.1

        )

        &

        (

            inverse_wavelength

            <

            3.3

        )

    )


    optical_wavelength = wavelength[

        optical

    ]


    optical_intrinsic_flux = intrinsic_flux[

        optical

    ]


    # ======================================================
    # Calculate and apply extinction
    # ======================================================

    extinction_curve = (

        ccm89_optical_extinction(

            wavelength=(

                optical_wavelength

            ),

            ebv=EBV,

            rv=RV

        )

    )


    extincted_flux = apply_extinction(

        flux=optical_intrinsic_flux,

        extinction=extinction_curve

    )


    # ======================================================
    # Calculate synthetic band fluxes
    # ======================================================

    johnson_b_flux = synthetic_band_flux(

        wavelength=optical_wavelength,

        flux=extincted_flux,

        passband=johnson_b

    )


    johnson_v_flux = synthetic_band_flux(

        wavelength=optical_wavelength,

        flux=extincted_flux,

        passband=johnson_v

    )


    # ======================================================
    # Convert fluxes to Vega magnitudes
    # ======================================================

    return {

        "Johnson B": pogson_magnitude(

            flux=johnson_b_flux,

            zero_point=(

                JOHNSON_B_VEGA_ZERO_POINT

            )

        ),

        "Johnson V": pogson_magnitude(

            flux=johnson_v_flux,

            zero_point=(

                JOHNSON_V_VEGA_ZERO_POINT

            )

        )

    }


# ==========================================================
# Generate mock observations from the reference model
# ==========================================================

reference_model = calculate_model_magnitudes(

    primary_temperature=6125.0,

    secondary_temperature=5875.0

)


measurements = [

    PhotometricMeasurement(

        passband="Johnson B",

        magnitude=(

            reference_model[

                "Johnson B"

            ]

        ),

        uncertainty=0.020

    ),

    PhotometricMeasurement(

        passband="Johnson V",

        magnitude=(

            reference_model[

                "Johnson V"

            ]

        ),

        uncertainty=0.015

    )

]


# ==========================================================
# Evaluate the generating model
# ==========================================================

reference_b_residual = photometric_residual(

    model_magnitude=(

        reference_model[

            "Johnson B"

        ]

    ),

    measurement=measurements[

        0

    ]

)


reference_v_residual = photometric_residual(

    model_magnitude=(

        reference_model[

            "Johnson V"

        ]

    ),

    measurement=measurements[

        1

    ]

)


reference_chi_square = photometric_chi_square(

    model_magnitudes=reference_model,

    measurements=measurements

)


# ==========================================================
# Evaluate a perturbed-temperature model
# ==========================================================

perturbed_model = calculate_model_magnitudes(

    primary_temperature=6000.0,

    secondary_temperature=5750.0

)


perturbed_chi_square = photometric_chi_square(

    model_magnitudes=perturbed_model,

    measurements=measurements

)


# ==========================================================
# Validate the generating model
# ==========================================================

assert np.isclose(

    reference_b_residual,

    0.0

)


assert np.isclose(

    reference_v_residual,

    0.0

)


assert np.isclose(

    reference_chi_square,

    0.0

)


# ==========================================================
# Validate the perturbed model
# ==========================================================

assert np.isfinite(

    perturbed_chi_square

)


assert perturbed_chi_square > 0.0


assert perturbed_chi_square > (

    reference_chi_square

)


# ==========================================================
# Report reference-model results
# ==========================================================

print()

print(

    "REFERENCE PHOTOMETRIC MODEL"

)

print(

    "=" * 70

)


print(

    "Johnson B magnitude"

)

print(

    reference_model[

        "Johnson B"

    ]

)


print()

print(

    "Johnson V magnitude"

)

print(

    reference_model[

        "Johnson V"

    ]

)


print()

print(

    "Johnson B residual"

)

print(

    reference_b_residual

)


print()

print(

    "Johnson V residual"

)

print(

    reference_v_residual

)


print()

print(

    "Photometric chi-square"

)

print(

    reference_chi_square

)


# ==========================================================
# Report perturbed-model results
# ==========================================================

print()

print(

    "PERTURBED-TEMPERATURE MODEL"

)

print(

    "=" * 70

)


print(

    "Primary temperature"

)

print(

    6000.0

)


print()

print(

    "Secondary temperature"

)

print(

    5750.0

)


print()

print(

    "Johnson B magnitude"

)

print(

    perturbed_model[

        "Johnson B"

    ]

)


print()

print(

    "Johnson V magnitude"

)

print(

    perturbed_model[

        "Johnson V"

    ]

)


print()

print(

    "Photometric chi-square"

)

print(

    perturbed_chi_square

)


print()

print(

    "Chi-square increased"

)

print(

    perturbed_chi_square

    >

    reference_chi_square

)