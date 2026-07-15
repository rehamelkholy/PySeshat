from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
)

from pyseshat.photometry.model import (
    binary_photometric_chi_square,
    synthetic_binary_magnitudes
)

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)


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
# Load atmosphere grid
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


# ==========================================================
# Load Johnson passbands
# ==========================================================

passbands = {

    "Johnson B": load_passband(

        filename=(

            passband_directory

            /

            "johnson_b.dat"

        ),

        name="Johnson B"

    ),

    "Johnson V": load_passband(

        filename=(

            passband_directory

            /

            "johnson_v.dat"

        ),

        name="Johnson V"

    )

}


# ==========================================================
# Define SVO Vega zero points
# ==========================================================

zero_points = {

    "Johnson B": 6.72553e-9,

    "Johnson V": 3.5833e-9

}


# ==========================================================
# Reference physical parameters
# ==========================================================

reference_parameters = np.array([

    6125.0,

    5875.0,

    1.20,

    0.90,

    0.20

])


# ==========================================================
# Generate mock observations
# ==========================================================

reference_magnitudes = (

    synthetic_binary_magnitudes(

        atmosphere_grid=grid,

        passbands=passbands,

        zero_points=zero_points,

        primary_temperature=(

            reference_parameters[
                0
            ]

        ),

        secondary_temperature=(

            reference_parameters[
                1
            ]

        ),

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        primary_radius=(

            reference_parameters[
                2
            ]

        ),

        secondary_radius=(

            reference_parameters[
                3
            ]

        ),

        distance=25.0,

        ebv=(

            reference_parameters[
                4
            ]

        ),

        rv=3.10

    )

)


measurements = [

    PhotometricMeasurement(

        passband="Johnson B",

        magnitude=(

            reference_magnitudes[

                "Johnson B"

            ]

        ),

        uncertainty=0.020

    ),

    PhotometricMeasurement(

        passband="Johnson V",

        magnitude=(

            reference_magnitudes[

                "Johnson V"

            ]

        ),

        uncertainty=0.015

    )

]


# ==========================================================
# Evaluate the generating parameter vector
# ==========================================================

reference_chi_square = (

    binary_photometric_chi_square(

        parameters=reference_parameters,

        atmosphere_grid=grid,

        passbands=passbands,

        zero_points=zero_points,

        measurements=measurements,

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        distance=25.0,

        rv=3.10

    )

)


# ==========================================================
# Evaluate a perturbed parameter vector
# ==========================================================

perturbed_parameters = np.array([

    6000.0,

    5750.0,

    1.20,

    0.90,

    0.20

])


perturbed_chi_square = (

    binary_photometric_chi_square(

        parameters=perturbed_parameters,

        atmosphere_grid=grid,

        passbands=passbands,

        zero_points=zero_points,

        measurements=measurements,

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        distance=25.0,

        rv=3.10

    )

)


# ==========================================================
# Validate objective values
# ==========================================================

assert isinstance(

    reference_chi_square,

    float

)


assert np.isclose(

    reference_chi_square,

    0.0

)


assert np.isfinite(

    perturbed_chi_square

)


assert perturbed_chi_square > 0.0


assert perturbed_chi_square > (

    reference_chi_square

)


# ==========================================================
# Validate parameter-vector rejection
# ==========================================================

invalid_vector_rejected = False


try:

    binary_photometric_chi_square(

        parameters=[

            6125.0,

            5875.0,

            1.20

        ],

        atmosphere_grid=grid,

        passbands=passbands,

        zero_points=zero_points,

        measurements=measurements,

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        distance=25.0,

        rv=3.10

    )


except ValueError:

    invalid_vector_rejected = True


assert invalid_vector_rejected


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "BINARY PHOTOMETRIC OBJECTIVE"

)

print(

    "=" * 70

)


print(

    "Reference parameters"

)

print(

    reference_parameters

)


print()

print(

    "Reference chi-square"

)

print(

    reference_chi_square

)


print()

print(

    "Perturbed parameters"

)

print(

    perturbed_parameters

)


print()

print(

    "Perturbed chi-square"

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


print()

print(

    "Invalid parameter vector rejected"

)

print(

    invalid_vector_rejected

)