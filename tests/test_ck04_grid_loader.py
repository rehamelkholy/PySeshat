from pathlib import Path

import numpy as np

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)


# ==========================================================
# CK04 solar-metallicity atmosphere files
# ==========================================================

directory = (

    Path(
        "validation"
    )

    /

    "atmosphere_models"

    /

    "ckp00"

)


filenames = [

    directory

    /

    "ckp00_5750.fits",

    directory

    /

    "ckp00_6000.fits",

    directory

    /

    "ckp00_6250.fits"

]


# ==========================================================
# Load combined atmosphere grid
# ==========================================================

grid = load_ck04_grid(

    filenames

)


# ==========================================================
# Display grid information
# ==========================================================

print()

print(

    "COMBINED CK04 GRID"

)

print(

    "=" * 70

)

print(

    "Number of atmosphere models"

)

print(

    len(
        grid
    )

)


print()

print(

    "EFFECTIVE TEMPERATURES"

)

print(

    "=" * 70

)

print(

    grid.temperatures

)


print()

print(

    "SURFACE GRAVITIES"

)

print(

    "=" * 70

)

print(

    grid.surface_gravities

)


print()

print(

    "METALLICITIES"

)

print(

    "=" * 70

)

print(

    grid.metallicities

)


# ==========================================================
# Validate expected grid
# ==========================================================

expected_temperatures = np.array([

    5750.0,

    6000.0,

    6250.0

])


expected_gravities = np.arange(

    0.0,

    5.0 + 0.5,

    0.5

)


assert np.array_equal(

    grid.temperatures,

    expected_temperatures

)


assert np.array_equal(

    grid.surface_gravities,

    expected_gravities

)


assert np.array_equal(

    grid.metallicities,

    np.array([

        0.0

    ])

)


# ==========================================================
# Inspect one exact atmosphere model
# ==========================================================

model = grid.get_model(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


print()

print(

    "SELECTED ATMOSPHERE MODEL"

)

print(

    "=" * 70

)

print(

    "Parameter point"

)

print(

    (

        model.teff,

        model.logg,

        model.metallicity

    )

)


print()

print(

    "Number of wavelength samples"

)

print(

    len(
        model.wavelength
    )

)


print()

print(

    "Wavelength range"

)

print(

    model.wavelength[
        0
    ],

    model.wavelength[
        -1
    ]

)


print()

print(

    "Flux range"

)

print(

    model.flux.min(),

    model.flux.max()

)


# ==========================================================
# Validate common wavelength grid
# ==========================================================

reference = grid.get_model(

    teff=5750.0,

    logg=4.5,

    metallicity=0.0

)


comparison = grid.get_model(

    teff=6250.0,

    logg=4.5,

    metallicity=0.0

)


print()

print(

    "COMMON WAVELENGTH GRID"

)

print(

    "=" * 70

)

print(

    np.array_equal(

        reference.wavelength,

        comparison.wavelength

    )

)

# ==========================================================
# Identify missing atmosphere parameter points
# ==========================================================

print()

print(

    "GRID COMPLETENESS"

)

print(

    "=" * 70

)


expected_points = {

    (

        temperature,

        gravity,

        0.0

    )

    for temperature in expected_temperatures

    for gravity in expected_gravities

}


available_points = set(

    grid.parameter_points

)


missing_points = sorted(

    expected_points

    -

    available_points

)


print(

    "Expected number of models"

)

print(

    len(
        expected_points
    )

)


print()

print(

    "Available number of models"

)

print(

    len(
        available_points
    )

)


print()

print(

    "Missing parameter points"

)

if missing_points:

    for point in missing_points:

        print(

            point

        )

else:

    print(

        "None"

    )