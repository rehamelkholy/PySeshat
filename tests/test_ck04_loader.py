from pathlib import Path

from pyseshat.stellar.atmospheres import (
    load_ck04_fits
)


# ==========================================================
# CK04 atmosphere file
# ==========================================================

filename = Path(

    "validation"
) / (

    "atmosphere_models"
) / (

    "ckm05"
) / (

    "ckm05_10000.fits"
)


# ==========================================================
# Load atmosphere grid
# ==========================================================

grid = load_ck04_fits(

    filename

)


# ==========================================================
# Display grid information
# ==========================================================

print()

print(

    "CK04 ATMOSPHERE GRID"

)

print(

    "=" * 60

)

print(

    "Number of loaded models"

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

    "=" * 60

)

print(

    grid.temperatures

)


print()

print(

    "SURFACE GRAVITIES"

)

print(

    "=" * 60

)

print(

    grid.surface_gravities

)


print()

print(

    "METALLICITIES"

)

print(

    "=" * 60

)

print(

    grid.metallicities

)


# ==========================================================
# Inspect one atmosphere model
# ==========================================================

logg = grid.surface_gravities[

    len(
        grid.surface_gravities
    )

    //

    2

]


model = grid.get_model(

    teff=10000.0,

    logg=logg,

    metallicity=-0.5

)


print()

print(

    "SELECTED MODEL"

)

print(

    "=" * 60

)

print(

    "Effective temperature"

)

print(

    model.teff

)

print()

print(

    "Surface gravity"

)

print(

    model.logg

)

print()

print(

    "Metallicity"

)

print(

    model.metallicity

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