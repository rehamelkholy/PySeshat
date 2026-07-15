from pathlib import Path

import numpy as np

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)

from pyseshat.stellar.sed_model import (
    composite_binary_sed,
    scale_surface_flux
)


# ==========================================================
# Load the CK04 solar-metallicity validation grid
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


grid = load_ck04_grid([

    directory

    /

    "ckp00_5750.fits",

    directory

    /

    "ckp00_6000.fits",

    directory

    /

    "ckp00_6250.fits"

])


# ==========================================================
# Interpolate component atmosphere models
# ==========================================================

primary = grid.interpolate(

    teff=6125.0,

    logg=4.25,

    metallicity=0.0

)


secondary = grid.interpolate(

    teff=5875.0,

    logg=4.50,

    metallicity=0.0

)


# ==========================================================
# Binary parameters
# ==========================================================

primary_radius = 1.20

secondary_radius = 0.90

distance = 25.0


# ==========================================================
# Construct unresolved composite SED
# ==========================================================

wavelength, composite_flux = (

    composite_binary_sed(

        primary=primary,

        secondary=secondary,

        primary_radius=primary_radius,

        secondary_radius=secondary_radius,

        distance=distance

    )

)


# ==========================================================
# Independently scale and combine the components
# ==========================================================

primary_observed = scale_surface_flux(

    atmosphere=primary,

    radius=primary_radius,

    distance=distance

)


secondary_observed = scale_surface_flux(

    atmosphere=secondary,

    radius=secondary_radius,

    distance=distance

)


expected_flux = (

    primary_observed.flux

    +

    secondary_observed.flux

)


# ==========================================================
# Validate the integrated pipeline
# ==========================================================

assert np.array_equal(

    wavelength,

    primary.wavelength

)


assert np.array_equal(

    wavelength,

    secondary.wavelength

)


assert np.all(

    np.isfinite(
        composite_flux
    )

)


assert np.all(

    composite_flux >= 0.0

)


assert np.allclose(

    composite_flux,

    expected_flux,

    rtol=0.0,

    atol=0.0

)


# ==========================================================
# Report atmosphere parameters
# ==========================================================

print()

print(

    "PRIMARY ATMOSPHERE"

)

print(

    "=" * 70

)

print(

    "Effective temperature"

)

print(

    primary.teff

)

print()

print(

    "Surface gravity"

)

print(

    primary.logg

)

print()

print(

    "Metallicity"

)

print(

    primary.metallicity

)


print()

print(

    "SECONDARY ATMOSPHERE"

)

print(

    "=" * 70

)

print(

    "Effective temperature"

)

print(

    secondary.teff

)

print()

print(

    "Surface gravity"

)

print(

    secondary.logg

)

print()

print(

    "Metallicity"

)

print(

    secondary.metallicity

)


# ==========================================================
# Report composite SED
# ==========================================================

print()

print(

    "COMPOSITE CK04 SED"

)

print(

    "=" * 70

)

print(

    "Number of wavelength samples"

)

print(

    len(
        wavelength
    )

)


print()

print(

    "Wavelength range"

)

print(

    wavelength[
        0
    ],

    wavelength[
        -1
    ]

)


print()

print(

    "Composite flux range"

)

print(

    composite_flux.min(),

    composite_flux.max()

)


print()

print(

    "Integrated component agreement"

)

print(

    np.array_equal(

        composite_flux,

        expected_flux

    )

)