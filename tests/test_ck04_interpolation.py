from pathlib import Path

import numpy as np

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)


# ==========================================================
# Load validation atmosphere grid
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
# Exact grid-point recovery
# ==========================================================

exact = grid.interpolate(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


reference = grid.get_model(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


print()

print(

    "EXACT GRID-POINT RECOVERY"

)

print(

    "=" * 70

)

print(

    "Flux arrays identical"

)

print(

    np.array_equal(

        exact.flux,

        reference.flux

    )

)


# ==========================================================
# Temperature interpolation
# ==========================================================

temperature_model = grid.interpolate(

    teff=6125.0,

    logg=4.5,

    metallicity=0.0

)


lower_temperature = grid.get_model(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


upper_temperature = grid.get_model(

    teff=6250.0,

    logg=4.5,

    metallicity=0.0

)


expected_temperature_flux = (

    0.5

    *

    (

        lower_temperature.flux

        +

        upper_temperature.flux

    )

)


print()

print(

    "TEMPERATURE INTERPOLATION"

)

print(

    "=" * 70

)

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            temperature_model.flux

            -

            expected_temperature_flux

        )

    )

)


# ==========================================================
# Surface-gravity interpolation
# ==========================================================

gravity_model = grid.interpolate(

    teff=6000.0,

    logg=4.25,

    metallicity=0.0

)


lower_gravity = grid.get_model(

    teff=6000.0,

    logg=4.0,

    metallicity=0.0

)


upper_gravity = grid.get_model(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


expected_gravity_flux = (

    0.5

    *

    (

        lower_gravity.flux

        +

        upper_gravity.flux

    )

)


print()

print(

    "SURFACE-GRAVITY INTERPOLATION"

)

print(

    "=" * 70

)

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            gravity_model.flux

            -

            expected_gravity_flux

        )

    )

)


# ==========================================================
# Bilinear interpolation
# ==========================================================

bilinear_model = grid.interpolate(

    teff=6125.0,

    logg=4.25,

    metallicity=0.0

)


corner_fluxes = [

    grid.get_model(

        teff=temperature,

        logg=gravity,

        metallicity=0.0

    ).flux

    for temperature in (

        6000.0,

        6250.0

    )

    for gravity in (

        4.0,

        4.5

    )

]


expected_bilinear_flux = (

    sum(
        corner_fluxes
    )

    /

    4.0

)


print()

print(

    "BILINEAR INTERPOLATION"

)

print(

    "=" * 70

)

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            bilinear_model.flux

            -

            expected_bilinear_flux

        )

    )

)