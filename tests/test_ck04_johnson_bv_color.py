from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
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
# Load CK04 solar-metallicity atmosphere grid
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
# Interpolate component atmospheres
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
# Construct unresolved binary SED
# ==========================================================

wavelength, composite_flux = (

    composite_binary_sed(

        primary=primary,

        secondary=secondary,

        primary_radius=1.20,

        secondary_radius=0.90,

        distance=25.0

    )

)


# ==========================================================
# Load real Johnson B and V passbands
# ==========================================================

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
# Calculate synthetic band fluxes
# ==========================================================

johnson_b_flux = synthetic_band_flux(

    wavelength=wavelength,

    flux=composite_flux,

    passband=johnson_b

)


johnson_v_flux = synthetic_band_flux(

    wavelength=wavelength,

    flux=composite_flux,

    passband=johnson_v

)


# ==========================================================
# Calculate Vega magnitudes
# ==========================================================

johnson_b_magnitude = pogson_magnitude(

    flux=johnson_b_flux,

    zero_point=(

        JOHNSON_B_VEGA_ZERO_POINT

    )

)


johnson_v_magnitude = pogson_magnitude(

    flux=johnson_v_flux,

    zero_point=(

        JOHNSON_V_VEGA_ZERO_POINT

    )

)


# ==========================================================
# Calculate Johnson B-V color
# ==========================================================

johnson_bv_color = (

    johnson_b_magnitude

    -

    johnson_v_magnitude

)


# ==========================================================
# Independently calculate the expected color
# ==========================================================

expected_color = (

    -2.5

    *

    np.log10(

        (

            johnson_b_flux

            /

            JOHNSON_B_VEGA_ZERO_POINT

        )

        /

        (

            johnson_v_flux

            /

            JOHNSON_V_VEGA_ZERO_POINT

        )

    )

)


# ==========================================================
# Validate the integrated calculation
# ==========================================================

assert np.isfinite(

    johnson_bv_color

)


assert np.isclose(

    johnson_bv_color,

    expected_color,

    rtol=1.0e-14,

    atol=1.0e-14

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "CK04 BINARY JOHNSON B-V COLOR"

)

print(

    "=" * 70

)


print(

    "Johnson B band flux"

)

print(

    johnson_b_flux

)


print()

print(

    "Johnson V band flux"

)

print(

    johnson_v_flux

)


print()

print(

    "Synthetic Johnson B magnitude"

)

print(

    johnson_b_magnitude

)


print()

print(

    "Synthetic Johnson V magnitude"

)

print(

    johnson_v_magnitude

)


print()

print(

    "Synthetic Johnson B-V color"

)

print(

    johnson_bv_color

)


print()

print(

    "Independently calculated B-V color"

)

print(

    expected_color

)


print()

print(

    "Absolute difference"

)

print(

    abs(

        johnson_bv_color

        -

        expected_color

    )

)