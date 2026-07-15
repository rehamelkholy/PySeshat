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
# SVO Generic/Johnson.V Vega calibration
# ==========================================================

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
# Load the real Johnson V passband
# ==========================================================

johnson_v = load_passband(

    filename=(

        passband_directory

        /

        "johnson_v.dat"

    ),

    name="Johnson V"

)


# ==========================================================
# Calculate synthetic Johnson V flux
# ==========================================================

johnson_v_flux = synthetic_band_flux(

    wavelength=wavelength,

    flux=composite_flux,

    passband=johnson_v

)


# ==========================================================
# Convert the synthetic flux to a Vega magnitude
# ==========================================================

johnson_v_magnitude = pogson_magnitude(

    flux=johnson_v_flux,

    zero_point=(

        JOHNSON_V_VEGA_ZERO_POINT

    )

)


# ==========================================================
# Independently calculate the expected magnitude
# ==========================================================

expected_magnitude = (

    -2.5

    *

    np.log10(

        johnson_v_flux

        /

        JOHNSON_V_VEGA_ZERO_POINT

    )

)


# ==========================================================
# Validate the integrated calculation
# ==========================================================

assert np.isfinite(

    johnson_v_magnitude

)


assert np.isclose(

    johnson_v_magnitude,

    expected_magnitude,

    rtol=0.0,

    atol=0.0

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "CK04 BINARY JOHNSON V MAGNITUDE"

)

print(

    "=" * 70

)


print(

    "Photometric system"

)

print(

    "Vega"

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

    "Johnson V Vega zero point"

)

print(

    JOHNSON_V_VEGA_ZERO_POINT

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

    "Independently calculated magnitude"

)

print(

    expected_magnitude

)


print()

print(

    "Absolute difference"

)

print(

    abs(

        johnson_v_magnitude

        -

        expected_magnitude

    )

)