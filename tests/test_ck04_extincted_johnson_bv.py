from pathlib import Path

import numpy as np

from pyseshat.photometry.extinction import (
    apply_extinction,
    ccm89_optical_extinction
)

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
# Construct intrinsic unresolved binary SED
# ==========================================================

wavelength, intrinsic_flux = (

    composite_binary_sed(

        primary=primary,

        secondary=secondary,

        primary_radius=1.20,

        secondary_radius=0.90,

        distance=25.0

    )

)


# ==========================================================
# Restrict the SED to the optical CCM89 domain
# ==========================================================

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


# ==========================================================
# Calculate the optical CCM89 extinction curve
# ==========================================================

extinction_curve = (

    ccm89_optical_extinction(

        wavelength=optical_wavelength,

        ebv=EBV,

        rv=RV

    )

)


# ==========================================================
# Apply extinction to the composite SED
# ==========================================================

extincted_flux = apply_extinction(

    flux=optical_intrinsic_flux,

    extinction=extinction_curve

)


# ==========================================================
# Load Johnson B and V passbands
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
# Calculate intrinsic Johnson fluxes
# ==========================================================

intrinsic_b_flux = synthetic_band_flux(

    wavelength=optical_wavelength,

    flux=optical_intrinsic_flux,

    passband=johnson_b

)


intrinsic_v_flux = synthetic_band_flux(

    wavelength=optical_wavelength,

    flux=optical_intrinsic_flux,

    passband=johnson_v

)


# ==========================================================
# Calculate extincted Johnson fluxes
# ==========================================================

extincted_b_flux = synthetic_band_flux(

    wavelength=optical_wavelength,

    flux=extincted_flux,

    passband=johnson_b

)


extincted_v_flux = synthetic_band_flux(

    wavelength=optical_wavelength,

    flux=extincted_flux,

    passband=johnson_v

)


# ==========================================================
# Calculate intrinsic Johnson magnitudes
# ==========================================================

intrinsic_b_magnitude = pogson_magnitude(

    flux=intrinsic_b_flux,

    zero_point=(

        JOHNSON_B_VEGA_ZERO_POINT

    )

)


intrinsic_v_magnitude = pogson_magnitude(

    flux=intrinsic_v_flux,

    zero_point=(

        JOHNSON_V_VEGA_ZERO_POINT

    )

)


# ==========================================================
# Calculate extincted Johnson magnitudes
# ==========================================================

extincted_b_magnitude = pogson_magnitude(

    flux=extincted_b_flux,

    zero_point=(

        JOHNSON_B_VEGA_ZERO_POINT

    )

)


extincted_v_magnitude = pogson_magnitude(

    flux=extincted_v_flux,

    zero_point=(

        JOHNSON_V_VEGA_ZERO_POINT

    )

)


# ==========================================================
# Calculate colors
# ==========================================================

intrinsic_bv = (

    intrinsic_b_magnitude

    -

    intrinsic_v_magnitude

)


extincted_bv = (

    extincted_b_magnitude

    -

    extincted_v_magnitude

)


synthetic_color_excess = (

    extincted_bv

    -

    intrinsic_bv

)


# ==========================================================
# Calculate effective broadband extinctions
# ==========================================================

effective_b_extinction = (

    extincted_b_magnitude

    -

    intrinsic_b_magnitude

)


effective_v_extinction = (

    extincted_v_magnitude

    -

    intrinsic_v_magnitude

)


# ==========================================================
# Validate physical behavior
# ==========================================================

assert np.all(

    np.isfinite(
        extinction_curve
    )

)


assert np.all(

    extinction_curve >= 0.0

)


assert np.all(

    extincted_flux

    <=

    optical_intrinsic_flux

)


assert extincted_b_flux < intrinsic_b_flux


assert extincted_v_flux < intrinsic_v_flux


assert extincted_b_magnitude > intrinsic_b_magnitude


assert extincted_v_magnitude > intrinsic_v_magnitude


assert effective_b_extinction > effective_v_extinction


assert extincted_bv > intrinsic_bv


assert synthetic_color_excess > 0.0


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "EXTINCTED CK04 JOHNSON BV PHOTOMETRY"

)

print(

    "=" * 70

)


print(

    "Input E(B-V)"

)

print(

    EBV

)


print()

print(

    "Input R_V"

)

print(

    RV

)


print()

print(

    "Intrinsic B magnitude"

)

print(

    intrinsic_b_magnitude

)


print()

print(

    "Extincted B magnitude"

)

print(

    extincted_b_magnitude

)


print()

print(

    "Effective broadband A_B"

)

print(

    effective_b_extinction

)


print()

print(

    "Intrinsic V magnitude"

)

print(

    intrinsic_v_magnitude

)


print()

print(

    "Extincted V magnitude"

)

print(

    extincted_v_magnitude

)


print()

print(

    "Effective broadband A_V"

)

print(

    effective_v_extinction

)


print()

print(

    "Intrinsic B-V color"

)

print(

    intrinsic_bv

)


print()

print(

    "Extincted B-V color"

)

print(

    extincted_bv

)


print()

print(

    "Synthetic broadband color excess"

)

print(

    synthetic_color_excess

)


print()

print(

    "Input minus synthetic color excess"

)

print(

    EBV

    -

    synthetic_color_excess

)