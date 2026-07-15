from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
)

from pyseshat.photometry.synthetic import (
    synthetic_band_flux
)

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
)

from pyseshat.stellar.sed_model import (
    composite_binary_sed
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
# Calculate synthetic Johnson V band flux
# ==========================================================

johnson_v_flux = synthetic_band_flux(

    wavelength=wavelength,

    flux=composite_flux,

    passband=johnson_v

)


# ==========================================================
# Independently reproduce the band-flux calculation
# ==========================================================

overlap = (

    (

        wavelength

        >=

        johnson_v.minimum_wavelength

    )

    &

    (

        wavelength

        <=

        johnson_v.maximum_wavelength

    )

)


integration_wavelength = wavelength[

    overlap

]


integration_flux = composite_flux[

    overlap

]


throughput = johnson_v.throughput_at(

    integration_wavelength

)


expected_flux = (

    np.trapezoid(

        (

            integration_flux

            *

            throughput

        ),

        integration_wavelength

    )

    /

    np.trapezoid(

        throughput,

        integration_wavelength

    )

)


# ==========================================================
# Validate the integrated calculation
# ==========================================================

assert np.isfinite(

    johnson_v_flux

)


assert johnson_v_flux > 0.0


assert np.isclose(

    johnson_v_flux,

    expected_flux,

    rtol=1.0e-14,

    atol=0.0

)


assert (

    johnson_v.minimum_wavelength

    >=

    wavelength[
        0
    ]

)


assert (

    johnson_v.maximum_wavelength

    <=

    wavelength[
        -1
    ]

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "CK04 BINARY JOHNSON V PHOTOMETRY"

)

print(

    "=" * 70

)


print(

    "Passband"

)

print(

    johnson_v.name

)


print()

print(

    "Passband wavelength range"

)

print(

    johnson_v.minimum_wavelength,

    johnson_v.maximum_wavelength

)


print()

print(

    "Composite SED wavelength range"

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

    "Synthetic Johnson V flux"

)

print(

    johnson_v_flux

)


print()

print(

    "Independently calculated flux"

)

print(

    expected_flux

)


print()

print(

    "Absolute difference"

)

print(

    abs(

        johnson_v_flux

        -

        expected_flux

    )

)