from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    PhotometricPassband
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
# Load CK04 solar-metallicity validation grid
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
# Construct validation passband
# ==========================================================

passband = PhotometricPassband(

    wavelength=np.array([

        4000.0,

        4500.0,

        5000.0,

        5500.0,

        6000.0

    ]),

    throughput=np.array([

        0.0,

        0.5,

        1.0,

        0.5,

        0.0

    ]),

    name="Test V"

)


# ==========================================================
# Calculate synthetic band flux
# ==========================================================

band_flux = synthetic_band_flux(

    wavelength=wavelength,

    flux=composite_flux,

    passband=passband

)


# ==========================================================
# Independently reproduce the calculation
# ==========================================================

overlap = (

    (

        wavelength

        >=

        passband.minimum_wavelength

    )

    &

    (

        wavelength

        <=

        passband.maximum_wavelength

    )

)


integration_wavelength = wavelength[

    overlap

]


integration_flux = composite_flux[

    overlap

]


throughput = passband.throughput_at(

    integration_wavelength

)


expected_flux = (

    np.trapezoid(

        integration_flux

        *

        throughput,

        integration_wavelength

    )

    /

    np.trapezoid(

        throughput,

        integration_wavelength

    )

)


# ==========================================================
# Validate integrated pipeline
# ==========================================================

assert np.isfinite(

    band_flux

)


assert band_flux > 0.0


assert np.isclose(

    band_flux,

    expected_flux,

    rtol=1.0e-14,

    atol=0.0

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "CK04 BINARY SYNTHETIC PHOTOMETRY"

)

print(

    "=" * 70

)

print(

    "Passband"

)

print(

    passband.name

)


print()

print(

    "SED wavelength range"

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

    "Passband wavelength range"

)

print(

    passband.minimum_wavelength,

    passband.maximum_wavelength

)


print()

print(

    "Synthetic band flux"

)

print(

    band_flux

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

        band_flux

        -

        expected_flux

    )

)