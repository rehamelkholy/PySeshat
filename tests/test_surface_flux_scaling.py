import numpy as np

from pyseshat.stellar.atmospheres import (
    AtmosphereModel
)

from pyseshat.stellar.sed_model import (
    PARSEC_CM,
    SOLAR_RADIUS_CM,
    scale_surface_flux
)


# ==========================================================
# Construct test atmosphere
# ==========================================================

wavelength = np.array([

    4000.0,

    5000.0,

    6000.0,

    7000.0

])


surface_flux = np.array([

    1.0e8,

    2.0e8,

    1.5e8,

    1.0e8

])


atmosphere = AtmosphereModel(

    wavelength=wavelength,

    flux=surface_flux,

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


# ==========================================================
# Scale to observer
# ==========================================================

radius = 1.0

distance = 10.0


observed = scale_surface_flux(

    atmosphere=atmosphere,

    radius=radius,

    distance=distance

)


# ==========================================================
# Independent expected result
# ==========================================================

expected_scale = (

    SOLAR_RADIUS_CM

    /

    (

        10.0

        *

        PARSEC_CM

    )

) ** 2


expected_flux = (

    surface_flux

    *

    expected_scale

)


# ==========================================================
# Report
# ==========================================================

print()

print(

    "GEOMETRIC SCALE FACTOR"

)

print(

    "=" * 70

)

print(

    expected_scale

)


print()

print(

    "EXPECTED OBSERVED FLUX"

)

print(

    "=" * 70

)

print(

    expected_flux

)


print()

print(

    "SCALED OBSERVED FLUX"

)

print(

    "=" * 70

)

print(

    observed.flux

)


print()

print(

    "MAXIMUM ABSOLUTE DIFFERENCE"

)

print(

    "=" * 70

)

print(

    np.max(

        np.abs(

            observed.flux

            -

            expected_flux

        )

    )

)


print()

print(

    "ATMOSPHERE PARAMETERS PRESERVED"

)

print(

    "=" * 70

)

print(

    observed.teff

)

print(

    observed.logg

)

print(

    observed.metallicity

)