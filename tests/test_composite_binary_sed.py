import numpy as np

from pyseshat.stellar.atmospheres import (
    AtmosphereModel
)

from pyseshat.stellar.sed_model import (
    PARSEC_CM,
    SOLAR_RADIUS_CM,
    composite_binary_sed
)


# ==========================================================
# Common wavelength grid
# ==========================================================

wavelength = np.array([

    4000.0,

    5000.0,

    6000.0,

    7000.0

])


# ==========================================================
# Primary atmosphere
# ==========================================================

primary_flux = np.array([

    2.0e8,

    3.0e8,

    2.5e8,

    2.0e8

])


primary = AtmosphereModel(

    wavelength=wavelength,

    flux=primary_flux,

    teff=6250.0,

    logg=4.5,

    metallicity=0.0

)


# ==========================================================
# Secondary atmosphere
# ==========================================================

secondary_flux = np.array([

    1.0e8,

    2.0e8,

    1.5e8,

    1.0e8

])


secondary = AtmosphereModel(

    wavelength=wavelength,

    flux=secondary_flux,

    teff=5750.0,

    logg=4.5,

    metallicity=0.0

)


# ==========================================================
# Binary parameters
# ==========================================================

primary_radius = 1.2

secondary_radius = 0.8

distance = 20.0


# ==========================================================
# Construct composite SED
# ==========================================================

composite_wavelength, composite_flux = (

    composite_binary_sed(

        primary=primary,

        secondary=secondary,

        primary_radius=primary_radius,

        secondary_radius=secondary_radius,

        distance=distance

    )

)


# ==========================================================
# Independent expected result
# ==========================================================

distance_cm = (

    distance

    *

    PARSEC_CM

)


primary_scale = (

    (

        primary_radius

        *

        SOLAR_RADIUS_CM

    )

    /

    distance_cm

) ** 2


secondary_scale = (

    (

        secondary_radius

        *

        SOLAR_RADIUS_CM

    )

    /

    distance_cm

) ** 2


expected_flux = (

    primary_flux

    *

    primary_scale

    +

    secondary_flux

    *

    secondary_scale

)


# ==========================================================
# Report
# ==========================================================

print()

print(

    "PRIMARY SCALE FACTOR"

)

print(

    "=" * 70

)

print(

    primary_scale

)


print()

print(

    "SECONDARY SCALE FACTOR"

)

print(

    "=" * 70

)

print(

    secondary_scale

)


print()

print(

    "EXPECTED COMPOSITE FLUX"

)

print(

    "=" * 70

)

print(

    expected_flux

)


print()

print(

    "COMPUTED COMPOSITE FLUX"

)

print(

    "=" * 70

)

print(

    composite_flux

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

            composite_flux

            -

            expected_flux

        )

    )

)


print()

print(

    "WAVELENGTH GRID PRESERVED"

)

print(

    "=" * 70

)

print(

    np.array_equal(

        composite_wavelength,

        wavelength

    )

)