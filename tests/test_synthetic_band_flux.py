import numpy as np

from pyseshat.photometry.filters import (
    PhotometricPassband
)

from pyseshat.photometry.synthetic import (
    synthetic_band_flux
)


# ==========================================================
# Construct a triangular passband
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
# Constant SED
# ==========================================================

wavelength = np.linspace(

    3500.0,

    6500.0,

    301

)


constant_flux = np.full(

    wavelength.shape,

    2.5e-10

)


constant_result = synthetic_band_flux(

    wavelength=wavelength,

    flux=constant_flux,

    passband=passband

)


constant_expected = 2.5e-10


# ==========================================================
# Linear SED
# ==========================================================

linear_flux = (

    1.0e-10

    +

    (

        2.0e-14

        *

        (

            wavelength

            -

            4000.0

        )

    )

)


linear_result = synthetic_band_flux(

    wavelength=wavelength,

    flux=linear_flux,

    passband=passband

)


# The passband is symmetric around 5000 Angstrom.
# Therefore, the weighted mean of this linear SED is
# its value at 5000 Angstrom.

linear_expected = (

    1.0e-10

    +

    (

        2.0e-14

        *

        (

            5000.0

            -

            4000.0

        )

    )

)


# ==========================================================
# Report constant-SED result
# ==========================================================

print()

print(

    "CONSTANT SED"

)

print(

    "=" * 70

)

print(

    "Expected band flux"

)

print(

    constant_expected

)

print()

print(

    "Computed band flux"

)

print(

    constant_result

)

print()

print(

    "Absolute difference"

)

print(

    abs(

        constant_result

        -

        constant_expected

    )

)


# ==========================================================
# Report linear-SED result
# ==========================================================

print()

print(

    "LINEAR SED"

)

print(

    "=" * 70

)

print(

    "Expected band flux"

)

print(

    linear_expected

)

print()

print(

    "Computed band flux"

)

print(

    linear_result

)

print()

print(

    "Absolute difference"

)

print(

    abs(

        linear_result

        -

        linear_expected

    )

)