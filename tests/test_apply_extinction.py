import numpy as np

from pyseshat.photometry.extinction import (
    apply_extinction
)


# ==========================================================
# Intrinsic validation fluxes
# ==========================================================

flux = np.array([

    1.0,

    2.0,

    4.0

])


# ==========================================================
# Zero extinction
# ==========================================================

zero_extinction_flux = apply_extinction(

    flux=flux,

    extinction=0.0

)


zero_extinction_expected = flux.copy()


# ==========================================================
# Uniform extinction of one magnitude
# ==========================================================

one_magnitude_flux = apply_extinction(

    flux=flux,

    extinction=1.0

)


one_magnitude_expected = (

    flux

    *

    np.power(

        10.0,

        -0.4

    )

)


# ==========================================================
# Wavelength-dependent extinction
# ==========================================================

extinction_curve = np.array([

    0.0,

    1.0,

    2.0

])


variable_extinction_flux = apply_extinction(

    flux=flux,

    extinction=extinction_curve

)


variable_extinction_expected = (

    flux

    *

    np.power(

        10.0,

        (

            -0.4

            *

            extinction_curve

        )

    )

)


# ==========================================================
# Scalar calculation
# ==========================================================

scalar_flux = apply_extinction(

    flux=10.0,

    extinction=2.5

)


scalar_expected = 1.0


# ==========================================================
# Validate results
# ==========================================================

assert np.array_equal(

    zero_extinction_flux,

    zero_extinction_expected

)


assert np.allclose(

    one_magnitude_flux,

    one_magnitude_expected

)


assert np.allclose(

    variable_extinction_flux,

    variable_extinction_expected

)


assert isinstance(

    scalar_flux,

    float

)


assert np.isclose(

    scalar_flux,

    scalar_expected

)


# ==========================================================
# Report zero-extinction result
# ==========================================================

print()

print(

    "ZERO EXTINCTION"

)

print(

    "=" * 70

)


print(

    "Computed flux"

)

print(

    zero_extinction_flux

)


print()

print(

    "Expected flux"

)

print(

    zero_extinction_expected

)


print()

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            zero_extinction_flux

            -

            zero_extinction_expected

        )

    )

)


# ==========================================================
# Report one-magnitude result
# ==========================================================

print()

print(

    "ONE MAGNITUDE OF EXTINCTION"

)

print(

    "=" * 70

)


print(

    "Computed flux"

)

print(

    one_magnitude_flux

)


print()

print(

    "Expected flux"

)

print(

    one_magnitude_expected

)


print()

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            one_magnitude_flux

            -

            one_magnitude_expected

        )

    )

)


# ==========================================================
# Report wavelength-dependent result
# ==========================================================

print()

print(

    "WAVELENGTH-DEPENDENT EXTINCTION"

)

print(

    "=" * 70

)


print(

    "Extinction curve"

)

print(

    extinction_curve

)


print()

print(

    "Computed flux"

)

print(

    variable_extinction_flux

)


print()

print(

    "Expected flux"

)

print(

    variable_extinction_expected

)


print()

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            variable_extinction_flux

            -

            variable_extinction_expected

        )

    )

)


# ==========================================================
# Report scalar result
# ==========================================================

print()

print(

    "SCALAR EXTINCTION"

)

print(

    "=" * 70

)


print(

    "Computed flux"

)

print(

    scalar_flux

)


print()

print(

    "Expected flux"

)

print(

    scalar_expected

)


print()

print(

    "Absolute difference"

)

print(

    abs(

        scalar_flux

        -

        scalar_expected

    )

)