import numpy as np

from pyseshat.photometry.extinction import (
    ccm89_optical_extinction
)


# ==========================================================
# Standard extinction parameters
# ==========================================================

ebv = 0.20

rv = 3.10


# ==========================================================
# CCM89 normalization wavelength
# ==========================================================

normalization_wavelength = (

    10000.0

    /

    1.82

)


normalization_extinction = (

    ccm89_optical_extinction(

        wavelength=(

            normalization_wavelength

        ),

        ebv=ebv,

        rv=rv

    )

)


normalization_expected = (

    rv

    *

    ebv

)


# ==========================================================
# Zero color excess
# ==========================================================

wavelength = np.array([

    4000.0,

    5000.0,

    6000.0,

    7000.0

])


zero_extinction = (

    ccm89_optical_extinction(

        wavelength=wavelength,

        ebv=0.0,

        rv=rv

    )

)


zero_expected = np.zeros_like(

    wavelength

)


# ==========================================================
# Optical extinction curve
# ==========================================================

optical_extinction = (

    ccm89_optical_extinction(

        wavelength=wavelength,

        ebv=ebv,

        rv=rv

    )

)


# ==========================================================
# Scalar-return behavior
# ==========================================================

scalar_extinction = (

    ccm89_optical_extinction(

        wavelength=5500.0,

        ebv=ebv,

        rv=rv

    )

)


# ==========================================================
# Validate results
# ==========================================================

assert isinstance(

    normalization_extinction,

    float

)


assert np.isclose(

    normalization_extinction,

    normalization_expected,

    rtol=0.0,

    atol=1.0e-14

)


assert np.array_equal(

    zero_extinction,

    zero_expected

)


assert np.all(

    np.isfinite(
        optical_extinction
    )

)


assert np.all(

    optical_extinction > 0.0

)


assert np.all(

    np.diff(
        optical_extinction
    )

    < 0.0

)


assert isinstance(

    scalar_extinction,

    float

)


# ==========================================================
# Report normalization
# ==========================================================

print()

print(

    "CCM89 OPTICAL NORMALIZATION"

)

print(

    "=" * 70

)


print(

    "Normalization wavelength"

)

print(

    normalization_wavelength

)


print()

print(

    "Expected A(lambda)"

)

print(

    normalization_expected

)


print()

print(

    "Computed A(lambda)"

)

print(

    normalization_extinction

)


print()

print(

    "Absolute difference"

)

print(

    abs(

        normalization_extinction

        -

        normalization_expected

    )

)


# ==========================================================
# Report zero-extinction result
# ==========================================================

print()

print(

    "ZERO COLOR EXCESS"

)

print(

    "=" * 70

)


print(

    zero_extinction

)


# ==========================================================
# Report optical extinction curve
# ==========================================================

print()

print(

    "OPTICAL CCM89 EXTINCTION CURVE"

)

print(

    "=" * 70

)


for wavelength_value, extinction_value in zip(

    wavelength,

    optical_extinction

):

    print(

        f"{wavelength_value:8.1f} "

        f"{extinction_value:.10f}"

    )


print()

print(

    "Extinction decreases with wavelength"

)

print(

    np.all(

        np.diff(

            optical_extinction

        )

        < 0.0

    )

)