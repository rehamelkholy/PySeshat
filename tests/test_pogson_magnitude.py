import numpy as np

from pyseshat.photometry.synthetic import (
    pogson_magnitude
)


# ==========================================================
# Johnson V Vega zero point
# ==========================================================

zero_point = 3.5833e-9


# ==========================================================
# Zero-magnitude source
# ==========================================================

zero_magnitude = pogson_magnitude(

    flux=zero_point,

    zero_point=zero_point

)


# ==========================================================
# Source ten times fainter
# ==========================================================

ten_times_fainter = pogson_magnitude(

    flux=(

        zero_point

        /

        10.0

    ),

    zero_point=zero_point

)


# ==========================================================
# Source one hundred times fainter
# ==========================================================

hundred_times_fainter = pogson_magnitude(

    flux=(

        zero_point

        /

        100.0

    ),

    zero_point=zero_point

)


# ==========================================================
# Array calculation
# ==========================================================

fluxes = np.array([

    zero_point,

    zero_point / 10.0,

    zero_point / 100.0

])


array_magnitudes = pogson_magnitude(

    flux=fluxes,

    zero_point=zero_point

)


expected_array = np.array([

    0.0,

    2.5,

    5.0

])


# ==========================================================
# Validate results
# ==========================================================

assert isinstance(

    zero_magnitude,

    float

)


assert np.isclose(

    zero_magnitude,

    0.0

)


assert np.isclose(

    ten_times_fainter,

    2.5

)


assert np.isclose(

    hundred_times_fainter,

    5.0

)


assert np.allclose(

    array_magnitudes,

    expected_array

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "POGSON MAGNITUDE CALIBRATION"

)

print(

    "=" * 70

)


print(

    "Zero-point flux"

)

print(

    zero_point

)


print()

print(

    "Magnitude at the zero-point flux"

)

print(

    zero_magnitude

)


print()

print(

    "Magnitude at one tenth of the zero-point flux"

)

print(

    ten_times_fainter

)


print()

print(

    "Magnitude at one hundredth of the zero-point flux"

)

print(

    hundred_times_fainter

)


print()

print(

    "ARRAY CALCULATION"

)

print(

    "=" * 70

)


print(

    "Computed magnitudes"

)

print(

    array_magnitudes

)


print()

print(

    "Expected magnitudes"

)

print(

    expected_array

)


print()

print(

    "Maximum absolute difference"

)

print(

    np.max(

        np.abs(

            array_magnitudes

            -

            expected_array

        )

    )

)