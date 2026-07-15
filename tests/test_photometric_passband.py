import numpy as np

from pyseshat.photometry.filters import (
    PhotometricPassband
)


# ==========================================================
# Construct a simple triangular passband
# ==========================================================

wavelength = np.array([

    4000.0,

    4500.0,

    5000.0,

    5500.0,

    6000.0

])


throughput = np.array([

    0.0,

    0.5,

    1.0,

    0.5,

    0.0

])


passband = PhotometricPassband(

    wavelength=wavelength,

    throughput=throughput,

    name="Test V"

)


# ==========================================================
# Interpolate throughput
# ==========================================================

sample_wavelength = np.array([

    3500.0,

    4250.0,

    4750.0,

    5250.0,

    5750.0,

    6500.0

])


interpolated = passband.throughput_at(

    sample_wavelength

)


expected = np.array([

    0.0,

    0.25,

    0.75,

    0.75,

    0.25,

    0.0

])


# ==========================================================
# Report
# ==========================================================

print()

print(

    "PHOTOMETRIC PASSBAND"

)

print(

    "=" * 70

)

print(

    "Name"

)

print(

    passband.name

)


print()

print(

    "Wavelength range"

)

print(

    passband.minimum_wavelength,

    passband.maximum_wavelength

)


print()

print(

    "Peak wavelength"

)

print(

    passband.peak_wavelength

)


print()

print(

    "INTERPOLATED THROUGHPUT"

)

print(

    "=" * 70

)

print(

    interpolated

)


print()

print(

    "EXPECTED THROUGHPUT"

)

print(

    "=" * 70

)

print(

    expected

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

            interpolated

            -

            expected

        )

    )

)