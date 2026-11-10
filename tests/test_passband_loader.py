from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
)


# ==========================================================
# Johnson V passband file
# ==========================================================

filename = (

    Path(
        "validation"
    )

    /

    "photometry"

    /

    "passbands"

    /

    "johnson_v.dat"

)


# ==========================================================
# Load passband
# ==========================================================

passband = load_passband(

    filename=filename,

    name="Johnson V"

)


# ==========================================================
# Validate loaded data
# ==========================================================

assert passband.name == "Johnson V"

# Maiz Apellaniz (2006) Table 3 curve, photon-counting (file header).
assert passband.photon_counting is True


assert len(

    passband.wavelength

) == 47


assert np.isclose(

    passband.minimum_wavelength,

    4700.0

)


assert np.isclose(

    passband.maximum_wavelength,

    7000.0

)


assert np.isclose(

    passband.peak_wavelength,

    5250.0

)


assert np.isclose(

    np.max(
        passband.throughput
    ),

    1.0

)


assert np.isclose(

    passband.throughput[
        0
    ],

    0.0

)


assert np.isclose(

    passband.throughput[
        -1
    ],

    0.0

)


# ==========================================================
# Report loaded passband
# ==========================================================

print()

print(

    "LOADED PHOTOMETRIC PASSBAND"

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

    "Number of wavelength samples"

)

print(

    len(
        passband.wavelength
    )

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

    "Maximum throughput"

)

print(

    np.max(
        passband.throughput
    )

)


print()

print(

    "Boundary throughputs"

)

print(

    passband.throughput[
        0
    ],

    passband.throughput[
        -1
    ]

)


print()

print(

    "FIRST FIVE PASSBAND SAMPLES"

)

print(

    "=" * 70

)


for wavelength, throughput in zip(

    passband.wavelength[
        :5
    ],

    passband.throughput[
        :5
    ]

):

    print(

        f"{wavelength:8.1f} "

        f"{throughput:.10f}"

    )