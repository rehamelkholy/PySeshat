from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
)

from pyseshat.photometry.model import (
    synthetic_binary_magnitudes
)

from pyseshat.stellar.atmospheres import (
    load_ck04_grid
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
# Load atmosphere grid
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
# Load Johnson passbands
# ==========================================================

passbands = {

    "Johnson B": load_passband(

        filename=(

            passband_directory

            /

            "johnson_b.dat"

        ),

        name="Johnson B"

    ),

    "Johnson V": load_passband(

        filename=(

            passband_directory

            /

            "johnson_v.dat"

        ),

        name="Johnson V"

    )

}


# ==========================================================
# Define SVO Vega zero points
# ==========================================================

zero_points = {

    "Johnson B": 6.72553e-9,

    "Johnson V": 3.5833e-9

}


# ==========================================================
# Calculate synthetic magnitudes
# ==========================================================

magnitudes = synthetic_binary_magnitudes(

    atmosphere_grid=grid,

    passbands=passbands,

    zero_points=zero_points,

    primary_temperature=6125.0,

    secondary_temperature=5875.0,

    primary_logg=4.25,

    secondary_logg=4.50,

    metallicity=0.0,

    primary_radius=1.20,

    secondary_radius=0.90,

    distance=25.0,

    ebv=0.20,

    rv=3.10

)


# ==========================================================
# Previously validated reference values
# ==========================================================

expected_b = 7.130442389055

expected_v = 6.296584811208877


# ==========================================================
# Validate forward-model output
# ==========================================================

assert set(

    magnitudes

) == {

    "Johnson B",

    "Johnson V"

}


assert isinstance(

    magnitudes[

        "Johnson B"

    ],

    float

)


assert isinstance(

    magnitudes[

        "Johnson V"

    ],

    float

)


assert np.isclose(

    magnitudes[

        "Johnson B"

    ],

    expected_b,

    rtol=0.0,

    atol=1.0e-12

)


assert np.isclose(

    magnitudes[

        "Johnson V"

    ],

    expected_v,

    rtol=0.0,

    atol=1.0e-12

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "SYNTHETIC BINARY MAGNITUDE MODEL"

)

print(

    "=" * 70

)


print(

    "Johnson B magnitude"

)

print(

    magnitudes[

        "Johnson B"

    ]

)


print()

print(

    "Expected Johnson B magnitude"

)

print(

    expected_b

)


print()

print(

    "Johnson V magnitude"

)

print(

    magnitudes[

        "Johnson V"

    ]

)


print()

print(

    "Expected Johnson V magnitude"

)

print(

    expected_v

)


print()

print(

    "Maximum absolute difference"

)

print(

    max(

        abs(

            magnitudes[

                "Johnson B"

            ]

            -

            expected_b

        ),

        abs(

            magnitudes[

                "Johnson V"

            ]

            -

            expected_v

        )

    )

)