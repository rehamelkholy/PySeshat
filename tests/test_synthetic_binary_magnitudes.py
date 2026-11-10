from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    PhotometricPassband,
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

# Regression values. Updated in 0.2.0 for the Maiz Apellaniz (2006)
# photon-counting B and V curves (previously 7.130442389055 and
# 6.296584811208877 with the old curves integrated in energy form).
expected_b = 7.139368535769073

expected_v = 6.295428757417945


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

# ==========================================================
# Construct a zero-extinction near-infrared test passband
# ==========================================================

near_infrared_passband = PhotometricPassband(

    name="Test J",

    wavelength=np.array([

        10000.0,

        11000.0,

        12000.0,

        13000.0,

        14000.0

    ]),

    throughput=np.array([

        0.0,

        0.5,

        1.0,

        0.5,

        0.0

    ])

)


# ==========================================================
# Evaluate outside the optical CCM89 domain
# ==========================================================

zero_extinction_magnitudes = (

    synthetic_binary_magnitudes(

        atmosphere_grid=grid,

        passbands={

            "Test J": (

                near_infrared_passband

            )

        },

        zero_points={

            "Test J": 1.0e-9

        },

        primary_temperature=6125.0,

        secondary_temperature=5875.0,

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        primary_radius=1.20,

        secondary_radius=0.90,

        distance=25.0,

        ebv=0.0,

        rv=3.10

    )

)


# ==========================================================
# Validate the zero-extinction bypass
# ==========================================================

assert set(

    zero_extinction_magnitudes

) == {

    "Test J"

}


assert isinstance(

    zero_extinction_magnitudes[

        "Test J"

    ],

    float

)


assert np.isfinite(

    zero_extinction_magnitudes[

        "Test J"

    ]

)


# ==========================================================
# Confirm that nonzero extinction remains restricted
# ==========================================================

nonzero_extinction_rejected = False


try:

    synthetic_binary_magnitudes(

        atmosphere_grid=grid,

        passbands={

            "Test J": (

                near_infrared_passband

            )

        },

        zero_points={

            "Test J": 1.0e-9

        },

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


except ValueError:

    nonzero_extinction_rejected = True


assert nonzero_extinction_rejected


# ==========================================================
# Report zero-extinction bypass results
# ==========================================================

print()

print(

    "ZERO-EXTINCTION CCM89 BYPASS"

)

print(

    "=" * 70

)


print(

    "Passband"

)

print(

    near_infrared_passband.name

)


print()

print(

    "Passband wavelength range"

)

print(

    near_infrared_passband.minimum_wavelength,

    near_infrared_passband.maximum_wavelength

)


print()

print(

    "Synthetic magnitude"

)

print(

    zero_extinction_magnitudes[

        "Test J"

    ]

)


print()

print(

    "Magnitude is finite"

)

print(

    np.isfinite(

        zero_extinction_magnitudes[

            "Test J"

        ]

    )

)


print()

print(

    "Nonzero extinction outside optical domain rejected"

)

print(

    nonzero_extinction_rejected

)

# ==========================================================
# Validate rejection of negative color excess
# ==========================================================

negative_ebv_rejected = False


try:

    synthetic_binary_magnitudes(

        atmosphere_grid=grid,

        passbands={

            "Test J": (

                near_infrared_passband

            )

        },

        zero_points={

            "Test J": 1.0e-9

        },

        primary_temperature=6125.0,

        secondary_temperature=5875.0,

        primary_logg=4.25,

        secondary_logg=4.50,

        metallicity=0.0,

        primary_radius=1.20,

        secondary_radius=0.90,

        distance=25.0,

        ebv=-0.10,

        rv=3.10

    )


except ValueError:

    negative_ebv_rejected = True


assert negative_ebv_rejected


print()

print(

    "Negative E(B-V) rejected"

)

print(

    negative_ebv_rejected

)