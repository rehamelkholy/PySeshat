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
# Load Tycho passbands
# ==========================================================

passbands = {

    "Tycho BT": load_passband(

        filename=(

            passband_directory

            /

            "tycho_bt.dat"

        ),

        name="Tycho BT"

    ),

    "Tycho VT": load_passband(

        filename=(

            passband_directory

            /

            "tycho_vt.dat"

        ),

        name="Tycho VT"

    )

}


# ==========================================================
# Official SVO Vega zero points
# ==========================================================

zero_points = {

    "Tycho BT": 6.50910e-9,

    "Tycho VT": 3.99504e-9

}


# ==========================================================
# Calculate synthetic Tycho magnitudes
# ==========================================================

magnitudes = (

    synthetic_binary_magnitudes(

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

)


# ==========================================================
# Validate returned passbands
# ==========================================================

assert set(

    magnitudes

) == {

    "Tycho BT",

    "Tycho VT"

}


assert isinstance(

    magnitudes[

        "Tycho BT"

    ],

    float

)


assert isinstance(

    magnitudes[

        "Tycho VT"

    ],

    float

)


assert np.isfinite(

    magnitudes[

        "Tycho BT"

    ]

)


assert np.isfinite(

    magnitudes[

        "Tycho VT"

    ]

)


# ==========================================================
# Physical sanity checks
# ==========================================================

assert magnitudes[

    "Tycho BT"

] > 0.0


assert magnitudes[

    "Tycho VT"

] > 0.0


assert (

    magnitudes[

        "Tycho BT"

    ]

    >

    magnitudes[

        "Tycho VT"

    ]

)


# ==========================================================
# Compute Tycho colour
# ==========================================================

bt_minus_vt = (

    magnitudes[

        "Tycho BT"

    ]

    -

    magnitudes[

        "Tycho VT"

    ]

)


assert np.isfinite(

    bt_minus_vt

)


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "CK04 BINARY TYCHO PHOTOMETRY"

)

print(

    "=" * 70

)


print(

    "Photometric system"

)

print(

    "Tycho Vega"

)


print()

print(

    "Tycho BT magnitude"

)

print(

    magnitudes[

        "Tycho BT"

    ]

)


print()

print(

    "Tycho VT magnitude"

)

print(

    magnitudes[

        "Tycho VT"

    ]

)


print()

print(

    "Tycho BT-VT colour"

)

print(

    bt_minus_vt

)


print()

print(

    "VALIDATION"

)

print(

    "=" * 70

)


print(

    "Finite BT magnitude"

)

print(

    np.isfinite(

        magnitudes[

            "Tycho BT"

        ]

    )

)


print()

print(

    "Finite VT magnitude"

)

print(

    np.isfinite(

        magnitudes[

            "Tycho VT"

        ]

    )

)


print()

print(

    "BT > VT"

)

print(

    magnitudes[

        "Tycho BT"

    ]

    >

    magnitudes[

        "Tycho VT"

    ]

)


print()

print(

    "Finite BT-VT colour"

)

print(

    np.isfinite(

        bt_minus_vt

    )

)