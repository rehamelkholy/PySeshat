from pathlib import Path

import numpy as np

from pyseshat.photometry.filters import (
    load_passband
)


# ==========================================================
# Repository validation-data directory
# ==========================================================

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
# Load Tycho BT
# ==========================================================

tycho_bt = load_passband(

    filename=(

        passband_directory

        /

        "tycho_bt.dat"

    ),

    name="Tycho BT"

)


# ==========================================================
# Load Tycho VT
# ==========================================================

tycho_vt = load_passband(

    filename=(

        passband_directory

        /

        "tycho_vt.dat"

    ),

    name="Tycho VT"

)


# ==========================================================
# Validate Tycho BT
# ==========================================================

assert tycho_bt.name == "Tycho BT"

assert np.all(

    np.diff(

        tycho_bt.wavelength

    ) > 0.0

)

assert np.isclose(

    tycho_bt.throughput[0],

    0.0

)

assert np.isclose(

    tycho_bt.throughput[-1],

    0.0

)


# ==========================================================
# Validate Tycho VT
# ==========================================================

assert tycho_vt.name == "Tycho VT"

# Maiz Apellaniz (2006) Table 1 curves are photon-counting.
assert tycho_bt.photon_counting and tycho_vt.photon_counting

assert np.all(

    np.diff(

        tycho_vt.wavelength

    ) > 0.0

)

assert np.isclose(

    tycho_vt.throughput[0],

    0.0

)

# Maiz Apellaniz (2006) Table 1 as printed ends at 6600 A with
# P = 0.028 (no zero end point).
assert np.isclose(

    tycho_vt.throughput[-1],

    0.028

)


# ==========================================================
# Report BT
# ==========================================================

print()

print(

    "TYCHO BT PASSBAND"

)

print(

    "=" * 70

)

print(

    "Number of samples"

)

print(

    tycho_bt.wavelength.size

)

print()

print(

    "Wavelength range"

)

print(

    tycho_bt.minimum_wavelength,

    tycho_bt.maximum_wavelength

)

print()

print(

    "Peak throughput"

)

print(

    np.max(

        tycho_bt.throughput

    )

)


# ==========================================================
# Report VT
# ==========================================================

print()

print(

    "TYCHO VT PASSBAND"

)

print(

    "=" * 70

)

print(

    "Number of samples"

)

print(

    tycho_vt.wavelength.size

)

print()

print(

    "Wavelength range"

)

print(

    tycho_vt.minimum_wavelength,

    tycho_vt.maximum_wavelength

)

print()

print(

    "Peak throughput"

)

print(

    np.max(

        tycho_vt.throughput

    )

)


# ==========================================================
# Validation
# ==========================================================

print()

print(

    "VALIDATION"

)

print(

    "=" * 70

)

print(

    "BT wavelengths increasing"

)

print(

    np.all(

        np.diff(

            tycho_bt.wavelength

        ) > 0.0

    )

)

print()

print(

    "VT wavelengths increasing"

)

print(

    np.all(

        np.diff(

            tycho_vt.wavelength

        ) > 0.0

    )

)

print()

print(

    "BT boundaries are zero"

)

print(

    np.isclose(

        tycho_bt.throughput[0],

        0.0

    )

    and

    np.isclose(

        tycho_bt.throughput[-1],

        0.0

    )

)

print()

print(

    "VT boundaries: 0 at start, 0.028 at end (as published)"

)

print(

    np.isclose(

        tycho_vt.throughput[0],

        0.0

    )

    and

    np.isclose(

        tycho_vt.throughput[-1],

        0.028

    )

)