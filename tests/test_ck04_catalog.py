from pathlib import Path

from pyseshat.stellar.atmospheres import (
    load_ck04_catalog
)


# ==========================================================
# CK04 catalog
# ==========================================================

filename = Path(

    "validation"

) / (

    "atmosphere_models"

) / (

    "catalog.fits"

)


# ==========================================================
# Load catalog
# ==========================================================

catalog = load_ck04_catalog(

    filename

)


# ==========================================================
# Catalog summary
# ==========================================================

temperatures = sorted({

    entry.teff

    for entry in catalog

})


metallicities = sorted({

    entry.metallicity

    for entry in catalog

})


surface_gravities = sorted({

    entry.logg

    for entry in catalog

})


print()

print(

    "CK04 CATALOG"

)

print(

    "=" * 70

)

print(

    "Number of entries"

)

print(

    len(
        catalog
    )

)


print()

print(

    "NUMBER OF TEMPERATURES"

)

print(

    "=" * 70

)

print(

    len(
        temperatures
    )

)


print()

print(

    "TEMPERATURE RANGE"

)

print(

    "=" * 70

)

print(

    temperatures[
        0
    ],

    temperatures[
        -1
    ]

)


print()

print(

    "METALLICITIES"

)

print(

    "=" * 70

)

print(

    metallicities

)


print()

print(

    "SURFACE GRAVITIES"

)

print(

    "=" * 70

)

print(

    surface_gravities

)


# ==========================================================
# Inspect known entry
# ==========================================================

target = next(

    entry

    for entry in catalog

    if (

        entry.teff == 10000.0

        and

        entry.metallicity == -0.5

        and

        entry.logg == 2.0

    )

)


print()

print(

    "KNOWN CATALOG ENTRY"

)

print(

    "=" * 70

)

print(

    "Parameter point"

)

print(

    target.parameter_point

)

print()

print(

    "Relative FITS path"

)

print(

    target.relative_path

)

print()

print(

    "FITS column"

)

print(

    target.column_name

)