from pathlib import Path

from pyseshat.stellar.atmospheres import (
    load_ck04_catalog
)


# ==========================================================
# Load CK04 catalog
# ==========================================================

catalog_path = (

    Path(
        "validation"
    )

    /

    "atmosphere_models"

    /

    "catalog.fits"

)


catalog = load_ck04_catalog(

    catalog_path

)


# ==========================================================
# Select solar-metallicity models
# ==========================================================

solar_entries = [

    entry

    for entry in catalog

    if entry.metallicity == 0.0

]


# ==========================================================
# Available solar-metallicity temperatures
# ==========================================================

temperatures = sorted({

    entry.teff

    for entry in solar_entries

})


print()

print(

    "SOLAR-METALLICITY CK04 GRID"

)

print(

    "=" * 70

)

print(

    "Number of atmosphere models"

)

print(

    len(
        solar_entries
    )

)


print()

print(

    "Number of temperatures"

)

print(

    len(
        temperatures
    )

)


print()

print(

    "Temperature range"

)

print(

    temperatures[
        0
    ],

    temperatures[
        -1
    ]

)


# ==========================================================
# Inspect temperatures around a solar-type star
# ==========================================================

minimum_temperature = 5000.0

maximum_temperature = 7000.0


selected_temperatures = [

    temperature

    for temperature in temperatures

    if (

        minimum_temperature

        <=

        temperature

        <=

        maximum_temperature

    )

]


print()

print(

    "TEMPERATURES FROM 5000 TO 7000 K"

)

print(

    "=" * 70

)

print(

    selected_temperatures

)


# ==========================================================
# Determine the corresponding FITS files
# ==========================================================

selected_entries = [

    entry

    for entry in solar_entries

    if entry.teff in selected_temperatures

]


relative_paths = sorted({

    entry.relative_path

    for entry in selected_entries

})


print()

print(

    "REQUIRED SOLAR-METALLICITY FITS FILES"

)

print(

    "=" * 70

)

for relative_path in relative_paths:

    print(

        relative_path

    )


# ==========================================================
# Inspect available gravities at each temperature
# ==========================================================

print()

print(

    "AVAILABLE SURFACE GRAVITIES"

)

print(

    "=" * 70

)


for temperature in selected_temperatures:

    gravities = sorted({

        entry.logg

        for entry in selected_entries

        if entry.teff == temperature

    })


    print(

        f"{temperature:7.1f} K : "

        f"{gravities}"

    )