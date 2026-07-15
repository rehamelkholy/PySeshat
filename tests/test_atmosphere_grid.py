import numpy as np

from pyseshat.stellar.atmospheres import (
    AtmosphereGrid,
    AtmosphereModel
)


# ==========================================================
# Common wavelength grid
# ==========================================================

wavelength = np.array([

    4000.0,

    5000.0,

    6000.0,

    7000.0

])


# ==========================================================
# Construct atmosphere models
# ==========================================================

model_5500 = AtmosphereModel(

    wavelength=wavelength,

    flux=np.array([

        1.0e-8,

        1.5e-8,

        1.3e-8,

        0.9e-8

    ]),

    teff=5500.0,

    logg=4.5,

    metallicity=0.0

)


model_6000 = AtmosphereModel(

    wavelength=wavelength,

    flux=np.array([

        1.2e-8,

        1.8e-8,

        1.5e-8,

        1.0e-8

    ]),

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


model_6500 = AtmosphereModel(

    wavelength=wavelength,

    flux=np.array([

        1.5e-8,

        2.1e-8,

        1.7e-8,

        1.1e-8

    ]),

    teff=6500.0,

    logg=4.0,

    metallicity=-0.5

)


# ==========================================================
# Construct atmosphere grid
# ==========================================================

grid = AtmosphereGrid([

    model_5500,

    model_6000,

    model_6500

])


# ==========================================================
# Retrieve exact model
# ==========================================================

retrieved = grid.get_model(

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


# ==========================================================
# Display results
# ==========================================================

print(

    "ATMOSPHERE GRID"

)

print(

    "=" * 60

)

print(

    "Number of models"

)

print(

    len(
        grid
    )

)


print(

    "\nTEMPERATURES"

)

print(

    "=" * 60

)

print(

    grid.temperatures

)


print(

    "\nSURFACE GRAVITIES"

)

print(

    "=" * 60

)

print(

    grid.surface_gravities

)


print(

    "\nMETALLICITIES"

)

print(

    "=" * 60

)

print(

    grid.metallicities

)


print(

    "\nPARAMETER POINTS"

)

print(

    "=" * 60

)

for point in grid.parameter_points:

    print(
        point
    )


print(

    "\nEXACT MODEL RETRIEVAL"

)

print(

    "=" * 60

)

print(

    "Effective temperature"

)

print(

    retrieved.teff

)

print(

    "\nSurface gravity"

)

print(

    retrieved.logg

)

print(

    "\nMetallicity"

)

print(

    retrieved.metallicity

)

print(

    "\nFlux"

)

print(

    retrieved.flux

)


print(

    "\nGRID MEMBERSHIP"

)

print(

    "=" * 60

)

print(

    (
        6000.0,
        4.5,
        0.0
    )

    in grid

)

print(

    (
        5750.0,
        4.5,
        0.0
    )

    in grid

)