import numpy as np

from pyseshat.stellar.atmospheres import (
    AtmosphereModel
)


# ==========================================================
# Construct a valid atmosphere model
# ==========================================================

wavelength = np.array([

    4000.0,

    5000.0,

    6000.0,

    7000.0

])

flux = np.array([

    1.2e-8,

    1.8e-8,

    1.5e-8,

    1.0e-8

])


model = AtmosphereModel(

    wavelength=wavelength,

    flux=flux,

    teff=6000.0,

    logg=4.5,

    metallicity=0.0

)


# ==========================================================
# Display results
# ==========================================================

print(

    "ATMOSPHERE PARAMETERS"

)

print(

    "=" * 60

)

print(

    "Effective temperature"

)

print(

    model.teff

)

print(

    "\nSurface gravity"

)

print(

    model.logg

)

print(

    "\nMetallicity"

)

print(

    model.metallicity

)


print(

    "\nSPECTRUM"

)

print(

    "=" * 60

)

print(

    "Number of wavelength samples"

)

print(

    len(
        model.wavelength
    )

)

print(

    "\nWavelength range"

)

print(

    model.wavelength[
        0
    ],

    model.wavelength[
        -1
    ]

)

print(

    "\nMinimum flux"

)

print(

    np.min(
        model.flux
    )

)

print(

    "\nMaximum flux"

)

print(

    np.max(
        model.flux
    )

)