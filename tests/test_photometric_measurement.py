import numpy as np

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)


# ==========================================================
# Construct Johnson B and V observations
# ==========================================================

johnson_b = PhotometricMeasurement(

    passband="Johnson B",

    magnitude=7.130442389055,

    uncertainty=0.020

)


johnson_v = PhotometricMeasurement(

    passband="  Johnson V  ",

    magnitude=np.float64(

        6.296584811208877

    ),

    uncertainty=np.float64(

        0.015

    )

)


# ==========================================================
# Validate stored values
# ==========================================================

assert johnson_b.passband == "Johnson B"


assert isinstance(

    johnson_b.magnitude,

    float

)


assert isinstance(

    johnson_b.uncertainty,

    float

)


assert np.isclose(

    johnson_b.magnitude,

    7.130442389055

)


assert np.isclose(

    johnson_b.uncertainty,

    0.020

)


# ==========================================================
# Validate normalization
# ==========================================================

assert johnson_v.passband == "Johnson V"


assert isinstance(

    johnson_v.magnitude,

    float

)


assert isinstance(

    johnson_v.uncertainty,

    float

)


assert np.isclose(

    johnson_v.magnitude,

    6.296584811208877

)


assert np.isclose(

    johnson_v.uncertainty,

    0.015

)


# ==========================================================
# Validate rejection of invalid uncertainty
# ==========================================================

invalid_uncertainty_rejected = False


try:

    PhotometricMeasurement(

        passband="Johnson V",

        magnitude=6.30,

        uncertainty=0.0

    )


except ValueError:

    invalid_uncertainty_rejected = True


assert invalid_uncertainty_rejected


# ==========================================================
# Validate rejection of an empty passband name
# ==========================================================

empty_passband_rejected = False


try:

    PhotometricMeasurement(

        passband="   ",

        magnitude=6.30,

        uncertainty=0.02

    )


except ValueError:

    empty_passband_rejected = True


assert empty_passband_rejected


# ==========================================================
# Report results
# ==========================================================

print()

print(

    "PHOTOMETRIC MEASUREMENTS"

)

print(

    "=" * 70

)


print(

    "Johnson B"

)

print(

    "Passband"

)

print(

    johnson_b.passband

)


print()

print(

    "Magnitude"

)

print(

    johnson_b.magnitude

)


print()

print(

    "Uncertainty"

)

print(

    johnson_b.uncertainty

)


print()

print(

    "Johnson V"

)

print(

    "Passband"

)

print(

    johnson_v.passband

)


print()

print(

    "Magnitude"

)

print(

    johnson_v.magnitude

)


print()

print(

    "Uncertainty"

)

print(

    johnson_v.uncertainty

)


print()

print(

    "VALIDATION"

)

print(

    "=" * 70

)


print(

    "Whitespace removed from passband name"

)

print(

    johnson_v.passband

    ==

    "Johnson V"

)


print()

print(

    "Zero uncertainty rejected"

)

print(

    invalid_uncertainty_rejected

)


print()

print(

    "Empty passband rejected"

)

print(

    empty_passband_rejected

)