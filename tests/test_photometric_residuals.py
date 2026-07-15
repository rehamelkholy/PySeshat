import numpy as np

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)

from pyseshat.photometry.residuals import (
    photometric_chi_square,
    photometric_residual
)


# ==========================================================
# Construct observed photometric measurements
# ==========================================================

johnson_b = PhotometricMeasurement(

    passband="Johnson B",

    magnitude=7.10,

    uncertainty=0.02

)


johnson_v = PhotometricMeasurement(

    passband="Johnson V",

    magnitude=6.30,

    uncertainty=0.01

)


# ==========================================================
# Define model magnitudes
# ==========================================================

model_magnitudes = {

    "Johnson B": 7.14,

    "Johnson V": 6.28

}


# ==========================================================
# Calculate individual standardized residuals
# ==========================================================

johnson_b_residual = photometric_residual(

    model_magnitude=(

        model_magnitudes[

            "Johnson B"

        ]

    ),

    measurement=johnson_b

)


johnson_v_residual = photometric_residual(

    model_magnitude=(

        model_magnitudes[

            "Johnson V"

        ]

    ),

    measurement=johnson_v

)


# ==========================================================
# Expected standardized residuals
# ==========================================================

expected_b_residual = (

    (

        7.14

        -

        7.10

    )

    /

    0.02

)


expected_v_residual = (

    (

        6.28

        -

        6.30

    )

    /

    0.01

)


# ==========================================================
# Calculate photometric chi-square
# ==========================================================

chi_square = photometric_chi_square(

    model_magnitudes=(

        model_magnitudes

    ),

    measurements=[

        johnson_b,

        johnson_v

    ]

)


expected_chi_square = (

    expected_b_residual**2

    +

    expected_v_residual**2

)


# ==========================================================
# Validate individual residuals
# ==========================================================

assert isinstance(

    johnson_b_residual,

    float

)


assert isinstance(

    johnson_v_residual,

    float

)


assert np.isclose(

    johnson_b_residual,

    expected_b_residual

)


assert np.isclose(

    johnson_v_residual,

    expected_v_residual

)


# ==========================================================
# Validate residual signs
# ==========================================================

assert johnson_b_residual > 0.0


assert johnson_v_residual < 0.0


# ==========================================================
# Validate chi-square
# ==========================================================

assert isinstance(

    chi_square,

    float

)


assert np.isclose(

    chi_square,

    expected_chi_square

)


assert np.isclose(

    chi_square,

    8.0

)


# ==========================================================
# Validate an exact model
# ==========================================================

exact_model = {

    "Johnson B": (

        johnson_b.magnitude

    ),

    "Johnson V": (

        johnson_v.magnitude

    )

}


exact_chi_square = photometric_chi_square(

    model_magnitudes=exact_model,

    measurements=[

        johnson_b,

        johnson_v

    ]

)


assert np.isclose(

    exact_chi_square,

    0.0

)


# ==========================================================
# Validate missing-passband rejection
# ==========================================================

missing_passband_rejected = False


try:

    photometric_chi_square(

        model_magnitudes={

            "Johnson B": 7.14

        },

        measurements=[

            johnson_b,

            johnson_v

        ]

    )


except ValueError:

    missing_passband_rejected = True


assert missing_passband_rejected


# ==========================================================
# Validate duplicate-measurement rejection
# ==========================================================

duplicate_measurement_rejected = False


try:

    photometric_chi_square(

        model_magnitudes=(

            model_magnitudes

        ),

        measurements=[

            johnson_b,

            johnson_b

        ]

    )


except ValueError:

    duplicate_measurement_rejected = True


assert duplicate_measurement_rejected


# ==========================================================
# Report individual residuals
# ==========================================================

print()

print(

    "PHOTOMETRIC RESIDUALS"

)

print(

    "=" * 70

)


print(

    "Johnson B residual"

)

print(

    johnson_b_residual

)


print()

print(

    "Expected Johnson B residual"

)

print(

    expected_b_residual

)


print()

print(

    "Johnson V residual"

)

print(

    johnson_v_residual

)


print()

print(

    "Expected Johnson V residual"

)

print(

    expected_v_residual

)


# ==========================================================
# Report chi-square
# ==========================================================

print()

print(

    "PHOTOMETRIC CHI-SQUARE"

)

print(

    "=" * 70

)


print(

    "Computed chi-square"

)

print(

    chi_square

)


print()

print(

    "Expected chi-square"

)

print(

    expected_chi_square

)


print()

print(

    "Exact-model chi-square"

)

print(

    exact_chi_square

)


# ==========================================================
# Report validation behavior
# ==========================================================

print()

print(

    "VALIDATION"

)

print(

    "=" * 70

)


print(

    "Positive B residual"

)

print(

    johnson_b_residual > 0.0

)


print()

print(

    "Negative V residual"

)

print(

    johnson_v_residual < 0.0

)


print()

print(

    "Missing passband rejected"

)

print(

    missing_passband_rejected

)


print()

print(

    "Duplicate measurement rejected"

)

print(

    duplicate_measurement_rejected

)