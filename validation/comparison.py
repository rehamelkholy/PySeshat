import pandas as pd

from metrics import (
    parameter_relative_error
)

def comparison_table(result):

    expected = result["expected"]

    fitted = result[
        "fit_result"
    ].elements

    expected_params = expected[
        "initial_parameters"
    ]

    rows = []

    for key, expected_value in (
        expected_params.items()
    ):

        fitted_value = getattr(
            fitted,
            key
        )

        rel_err = parameter_relative_error(
            key,
            expected_value,
            fitted_value
        )

        rows.append({

            "Parameter": key,

            "Expected":
                expected_value,

            "PySeshat":
                fitted_value,

            "Relative Error":
                rel_err
        })

    return pd.DataFrame(rows)