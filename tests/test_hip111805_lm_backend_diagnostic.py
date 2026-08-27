import numpy as np

from scipy.optimize import least_squares

from test_hip111805_orbit3 import load_hip111805

from pyseshat.triple.orbit3_fit import (
    orbit3_fit_flag_residuals,
)

from pyseshat.triple.triple_fit import (
    orbit3_to_array,
)


def test_hip111805_orbit3_lm_jacobian_interface_diagnostic():

    result = load_hip111805()

    elements = result["elements"]
    observations = result["observations"]
    rv_observations = result["rv_observations"]

    mask = np.asarray(
        elements.fitted_mask(),
        dtype=bool,
    )

    p0 = (
        orbit3_to_array(
            elements
        )
    )[mask]

    args = (
        elements,
        observations,
        rv_observations,
    )

    def residual_function(p):

        return orbit3_fit_flag_residuals(
            p,
            *args,
        )

    print()
    print("=" * 110)
    print("HIP111805 ORBIT3 LM JACOBIAN INTERFACE DIAGNOSTIC")
    print("=" * 110)

    r0 = residual_function(p0)

    print()
    print(
        f"Initial chi2 = {np.sum(r0**2):.12f}"
    )

    # ------------------------------------------------------
    # SciPy local numerical Jacobian
    # ------------------------------------------------------

    numerical = least_squares(

        residual_function,

        p0,

        method="lm",

        jac="2-point",

        loss="linear",

        max_nfev=30,

    )

    r_numeric = residual_function(
        numerical.x
    )

    print()
    print("LOCAL NUMERICAL JACOBIAN")
    print("-" * 110)

    print(
        f"success : {numerical.success}"
    )

    print(
        f"status  : {numerical.status}"
    )

    print(
        f"message : {numerical.message}"
    )

    print(
        f"nfev    : {numerical.nfev}"
    )

    print(
        f"chi2    : {np.sum(r_numeric**2):.12f}"
    )

    print(
        "max |delta parameter| = "
        f"{np.max(np.abs(numerical.x - p0)):.12e}"
    )

    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    print()
    print("RESULT")
    print("-" * 110)

    assert numerical.success

    assert (
        np.sum(r_numeric**2)
        <
        np.sum(r0**2)
    )