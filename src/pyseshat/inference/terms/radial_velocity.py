import numpy as np

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.orbit.rv_model import (
    compute_rv_model,
)


def rv_residuals(
    fitted_values,
    elements: OrbitalElements,
    rv_data,
):
    """
    Compute weighted radial-velocity residuals.
    """

    if rv_data is None or len(rv_data) == 0:
        return np.array([])

    updated = update_elements(
        elements,
        fitted_values,
    )

    rv_obs = rv_data[:, 1]
    sigma = rv_data[:, 2]

    rv_model = compute_rv_model(
        rv_data,
        updated,
    )

    return (rv_obs - rv_model) / sigma