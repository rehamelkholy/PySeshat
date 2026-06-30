import numpy as np


def build_triple_bounds(
    elements
):
    """
    Construct optimizer bounds
    for fitted parameters only.
    """

    lower_full = np.array([

        # outer

        1e-6,
        -np.inf,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        # inner

        1e-6,
        -np.inf,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        # B

        0.0
    ])

    upper_full = np.array([

        np.inf,
        np.inf,
        0.99,
        np.inf,
        360.0,
        360.0,
        180.0,

        np.inf,
        np.inf,
        0.99,
        np.inf,
        360.0,
        360.0,
        180.0,

        np.inf
    ])

    mask = elements.fitted_mask()

    return (
        lower_full[mask],
        upper_full[mask]
    )