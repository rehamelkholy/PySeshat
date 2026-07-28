import numpy as np

from pyseshat.inference.likelihood import (
    combined_residuals,
)

from pyseshat.inference.terms.joint import (
    joint_residuals,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_joint_residuals_matches_combined():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
    )

    elements.fit_flags.update(
        P=True,
        T0=False,
        e=False,
        a=False,
        Omega=False,
        omega=False,
        i=False,
        K1=False,
        K2=False,
        V0=False,
    )

    t = np.array([0.0, 2.5])

    x = np.array([0.1, 0.2])
    y = np.array([0.3, 0.4])

    sx = np.full(2, 0.01)
    sy = np.full(2, 0.01)

    params = elements.fitted_parameters()

    np.testing.assert_allclose(

        combined_residuals(
            params,
            elements,
            t,
            x,
            y,
            sx,
            sy,
        ),

        joint_residuals(
            params,
            elements,
            t,
            x,
            y,
            sx,
            sy,
        ),
    )