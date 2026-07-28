import numpy as np

from pyseshat.inference.likelihood import (
    rv_residuals as legacy_rv,
)

from pyseshat.inference.terms.radial_velocity import (
    rv_residuals as modular_rv,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_modular_rv_residuals_matches_legacy():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
        K1=20.0,
        K2=30.0,
        V0=5.0,
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

    rv_data = np.array([
        [0.0, 12.0, 1.0, 1],
        [2.0, 15.0, 1.0, 2],
    ])

    params = elements.fitted_parameters()

    np.testing.assert_allclose(
        legacy_rv(
            params,
            elements,
            rv_data,
        ),
        modular_rv(
            params,
            elements,
            rv_data,
        ),
    )