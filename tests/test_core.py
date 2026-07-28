import numpy as np

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_update_elements_updates_only_fitted_parameters():

    elements = OrbitalElements(
        P=10.0,
        T0=100.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
    )

    elements.fit_flags.update(
        P=True,
        T0=False,
        e=True,
        a=False,
        Omega=False,
        omega=False,
        i=False,
        K1=False,
        K2=False,
        V0=False,
    )

    updated = update_elements(
        elements,
        np.array([20.0, 0.5]),
    )

    assert updated.P == 20.0
    assert updated.e == 0.5

    assert updated.T0 == elements.T0
    assert updated.a == elements.a
    assert updated.Omega == elements.Omega