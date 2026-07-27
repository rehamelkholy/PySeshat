import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)
from pyseshat.triple.triple_elements import (
    TripleOrbitalElements,
)
from pyseshat.triple.triple_orbit import (
    compute_triple_orbit,
)


def make_triple():

    outer = OrbitalElements(
        P=50,
        T0=2000,
        e=0.3,
        a=1.0,
        Omega=100,
        omega=50,
        i=60,
    )

    inner = OrbitalElements(
        P=5,
        T0=2001,
        e=0.2,
        a=0.2,
        Omega=20,
        omega=120,
        i=70,
    )

    return TripleOrbitalElements(
        outer=outer,
        inner=inner,
        B=0.5,
    )


def test_compute_triple_orbit():

    epochs = np.linspace(
        2000,
        2020,
        100,
    )

    orbit = compute_triple_orbit(
        epochs,
        make_triple(),
    )

    assert {
        "X",
        "Y",
        "rho",
        "theta",
    }.issubset(orbit)

    assert orbit["X"].shape == (100,)
    assert orbit["Y"].shape == (100,)
    assert orbit["rho"].shape == (100,)
    assert orbit["theta"].shape == (100,)
    assert orbit["outer"]["X"].shape == (100,)
    assert orbit["inner"]["X"].shape == (100,)