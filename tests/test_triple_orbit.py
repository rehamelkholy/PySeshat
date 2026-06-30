import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple.triple_orbit import (
    compute_triple_orbit
)

outer = OrbitalElements(
    P=50,
    T0=2000,
    e=0.3,
    a=1.0,
    Omega=100,
    omega=50,
    i=60
)

inner = OrbitalElements(
    P=5,
    T0=2001,
    e=0.2,
    a=0.2,
    Omega=20,
    omega=120,
    i=70
)

triple = TripleOrbitalElements(
    outer=outer,
    inner=inner
)

epochs = np.linspace(
    2000,
    2020,
    100
)

orbit = compute_triple_orbit(
    epochs,
    triple
)

print()

print(
    orbit.keys()
)

print()

print(
    orbit["X"].shape
)

print(
    orbit["Y"].shape
)

print(
    orbit["rho"].shape
)

print(
    orbit["theta"].shape
)