from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.inference.node_disambiguation import (
    node_flip
)

elements = OrbitalElements(
    P=10.0,
    T0=2000.0,
    e=0.2,
    a=1.0,
    Omega=100.0,
    omega=50.0,
    i=60.0
)

flipped = node_flip(
    elements
)

print()

print(
    "Original Omega:",
    elements.Omega
)

print(
    "Flipped Omega:",
    flipped.Omega
)

print()

print(
    "Original omega:",
    elements.omega
)

print(
    "Flipped omega:",
    flipped.omega
)