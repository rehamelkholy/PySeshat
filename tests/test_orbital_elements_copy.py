from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

elements = OrbitalElements(
    P=10,
    T0=2000,
    e=0.3,
    a=1.2,
    Omega=100,
    omega=50,
    i=70
)

copied = elements.copy()

print()
print("Original:")
print(elements)

print()
print("Copy:")
print(copied)

print()
print("Same object:")
print(elements is copied)

print()
print("Same fit_flags object:")
print(
    elements.fit_flags
    is
    copied.fit_flags
)