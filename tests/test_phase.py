from pyseshat.orbit.phase import (
    orbital_phase
)

phase = orbital_phase(
    [2000.0, 2000.25, 2000.50, 2000.75],
    P=1.0,
    T0=2000.0
)

print(phase)