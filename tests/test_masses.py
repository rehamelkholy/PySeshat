from pyseshat.analysis.masses import (
    compute_total_mass,
    total_mass_from_elements,
    compute_mass_function,
    component_masses_from_elements,
    minimum_secondary_mass
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

# ------------------------------------------------------
# Analytic test
# ------------------------------------------------------

mass = compute_total_mass(

    period_years=10.0,

    semimajor_axis_arcsec=0.5,

    parallax_mas=50.0
)

print()

print(
    "Analytic mass:",
    mass
)

# ------------------------------------------------------
# Elements wrapper test
# ------------------------------------------------------

elements = OrbitalElements(

    P=10.0,

    T0=2000.0,

    e=0.0,

    a=0.5,

    Omega=0.0,

    omega=0.0,

    i=90.0
)

mass2 = total_mass_from_elements(

    elements,

    parallax_mas=50.0
)

print()

print(
    "Wrapper mass:",
    mass2
)

fm = compute_mass_function(

    period_years=1.0,

    eccentricity=0.0,

    K1=30.0
)

print()

print(
    "Mass function:",
    fm
)

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

masses = component_masses_from_elements(
    parsed["elements"]
)

print()

for k, v in masses.items():

    print(
        k,
        v
    )

m2 = minimum_secondary_mass(

    mass_function=0.01,

    primary_mass=1.0
)

print()

print(
    "Minimum companion mass:",
    m2
)