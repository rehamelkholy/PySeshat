"""
Example 6: From your own data.

Points at two annotated templates (examples/templates/) showing
exactly which fields PySeshat's two main input paths need, so a
new user can substitute their own system without reading the
parser source directly:

    examples/templates/your_orbit.inp
        Tokovinin ORBIT-format visual/RV data file, as parsed by
        io.orbit_parser.parse_orbit_file() and fit in Example 1.

    examples/templates/your_sed_system.json
        grid-based unresolved-binary photometric SED system,
        in the schema validation/expected_results/*.json uses, as
        fit in Example 4.

This script does not do a real fit -- it just parses/loads both
templates and prints back what PySeshat read from them, to confirm
the templates are valid and to show what to expect when you swap
in your own numbers.

Run

    python examples/06_your_own_data_template.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def show_orbit_template():

    print("=" * 60)
    print("your_orbit.inp")
    print("=" * 60)

    parsed = parse_orbit_file(
        str(TEMPLATES_DIR / "your_orbit.inp")
    )

    elements = parsed["elements"]

    print(
        f"Object: {parsed['metadata'].get('object')}"
    )

    print()

    print("Starting elements read from the file:")

    for name in elements.PARAMETER_ORDER:

        fixed = (
            " (fixed)"
            if not elements.fit_flags.get(name, True)
            else ""
        )

        print(
            f"  {name:>6} = {getattr(elements, name):.4f}"
            f"{fixed}"
        )

    print()

    print(
        f"Visual observations: {len(parsed['visual_data'])} rows "
        "(columns: epoch, theta[deg], rho[arcsec], sigma[arcsec])"
    )

    print(
        f"RV observations: {len(parsed['rv_data'])} rows "
        "(columns: epoch, RV[km/s], sigma[km/s], component)"
    )

    print()

    print(
        "Next step: replace every value in "
        "examples/templates/your_orbit.inp with your own system's "
        "(see the comments in that file for exactly which columns "
        "matter), then adapt Example 1 "
        "(examples/01_visual_rv_binary_fit.py) to point at it."
    )


def show_sed_template():

    print()
    print("=" * 60)
    print("your_sed_system.json")
    print("=" * 60)

    with open(
        TEMPLATES_DIR / "your_sed_system.json"
    ) as f:

        template = json.load(f)

    required_fields = (
        "atmosphere_files",
        "measurements",
        "initial_parameters",
        "free_parameters",
        "literature_component_parameters",
        "primary_logg",
        "secondary_logg",
        "metallicity",
        "distance_pc",
    )

    missing = [
        field
        for field in required_fields
        if field not in template
    ]

    if missing:

        raise ValueError(
            f"Template is missing required fields: {missing}"
        )

    print(
        f"System: {template.get('system_name')}"
    )

    print(
        f"Free parameters: {template['free_parameters']}"
    )

    print(
        f"Measurements: {len(template['measurements'])} bands "
        f"({[m['passband'] for m in template['measurements']]})"
    )

    print(
        f"Atmosphere grid files: {template['atmosphere_files']}"
    )

    print()

    print(
        "All required fields present. Next step: replace every "
        "'_comment*' key's sibling values in "
        "examples/templates/your_sed_system.json with your "
        "own system's, delete the '_comment*' keys themselves "
        "(they're documentation only, not read by any PySeshat "
        "code), then adapt Example 4 "
        "(examples/04_binary_sed_fit.py) to load it."
    )


def main():

    show_orbit_template()

    show_sed_template()


if __name__ == "__main__":

    main()
