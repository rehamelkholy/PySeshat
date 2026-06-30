from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)

filename = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

result = parse_triple_file(
    filename
)

observations = result["observations"]

print()
print("a2        B         a2*B")

for a2 in np.linspace(
    0.08,
    0.16,
    9
):

    elements = result["elements"]

    elements.inner.a = a2

    elements.B = (
        0.38467 / a2
    )

    elements.fit_flags["a2"] = False
    elements.fit_flags["B"] = False

    fit = fit_triple_orbit(
        elements,
        observations
    )

    print(
        f"{elements.inner.a:.4f}   "
        f"{elements.B:.4f}   "
        f"{elements.inner.a * elements.B:.6f}"
    )