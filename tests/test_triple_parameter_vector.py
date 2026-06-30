from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    triple_to_array,
    array_to_triple
)

# ==========================================================
# Load benchmark
# ==========================================================

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

triple = result["elements"]

# ==========================================================
# Convert
# ==========================================================

p = triple_to_array(
    triple
)

triple2 = array_to_triple(
    p,
    triple
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Parameter vector shape:"
)

print(
    p.shape
)

print()

print(
    "Original outer:"
)

print(
    triple.outer
)

print()

print(
    "Recovered outer:"
)

print(
    triple2.outer
)

print()

print(
    "Original inner:"
)

print(
    triple.inner
)

print()

print(
    "Recovered inner:"
)

print(
    triple2.inner
)

print()

print(
    "Original B:"
)

print(
    triple.B
)

print()

print(
    "Recovered B:"
)

print(
    triple2.B
)

print()

print(
    "Maximum absolute difference:"
)

print(
    np.max(
        np.abs(
            p
            -
            triple_to_array(
                triple2
            )
        )
    )
)