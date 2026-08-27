from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.ephemeris.ephemeris import (
    generate_combined_ephemeris
)

# This file has no `def test_...` functions and no assertions --
# it is a demo script, not a real pytest test (pytest reports 0
# tests collected from it, but still executes this module-level
# code on collection). It used to write fin379_ephemeris.csv/.txt
# with bare relative paths, which landed wherever pytest's cwd
# happened to be (the repo root, in normal use) rather than
# anywhere clearly marked as generated output -- see
# docs/project_decisions.md, "FIN379 files stranded at repo root"
# (2026-08-27). Anchored to this file's own location instead, so
# the output lands in the same place regardless of invocation cwd.

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "artifacts"
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Load benchmark orbit
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

# ==========================================================
# Generate ephemeris
# ==========================================================

epochs = np.arange(
    2025,
    2031,
    1
)

eph = generate_combined_ephemeris(
    elements,
    epochs
)

# ==========================================================
# DataFrame
# ==========================================================

print()

print(
    eph.to_dataframe()
)

# ==========================================================
# Text table
# ==========================================================

print()

print(
    eph.to_text()
)

# ==========================================================
# Save files
# ==========================================================

eph.save_csv(
    str(ARTIFACTS_DIR / "fin379_ephemeris.csv")
)

eph.save_text(
    str(ARTIFACTS_DIR / "fin379_ephemeris.txt")
)

print()

print(
    "Files written successfully."
)