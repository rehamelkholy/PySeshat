from pathlib import Path

# ==========================================================
# Input file
# ==========================================================

INPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

# ==========================================================
# Parse header
# ==========================================================

parameters = {}

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        parts = line.split()

        if len(parts) < 2:
            continue

        key = parts[0]

        try:

            value = float(parts[1])

            parameters[key] = value

        except ValueError:

            continue

# ==========================================================
# Output
# ==========================================================

print()
print("OUTER ORBIT")
print("=" * 60)

for key in [
    "P",
    "T",
    "e",
    "a",
    "W",
    "w",
    "i"
]:

    print(
        f"{key:<6} = {parameters.get(key)}"
    )

print()
print("INNER ORBIT")
print("=" * 60)

for key in [
    "P2",
    "T2",
    "e2",
    "a2",
    "W2",
    "w2",
    "i2"
]:

    print(
        f"{key:<6} = {parameters.get(key)}"
    )

print()
print("SCALE FACTOR")
print("=" * 60)

print(
    f"B = {parameters.get('B')}"
)