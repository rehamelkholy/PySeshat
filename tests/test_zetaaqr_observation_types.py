from pathlib import Path
from collections import Counter

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
# Count observation types
# ==========================================================

counter = Counter()

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        parts = line.split()

        if len(parts) < 5:
            continue

        tag = parts[4]

        if tag in [
            "I1",
            "I2"
        ]:

            counter[tag] += 1

# ==========================================================
# Output
# ==========================================================

print()
print("OBSERVATION COUNTS")
print("=" * 60)

for key in sorted(counter):

    print(
        f"{key:<4} : {counter[key]}"
    )

print()
print(
    f"Total observations = {sum(counter.values())}"
)