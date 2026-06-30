from pathlib import Path

filename = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

parameters = {}

with open(
    filename,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        parts = line.split()

        if len(parts) < 2:
            continue

        try:

            value = float(parts[1])

            parameters[parts[0]] = value

        except ValueError:

            pass

print()
print(sorted(parameters.keys()))