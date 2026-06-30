from pathlib import Path

filename = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

with open(
    filename,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line_number, line in enumerate(f, start=1):

        if "a2" in line:

            print()
            print(f"Line {line_number}")
            print(repr(line))