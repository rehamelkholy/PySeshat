from pathlib import Path

from benchmark_runner import (
    run_benchmark
)

config = {

    "data_file": (
        Path(
            "validation/benchmark_systems/FIN379/FIN379.inp"
        )
    ),

    "expected_file": (
        Path(
            "validation/expected_results/FIN379.json"
        )
    )
}

result = run_benchmark(
    "FIN379",
    config
)

print()

print("=== STATISTICS ===")

print(
    result["statistics"]
)

print()

print("=== PARAMETER ERRORS ===")

for name, values in (
    result["parameter_errors"].items()
):

    print()

    print(name)

    print(values)