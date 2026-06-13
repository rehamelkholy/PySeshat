from metrics import (
    parameter_relative_error
)

print()
print("Omega test")
print()

print(
    parameter_relative_error(
        "Omega",
        183.9,
        3.9
    )
)

print()

print("omega test")
print()

print(
    parameter_relative_error(
        "omega",
        191.0,
        11.0
    )
)