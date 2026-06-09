from pyseshat.io.coordinates import (
    parse_ra,
    parse_dec
)

print()

print(
    parse_ra("12:00:00")
)

print(
    parse_ra("19.4040")
)

print(
    parse_dec("+30:00:00")
)

print(
    parse_dec("-30:00:00")
)