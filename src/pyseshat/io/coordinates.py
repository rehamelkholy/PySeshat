def parse_ra(ra_string):
    """
    Convert RA string to decimal degrees.

    Examples
    --------
    19.4040
    19:24:14.4
    19 24 14.4
    """

    ra_string = str(ra_string).strip()

    if ":" in ra_string:

        h, m, s = map(
            float,
            ra_string.split(":")
        )

        hours = (
            h
            + m / 60.0
            + s / 3600.0
        )

    elif " " in ra_string:

        h, m, s = map(
            float,
            ra_string.split()
        )

        hours = (
            h
            + m / 60.0
            + s / 3600.0
        )

    else:

        hours = float(
            ra_string
        )

    return hours * 15.0

def parse_dec(dec_string):
    """
    Convert Dec string to decimal degrees.

    Examples
    --------
    +15.25
    -23:14:12
    +15 24 36
    """

    dec_string = str(
        dec_string
    ).strip()

    if ":" in dec_string:

        parts = dec_string.split(":")

    elif " " in dec_string:

        parts = dec_string.split()

    else:

        return float(
            dec_string
        )

    d = float(parts[0])

    m = float(parts[1])

    s = float(parts[2])

    sign = 1.0

    if d < 0:

        sign = -1.0

    return sign * (

        abs(d)

        + m / 60.0

        + s / 3600.0
    )