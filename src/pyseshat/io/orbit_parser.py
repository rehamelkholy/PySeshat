import numpy as np

from pyseshat.orbit.orbital_elements import OrbitalElements

from pyseshat.io.coordinates import (
    parse_ra,
    parse_dec
)


# ==========================================================
# Time conversion utilities
# ==========================================================

def jd_to_decimal_year(jd):
    """
    Convert a reduced-Julian-Date epoch (JD - 2400000, the
    convention used by RV columns in this codebase's ``.inp``
    file format) to a decimal-year approximation.

    Uses a fixed 365.25-day Julian year (no leap-second/calendar
    correction) -- adequate for the observational timescales
    (years to decades) these benchmark systems span, not intended
    as a general-purpose calendar-accurate JD converter.

    Parameters
    ----------
    jd : float or array_like
        Reduced Julian date (JD - 2400000).

    Returns
    -------
    float or ndarray
        Approximate decimal year.
    """

    # restore full JD

    jd_full = jd + 2400000.0

    # Julian year approximation

    return (
        2000.0
        +
        (jd_full - 2451545.0) / 365.25
    )


# ==========================================================
# ORBIT-compatible mixed time normalization
# ==========================================================

def normalize_time(t):
    """
    Normalize ORBIT time conventions.

    Rules
    -----
    t > 3000
        Treat ``t`` as JD-2400000 and convert it to
        decimal year.

    t <= 3000
        Treat ``t`` as an already supplied decimal year.

    Notes
    -----
    The GL 765 benchmark intentionally contains visual
    observations in decimal years and radial-velocity
    observations in JD-2400000 format. Therefore the
    normalization is performed per observation rather than
    assuming that every record uses the same numerical
    representation.
    """

    if t > 3000:

        return jd_to_decimal_year(
            t
        )

    return t


# ==========================================================
# ORBIT element tag mapping
# ==========================================================

ELEMENT_MAP = {
    "P": "P",
    "T": "T0",
    "e": "e",
    "a": "a",
    "W": "Omega",
    "w": "omega",
    "i": "i",
    "K1": "K1",
    "K2": "K2",
    "V0": "V0"
}


# ==========================================================
# Parse ORBIT-format file
# ==========================================================

def parse_orbit_file(filename):
    """
    Parse Tokovinin ORBIT-format input file.

    Current implementation supports:

    - visual binary orbital elements
    - fit/fixed flags
    - visual astrometric observations
    - radial-velocity observations
    - mixed decimal-year / JD-2400000 epochs

    Returns
    -------
    dict
        Dictionary containing:

        elements
            OrbitalElements instance.

        visual_data
            NumPy array with columns

                [epoch, theta, rho, sigma]

        rv_data
            NumPy array with columns

                [epoch, velocity, sigma, component]

            where

                0 = primary (Va)
                1 = secondary (Vb)

        metadata
            Parsed object metadata.
    """

    with open(
        filename,
        "r"
    ) as f:

        lines = f.readlines()


    # ------------------------------------------------------
    # Storage
    # ------------------------------------------------------

    metadata = {}


    element_values = {
        "P": 0.0,
        "T0": 0.0,
        "e": 0.0,
        "a": 0.0,
        "Omega": 0.0,
        "omega": 0.0,
        "i": 0.0,
        "K1": 0.0,
        "K2": 0.0,
        "V0": 0.0
    }


    fit_flags = {
        "P": True,
        "T0": True,
        "e": True,
        "a": True,
        "Omega": True,
        "omega": True,
        "i": True,
        "K1": False,
        "K2": False,
        "V0": False
    }


    visual_rows = []

    rv_rows = []


    # ------------------------------------------------------
    # Parse line-by-line
    # ------------------------------------------------------

    for line in lines:

        line = line.strip()


        # --------------------------------------------------
        # Skip blank lines
        # --------------------------------------------------

        if not line:

            continue


        # --------------------------------------------------
        # Skip ORBIT comments
        # --------------------------------------------------

        if line.startswith("C"):

            continue


        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        if line.startswith("Object:"):

            metadata["object"] = (
                line.split(
                    ":",
                    1
                )[1].strip()
            )

            continue


        if line.startswith("R.A.:"):

            ra_string = (
                line.split(
                    ":",
                    1
                )[1].strip()
            )

            metadata["ra"] = (
                ra_string
            )

            metadata["ra_deg"] = (
                parse_ra(
                    ra_string
                )
            )

            continue


        if line.startswith("Dec.:"):

            dec_string = (
                line.split(
                    ":",
                    1
                )[1].strip()
            )

            metadata["dec"] = (
                dec_string
            )

            metadata["dec_deg"] = (
                parse_dec(
                    dec_string
                )
            )

            continue


        # --------------------------------------------------
        # Tokenize
        # --------------------------------------------------

        tokens = line.split()


        # --------------------------------------------------
        # Orbital elements
        # --------------------------------------------------

        if len(tokens) >= 2:

            tag = (
                tokens[0]
                .replace(
                    ":",
                    ""
                )
            )

            value = tokens[1]


            is_fixed = False


            # ORBIT fixed-element syntax

            if tag.startswith("*"):

                is_fixed = True

                tag = tag[1:]


            if tag in ELEMENT_MAP:

                attr = (
                    ELEMENT_MAP[tag]
                )


                try:

                    element_values[attr] = (
                        float(value)
                    )

                except ValueError as exc:

                    element_values[attr] = 0.0

                    print(
                        "Warning: failed to parse element "
                        f"line for '{attr}':\n"
                        f"{line}\n"
                        f"{exc}"
                    )


                fit_flags[attr] = (
                    not is_fixed
                )


                continue


        # --------------------------------------------------
        # Visual observations
        # --------------------------------------------------
        #
        # ORBIT recognizes an astrometric observation when
        # the line contains the I1 tag. Additional columns
        # may occur between sigma and I1.
        # --------------------------------------------------

        if "I1" in tokens:

            try:

                t = float(
                    tokens[0]
                )

                theta = float(
                    tokens[1]
                )

                rho = float(
                    tokens[2]
                )

                sigma = float(
                    tokens[3]
                )


                visual_rows.append([
                    t,
                    theta,
                    rho,
                    sigma
                ])


            except (
                ValueError,
                IndexError
            ) as exc:

                print(
                    "Warning: failed to parse visual line:\n"
                    f"{line}\n"
                    f"{exc}"
                )


            continue


        # --------------------------------------------------
        # Radial-velocity observations
        # --------------------------------------------------
        #
        # ORBIT allows the Va/Vb marker to occur after
        # additional columns:
        #
        #   time RV sigma ... Va ...
        #
        # Therefore the component marker must NOT be
        # assumed to occupy tokens[3].
        #
        # Only exact Va and Vb tags are accepted here.
        # Special tags such as Va+b are intentionally left
        # unresolved until their ORBIT semantics are
        # explicitly established.
        # --------------------------------------------------

        component = None


        if "Va" in tokens:

            component = 0


        elif "Vb" in tokens:

            component = 1


        if component is not None:

            try:

                t_rv = float(
                    tokens[0]
                )

                rv = float(
                    tokens[1]
                )

                sigma_rv = float(
                    tokens[2]
                )


                rv_rows.append([
                    t_rv,
                    rv,
                    sigma_rv,
                    component
                ])


            except (
                ValueError,
                IndexError
            ) as exc:

                print(
                    "Warning: failed to parse RV line:\n"
                    f"{line}\n"
                    f"{exc}"
                )


            continue


    # ======================================================
    # Create orbital elements object
    # ======================================================

    elements = OrbitalElements(

        P=element_values["P"],

        T0=element_values["T0"],

        e=element_values["e"],

        a=element_values["a"],

        Omega=element_values["Omega"],

        omega=element_values["omega"],

        i=element_values["i"],

        K1=element_values["K1"],

        K2=element_values["K2"],

        V0=element_values["V0"],

        fit_flags=fit_flags

    )


    # ======================================================
    # Normalize orbital epoch
    # ======================================================

    if elements.T0 > 3000:

        elements.T0 = normalize_time(
            elements.T0
        )

        # When the orbital epoch is expressed in
        # JD-2400000, the period is supplied in days.
        # Convert it to years because the internal orbit
        # representation uses years.

        elements.P /= 365.25


    # ======================================================
    # Convert visual observations to NumPy array
    # ======================================================

    if len(visual_rows) == 0:

        visual_data = np.empty(
            (0, 4),
            dtype=float
        )

    else:

        visual_data = np.array(
            visual_rows,
            dtype=float
        )


    # ------------------------------------------------------
    # Normalize visual observation times
    # ------------------------------------------------------

    if len(visual_data) > 0:

        visual_data[:, 0] = np.array([

            normalize_time(t)

            for t in visual_data[:, 0]

        ])


    # ======================================================
    # Convert RV observations to NumPy array
    # ======================================================

    if len(rv_rows) == 0:

        rv_data = np.empty(
            (0, 4),
            dtype=float
        )

    else:

        rv_data = np.array(
            rv_rows,
            dtype=float
        )


    # ------------------------------------------------------
    # Normalize RV observation times
    # ------------------------------------------------------

    if len(rv_data) > 0:

        rv_data[:, 0] = np.array([

            normalize_time(t)

            for t in rv_data[:, 0]

        ])


    # ======================================================
    # Return parsed representation
    # ======================================================

    return {

        "elements": elements,

        "visual_data": visual_data,

        "rv_data": rv_data,

        "metadata": metadata

    }