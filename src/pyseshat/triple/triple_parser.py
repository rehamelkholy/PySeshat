from pathlib import Path
import re


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_observation import (
    TripleObservation
)


# ==========================================================
# ORBIT parameter definitions
# ==========================================================

_LEGACY_PARAMETER_KEYS = {

    "P",
    "T",
    "e",
    "a",
    "W",
    "w",
    "i",

    "P2",
    "T2",
    "e2",
    "a2",
    "W2",
    "w2",
    "i2",

    "B"

}


_ORBIT3_PARAMETER_KEYS = {

    "P",
    "T",
    "e",
    "a",
    "W",
    "w",
    "i",

    "K1",
    "K2",
    "f",

    "P2",
    "T2",
    "e2",
    "a2",
    "W2",
    "w2",
    "i2",

    "K3",
    "K4",
    "V0"

}


_PARAMETER_MAPPING = {

    "P": "P",
    "T": "T0",
    "e": "e",
    "a": "a",
    "W": "Omega",
    "w": "omega",
    "i": "i",

    "K1": "K1",
    "K2": "K2",
    "f": "f",

    "P2": "P2",
    "T2": "T02",
    "e2": "e2",
    "a2": "a2",
    "W2": "Omega2",
    "w2": "omega2",
    "i2": "i2",

    "K3": "K3",
    "K4": "K4",
    "V0": "V0",

    "B": "B"

}


# ==========================================================
# ORBIT epoch conversion
# ==========================================================

_JD240000_TO_BESSELIAN_YEAR_OFFSET = 15020.31352

_DAYS_PER_BESSELIAN_YEAR = 365.242198781


def _rv_epoch_to_besselian_year(
    epoch
):
    """
    Convert an ORBIT3 radial-velocity epoch from the
    reduced Julian-date convention to Besselian year.

    ORBIT3 stores RV epochs in the form

        JD - 2400000

    and converts them to Besselian years using

        1900
        +
        (epoch - 15020.31352)
        / 365.242198781

    before evaluating the orbital model.

    Parameters
    ----------
    epoch : float
        Radial-velocity epoch as stored in an ORBIT3
        input record.

    Returns
    -------
    float
        Epoch expressed as Besselian year.
    """

    return (

        1900.0

        +

        (

            epoch
            -
            _JD240000_TO_BESSELIAN_YEAR_OFFSET

        )

        /

        _DAYS_PER_BESSELIAN_YEAR

    )


# ==========================================================
# ORBIT3 coordinate conversion
# ==========================================================

def _orbit3_getcoord(
    value
):
    """
    Reproduce the ORBIT3 ``getcoord`` function.

    ORBIT3 interprets packed sexagesimal coordinates such as

        22.3848 -> 22 deg 38 min 48 sec

    and

        44.19 -> 44 deg 19 min 00 sec.

    The conversion is intentionally not replaced by ordinary
    decimal-degree interpretation because ORBIT3 explicitly
    uses packed sexagesimal input.
    """

    text = str(
        value
    ).strip()


    if "." not in text:

        return float(
            text
        )


    decimal_position = text.find(
        "."
    )


    degrees = int(
        text[:decimal_position]
    )


    minutes = int(
        text[
            decimal_position + 1 :
            decimal_position + 3
        ]
        or "0"
    )


    seconds_text = text[
        decimal_position + 3 :
    ]


    seconds = (

        float(
            seconds_text
        )

        if seconds_text

        else 0.0

    )


    result = (

        abs(degrees)

        +

        minutes / 60.0

        +

        seconds / 3600.0

    )


    if degrees < 0:

        result = -result


    return result


# ==========================================================
# ORBIT3 observation epoch correction
# ==========================================================

def _orbit3_correct_observation_epochs(
    observations,
    orbital_epoch
):
    """
    Reproduce the ORBIT3 ``correct`` procedure for
    astrometric observations.

    ORBIT3 changes between:

        calendar/Besselian years

    and

        JD - 2400000

    according to the representation of the orbital
    reference epoch.

    The conversion is applied to the observation epochs
    before the ORBIT3 precession correction.
    """

    corrected = []


    for observation in observations:

        epoch = observation.epoch


        # --------------------------------------------------
        # Years -> JD - 2400000
        # --------------------------------------------------

        if (

            epoch < 3000.0
            and orbital_epoch > 3000.0

        ):

            epoch = (

                _DAYS_PER_BESSELIAN_YEAR
                *
                (
                    epoch
                    -
                    1900.0
                )
                +
                _JD240000_TO_BESSELIAN_YEAR_OFFSET

            )


        # --------------------------------------------------
        # JD - 2400000 -> Besselian years
        # --------------------------------------------------

        elif (

            epoch > 3000.0
            and orbital_epoch < 3000.0

        ):

            epoch = (

                1900.0

                +

                (
                    epoch
                    -
                    _JD240000_TO_BESSELIAN_YEAR_OFFSET
                )
                /
                _DAYS_PER_BESSELIAN_YEAR

            )


        if epoch == observation.epoch:

            corrected.append(
                observation
            )

        else:

            corrected.append(

                TripleObservation(

                    epoch=epoch,

                    theta=observation.theta,

                    rho=observation.rho,

                    sigma=observation.sigma,

                    orbit_id=observation.orbit_id,

                    velocity=observation.velocity

                )

            )


    return corrected


# ==========================================================
# ORBIT3 precession correction
# ==========================================================

def _orbit3_precession_rate(
    metadata,
    orbital_epoch
):
    """
    Reproduce the ORBIT3 precession-rate calculation.

    ORBIT3 uses

        PR =
            0.0057
            * sin(RA)
            / cos(Dec)

    in degrees per year.

    RA is interpreted as hours in packed sexagesimal form
    and converted to degrees by multiplication by 15.

    The correction is enabled only when:

        RA > 0

    and

        T0 < 3000

    exactly as in ORBIT3 ``readinp``.
    """

    ra_value = metadata.get(
        "ra"
    )

    dec_value = metadata.get(
        "dec"
    )


    if (

        ra_value is None
        or dec_value is None

    ):

        return None


    if orbital_epoch >= 3000.0:

        return None


    try:

        ra_hours = _orbit3_getcoord(
            ra_value
        )

        dec_degrees = _orbit3_getcoord(
            dec_value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    ra_degrees = (
        15.0
        *
        ra_hours
    )


    if ra_degrees <= 0.0:

        return None


    return (

        0.0057

        *

        __import__(
            "math"
        ).sin(
            __import__(
                "math"
            ).radians(
                ra_degrees
            )
        )

        /

        __import__(
            "math"
        ).cos(
            __import__(
                "math"
            ).radians(
                dec_degrees
            )
        )

    )


def _orbit3_apply_precession(
    observations,
    metadata,
    orbital_epoch
):
    """
    Apply the ORBIT3 ``readinp`` precession correction
    to visual observations.

    ORBIT3 performs:

        theta_corrected =
            theta_original
            +
            (2000 - epoch) * PR

    for both POS1 and POS2.

    The returned TripleObservation objects contain the
    corrected theta because that is the angle subsequently
    converted by ORBIT3 into its fitting X/Y coordinates.
    """

    precession_rate = _orbit3_precession_rate(
        metadata,
        orbital_epoch
    )


    if precession_rate is None:

        return observations


    corrected = []


    for observation in observations:

        if observation.orbit_id not in (
            "I1",
            "I2"
        ):

            corrected.append(
                observation
            )

            continue


        corrected_theta = (

            observation.theta

            +

            (
                2000.0
                -
                observation.epoch
            )
            *
            precession_rate

        )


        corrected.append(

            TripleObservation(

                epoch=observation.epoch,

                theta=corrected_theta,

                rho=observation.rho,

                sigma=observation.sigma,

                orbit_id=observation.orbit_id,

                velocity=observation.velocity

            )

        )


    return corrected


# ==========================================================
# Comment handling
# ==========================================================

def _is_comment(
    line
):
    """
    Return True if an ORBIT input line is a comment.

    ORBIT uses lines beginning with ``C`` for comments.
    """

    stripped = line.lstrip()


    if not stripped:

        return True


    return stripped[0].upper() == "C"


# ==========================================================
# Header metadata
# ==========================================================

def _parse_header_metadata(
    lines
):
    """
    Parse metadata from an ORBIT input file.

    Supported forms

    ---------------

    Labeled records:

        Object: ...
        RA: ...
        R.A.: ...
        Dec.: ...
        WDS: ...

    ORBIT3 WDS record:

        WDS 22388+4419 ADS 16138 ...

    Positional records:

        object
        RA
        Dec
        WDS

    For a WDS record containing additional catalog
    information, only the first token after WDS is stored.
    """

    metadata = {

        "object_name": None,

        "ra": None,

        "dec": None,

        "wds_code": None

    }


    labeled_patterns = {

        "object_name": re.compile(

            r"^\s*object\s*:\s*(.*?)\s*$",

            re.IGNORECASE

        ),

        "ra": re.compile(

            r"^\s*r\.?\s*a\.?\s*:\s*(.*?)\s*$",

            re.IGNORECASE

        ),

        "dec": re.compile(

            r"^\s*dec(?:lination)?\.?\s*:\s*(.*?)\s*$",

            re.IGNORECASE

        ),

        "wds_code": re.compile(

            r"^\s*wds\s*:\s*(.*?)\s*$",

            re.IGNORECASE

        )

    }


    for line in lines:

        stripped = line.strip()


        if not stripped:

            continue


        if _is_comment(line):

            continue


        for field, pattern in (
            labeled_patterns.items()
        ):

            match = pattern.match(
                stripped
            )


            if match:

                value = match.group(
                    1
                ).strip()


                if field == "wds_code":

                    parts = value.split()


                    metadata[field] = (

                        parts[0]

                        if parts

                        else None

                    )

                else:

                    metadata[field] = (

                        value

                        if value

                        else None

                    )


                break


    positional = []


    for line in lines:

        stripped = line.strip()


        if not stripped:

            if positional:

                break

            continue


        if _is_comment(line):

            if positional:

                continue

            continue


        wds_match = re.match(

            r"^\s*wds(?:\s+|:)(.*)$",

            stripped,

            re.IGNORECASE

        )


        if wds_match:

            value = wds_match.group(
                1
            ).strip()


            parts = value.split()


            if (

                metadata["wds_code"] is None
                and parts

            ):

                metadata["wds_code"] = (
                    parts[0]
                )


            continue


        is_labeled = any(

            pattern.match(
                stripped
            )

            for pattern
            in labeled_patterns.values()

        )


        if is_labeled:

            continue


        parts = stripped.split()


        if not parts:

            continue


        raw_key = parts[0].lstrip(
            "*"
        )


        if raw_key in (

            _LEGACY_PARAMETER_KEYS
            |
            _ORBIT3_PARAMETER_KEYS

        ):

            break


        if len(parts) >= 4:

            if any(

                tag in parts[3:]

                for tag in (
                    "I1",
                    "I2",
                    "Va",
                    "Vb",
                    "Vc"
                )

            ):

                break


        positional.append(
            stripped
        )


    if (

        metadata["object_name"] is None
        and len(positional) >= 1

    ):

        metadata["object_name"] = (
            positional[0]
        )


    if (

        metadata["ra"] is None
        and len(positional) >= 2

    ):

        metadata["ra"] = (
            positional[1]
        )


    if (

        metadata["dec"] is None
        and len(positional) >= 3

    ):

        metadata["dec"] = (
            positional[2]
        )


    if (

        metadata["wds_code"] is None
        and len(positional) >= 4

    ):

        metadata["wds_code"] = (
            positional[3].split()[0]
        )


    return metadata


# ==========================================================
# Parameter parsing
# ==========================================================

def _parse_parameter_line(
    parts
):
    """
    Parse an ORBIT orbital-parameter line.

    A leading ``*`` means that the parameter is FIXED.

    An unstarred parameter is FITTED.
    """

    if len(parts) < 2:

        return None


    raw_key = parts[0]


    fixed = raw_key.startswith(
        "*"
    )


    fitted = not fixed


    key = raw_key.lstrip(
        "*"
    )


    if key not in (

        _LEGACY_PARAMETER_KEYS
        |
        _ORBIT3_PARAMETER_KEYS

    ):

        return None


    try:

        value = float(
            parts[1]
        )

    except ValueError:

        return None


    error = None


    if len(parts) >= 3:

        try:

            error = float(
                parts[2]
            )

        except ValueError:

            error = None


    return (

        key,

        value,

        fitted,

        error

    )


# ==========================================================
# Astrometric observation parsing
# ==========================================================

def _parse_position_line(
    parts
):
    """
    Parse an ORBIT positional observation.

    Expected form includes:

        epoch theta rho sigma ... I1

    or:

        epoch theta rho sigma ... I2
    """

    if len(parts) < 5:

        return None


    tag = None


    for candidate in (
        "I1",
        "I2"
    ):

        if candidate in parts[4:]:

            tag = candidate

            break


    if tag is None:

        return None


    try:

        epoch = float(
            parts[0]
        )

        theta = float(
            parts[1]
        )

        rho = float(
            parts[2]
        )

        sigma = float(
            parts[3]
        )

    except ValueError:

        return None


    return TripleObservation(

        epoch=epoch,

        theta=theta,

        rho=rho,

        sigma=sigma,

        orbit_id=tag,

        velocity=None

    )


# ==========================================================
# Radial-velocity observation parsing
# ==========================================================

def _parse_rv_line(
    parts
):
    """
    Parse an ORBIT3 radial-velocity observation.

    Expected forms include:

        epoch RV sigma ... Va

        epoch RV sigma ... Vb

        epoch RV sigma ... Vc

    ORBIT3 stores these RV epochs as reduced Julian dates
    (JD - 2400000) and converts them to Besselian years
    before evaluating the orbital model.
    """

    if len(parts) < 4:

        return None


    component = None


    for candidate in (
        "Va",
        "Vb",
        "Vc"
    ):

        if candidate in parts[3:]:

            component = candidate

            break


    if component is None:

        return None


    try:

        epoch = float(
            parts[0]
        )

        velocity = float(
            parts[1]
        )

        sigma = float(
            parts[2]
        )

    except ValueError:

        return None


    epoch = _rv_epoch_to_besselian_year(
        epoch
    )


    return TripleObservation(

        epoch=epoch,

        theta=None,

        rho=None,

        sigma=sigma,

        orbit_id=component,

        velocity=velocity

    )


# ==========================================================
# Main parser
# ==========================================================

def parse_triple_file(
    filename
):
    """
    Parse an ORBIT triple-star input file.

    Supported formats:

        .in2
            Legacy ORBITT2 hierarchical-triple format.

        .in3
            Modern ORBIT3 hierarchical-triple format.

    Returns
    -------

    dict

        elements
            TripleOrbitalElements

        observations
            Visual observations

        rv_observations
            Radial-velocity observations

        parameter_errors
            Parameter uncertainties supplied in the file

        metadata
            Object metadata and detected format
    """

    filename = Path(
        filename
    )


    if not filename.exists():

        raise FileNotFoundError(
            filename
        )


    parameters = {}

    parameter_errors = {}


    fit_flags = {

        "P": False,
        "T0": False,
        "e": False,
        "a": False,
        "Omega": False,
        "omega": False,
        "i": False,

        "K1": False,
        "K2": False,
        "f": False,

        "P2": False,
        "T02": False,
        "e2": False,
        "a2": False,
        "Omega2": False,
        "omega2": False,
        "i2": False,

        "K3": False,
        "K4": False,
        "V0": False,

        "B": False

    }


    observations = []

    rv_observations = []


    metadata = {

        "format": (

            "in3"

            if filename.suffix.lower() == ".in3"

            else "in2"

        )

    }


    with open(

        filename,

        "r",

        encoding="utf-8",

        errors="ignore"

    ) as f:

        lines = f.readlines()


    # ======================================================
    # Header
    # ======================================================

    metadata.update(

        _parse_header_metadata(
            lines
        )

    )


    # ======================================================
    # Parse all records
    # ======================================================

    for line in lines:

        if _is_comment(line):

            continue


        parts = line.split()


        parsed_parameter = (

            _parse_parameter_line(
                parts
            )

        )


        if parsed_parameter is not None:

            (

                key,

                value,

                fitted,

                error

            ) = parsed_parameter


            parameters[key] = value


            if error is not None:

                parameter_errors[key] = error


            if fitted:

                mapped_key = (
                    _PARAMETER_MAPPING[key]
                )


                fit_flags[
                    mapped_key
                ] = True


            continue


        positional = (

            _parse_position_line(
                parts
            )

        )


        if positional is not None:

            observations.append(
                positional
            )

            continue


        radial_velocity = (

            _parse_rv_line(
                parts
            )

        )


        if radial_velocity is not None:

            rv_observations.append(
                radial_velocity
            )


    # ======================================================
    # Required orbital parameters
    # ======================================================

    required_parameters = {

        "P",
        "T",
        "e",
        "a",
        "W",
        "w",
        "i",

        "P2",
        "T2",
        "e2",
        "a2",
        "W2",
        "w2",
        "i2"

    }


    missing = sorted(

        required_parameters
        -
        set(parameters)

    )


    if missing:

        raise ValueError(

            "Missing required triple orbital "
            "parameters: "
            +
            ", ".join(
                missing
            )

        )


    # ======================================================
    # ORBIT3 spectroscopic parameters
    # ======================================================

    K1 = parameters.get(
        "K1",
        0.0
    )


    K2 = parameters.get(
        "K2",
        0.0
    )


    f = parameters.get(
        "f",
        0.0
    )


    K3 = parameters.get(
        "K3",
        0.0
    )


    K4 = parameters.get(
        "K4",
        0.0
    )


    V0 = parameters.get(
        "V0",
        0.0
    )


    # ======================================================
    # Legacy B
    # ======================================================

    B = parameters.get(
        "B",
        1.0
    )


    # ======================================================
    # Determine representation
    # ======================================================

    is_orbit3 = (

        metadata["format"] == "in3"

    )


    # ======================================================
    # Modern ORBIT3
    # ======================================================

    if is_orbit3:

        inner = OrbitalElements(

            P=parameters["P"],

            T0=parameters["T"],

            e=parameters["e"],

            a=parameters["a"],

            Omega=parameters["W"],

            omega=parameters["w"],

            i=parameters["i"]

        )


        outer = OrbitalElements(

            P=parameters["P2"],

            T0=parameters["T2"],

            e=parameters["e2"],

            a=parameters["a2"],

            Omega=parameters["W2"],

            omega=parameters["w2"],

            i=parameters["i2"]

        )


        # --------------------------------------------------
        # ORBIT3 readinp preprocessing
        # --------------------------------------------------
        #
        # ORBIT3 performs:
        #
        #   correct(pos,  t0)
        #   correct(pos2, t0)
        #
        # followed by the precession correction to both
        # POS1 and POS2.
        #
        # The primary orbital epoch is T = el[1].
        # --------------------------------------------------

        orbital_epoch = parameters["T"]


        observations = (
            _orbit3_correct_observation_epochs(
                observations,
                orbital_epoch
            )
        )


        observations = (
            _orbit3_apply_precession(
                observations,
                metadata,
                orbital_epoch
            )
        )


    # ======================================================
    # Legacy ORBITT2
    # ======================================================

    else:

        outer = OrbitalElements(

            P=parameters["P"],

            T0=parameters["T"],

            e=parameters["e"],

            a=parameters["a"],

            Omega=parameters["W"],

            omega=parameters["w"],

            i=parameters["i"]

        )


        inner = OrbitalElements(

            P=parameters["P2"],

            T0=parameters["T2"],

            e=parameters["e2"],

            a=parameters["a2"],

            Omega=parameters["W2"],

            omega=parameters["w2"],

            i=parameters["i2"]

        )


    # ======================================================
    # Construct TripleOrbitalElements
    # ======================================================

    triple = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        K1=K1,

        K2=K2,

        f=f,

        K3=K3,

        K4=K4,

        V0=V0,

        B=B,

        fit_flags=fit_flags

    )


    return {

        "elements": triple,

        "observations": observations,

        "rv_observations": rv_observations,

        "parameter_errors": parameter_errors,

        "metadata": metadata

    }