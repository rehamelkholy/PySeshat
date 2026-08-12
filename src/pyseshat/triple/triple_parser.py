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

    The parser accepts both:

    1. The documented positional header:

           object
           R.A.
           Dec.
           WDS

    2. Labeled headers such as:

           Object: ...
           R.A.: ...
           Dec.: ...
           WDS: ...

    Returns
    -------
    dict
        Metadata dictionary.
    """

    metadata = {

        "object_name": None,

        "ra": None,

        "dec": None,

        "wds_code": None

    }


    # ------------------------------------------------------
    # First pass: explicitly labeled fields.
    #
    # We deliberately use regular expressions here rather
    # than startswith() so that whitespace and capitalization
    # variations do not affect metadata extraction.
    # ------------------------------------------------------

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

                metadata[field] = (
                    value
                    if value
                    else None
                )


    # ------------------------------------------------------
    # Second pass: positional header.
    #
    # We use only unlabeled lines for this pass. This prevents
    # labeled metadata lines from shifting the positional
    # interpretation.
    # ------------------------------------------------------

    unlabeled_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if _is_comment(line):
            continue

        is_labeled = any(

            pattern.match(
                stripped
            )

            for pattern
            in labeled_patterns.values()

        )

        if not is_labeled:

            unlabeled_lines.append(
                stripped
            )


    # ------------------------------------------------------
    # Fill only metadata fields not already obtained from
    # labeled records.
    # ------------------------------------------------------

    if (
        metadata["object_name"] is None
        and len(unlabeled_lines) >= 1
    ):

        metadata["object_name"] = (
            unlabeled_lines[0]
        )


    if (
        metadata["ra"] is None
        and len(unlabeled_lines) >= 2
    ):

        metadata["ra"] = (
            unlabeled_lines[1]
        )


    if (
        metadata["dec"] is None
        and len(unlabeled_lines) >= 3
    ):

        metadata["dec"] = (
            unlabeled_lines[2]
        )


    if (
        metadata["wds_code"] is None
        and len(unlabeled_lines) >= 4
    ):

        metadata["wds_code"] = (
            unlabeled_lines[3]
        )


    return metadata


# ==========================================================
# Parameter parsing
# ==========================================================

def _parse_parameter_line(
    parts
):
    """
    Parse an ORBIT orbital-element line.

    Returns
    -------
    tuple or None

        (key, value, fitted, error)

    The leading ``*`` indicates a fitted parameter.
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

    The documented form contains:

        time theta rho error ... I1 ...

    or:

        time theta rho error ... I2 ...

    The I1/I2 tag may occur after additional fields.
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

        orbit_id=tag

    )


# ==========================================================
# Radial-velocity observation parsing
# ==========================================================

def _parse_rv_line(
    parts
):
    """
    Parse an ORBIT3 radial-velocity observation.

    The documented form is:

        time RV error ... Va ...

        time RV error ... Vb ...

        time RV error ... Vc ...

    where:

        Va = first component
        Vb = second component
        Vc = third component
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


    return {

        "epoch": epoch,

        "velocity": velocity,

        "sigma": sigma,

        "component": component

    }


# ==========================================================
# Main parser
# ==========================================================

def parse_triple_file(
    filename
):
    """
    Parse an ORBIT triple-star input file.

    Both legacy ``.in2`` files used by ``orbitT2.pro`` and
    modern ``.in3`` files used by ``orbit3.pro`` are
    supported.

    Legacy ``.in2`` files use:

        P, T, e, a, W, w, i,
        P2, T2, e2, a2, W2, w2, i2,
        B

    with the first orbit representing the outer orbit.

    Modern ``.in3`` files use:

        P, T, e, a, W, w, i,
        K1, K2, f,
        P2, T2, e2, a2, W2, w2, i2,
        K3, K4, V0

    with the first orbit representing the inner orbit.

    Returns
    -------
    dict

        ``elements``
            TripleOrbitalElements.

        ``observations``
            Astrometric TripleObservation objects.

        ``rv_observations``
            Radial-velocity records.

        ``parameter_errors``
            Supplied parameter errors.

        ``metadata``
            Parsed object metadata and detected format.
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

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        "P": False,
        "T0": False,
        "e": False,
        "a": False,
        "Omega": False,
        "omega": False,
        "i": False,

        # --------------------------------------------------
        # Inner spectroscopy / wobble
        # --------------------------------------------------

        "K1": False,
        "K2": False,
        "f": False,

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        "P2": False,
        "T02": False,
        "e2": False,
        "a2": False,
        "Omega2": False,
        "omega2": False,
        "i2": False,

        # --------------------------------------------------
        # Outer spectroscopy
        # --------------------------------------------------

        "K3": False,
        "K4": False,
        "V0": False,

        # --------------------------------------------------
        # Legacy ORBITT2 parameter
        # --------------------------------------------------

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
    # Header metadata
    # ======================================================

    metadata.update(

        _parse_header_metadata(
            lines
        )

    )


    # ======================================================
    # Parse records
    # ======================================================

    for line in lines:

        if _is_comment(line):
            continue


        parts = line.split()


        # --------------------------------------------------
        # Orbital parameter
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Astrometric observation
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Radial-velocity observation
        # --------------------------------------------------

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
    # Modern ORBIT3 parameters
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
    # Determine file formulation
    # ======================================================

    is_orbit3 = (
        metadata["format"] == "in3"
    )


    # ======================================================
    # ORBIT3 formulation
    # ======================================================

    if is_orbit3:

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        inner = OrbitalElements(

            P=parameters["P"],

            T0=parameters["T"],

            e=parameters["e"],

            a=parameters["a"],

            Omega=parameters["W"],

            omega=parameters["w"],

            i=parameters["i"]

        )


        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        outer = OrbitalElements(

            P=parameters["P2"],

            T0=parameters["T2"],

            e=parameters["e2"],

            a=parameters["a2"],

            Omega=parameters["W2"],

            omega=parameters["w2"],

            i=parameters["i2"]

        )


    # ======================================================
    # Legacy ORBITT2 formulation
    # ======================================================

    else:

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        outer = OrbitalElements(

            P=parameters["P"],

            T0=parameters["T"],

            e=parameters["e"],

            a=parameters["a"],

            Omega=parameters["W"],

            omega=parameters["w"],

            i=parameters["i"]

        )


        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

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
    # Construct triple elements
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