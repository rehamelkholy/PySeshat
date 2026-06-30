from pathlib import Path

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple.triple_observation import (
    TripleObservation
)


def parse_triple_file(
    filename
):
    """
    Parse ORBIT .in2 file.

    Returns
    -------
    dict
    """

    filename = Path(
        filename
    )

    parameters = {}

    fit_flags = {
        "P": False,
        "T0": False,
        "e": False,
        "a": False,
        "Omega": False,
        "omega": False,
        "i": False,

        "P2": False,
        "T02": False,
        "e2": False,
        "a2": False,
        "Omega2": False,
        "omega2": False,
        "i2": False,

        "B": False
    }

    observations = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            parts = line.split()

            # ------------------------------------------
            # Parameter line
            # ------------------------------------------

            KNOWN_PARAMETERS = {

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

            if len(parts) >= 2:

                raw_key = parts[0]

                fitted = raw_key.startswith("*")

                key = raw_key.lstrip("*")

                if key in KNOWN_PARAMETERS:

                    try:

                        parameters[key] = float(
                            parts[1]
                        )

                        if fitted:

                            mapping = {

                                "P": "P",
                                "T": "T0",
                                "e": "e",
                                "a": "a",
                                "W": "Omega",
                                "w": "omega",
                                "i": "i",

                                "P2": "P2",
                                "T2": "T02",
                                "e2": "e2",
                                "a2": "a2",
                                "W2": "Omega2",
                                "w2": "omega2",
                                "i2": "i2",

                                "B": "B"
                            }

                            if key in mapping:

                                fit_flags[
                                    mapping[key]
                                ] = True

                    except ValueError:

                        pass

            # ------------------------------------------
            # Observation line
            # ------------------------------------------

            if len(parts) >= 5:

                tag = parts[4]

                if tag in [
                    "I1",
                    "I2"
                ]:

                    observations.append(

                        TripleObservation(

                            epoch=float(
                                parts[0]
                            ),

                            theta=float(
                                parts[1]
                            ),

                            rho=float(
                                parts[2]
                            ),

                            sigma=float(
                                parts[3]
                            ),

                            orbit_id=tag
                        )
                    )

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

    triple = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        B=parameters["B"],

        fit_flags=fit_flags
    )

    return {

        "elements": triple,

        "observations": observations
    }