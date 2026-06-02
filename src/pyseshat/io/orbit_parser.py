import numpy as np

from pyseshat.orbit.orbital_elements import OrbitalElements

# ==========================================================
# Time conversion utilities
# ==========================================================

def jd_to_decimal_year(jd):
    """
    Convert JD-2400000 time convention
    to decimal year approximation.
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

    Rules:
    ------
    t > 3000  -> JD-2400000
    otherwise -> decimal year
    """

    if t > 3000:

        return jd_to_decimal_year(t)

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

    Returns
    -------
    dict with keys:
        elements
        visual_data
        metadata
    """

    with open(filename, "r") as f:
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

        # skip blank lines
        if not line:
            continue

        # skip comments
        if line.startswith("C"):
            continue

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        if line.startswith("Object:"):
            metadata["object"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("R.A.:"):
            metadata["ra"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("Dec.:"):
            metadata["dec"] = line.split(":", 1)[1].strip()
            continue

        # --------------------------------------------------
        # Orbital elements
        # --------------------------------------------------

        tokens = line.split()

        if len(tokens) >= 2:

            tag = tokens[0].replace(":", "")
            value = tokens[1]

            is_fixed = False

            # ORBIT fixed-element syntax
            if tag.startswith("*"):

                is_fixed = True
                tag = tag[1:]

            if tag in ELEMENT_MAP:

                attr = ELEMENT_MAP[tag]

                try:
                    element_values[attr] = float(value)
                except ValueError:
                    element_values[attr] = 0.0

                fit_flags[attr] = not is_fixed

                continue

        # --------------------------------------------------
        # Visual observations
        # --------------------------------------------------

        if "I1" in line:

            try:
                cols = line.split()

                t = float(cols[0])
                theta = float(cols[1])
                rho = float(cols[2])
                
                sigma_str = cols[3].replace(
                    "I1",
                    ""
                )

                sigma = float(sigma_str)

                visual_rows.append([t, theta, rho, sigma])

            except Exception as e:
                print(
                    f"Warning: failed to parse visual line:\n"
                    f"{line}\n"
                    f"{e}"
                )
        
        # --------------------------------------------------
        # Radial velocity observations
        # --------------------------------------------------

        if ("Va" in line) or ("Vb" in line):

            try:

                cols = line.split()

                t_rv = float(cols[0])

                rv = float(cols[1])

                sigma_rv = float(cols[2])

                component_tag = cols[3]

                # component encoding
                # 0 = primary
                # 1 = secondary

                if component_tag == "Va":

                    component = 0

                elif component_tag == "Vb":

                    component = 1

                else:

                    continue

                rv_rows.append([
                    t_rv,
                    rv,
                    sigma_rv,
                    component
                ])

            except Exception as e:

                print(
                    f"Warning: failed to parse RV line:\n"
                    f"{line}\n"
                    f"{e}"
                )

    # ------------------------------------------------------
    # Create orbital elements object
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Normalize orbital epoch
    # ------------------------------------------------------

    if elements.T0 > 3000:

        elements.T0 = normalize_time(
            elements.T0
        )

        # period must also convert
        elements.P /= 365.25

    # ------------------------------------------------------
    # Convert observations to NumPy array
    # ------------------------------------------------------

    visual_data = np.array(visual_rows, dtype=float)

    # ------------------------------------------------------
    # Normalize visual observation times
    # ------------------------------------------------------

    if len(visual_data) > 0:

        visual_data[:,0] = np.array([
            normalize_time(t)
            for t in visual_data[:,0]
        ])

    # ------------------------------------------------------
    # Convert RV rows to NumPy array
    # ------------------------------------------------------
    if len(rv_rows) == 0:

        rv_data = np.empty((0,4))

    else:

        rv_data = np.array(
            rv_rows,
            dtype=float
        )

    # ------------------------------------------------------
    # Normalize RV observation times
    # ------------------------------------------------------

    if len(rv_data) > 0:

        rv_data[:,0] = np.array([
            normalize_time(t)
            for t in rv_data[:,0]
        ])

    return {
        "elements": elements,
        "visual_data": visual_data,
        "rv_data": rv_data,
        "metadata": metadata
    }