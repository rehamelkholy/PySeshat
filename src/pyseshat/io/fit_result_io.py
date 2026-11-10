import json
import numpy as np

from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone

from scipy.optimize import (
    OptimizeResult
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.inference.fit_result import (
    FitResult
)

def orbital_elements_to_dict(
    elements
):
    """
    Convert OrbitalElements to a
    JSON-safe dictionary.
    """

    return asdict(
        elements
    )

def fit_result_to_dict(
    fit_result,
    system_name=None
):
    """
    Convert FitResult to a
    JSON-serializable dictionary.
    """

    return {

        "metadata": {

            "system":
                system_name,

            "created_utc":
                datetime.now(timezone.utc).isoformat(),

            "format":
                "PySeshat FitResult",

            "version":
                "0.1"
        },

        "elements":
            orbital_elements_to_dict(
                fit_result.elements
            ),

        "uncertainties":
            fit_result.uncertainties,

        "covariance":
            fit_result.covariance.tolist(),

        "residuals":
            fit_result.residuals.tolist(),

        "success":
            fit_result.optimizer_result.success,

        "message":
            str(
                fit_result.optimizer_result.message
            )
    }

def save_fit_result(
    fit_result,
    filename,
    system_name=None
):
    """
    Save FitResult to JSON.
    """

    payload = fit_result_to_dict(
        fit_result,
        system_name=system_name
    )

    filename = Path(
        filename
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            indent=4
        )

    return filename

def dict_to_orbital_elements(
    data
):
    """
    Reconstruct OrbitalElements
    from dictionary.
    """

    return OrbitalElements(
        P=data["P"],
        T0=data["T0"],
        e=data["e"],
        a=data["a"],
        Omega=data["Omega"],
        omega=data["omega"],
        i=data["i"],
        K1=data["K1"],
        K2=data["K2"],
        V0=data["V0"],
        fit_flags=data["fit_flags"]
    )

def load_fit_result(
    filename
):
    """
    Load FitResult from JSON.
    """

    filename = Path(
        filename
    )

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)
    
        elements = dict_to_orbital_elements(
        data["elements"]
        )

        covariance = np.array(
            data["covariance"],
            dtype=float
        )

        residuals = np.array(
            data["residuals"],
            dtype=float
        )

        optimizer_result = OptimizeResult(

            success=data["success"],

            message=data["message"]
        )

        fit_result = FitResult(

            elements=elements,

            optimizer_result=optimizer_result,

            residuals=residuals,

            covariance=covariance,

            uncertainties=data[
                "uncertainties"
            ],

            metadata=data.get(
                "metadata"
            )
        )

        return fit_result