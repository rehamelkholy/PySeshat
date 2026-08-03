from dataclasses import dataclass

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


@dataclass(frozen=True)
class PhotometricInferenceProblem:
    """
    Canonical inference problem for unresolved-binary
    photometric inference.

    Parameters
    ----------
    parameters : PhotometricParameters
        Current photometric model parameters.

    context : PhotometricContext
        Fixed observational context.
    """

    parameters: PhotometricParameters

    context: PhotometricContext