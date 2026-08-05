from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pyseshat.photometry.parameters import (
    update_parameters,
)

from pyseshat.orbit.orbital_elements import OrbitalElements

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


@dataclass(frozen=True)
class JointParameters:
    """
    Canonical parameter vector for simultaneous orbital
    and photometric inference.

    Parameters
    ----------
    orbital
        Orbital elements.

    photometric
        Photometric model parameters.
    """

    orbital: OrbitalElements

    photometric: PhotometricParameters

    def fitted_vector(self) -> np.ndarray:
        """
        Return the concatenated fitted parameter vector.

        Returns
        -------
        numpy.ndarray
        """

        orbital = np.asarray(
            self.orbital.fitted_parameters(),
            dtype=float,
        )

        photometric = np.asarray(
            self.photometric.fitted_parameters(),
            dtype=float,
        )

        return np.concatenate(
            (
                orbital,
                photometric,
            )
        )

    @property
    def n_orbital(self) -> int:
        """
        Number of fitted orbital parameters.
        """

        return len(
            self.orbital.fitted_parameters()
        )

    @property
    def n_photometric(self) -> int:
        """
        Number of fitted photometric parameters.
        """

        return len(
            self.photometric.fitted_parameters()
        )

    @property
    def size(self) -> int:
        """
        Total number of fitted parameters.
        """

        return (
            self.n_orbital
            +
            self.n_photometric
        )

    def update_from_vector(
        self,
        fitted_values,
    ):
        """
        Return a new JointParameters instance with the
        fitted parameters updated from a flat optimization
        vector.

        Parameters
        ----------
        fitted_values : array_like
            Concatenated orbital and photometric fitted
            parameter vector.

        Returns
        -------
        JointParameters
        """

        fitted_values = np.asarray(
            fitted_values,
            dtype=float,
        )

        if fitted_values.ndim != 1:

            raise ValueError(
                "fitted_values must be one-dimensional."
            )

        if fitted_values.size != self.size:

            raise ValueError(
                f"Expected {self.size} fitted parameters "
                f"but received {fitted_values.size}."
            )

        n_orbit = self.n_orbital

        orbital_values = fitted_values[:n_orbit]

        photometric_values = fitted_values[n_orbit:]

        orbital = self.orbital.copy()

        orbital.update_fitted(
            orbital_values,
        )

        orbital.normalize_angles()

        photometric = update_parameters(
            self.photometric,
            photometric_values,
        )

        return replace(
            self,
            orbital=orbital,
            photometric=photometric,
        )