from dataclasses import dataclass

import numpy as np


@dataclass
class PhotometricPassband:
    """
    Photometric passband transmission curve.

    Parameters
    ----------
    wavelength : array_like
        Passband wavelength grid in Angstrom.

    throughput : array_like
        Dimensionless total-system throughput.

    name : str, default=""
        Descriptive passband name.

    Notes
    -----
    Throughput values must be finite, non-negative,
    and need not be normalized;
    any constant scaling is removed during synthetic-photometry
    integration.
    """

    wavelength: np.ndarray

    throughput: np.ndarray

    name: str = ""

    def __post_init__(
        self
    ):
        """
        Validate and normalize the passband data.
        """

        # ==================================================
        # Normalize arrays
        # ==================================================

        self.wavelength = np.asarray(

            self.wavelength,

            dtype=float

        )

        self.throughput = np.asarray(

            self.throughput,

            dtype=float

        )

        self.name = str(

            self.name

        ).strip()

        # ==================================================
        # Validate dimensions
        # ==================================================

        if self.wavelength.ndim != 1:

            raise ValueError(

                "wavelength must be a "
                "one-dimensional array."

            )

        if self.throughput.ndim != 1:

            raise ValueError(

                "throughput must be a "
                "one-dimensional array."

            )

        # ==================================================
        # Validate array lengths
        # ==================================================

        if (

            len(
                self.wavelength
            )

            !=

            len(
                self.throughput
            )

        ):

            raise ValueError(

                "wavelength and throughput must "
                "have the same length."

            )

        if len(
            self.wavelength
        ) < 2:

            raise ValueError(

                "A photometric passband must contain "
                "at least two wavelength samples."

            )

        # ==================================================
        # Validate finite values
        # ==================================================

        if not np.all(

            np.isfinite(
                self.wavelength
            )

        ):

            raise ValueError(

                "wavelength must contain only "
                "finite values."

            )

        if not np.all(

            np.isfinite(
                self.throughput
            )

        ):

            raise ValueError(

                "throughput must contain only "
                "finite values."

            )

        # ==================================================
        # Validate wavelength grid
        # ==================================================

        if np.any(

            self.wavelength <= 0.0

        ):

            raise ValueError(

                "All wavelengths must be positive."

            )

        if np.any(

            np.diff(
                self.wavelength
            )

            <= 0.0

        ):

            raise ValueError(

                "wavelength must be strictly increasing."

            )

        # ==================================================
        # Validate throughput
        # ==================================================

        if np.any(

            self.throughput < 0.0

        ):

            raise ValueError(

                "throughput must not contain "
                "negative values."

            )

        if not np.any(

            self.throughput > 0.0

        ):

            raise ValueError(

                "throughput must contain at least "
                "one positive value."

            )

    # ======================================================
    # Passband properties
    # ======================================================

    @property
    def minimum_wavelength(
        self
    ):
        """
        Minimum passband wavelength in Angstrom.
        """

        return float(

            self.wavelength[
                0
            ]

        )

    @property
    def maximum_wavelength(
        self
    ):
        """
        Maximum passband wavelength in Angstrom.
        """

        return float(

            self.wavelength[
                -1
            ]

        )

    @property
    def peak_wavelength(
        self
    ):
        """
        Wavelength of maximum throughput in Angstrom.
        """

        index = int(

            np.argmax(
                self.throughput
            )

        )

        return float(

            self.wavelength[
                index
            ]

        )

    def throughput_at(
        self,
        wavelength
    ):
        """
        Interpolate the passband throughput.

        Parameters
        ----------
        wavelength : array_like
            Requested wavelengths in Angstrom.

        Returns
        -------
        numpy.ndarray
            Interpolated dimensionless throughput.

        Notes
        -----
        Throughput is set to zero outside the tabulated
        passband wavelength range.
        """

        wavelength = np.asarray(

            wavelength,

            dtype=float

        )

        if not np.all(

            np.isfinite(
                wavelength
            )

        ):

            raise ValueError(

                "wavelength must contain only "
                "finite values."

            )

        return np.interp(

            wavelength,

            self.wavelength,

            self.throughput,

            left=0.0,

            right=0.0

        )
    
def load_passband(
    filename,
    name=""
):
    """
    Load a photometric passband from a two-column
    ASCII text file.

    Parameters
    ----------
    filename : str or path-like
        Path to the passband transmission file.

        The file must contain two whitespace-separated
        numeric columns:

            wavelength    throughput

        Wavelengths must be expressed in Angstrom and
        throughput must be dimensionless.

    name : str, default=""
        Descriptive passband name.

    Returns
    -------
    PhotometricPassband
        Validated photometric passband.

    Raises
    ------
    FileNotFoundError
        If the passband file does not exist.

    ValueError
        If the file cannot be interpreted as a
        two-column numeric table.

    Notes
    -----
    Blank lines and lines beginning with ``#`` are
    ignored by NumPy's text loader.
    """

    # ======================================================
    # Normalize and validate path
    # ======================================================

    from pathlib import Path


    path = Path(

        filename

    )


    if not path.is_file():

        raise FileNotFoundError(

            "Passband file does not exist: "

            f"{path}"

        )

    # ======================================================
    # Load numeric table
    # ======================================================

    try:

        data = np.loadtxt(

            path,

            dtype=float,

            comments="#",

            ndmin=2

        )

    except ValueError as error:

        raise ValueError(

            "Passband file must contain a valid "
            "two-column numeric table: "

            f"{path}"

        ) from error

    # ======================================================
    # Validate table structure
    # ======================================================

    if (

        data.ndim != 2

        or

        data.shape[1] != 2

    ):

        raise ValueError(

            "Passband file must contain exactly "
            "two columns: wavelength and throughput."

        )

    if data.shape[0] < 2:

        raise ValueError(

            "Passband file must contain at least "
            "two wavelength samples."

        )

    # ======================================================
    # Construct validated passband
    # ======================================================

    return PhotometricPassband(

        wavelength=data[

            :,

            0

        ],

        throughput=data[

            :,

            1

        ],

        name=name

    )