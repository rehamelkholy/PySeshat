"""
Castelli & Kurucz (2004) ATLAS9 (CK04) stellar-atmosphere grid.

Grid coverage (full grid, as downloaded to
``validation/atmosphere_models/`` -- confirmed complete by the
2026-09-02 grid-completeness audit, see ``docs/project_decisions.md``
and ``docs/guide/alwardat_sed.md``):

- Effective temperature: 3500-50000 K (76 grid points, non-uniform
  spacing).
- Surface gravity: log g = 0.0-5.0 cgs (0.5-dex steps).
- Metallicity: [M/H] in {-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, +0.2, +0.5}
  dex (608 FITS files total: 76 Teff points x 8 metallicities).

This package's documented default scope for general use is roughly
[Fe/H] = -1.0 to +0.5 dex (solar-neighborhood F/G/K dwarfs/subgiants,
the Al-Wardat method's target population) -- comfortably inside the
downloaded grid's actual range, which extends further to -2.5 dex.

``AtmosphereGrid.interpolate()`` bilinearly interpolates in
(teff, logg) but selects metallicity **exactly** -- it is not
interpolated across metallicity, so a target star's [Fe/H] must be
rounded to the nearest available grid step before fitting (see
``load_ck04_grid()``, and the per-system ``atmosphere_files``/
``metallicity`` fields under ``validation/expected_results/`` for
worked examples, including cases where the nearest grid step is a
genuine tie between two adjacent metallicities).
"""

import numpy as np
import re

from dataclasses import dataclass
from pathlib import Path
from astropy.io import fits

@dataclass
class AtmosphereModel:
    """
    Single stellar-atmosphere spectrum.

    Parameters
    ----------
    wavelength : array_like
        Wavelength grid in Angstrom.

    flux : array_like
        Monochromatic model flux F_lambda in
        erg s^-1 cm^-2 Angstrom^-1.

    teff : float
        Effective temperature in Kelvin.

    logg : float
        Base-10 logarithm of the surface gravity
        in cgs units.

    metallicity : float, default=0.0
        Metallicity [M/H] relative to solar.
    """

    wavelength: np.ndarray

    flux: np.ndarray

    teff: float

    logg: float

    metallicity: float = 0.0

    def __post_init__(
        self
    ):
        """
        Validate and normalize the atmosphere data.
        """

        # --------------------------------------------------
        # Convert spectral data to NumPy arrays
        # --------------------------------------------------

        self.wavelength = np.asarray(

            self.wavelength,

            dtype=float

        )

        self.flux = np.asarray(

            self.flux,

            dtype=float

        )

        # --------------------------------------------------
        # Normalize scalar parameters
        # --------------------------------------------------

        self.teff = float(

            self.teff

        )

        self.logg = float(

            self.logg

        )

        self.metallicity = float(

            self.metallicity

        )

        # --------------------------------------------------
        # Validate dimensions
        # --------------------------------------------------

        if self.wavelength.ndim != 1:

            raise ValueError(

                "wavelength must be a "
                "one-dimensional array."

            )

        if self.flux.ndim != 1:

            raise ValueError(

                "flux must be a "
                "one-dimensional array."

            )

        # --------------------------------------------------
        # Validate array lengths
        # --------------------------------------------------

        if (

            len(
                self.wavelength
            )

            !=

            len(
                self.flux
            )

        ):

            raise ValueError(

                "wavelength and flux must "
                "have the same length."

            )

        if len(
            self.wavelength
        ) < 2:

            raise ValueError(

                "An atmosphere model must "
                "contain at least two "
                "wavelength samples."

            )

        # --------------------------------------------------
        # Validate finite values
        # --------------------------------------------------

        if not np.all(

            np.isfinite(
                self.wavelength
            )

        ):

            raise ValueError(

                "wavelength must contain "
                "only finite values."

            )

        if not np.all(

            np.isfinite(
                self.flux
            )

        ):

            raise ValueError(

                "flux must contain only "
                "finite values."

            )

        # --------------------------------------------------
        # Validate wavelength grid
        # --------------------------------------------------

        if np.any(

            self.wavelength <= 0.0

        ):

            raise ValueError(

                "All wavelengths must "
                "be positive."

            )

        if np.any(

            np.diff(
                self.wavelength
            )

            <= 0.0

        ):

            raise ValueError(

                "wavelength must be "
                "strictly increasing."

            )

        # --------------------------------------------------
        # Validate flux
        # --------------------------------------------------

        if np.any(

            self.flux < 0.0

        ):

            raise ValueError(

                "flux must not contain "
                "negative values."

            )

        # --------------------------------------------------
        # Validate atmosphere parameters
        # --------------------------------------------------

        if (

            not np.isfinite(
                self.teff
            )

            or

            self.teff <= 0.0

        ):

            raise ValueError(

                "teff must be a positive "
                "finite value."

            )

        if not np.isfinite(

            self.logg

        ):

            raise ValueError(

                "logg must be finite."

            )

        if not np.isfinite(

            self.metallicity

        ):

            raise ValueError(

                "metallicity must be finite."

            )

@dataclass(frozen=True)
class CK04CatalogEntry:
    """
    Single entry in an STScI CK04 atmosphere catalog.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.

    metallicity : float
        Metallicity [M/H] relative to solar.

    logg : float
        Base-10 surface gravity in cgs units.

    relative_path : str
        Relative path to the CK04 FITS file.

    column_name : str
        FITS-table column containing the atmosphere
        spectrum.
    """

    teff: float

    metallicity: float

    logg: float

    relative_path: str

    column_name: str

    def __post_init__(
        self
    ):
        """
        Validate and normalize the catalog entry.
        """

        object.__setattr__(

            self,

            "teff",

            float(
                self.teff
            )

        )

        object.__setattr__(

            self,

            "metallicity",

            float(
                self.metallicity
            )

        )

        object.__setattr__(

            self,

            "logg",

            float(
                self.logg
            )

        )

        object.__setattr__(

            self,

            "relative_path",

            str(
                self.relative_path
            ).strip()

        )

        object.__setattr__(

            self,

            "column_name",

            str(
                self.column_name
            ).strip()

        )

        if (

            not np.isfinite(
                self.teff
            )

            or

            self.teff <= 0.0

        ):

            raise ValueError(

                "teff must be a positive "
                "finite value."

            )

        if not np.isfinite(

            self.metallicity

        ):

            raise ValueError(

                "metallicity must be finite."

            )

        if not np.isfinite(

            self.logg

        ):

            raise ValueError(

                "logg must be finite."

            )

        if not self.relative_path:

            raise ValueError(

                "relative_path must not be empty."

            )

        if not re.fullmatch(

            r"g\d{2}",

            self.column_name

        ):

            raise ValueError(

                "column_name must follow the "
                "CK04 gravity-column convention "
                "'gXX'."

            )

    @property
    def parameter_point(
        self
    ):
        """
        Atmosphere-grid parameter point.

        Returns
        -------
        tuple
            Parameters in the order

                (teff, logg, metallicity)
        """

        return (

            self.teff,

            self.logg,

            self.metallicity

        )

class AtmosphereGrid:
    """
    Collection of stellar-atmosphere models indexed by
    effective temperature, surface gravity, and metallicity.

    Parameters
    ----------
    models : sequence of AtmosphereModel, optional
        Atmosphere models used to initialize the grid.

    Notes
    -----
    Each atmosphere model is uniquely identified by

        (teff, logg, metallicity)

    Exact model retrieval is supported at this stage.
    Interpolation between grid points is implemented
    separately.
    """

    def __init__(
        self,
        models=None
    ):
        """
        Initialize an atmosphere-model grid.
        """

        self._models = {}

        if models is not None:

            for model in models:

                self.add_model(
                    model
                )

    # ======================================================
    # Internal grid key
    # ======================================================

    @staticmethod
    def _model_key(
        teff,
        logg,
        metallicity
    ):
        """
        Construct the parameter key for an atmosphere model.

        Parameters
        ----------
        teff : float
            Effective temperature in Kelvin.

        logg : float
            Base-10 surface gravity in cgs units.

        metallicity : float
            Metallicity [M/H] relative to solar.

        Returns
        -------
        tuple
            Model-grid key.
        """

        return (

            float(
                teff
            ),

            float(
                logg
            ),

            float(
                metallicity
            )

        )

        # ======================================================
    # Find interpolation bounds
    # ======================================================

    @staticmethod
    def _bracket_value(
        values,
        target,
        name
    ):
        """
        Find the grid values bracketing a target value.

        Parameters
        ----------
        values : array_like
            Available coordinate values.

        target : float
            Requested coordinate value.

        name : str
            Coordinate name used in error messages.

        Returns
        -------
        lower : float
            Lower bounding grid value.

        upper : float
            Upper bounding grid value.

        Raises
        ------
        ValueError
            If the requested value lies outside the
            available coordinate range.
        """

        values = np.asarray(

            values,

            dtype=float

        )

        target = float(

            target

        )

        if values.ndim != 1:

            raise ValueError(

                f"{name} grid values must be "
                "one-dimensional."

            )

        if len(
            values
        ) == 0:

            raise ValueError(

                f"No {name} grid values are available."

            )

        values = np.unique(

            values

        )

        if (

            target < values[0]

            or

            target > values[-1]

        ):

            raise ValueError(

                f"Requested {name}={target} lies outside "
                "the available grid range "

                f"[{values[0]}, {values[-1]}]."

            )

        exact = values[

            np.isclose(

                values,

                target,

                rtol=0.0,

                atol=1.0e-12

            )

        ]

        if len(
            exact
        ) > 0:

            value = float(

                exact[0]

            )

            return (

                value,

                value

            )

        upper_index = int(

            np.searchsorted(

                values,

                target,

                side="right"

            )

        )

        lower = float(

            values[
                upper_index - 1
            ]

        )

        upper = float(

            values[
                upper_index
            ]

        )

        return (

            lower,

            upper

        )

    # ======================================================
    # Add atmosphere model
    # ======================================================

    def add_model(
        self,
        model
    ):
        """
        Add an atmosphere model to the grid.

        Parameters
        ----------
        model : AtmosphereModel
            Atmosphere spectrum to add.

        Raises
        ------
        TypeError
            If model is not an AtmosphereModel.

        ValueError
            If a model with the same atmosphere parameters
            is already present.
        """

        if not isinstance(

            model,

            AtmosphereModel

        ):

            raise TypeError(

                "model must be an "
                "AtmosphereModel instance."

            )

        key = self._model_key(

            model.teff,

            model.logg,

            model.metallicity

        )

        if key in self._models:

            raise ValueError(

                "An atmosphere model already exists "
                "at "

                f"teff={model.teff}, "

                f"logg={model.logg}, "

                f"metallicity={model.metallicity}."

            )

        self._models[
            key
        ] = model

    # ======================================================
    # Retrieve exact atmosphere model
    # ======================================================

    def get_model(
        self,
        teff,
        logg,
        metallicity=0.0
    ):
        """
        Retrieve an atmosphere model at an exact grid point.

        Parameters
        ----------
        teff : float
            Effective temperature in Kelvin.

        logg : float
            Base-10 surface gravity in cgs units.

        metallicity : float, default=0.0
            Metallicity [M/H] relative to solar.

        Returns
        -------
        AtmosphereModel
            Atmosphere model at the requested grid point.

        Raises
        ------
        KeyError
            If the requested grid point is unavailable.
        """

        key = self._model_key(

            teff,

            logg,

            metallicity

        )

        try:

            return self._models[
                key
            ]

        except KeyError as error:

            raise KeyError(

                "No atmosphere model is available "
                "at "

                f"teff={float(teff)}, "

                f"logg={float(logg)}, "

                f"metallicity={float(metallicity)}."

            ) from error

        # ======================================================
    # Interpolate atmosphere model
    # ======================================================

    def interpolate(
        self,
        teff,
        logg,
        metallicity=0.0
    ):
        """
        Interpolate an atmosphere spectrum in effective
        temperature and surface gravity.

        Metallicity is selected exactly and is not
        interpolated.

        Parameters
        ----------
        teff : float
            Requested effective temperature in Kelvin.

        logg : float
            Requested base-10 surface gravity in cgs units.

        metallicity : float, default=0.0
            Required metallicity [M/H].

        Returns
        -------
        AtmosphereModel
            Atmosphere spectrum at the requested parameters.

        Raises
        ------
        ValueError
            If the metallicity is unavailable, the requested
            temperature or gravity lies outside the grid, or
            the required interpolation cell is incomplete.

        Notes
        -----
        Bilinear interpolation is performed wavelength by
        wavelength in

            (teff, logg).

        If one or both requested coordinates lie exactly on
        grid points, the interpolation reduces to linear
        interpolation or exact retrieval.
        """

        # ==================================================
        # Normalize requested parameters
        # ==================================================

        teff = float(

            teff

        )

        logg = float(

            logg

        )

        metallicity = float(

            metallicity

        )

        if (

            not np.isfinite(
                teff
            )

            or

            teff <= 0.0

        ):

            raise ValueError(

                "teff must be a positive finite value."

            )

        if not np.isfinite(

            logg

        ):

            raise ValueError(

                "logg must be finite."

            )

        if not np.isfinite(

            metallicity

        ):

            raise ValueError(

                "metallicity must be finite."

            )

        # ==================================================
        # Select exact metallicity slice
        # ==================================================

        metallicity_matches = [

            value

            for value in self.metallicities

            if np.isclose(

                value,

                metallicity,

                rtol=0.0,

                atol=1.0e-12

            )

        ]

        if not metallicity_matches:

            raise ValueError(

                "Requested metallicity "

                f"{metallicity} is unavailable. "

                "Available metallicities are "

                f"{self.metallicities.tolist()}."

            )

        selected_metallicity = float(

            metallicity_matches[0]

        )

        # ==================================================
        # Determine coordinates available in this slice
        # ==================================================

        points = [

            point

            for point in self.parameter_points

            if np.isclose(

                point[2],

                selected_metallicity,

                rtol=0.0,

                atol=1.0e-12

            )

        ]

        temperatures = np.asarray(

            sorted({

                point[0]

                for point in points

            }),

            dtype=float

        )

        gravities = np.asarray(

            sorted({

                point[1]

                for point in points

            }),

            dtype=float

        )

        lower_teff, upper_teff = (

            self._bracket_value(

                temperatures,

                teff,

                "teff"

            )

        )

        lower_logg, upper_logg = (

            self._bracket_value(

                gravities,

                logg,

                "logg"

            )

        )

        # ==================================================
        # Retrieve required interpolation corners
        # ==================================================

        required_points = {

            (

                lower_teff,

                lower_logg,

                selected_metallicity

            ),

            (

                lower_teff,

                upper_logg,

                selected_metallicity

            ),

            (

                upper_teff,

                lower_logg,

                selected_metallicity

            ),

            (

                upper_teff,

                upper_logg,

                selected_metallicity

            )

        }

        missing_points = sorted(

            point

            for point in required_points

            if point not in self

        )

        if missing_points:

            raise ValueError(

                "The atmosphere interpolation cell is "
                "incomplete. Missing grid point(s): "

                + ", ".join(

                    str(
                        point
                    )

                    for point in missing_points

                )

            )

        models = {

            point:

            self.get_model(

                teff=point[0],

                logg=point[1],

                metallicity=point[2]

            )

            for point in required_points

        }

        # ==================================================
        # Validate common wavelength grid
        # ==================================================

        reference_model = next(

            iter(
                models.values()
            )

        )

        wavelength = (

            reference_model
            .wavelength
            .copy()

        )

        for model in models.values():

            if not np.array_equal(

                model.wavelength,

                wavelength

            ):

                raise ValueError(

                    "Atmosphere models required for "
                    "interpolation do not share an "
                    "identical wavelength grid."

                )

        # ==================================================
        # Exact grid-point retrieval
        # ==================================================

        if (

            lower_teff == upper_teff

            and

            lower_logg == upper_logg

        ):

            flux = (

                models[

                    (

                        lower_teff,

                        lower_logg,

                        selected_metallicity

                    )

                ]

                .flux

                .copy()

            )

        # ==================================================
        # Linear interpolation in surface gravity
        # ==================================================

        elif lower_teff == upper_teff:

            lower_model = models[

                (

                    lower_teff,

                    lower_logg,

                    selected_metallicity

                )

            ]

            upper_model = models[

                (

                    lower_teff,

                    upper_logg,

                    selected_metallicity

                )

            ]

            gravity_weight = (

                (

                    logg

                    -

                    lower_logg

                )

                /

                (

                    upper_logg

                    -

                    lower_logg

                )

            )

            flux = (

                (

                    1.0

                    -

                    gravity_weight

                )

                *

                lower_model.flux

                +

                gravity_weight

                *

                upper_model.flux

            )

        # ==================================================
        # Linear interpolation in effective temperature
        # ==================================================

        elif lower_logg == upper_logg:

            lower_model = models[

                (

                    lower_teff,

                    lower_logg,

                    selected_metallicity

                )

            ]

            upper_model = models[

                (

                    upper_teff,

                    lower_logg,

                    selected_metallicity

                )

            ]

            temperature_weight = (

                (

                    teff

                    -

                    lower_teff

                )

                /

                (

                    upper_teff

                    -

                    lower_teff

                )

            )

            flux = (

                (

                    1.0

                    -

                    temperature_weight

                )

                *

                lower_model.flux

                +

                temperature_weight

                *

                upper_model.flux

            )

        # ==================================================
        # Bilinear interpolation
        # ==================================================

        else:

            temperature_weight = (

                (

                    teff

                    -

                    lower_teff

                )

                /

                (

                    upper_teff

                    -

                    lower_teff

                )

            )

            gravity_weight = (

                (

                    logg

                    -

                    lower_logg

                )

                /

                (

                    upper_logg

                    -

                    lower_logg

                )

            )

            flux_00 = models[

                (

                    lower_teff,

                    lower_logg,

                    selected_metallicity

                )

            ].flux

            flux_01 = models[

                (

                    lower_teff,

                    upper_logg,

                    selected_metallicity

                )

            ].flux

            flux_10 = models[

                (

                    upper_teff,

                    lower_logg,

                    selected_metallicity

                )

            ].flux

            flux_11 = models[

                (

                    upper_teff,

                    upper_logg,

                    selected_metallicity

                )

            ].flux

            flux = (

                (

                    1.0

                    -

                    temperature_weight

                )

                *

                (

                    1.0

                    -

                    gravity_weight

                )

                *

                flux_00

                +

                (

                    1.0

                    -

                    temperature_weight

                )

                *

                gravity_weight

                *

                flux_01

                +

                temperature_weight

                *

                (

                    1.0

                    -

                    gravity_weight

                )

                *

                flux_10

                +

                temperature_weight

                *

                gravity_weight

                *

                flux_11

            )

        # ==================================================
        # Construct interpolated atmosphere model
        # ==================================================

        return AtmosphereModel(

            wavelength=wavelength,

            flux=flux,

            teff=teff,

            logg=logg,

            metallicity=selected_metallicity

        )

    # ======================================================
    # Available parameter values
    # ======================================================

    @property
    def temperatures(
        self
    ):
        """
        Sorted effective temperatures in the grid.
        """

        return np.asarray(

            sorted({

                key[0]

                for key in self._models

            }),

            dtype=float

        )

    @property
    def surface_gravities(
        self
    ):
        """
        Sorted surface gravities in the grid.
        """

        return np.asarray(

            sorted({

                key[1]

                for key in self._models

            }),

            dtype=float

        )

    @property
    def metallicities(
        self
    ):
        """
        Sorted metallicities in the grid.
        """

        return np.asarray(

            sorted({

                key[2]

                for key in self._models

            }),

            dtype=float

        )

    # ======================================================
    # Grid information
    # ======================================================

    @property
    def parameter_points(
        self
    ):
        """
        Sorted atmosphere-parameter points in the grid.

        Returns
        -------
        tuple of tuple
            Grid points in the form

                (teff, logg, metallicity)
        """

        return tuple(

            sorted(
                self._models
            )

        )

    def __len__(
        self
    ):
        """
        Return the number of atmosphere models.
        """

        return len(
            self._models
        )

    def __contains__(
        self,
        parameters
    ):
        """
        Test whether an exact parameter point is available.

        Parameters
        ----------
        parameters : tuple
            Atmosphere parameters in the form

                (teff, logg, metallicity)
        """

        if (

            not isinstance(
                parameters,
                tuple
            )

            or

            len(
                parameters
            ) != 3

        ):

            return False

        key = self._model_key(

            *parameters

        )

        return key in self._models

def load_ck04_fits(
    filename
):
    """
    Load Castelli & Kurucz (2004) ATLAS9 atmosphere
    models from an STScI CK04 FITS table.

    A CK04 FITS file contains one effective temperature
    and one metallicity. Individual table columns contain
    atmosphere spectra at different surface gravities.

    Parameters
    ----------
    filename : str or path-like
        Path to an STScI CK04 atmosphere FITS file.

    Returns
    -------
    AtmosphereGrid
        Grid containing one AtmosphereModel for every
        available surface-gravity column.

    Raises
    ------
    FileNotFoundError
        If the requested FITS file does not exist.

    ValueError
        If required atmosphere metadata, the wavelength
        column, or valid gravity spectra are unavailable.
    """

    # ======================================================
    # Validate path
    # ======================================================

    path = Path(
        filename
    )

    if not path.is_file():

        raise FileNotFoundError(

            "CK04 atmosphere file does not exist: "

            f"{path}"

        )

    # ======================================================
    # Open FITS file
    # ======================================================

    with fits.open(
        path
    ) as hdul:

        if len(
            hdul
        ) < 2:

            raise ValueError(

                "The CK04 FITS file does not contain "
                "a binary-table extension."

            )

        primary_header = hdul[
            0
        ].header

        table_hdu = hdul[
            1
        ]

        table = table_hdu.data

        if table is None:

            raise ValueError(

                "The CK04 FITS table contains no data."

            )

        # ==================================================
        # Read atmosphere metadata
        # ==================================================

        if "TEFF" not in primary_header:

            raise ValueError(

                "The CK04 FITS header does not contain "
                "the TEFF keyword."

            )

        if "LOG_Z" not in primary_header:

            raise ValueError(

                "The CK04 FITS header does not contain "
                "the LOG_Z keyword."

            )

        teff = float(

            primary_header[
                "TEFF"
            ]

        )

        metallicity = float(

            primary_header[
                "LOG_Z"
            ]

        )

        # ==================================================
        # Identify table columns
        # ==================================================

        column_names = list(

            table.columns.names

        )

        wavelength_name = next(

            (

                name

                for name in column_names

                if name.lower()
                == "wavelength"

            ),

            None

        )

        if wavelength_name is None:

            raise ValueError(

                "The CK04 FITS table does not contain "
                "a wavelength column."

            )

        wavelength = np.asarray(

            table[
                wavelength_name
            ],

            dtype=float

        )

        # ==================================================
        # Construct atmosphere grid
        # ==================================================

        grid = AtmosphereGrid()

        for column_name in column_names:

            normalized_name = (
                column_name
                .strip()
                .lower()
            )

            if (

                len(
                    normalized_name
                ) != 3

                or

                not normalized_name.startswith(
                    "g"
                )

                or

                not normalized_name[
                    1:
                ].isdigit()

            ):

                continue

            flux = np.asarray(

                table[
                    column_name
                ],

                dtype=float

            )

            # CK04 uses all-zero columns for unavailable
            # temperature-gravity combinations.

            if np.all(

                flux == 0.0

            ):

                continue

            logg = (

                float(
                    normalized_name[
                        1:
                    ]
                )

                /

                10.0

            )

            model = AtmosphereModel(

                wavelength=(
                    wavelength.copy()
                ),

                flux=(
                    flux.copy()
                ),

                teff=teff,

                logg=logg,

                metallicity=metallicity

            )

            grid.add_model(
                model
            )

    # ======================================================
    # Validate loaded grid
    # ======================================================

    if len(
        grid
    ) == 0:

        raise ValueError(

            "The CK04 FITS file contains no "
            "available atmosphere spectra."

        )

    return grid

def load_ck04_grid(
    filenames
):
    """
    Load multiple CK04 atmosphere FITS files into one grid.

    Parameters
    ----------
    filenames : iterable of str or path-like
        Paths to STScI Castelli & Kurucz (2004)
        atmosphere FITS files.

    Returns
    -------
    AtmosphereGrid
        Combined atmosphere grid containing all available
        models from the supplied FITS files.

    Raises
    ------
    ValueError
        If no files are supplied, if duplicate atmosphere
        parameter points are encountered, or if the files
        do not use the same wavelength grid.

    Notes
    -----
    All CK04 spectra in a combined grid are required to use
    an identical wavelength grid. This ensures that later
    interpolation can be performed wavelength by wavelength
    without resampling the spectra.
    """

    # ======================================================
    # Normalize input paths
    # ======================================================

    try:

        paths = [

            Path(
                filename
            )

            for filename in filenames

        ]

    except TypeError as error:

        raise TypeError(

            "filenames must be an iterable of "
            "file paths."

        ) from error

    if not paths:

        raise ValueError(

            "At least one CK04 FITS file "
            "must be supplied."

        )

    # ======================================================
    # Initialize combined grid
    # ======================================================

    combined_grid = AtmosphereGrid()

    reference_wavelength = None

    # ======================================================
    # Load individual CK04 files
    # ======================================================

    for path in paths:

        file_grid = load_ck04_fits(

            path

        )

        # ==================================================
        # Add models to the combined grid
        # ==================================================

        for parameter_point in (

            file_grid.parameter_points

        ):

            model = file_grid.get_model(

                teff=parameter_point[0],

                logg=parameter_point[1],

                metallicity=parameter_point[2]

            )

            # ==============================================
            # Establish or validate wavelength grid
            # ==============================================

            if reference_wavelength is None:

                reference_wavelength = (

                    model.wavelength.copy()

                )

            else:

                same_shape = (

                    model.wavelength.shape

                    ==

                    reference_wavelength.shape

                )

                same_values = (

                    same_shape

                    and

                    np.array_equal(

                        model.wavelength,

                        reference_wavelength

                    )

                )

                if not same_values:

                    raise ValueError(

                        "CK04 atmosphere files do not "
                        "share an identical wavelength "
                        "grid. Incompatible file: "

                        f"{path}"

                    )

            # ==============================================
            # Add model
            # ==============================================

            try:

                combined_grid.add_model(

                    model

                )

            except ValueError as error:

                raise ValueError(

                    "Duplicate atmosphere parameter "
                    "point encountered while loading "

                    f"{path}: "

                    f"{parameter_point}."

                ) from error

    return combined_grid

def load_ck04_catalog(
    filename
):
    """
    Load an STScI CK04 atmosphere catalog.

    Parameters
    ----------
    filename : str or path-like
        Path to the CK04 catalog FITS file.

    Returns
    -------
    tuple of CK04CatalogEntry
        Catalog entries sorted by effective temperature,
        surface gravity, and metallicity.

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist.

    ValueError
        If the catalog structure or an entry is invalid.
    """

    # ======================================================
    # Validate path
    # ======================================================

    path = Path(
        filename
    )

    if not path.is_file():

        raise FileNotFoundError(

            "CK04 catalog file does not exist: "

            f"{path}"

        )

    # ======================================================
    # Read catalog table
    # ======================================================

    with fits.open(
        path
    ) as hdul:

        if len(
            hdul
        ) < 2:

            raise ValueError(

                "The CK04 catalog does not contain "
                "a binary-table extension."

            )

        table = hdul[
            1
        ].data

        if table is None:

            raise ValueError(

                "The CK04 catalog table contains "
                "no data."

            )

        column_names = {

            name.upper():

            name

            for name in table.columns.names

        }

        required = {

            "INDEX",

            "FILENAME"

        }

        missing = (

            required

            -

            set(
                column_names
            )

        )

        if missing:

            raise ValueError(

                "The CK04 catalog is missing "
                "required column(s): "

                + ", ".join(
                    sorted(
                        missing
                    )
                )

            )

        index_column = column_names[
            "INDEX"
        ]

        filename_column = column_names[
            "FILENAME"
        ]

        entries = []

        # ==================================================
        # Parse catalog rows
        # ==================================================

        filename_pattern = re.compile(

            r"^(?P<path>.+\.fits)"

            r"\[(?P<column>g\d{2})\]$",

            flags=re.IGNORECASE

        )

        for row_number, row in enumerate(

            table

        ):

            index_text = str(

                row[
                    index_column
                ]

            ).strip()

            filename_text = str(

                row[
                    filename_column
                ]

            ).strip()

            values = [

                value.strip()

                for value in index_text.split(
                    ","
                )

            ]

            if len(
                values
            ) != 3:

                raise ValueError(

                    "Invalid CK04 INDEX value "
                    f"at row {row_number}: "

                    f"{index_text!r}"

                )

            try:

                teff = float(
                    values[0]
                )

                metallicity = float(
                    values[1]
                )

                logg = float(
                    values[2]
                )

            except ValueError as error:

                raise ValueError(

                    "Non-numeric CK04 INDEX value "
                    f"at row {row_number}: "

                    f"{index_text!r}"

                ) from error

            match = filename_pattern.fullmatch(

                filename_text

            )

            if match is None:

                raise ValueError(

                    "Invalid CK04 FILENAME value "
                    f"at row {row_number}: "

                    f"{filename_text!r}"

                )

            entry = CK04CatalogEntry(

                teff=teff,

                metallicity=metallicity,

                logg=logg,

                relative_path=match.group(
                    "path"
                ),

                column_name=match.group(
                    "column"
                ).lower()

            )

            entries.append(
                entry
            )

    # ======================================================
    # Validate catalog
    # ======================================================

    if not entries:

        raise ValueError(

            "The CK04 catalog contains no entries."

        )

    parameter_points = [

        entry.parameter_point

        for entry in entries

    ]

    if (

        len(
            parameter_points
        )

        !=

        len(
            set(
                parameter_points
            )
        )

    ):

        raise ValueError(

            "The CK04 catalog contains duplicate "
            "atmosphere parameter points."

        )

    # ======================================================
    # Return deterministic ordering
    # ======================================================

    return tuple(

        sorted(

            entries,

            key=lambda entry: (

                entry.teff,

                entry.logg,

                entry.metallicity

            )

        )

    )