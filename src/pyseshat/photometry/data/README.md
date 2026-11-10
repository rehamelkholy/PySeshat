# Photometric calibration data

`alpha_lyr_stis_003.fits` — CALSPEC spectrum of Vega (alpha Lyr) from
HST/STIS, version stis_003 (Bohlin 2007, ASP Conf. Ser., 364, 315; file
header: SOURCE = 'Bohlin 2007, ASP Conf. Series, Vol. xxx, ed. C.
Sterken'). It is the Vega spectrum for which the standard-system Vega
magnitudes of Maíz Apellániz (2007, ASP Conf. Ser., 364, 227, Table 4)
are defined, and is used by `pyseshat.photometry.calibration` to compute
zero points through PySeshat's own passband curves.

Source: STScI CALSPEC database
(https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec);
this copy was taken from the `pyphot` 2.1.1 distribution (`pyphot/libs/`),
which redistributes the CALSPEC file unchanged.
