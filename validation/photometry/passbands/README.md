# Passband sensitivity curves

Each file has two columns, wavelength (Angstrom) and normalized sensitivity,
and a header naming its source and its detector type (`# detector: photon` or
`# detector: energy`), which `pyseshat.photometry.filters.load_passband` reads.
Photon-counting curves are integrated with the lambda weighting of
Maíz Apellániz (2006), Eq. (A1).

| File | Band | Source | Detector |
|---|---|---|---|
| `johnson_u.dat` | Johnson U | Maíz Apellániz (2006), Table 3 (new U curve) | photon |
| `johnson_b.dat` | Johnson B | Maíz Apellániz (2006), Table 3 (Bessell 1990 B, converted) | photon |
| `johnson_v.dat` | Johnson V | Maíz Apellániz (2006), Table 3 (Bessell 1990 V, converted) | photon |
| `cousins_r.dat` | Cousins R | Bessell (1983), Table AII | energy |
| `stromgren_u.dat` | Strömgren u | Maíz Apellániz (2006), Table 2 (new u curve) | photon |
| `stromgren_v.dat` | Strömgren v | Maíz Apellániz (2006), Table 2 (Matsushima 1969, converted) | photon |
| `stromgren_b.dat` | Strömgren b | Maíz Apellániz (2006), Table 2 (Matsushima 1969, converted) | photon |
| `stromgren_y.dat` | Strömgren y | Maíz Apellániz (2006), Table 2 (Matsushima 1969, converted) | photon |
| `tycho_bt.dat` | Tycho BT | Maíz Apellániz (2006), Table 1 (Bessell 2000, converted) | photon |
| `tycho_vt.dat` | Tycho VT | Maíz Apellániz (2006), Table 1 (Bessell 2000, converted) | photon |

Values are transcribed verbatim from the printed tables. The Tycho VT table
ends at 6600 Å with P = 0.028; the Bessell (1983) R table ends at 865 nm with
0.004. Neither is extended.

These are the curves for which the Vega magnitudes in
`pyseshat.photometry.calibration.VEGA_STANDARD_MAGNITUDES` (Maíz Apellániz
2007, Table 4) are defined: that table cites Maíz Apellániz (2006) for Johnson
UBV and Strömgren uvby and Bessell (1983) for Cousins R, and notes that the
Bessell (1983) curves are energy-integrating.

Changed in 0.2.0: the previous files for Johnson U, B, V, Strömgren u, v, b, y
and Tycho VT were not these curves and their origin was not recorded; all
curves were integrated as energy-integrating; and the R band was called
"Johnson R" with a curve of unknown origin. Tycho BT was already the Table 1
curve.

## References

- Bessell, M. S. 1983, PASP, 95, 480
- Bessell, M. S. 1990, PASP, 102, 1181
- Bessell, M. S. 2000, PASP, 112, 961
- Maíz Apellániz, J. 2006, AJ, 131, 1184, doi:10.1086/499158
- Maíz Apellániz, J. 2007, in The Future of Photometric, Spectrophotometric
  and Polarimetric Standardization, ed. C. Sterken, ASP Conf. Ser.
  (arXiv:astro-ph/0609430)
- Matsushima, S. 1969, ApJ, 158, 1137
