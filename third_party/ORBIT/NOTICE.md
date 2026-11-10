# Third-party material from ORBIT and ORBIT3 (A. Tokovinin)

The following input files used by PySeshat's validation suite are example
data distributed with ORBIT, "IDL software for visual, spectroscopic, and
combined orbits", by Andrei Tokovinin (Zenodo, v1, 2016,
doi:10.5281/zenodo.61119), which is released under the
**GNU General Public License, version 2 only**:

| File (in `validation/benchmark_systems/`) | Origin |
|---|---|
| `FIN379/FIN379.inp` | ORBIT v1 example (Zenodo 61119) |
| `GL765/GL765.inp` | ORBIT v1 example (Zenodo 61119) |
| `ZetaAqr/ZetaAqr.in2` | ORBIT v1 example (Zenodo 61119) |

These files are redistributed unmodified (apart from any line-ending
normalisation) under the terms of the GPL-2.0; the full licence text is in
`LICENSE-GPL-2.0.txt` in this directory. They are test data only and are not
imported by the PySeshat package. The remainder of PySeshat is licensed
under the MIT licence (see the top-level `LICENSE`).

PySeshat's orbit-fitting code is written as a Python implementation of the
published ORBIT/ORBIT3 formalism (see the references in the PySeshat paper and
`refs.bib`) and is not distributed with ORBIT's IDL source. Please cite
Tokovinin (2016, Zenodo, doi:10.5281/zenodo.61119) and the original method
papers when using these files.

## ORBIT3 example file (CC BY 4.0)

`validation/benchmark_systems/HIP111805/HIP111805.in3` is the example input
file distributed with "ORBIT3 - orbits of triple stars" by Andrei Tokovinin
(Zenodo, v1, 2017, doi:10.5281/zenodo.321854), which is licensed under the
**Creative Commons Attribution 4.0 International** licence
(https://creativecommons.org/licenses/by/4.0/). Attribution is given here and
in the PySeshat paper. Any change to the file other than line endings is
noted in the file header. The ORBIT3 code itself is described in Tokovinin &
Latham (2017, ApJ 844, 103); the HIP 111805 solution in the file is the one published in Tokovinin &
Latham (2017, ApJ 838, 54, Table 2).
