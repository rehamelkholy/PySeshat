# Catalogue photometry for the Capella and alpha Cen SED benchmarks

Input to `scripts/build_real_sed_benchmarks.py`, which writes the measurements
into `validation/expected_results/Capella_photometry.json` and
`AlphaCenAB_photometry.json`.

These VizieR query results are **not redistributed** with PySeshat. To rebuild
the benchmark files, download them into this folder under these names:

| File | Query |
|---|---|
| `HD34029_II168_ubvmeans.tsv` | https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=II/168/ubvmeans&-c=HD34029&-c.rs=60&-out.all |
| `HD128620_II168_ubvmeans.tsv` | https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=II/168/ubvmeans&-c=HD128620&-c.rs=30&-out.all |
| `HD34029_II215.tsv` | https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=II/215&-c=HD34029&-c.rs=60&-out.all |
| `HD128620_II215.tsv` | https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=II/215/catalog&HD=128620..128621&-out.all |

The alpha Cen II/168 query returns both HD 128620 (A) and HD 128621 (B). The
II/215 file for alpha Cen is not used: its only entry (HD 128620, flagged "AB")
has V = 0.000, the V of A alone, so it is unclear whether it refers to A or to
the pair.

Catalogues: Mermilliod, J.-C. 1991, Homogeneous Means in the UBV System
(VizieR II/168); Hauck, B., & Mermilliod, M. 1998, A&AS, 129, 431 (VizieR
II/215).

If you use these data, acknowledge VizieR: "This research has made use of the
VizieR catalogue access tool, CDS, Strasbourg, France (DOI: 10.26093/cds/vizier).
The original description of the VizieR service was published in Ochsenbein et
al. (2000, A&AS 143, 23)."
