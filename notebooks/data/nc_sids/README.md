# North Carolina SIDS, 1974-78 and 1979-84

Sudden infant death syndrome (SIDS) counts and live births for the 100 counties of North
Carolina, as distributed with libpysal (`libpysal.examples`, dataset `sids2`, BSD-3-Clause) and the
GeoDa Center (<https://geodacenter.github.io/data-and-lab/sids2/>).

Source: Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, pp. 386-389.

Coordinates are longitude / latitude (NAD27; the shapefile has no `.prj`). Used by
`notebooks/nc_sids_case.livemd` and `notebooks/es/caso_sids_carolina_del_norte.livemd`.

| Column | Meaning |
|---|---|
| `NAME`, `FIPS` | county name and FIPS code |
| `BIR74`, `SID74`, `NWBIR74` | live births, SIDS deaths, non-white live births, 1974-78 |
| `BIR79`, `SID79`, `NWBIR79` | the same for 1979-84 |
| `SIDR74`, `SIDR79` | SIDS deaths per 1,000 births |
| `NWR74`, `NWR79` | non-white births per 1,000 births |
