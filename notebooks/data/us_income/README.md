# US state per-capita income, 1929-2009

Per-capita personal income (current dollars) of the 48 contiguous US states for every year from
1929 to 2009, with state boundaries. Distributed with libpysal (`libpysal.examples`, dataset
`us_income`, BSD-3-Clause). Original source: Regional Economic Information System, Bureau of
Economic Analysis, U.S. Department of Commerce.

Coordinates are longitude / latitude (the shapefile has no `.prj`). Used by
`notebooks/us_income_case.livemd` and `notebooks/es/caso_ingresos_eeuu.livemd`.

| File | Contents |
|---|---|
| `us48.shp` / `.shx` / `.dbf` | state polygons; `STATE_NAME`, `STATE_ABBR`, `STATE_FIPS` (text), `SUB_REGION` (Census division) |
| `usjoin.csv` | `Name`, `STATE_FIPS` (integer) and one column per year, `1929` to `2009` |
