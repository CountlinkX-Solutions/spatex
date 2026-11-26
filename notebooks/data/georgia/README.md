# Georgia counties, 1990 census

Socio-economic variables for the 159 counties of Georgia in the 1990 census. Distributed with
libpysal (`libpysal.examples`, dataset `georgia`, BSD-3-Clause); the example of Fotheringham,
Brunsdon and Charlton (2002), *Geographically Weighted Regression*, Wiley, and of PySAL's `mgwr`.

Polygons in UTM (meters; the shapefile has no `.prj`); county centroids in `X`, `Y`. Used by
`notebooks/georgia_case.livemd` and `notebooks/es/caso_educacion_georgia.livemd`.

| Column | Meaning |
|---|---|
| `PctBach` | adults with a bachelor's degree, % |
| `PctFB`, `PctBlack`, `PctRural` | foreign-born, Black and rural population, % |
| `PctEld`, `PctPov` | elderly population and population in poverty, % |
| `TotPop90` | population, 1990 |
| `X`, `Y`, `Latitude`, `Longitud` | centroid in UTM and in degrees |
| `AreaKey` | county FIPS code |
