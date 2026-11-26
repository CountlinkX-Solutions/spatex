# Baltimore house sales, 1978

211 house sales in Baltimore City and Baltimore County in 1978, with price (thousands of dollars),
house features and location. Distributed with libpysal (`libpysal.examples`, dataset `baltim`,
BSD-3-Clause). Source: Dubin, R. A. (1992). Spatial autocorrelation and neighborhood quality.
*Regional Science and Urban Economics* 22(3), 433-452.

Point geometry in a local planar coordinate system (also in `X`, `Y`). Used by
`notebooks/baltimore_case.livemd` and `notebooks/es/caso_vivienda_baltimore.livemd`.

| Column | Meaning |
|---|---|
| `PRICE` | sale price, thousands of dollars |
| `NROOM`, `NBATH`, `NSTOR` | rooms, bathrooms, stories |
| `PATIO`, `FIREPL`, `AC`, `BMENT`, `GAR` | patio, fireplace, air conditioning, basement, garage indicators |
| `DWELL` | detached dwelling indicator |
| `AGE`, `LOTSZ`, `SQFT` | age in years, lot size, living area |
| `CITCOU` | 1 for Baltimore County, 0 for Baltimore City |
