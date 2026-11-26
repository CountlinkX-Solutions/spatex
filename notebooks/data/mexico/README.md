# Mexican states, per-capita GDP 1940-2000

Per-capita gross domestic product of the 32 Mexican states every ten years from 1940 to 2000, with
regional schemes from the literature. Distributed with libpysal (`libpysal.examples`, dataset
`mexico`, BSD-3-Clause). Source: Rey, S. J. and Sastre Gutiérrez, M. L. (2010). Interregional
inequality dynamics in Mexico. *Spatial Economic Analysis* 5, 277-298.

Used by `notebooks/mexico_case.livemd` and `notebooks/es/caso_regiones_mexico.livemd`.

| Column | Meaning |
|---|---|
| `NAME` | state |
| `PCGDP1940` ... `PCGDP2000` | per-capita GDP by decade |
| `LPCGDP40` ... `LPCGDP00` | their logarithms |
| `INEGI`, `INEGI2` | regions of INEGI, the national statistics office |
| `HANSON98`, `HANSON03`, `ESQUIVEL99` | regional schemes of Hanson (1998, 2003) and Esquivel (1999) |
| `MAXP` | a Max-p regionalization from the source study |
