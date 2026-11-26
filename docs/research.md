# Research synthesis

Date: 2026-10-25. Scope: what is needed to port PySAL faithfully to Elixir + Rust.

## 1. PySAL ecosystem

Meta-package PySAL 26.7 ([PyPI](https://pypi.org/pypi/pysal/json)).

| Group | Package | Version | spatex relevance |
|---|---|---|---|
| lib | libpysal (weights, Graph) | 4.15.1 | M0-M1 |
| explore | esda | 2.10.0 | M2-M3 |
| explore | giddy | 2.3.9 | M8 |
| explore | pointpats | 2.6.0 | M6 |
| explore | inequality | - | M8 |
| explore | segregation, spaghetti, momepy, access | - | out of scope |
| model | spreg | 1.9.1 | M7 |
| model | mgwr | 2.2.1 | M9 |
| model | spopt | 0.8.0 | M9 |
| model | spglm, spint, spml, tobler | - | out of scope |
| viz | mapclassify | 2.11.0 | M4 |
| viz | splot | - | out of scope |

## 2. Graph model

Source: [libpysal/graph/base.py](https://github.com/pysal/libpysal/blob/main/libpysal/graph/base.py); [W to Graph migration guide](https://pysal.org/libpysal/dev/user-guide/graph/w_g_migration.html). `Graph` replaces the legacy `W`.

- Storage: adjacency Series indexed by `(focal, neighbor)`, sorted by focal order. Every node appears at least once; islands are zero-weight self-loops.
- Constructors: `from_arrays`, `from_adjacency`, `from_sparse`, `from_dicts`.
- Builders: `build_contiguity` (rook / queen / by_perimeter), `build_knn`, `build_kernel`, `build_distance_band`, `build_block_contiguity`, `build_fuzzy_contiguity`, `build_triangulation` (delaunay, gabriel, relative_neighborhood, voronoi).
- Transforms:

| Code | Name | Rule |
|---|---|---|
| R | row-standardize | w / row sum (isolates stay 0) |
| D | double-standardize | w / global sum |
| B | binary | nonzero becomes 1 |
| V | variance-stabilizing | row-normalize by sqrt(sum w^2), then scale by n / sum(all) |

## 3. Algorithm details for a faithful port

### 3.1 Contiguity hashing

Source: `libpysal/graph/_contiguity.py` (`_vertex_set_intersection`).

- Queen: hash map from coordinate tuple to the set of geometries containing it.
- Rook: hash map from sorted vertex-pair edge to the set of geometries.
- Every set with size >= 2 becomes a clique of neighbors.
- Multipolygon phantom edges (between parts of the same geometry) are skipped.
- Rust: hash exact f64 bit patterns (no tolerance), matching PySAL's exact-coordinate behavior.

### 3.2 Global Moran's I

Source: [esda/moran.py](https://github.com/pysal/esda/blob/main/esda/moran.py). With z = y - mean(y), weights W, S0/S1/S2 the usual weight sums:

- I = n / S0 * sum_i z_i (Wz)_i / sum z^2
- E[I] = -1 / (n - 1)
- Normality: VI_norm = (n^2 S1 - n S2 + 3 S0^2) / ((n-1)(n+1) S0^2) - E[I]^2
- Randomization, with k = m4 / m2^2:
  - A = n ((n^2 - 3n + 3) S1 - n S2 + 3 S0^2)
  - B = k ((n^2 - n) S1 - 2 n S2 + 6 S0^2)
  - VI_rand = (A - B) / ((n-1)(n-2)(n-3) S0^2) - E[I]^2
- Analytic p-value: normal survival function if z > 0, else CDF; doubled when `two_tailed`.
- Permutations (default 999): `EI_sim` = mean of sims; `seI_sim` = std with ddof=0; `z_sim = (I - EI_sim) / seI_sim`.
- `p_sim = (larger + 1) / (P + 1)`, `larger` = count of sims >= observed, flipped to `P - larger` when that is smaller than `larger`.
- Default transform: `"r"` (row-standardized).

### 3.3 Local Moran

Sources: esda/moran.py, `esda/crand.py`.

- z standardized with ddof=0; I_i = (n - 1) z_i (Wz)_i / sum z^2.
- Quadrants: 1 = HH, 2 = LH, 3 = LL, 4 = HL (`geoda_quads` swaps 2 and 3).
- Conditional randomization: hold z_i fixed, draw card(i) of the other n - 1 values without replacement, recompute the local statistic.
- Islands: lag 0, I = 0.

### 3.4 Pseudo p-value alternatives

Source: `esda/significance.py`.

| Alternative | Rule |
|---|---|
| `:directed` (default) | folded: tail chosen by sign of the statistic relative to simulated mean |
| `:two_sided` | min tail x 2, capped at 1 |
| `:greater` | upper tail |
| `:lesser` | lower tail |

All use the (count + 1) / (P + 1) convention.

## 4. Rust crate choices

Versions from crates.io as of 2026-10.

| Need | Crate | Version | Rationale |
|---|---|---|---|
| NIF binding | rustler | 0.39.0 | Same as GeoFrame; Mix package `rustler ~> 0.38`, with `rustler_precompiled` 0.10.1 for the precompiled release (M10) |
| Geometry | geo + wkb | 0.33.1 | WKB polygon parsing for contiguity; geometry types and predicates |
| Spatial index | rstar | 0.13.0 | R-tree for distance band / kNN fallback |
| Spatial index | kiddo | 6.3.0 | Fast kd-tree kNN |
| Triangulation | spade | - | Delaunay (Gabriel / RNG derived from it) |
| Linear algebra | faer | 0.24.4 | Pure Rust dense and sparse; no system BLAS or LAPACK, so precompiled NIFs stay portable. Preferred over `ndarray-linalg` (needs system BLAS/LAPACK) |
| Linear algebra (alt.) | nalgebra | 0.35.0 | Considered; weaker sparse and large-matrix performance |
| Sparse (alt.) | sprs | 0.11.6 | Considered; CSR is simple enough to own |
| RNG | rand + rand_chacha | 0.10.3 / 0.10.0 | `ChaCha` with `seed_from_u64`; reproducible across platforms and versions |
| Parallelism | rayon | 1.12.0 | Permutation loops; per-site / per-chunk derived streams give thread-count-independent results |
| Distributions | statrs | 0.19.1 | Normal CDF/SF, t, chi-square, F |
| Optimization | argmin | 0.11.0 | Brent for ML rho (or hand-rolled Brent to avoid the dependency) |
| Graph algorithms | petgraph | 0.8.3 | Connected components, MST (SKATER) |

Crates marked "-" had no version pinned when this synthesis was written.

## 5. Elixir ecosystem gap

- Hex has `ex_rstar` (spatial index) and an image-only `moransi`; no ESDA, weights, or spatial regression library.
- Rust: `wbspatialstats`, `mermin-stats`, `scirs2-spatial` exist but are unvetted and not a PySAL-grade ESDA suite.
- Conclusion: spatex fills a real gap, and the pure-Rust `spatex_core` can serve the wider Rust community.

## 6. Determinism and parity stance

- Same seed gives identical results across runs and thread counts.
- Bit-for-bit equality with numpy's RNG is not a goal.
- Deterministic statistics must match PySAL to 1e-9; permutation p-values are compared statistically.
