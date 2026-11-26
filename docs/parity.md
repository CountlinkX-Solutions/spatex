# PySAL parity matrix

Each row maps a PySAL function or class to its Spatex counterpart, the milestone that delivered it
(see the [roadmap](roadmap.md)) and its status: `available`, with notes on any difference, or not
planned or not ported, with the reason. The notes under each table record how the oracle tests
compare Spatex with PySAL and every known deviation.

## libpysal (weights / Graph)

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| Graph.build_contiguity | Spatex.Graph.build_contiguity/2 | M1 | available (vertex and strict methods, `by_perimeter`, lines) |
| Graph.build_knn | Spatex.Graph.build_knn/3 | M1 | available |
| Graph.build_distance_band | Spatex.Graph.build_distance_band/3 | M1 | available |
| Graph.build_kernel | Spatex.Graph.build_kernel/2 | M1 | available |
| Graph.build_triangulation | Spatex.Graph.build_triangulation/2 | M1 | available (`:voronoi` rook only) |
| Graph.build_block_contiguity | Spatex.Graph.build_block_contiguity/2 | M1 | available (keeps input node order) |
| Graph.from_arrays | Spatex.Graph.from_arrays/4 | M0 | available |
| Graph.from_adjacency | Spatex.Graph.from_adjacency/1 | M0 | available |
| Graph.transform | Spatex.Graph.transform/2 | M0 | available |
| Graph.lag | Spatex.Graph.lag/2 | M0 | available |
| Graph.cardinalities | Spatex.Graph.cardinalities/1 | M0 | available |
| Graph.isolates | Spatex.Graph.isolates/1 | M0 | available |
| Graph.n | Spatex.Graph.n/1 | M0 | available |
| Graph.s0 / s1 / s2 | Spatex.Graph.s0/1, s1/1, s2/1 | M0 | available |
| Graph.asymmetry / symmetric check | Spatex.Graph.symmetric?/1 | M0 | available |
| Graph.adjacency | Spatex.Graph.to_adjacency/1 | M0 | available |
| Graph.sparse | Spatex.Graph.to_sparse/1 | M0 | available |
| GAL / GWT read and write | Spatex.Graph.read_gal/2, write_gal/2, read_gwt/2, write_gwt/2, to_gal/1, to_gwt/1 (`Spatex.Graph.IO`, also `parse_gal/2`, `parse_gwt/2`) | M1 | available |
| Graph.build_fuzzy_contiguity | Spatex.Graph.build_fuzzy_contiguity/2 | M1 | available (`:tolerance`, `:buffer`, `:predicate`; exact distance instead of GEOS buffers, see Fuzzy contiguity) |

### Deviations: KNN and distance band

Metrics: `metric: :euclidean` (default) and `:haversine` (great-circle on `{lon, lat}` degrees, kilometres, Earth radius 6371 km) for `build_knn/3`, `build_distance_band/3` and `build_kernel/2`; libpysal's other scikit-learn metrics are not offered. Points go through `Spatex.Input`; graph ids default to `0..n-1` and `ids:` relabels them.

| Topic | libpysal 4.15.1 | Spatex |
|---|---|---|
| KNN ties among equidistant candidates | scikit-learn KD-tree order | lowest point index wins; fixtures with ties (`knn_grid_ties`) compare per-row neighbor distances |
| Coincident points (`coplanar=`) | `"raise"` (default), `"clique"`, `"jitter"`: `_jitter_geoms` adds `r + sin(theta)` (about one coordinate unit, a bug for `r * sin(theta)`) to *every* point, unseeded | `coincident: :raise` (default), `:clique` (same site semantics, including cardinality above `k`) and `:jitter`: seeded (`:seed`), moves only points of coincident groups by at most `max(abs(x), abs(y), 1) * 2^-30`, redrawing until distinct; not comparable to libpysal's random output |
| `k` out of range | scikit-learn `ValueError` | `ArgumentError` unless `1 <= k <= n - 1` (distinct locations minus one with `:clique`) |
| Distance band threshold | `d <= threshold` (scipy `sparse_distance_matrix`) | same, on squared distance `dx^2 + dy^2 <= threshold^2`; `threshold` must be positive |
| Inverse distance with coincident points | weight `inf` | `ArgumentError` (negative `alpha`) |
| Kernel weights over a distance band (`kernel=`, `bandwidth=`) | supported | supported with `binary: false, kernel: ..., bandwidth: ...` (see Kernel weights); `kernel:` with `binary: true` raises (libpysal ignores it) |
| Haversine distance unit | radians: `_knn`'s `d_linear_flat * 6371` is a no-op statement | kilometres everywhere (libpysal's stated intent); neighbor sets and data-driven kernel bandwidths are unit-free and match (`haversine_*` fixtures), fixed bandwidths and thresholds are km |
| Haversine kernel without `k` | degrees passed unflipped to `pairwise_distances(metric="haversine")`, which expects `(lat, lon)` radians: wrong distances | correct great-circle distances |
| Haversine distance band | no `metric` parameter | `metric: :haversine`, threshold in km |
| Haversine coordinate range | `-180 < lon < 180`, `-90 < lat < 90` (strict) or `ValueError` | same, `ArgumentError` naming the point |

### Kernel weights

`build_kernel(points, kernel: :gaussian, bandwidth: nil, k: nil, taper: true, decay: false,
coincident: :raise, metric: :euclidean, ids: nil)` follows `libpysal.graph.Graph.build_kernel`: the
nine libpysal kernels with their formulas, the 25th-percentile default bandwidth (numpy linear
interpolation), the largest k-NN distance as default with `k`, adaptive bandwidth
(`k`-th neighbor distance * 1.0000001 per row), `taper` and `decay`, and no self weights.
Verified on a line, columbus centroids and seeded random points with exact adjacency/order
and weights within 1e-12.

<a id="deviations-kernel-weights"></a>

| Topic | libpysal 4.15.1 | Spatex |
|---|---|---|
| Coincident pairs with a fixed numeric bandwidth | silently dropped by the sparse fast path (`eliminate_zeros`); kept on the dense default-bandwidth path | kept (distance 0) on every path (`known_deviations.exs`) |
| `bandwidth="auto"` (entropy optimisation) | supported | `ArgumentError` (unsupported) |
| Callable kernels, numeric `taper` threshold | supported | not supported: `kernel` is one of the libpysal names, `taper` is a boolean |
| `kernel` aliases | `discrete` = boxcar | also `:uniform`, `:epanechnikov` (= `:parabolic`); boxcar weight is 1 (`d < h`), not 1/2 |
| `decay` with the identity kernel | division by zero | `ArgumentError` |
| Non-positive or zero default bandwidth (many coincident points) | `nan` weights | `ArgumentError` naming the node |
| Memory | sparse fast path for compact kernels or `taper=True` with a numeric bandwidth | same candidate rule (rstar range query); the default bandwidth and `taper: false` with a non-compact kernel are O(n^2) like libpysal's dense path |

### Triangulation graphs

`build_triangulation(points, method: :delaunay, bandwidth: :infinity, kernel: :boxcar,
coincident: :raise, taper: true, decay: false, ids: nil)` follows
`libpysal.graph.Graph.build_triangulation`. The Delaunay triangulation comes from the `spade`
crate (pinned `=2.15.1`: pure Rust, exact robust predicates, MIT/Apache-2.0, no C/Qhull build
dependency). For points in general position the triangulation is unique, so edge sets equal
scipy's Qhull exactly; `:gabriel` and `:relative_neighborhood` are Delaunay subgraphs using
libpysal's own tests (Gabriel: strict, Delaunay neighbors of an endpoint only; relative
neighborhood: non-strict over all points). Verified on columbus centroids, 200 seeded random
points, a 7-point hand set and a clique case with exact adjacency/order and weights within 1e-12.

<a id="deviations-triangulation"></a>

| Topic | libpysal 4.15.1 | Spatex |
|---|---|---|
| Cocircular points (e.g. a square) | Qhull picks one diagonal | spade may pick the other; Gabriel/RNG follow it. Fixtures use general position |
| Node whose edges are all tapered away by a finite bandwidth | the node disappears from the graph | kept as an island (zero self-loop) (`known_deviations.exs`) |
| `method="voronoi"` | rook contiguity of GEOS Voronoi cells clipped to `clip` (vertex equality) | the Delaunay edges whose dual Voronoi edge has positive length inside the clip region (Cyrus-Beck); same graph without GEOS (`voronoi.json`: random and lattice points, all three clips). `clip: nil` reproduces GEOS's default extent (bounding box grown by its largest side); `:alpha_shape` and polygon clips and `rook: false` are not supported; weights are 1 as in libpysal (which ignores `kernel` there; Spatex applies a non-default one) |
| Collinear input or fewer than 3 distinct points | `QhullError` | `ArgumentError` |
| `coplanar="jitter"`, `bandwidth="auto"`, callable kernels | supported | not supported |
| Default weights | int8 `1` when `kernel=None`; float `1.0` via the default boxcar | float `1.0` |

### Contiguity

`build_contiguity(wkbs, rook: true, strict: false, ids: nil)` replicates libpysal's default
vertex-set intersection from WKB `Polygon`/`MultiPolygon` input, verified on columbus, sids2,
mexico and synthetic cases (holes, T-junctions, multipolygons, `-0.0`, islands) with exact
adjacency and order. Quirks reproduced on purpose: coordinates compare exactly (`-0.0 == 0.0`),
rook means an identical edge (a T-junction is not rook contiguity), no phantom edge between
multipolygon parts, but the step from an exterior ring to its first hole counts as an edge.

- `rook` defaults to `true`, as in libpysal's `Graph.build_contiguity`.
- `strict: true` reproduces shapely's `touches` (queen) and DE-9IM `F***1****` (rook) on
  polygons with exact orientation tests on the boundary segments, and `by_perimeter: true` the
  length of `intersection(geom_i, geom_j)` (0 for a point contact, which is stored but left out
  of the cardinalities, as in libpysal); overlapping neighbors raise. Verified on columbus and
  synthetic T-junction, corner, hole-island and overlap cases (`contiguity_strict.json`, every
  rook / strict / by_perimeter combination). Strict and perimeter contiguity need polygonal
  geometries; libpysal's float ids on the perimeter path are not reproduced.
- `LineString` / `MultiLineString` geometries use the vertex method, each line a part.
- NaN or infinite coordinates raise `ArgumentError`.

### Fuzzy contiguity

`build_fuzzy_contiguity(wkbs, tolerance: nil, buffer: nil, predicate: :intersects, ids: nil)`
follows libpysal's `_fuzzy_contiguity`: `tolerance` (a share of half the shorter side of the total
bounds, `buffer = tolerance * 0.5 * min(width, height)`) and `buffer` are exclusive, and the
pairs `(i, j)` with `predicate(geom_i, geom_j)` become binary edges; islands as elsewhere.
Neighbor sets equal libpysal's on columbus, sids2 and synthetic multipolygon, hole, nested,
equal and island cases for every `tolerance`, `buffer` and predicate below
(`contiguity_fuzzy.json`).

- **No buffer geometry.** Spatex has no GEOS. Two round buffers of radius `b` intersect exactly
  when the polygons are at most `2b` apart, so `:intersects` with a buffer is an exact
  polygon-to-polygon distance test (0 when the polygons intersect). GEOS draws arcs with 16
  chords per quadrant, so its buffers are up to 0.12 % smaller: a pair whose distance lies
  between about 0.9985 and 1 times `2b` can be a neighbor in Spatex and not in libpysal (never
  the reverse). Measured on 113 synthetic near-threshold pairs (`near_threshold`: edge, corner,
  triangle tip and disc pairs at 0.99 to 1.01 times `2b`): axis-aligned edges agree exactly;
  12 pairs (corners off the arc vertices, tips, discs), all between 0.999 and 0.9999 times
  `2b`, differ. Real data (columbus, sids2) agrees for every tested tolerance and buffer. Spatex
  documents the exact semantics and records the band in `known_deviations.exs`.
- **Predicates** are directed, `predicate(focal, neighbor)`, as in `sindex.query`:
  `:intersects` (default), `:touches`, `:overlaps`, `:within`, `:contains`, `:covers`,
  `:covered_by`, `:contains_properly`, `:crosses` (never true for two polygons) and `nil`
  (bounding boxes intersect, grown by the buffer), computed exactly from the boundary segments
  with robust orientation tests. With a buffer only `:intersects` and `nil` are defined (the
  other predicates would need the buffered geometries): anything else raises `ArgumentError`,
  as do `:dwithin` (needs a distance) and unknown names. A zero buffer or tolerance is no buffer.
- A negative `buffer` or `tolerance` raises (libpysal erodes the polygons). `**kwargs` passed to
  `GeoSeries.buffer` (`quad_segs`, cap and join styles) are not ported. Only `Polygon` and
  `MultiPolygon` input; cost is `O(m_i m_j)` segment tests per candidate pair.

### Block contiguity

`build_block_contiguity(regimes, ids: nil)` connects all nodes of a regime. libpysal orders
nodes regime by regime (regimes `[a, b, a, c, b]` give node order `[0, 2, 1, 4, 3]`); Spatex
keeps the input order. Neighbor sets are identical and are compared by id. Recorded in
`test/reference/known_deviations.exs`.

### Deviations: GAL and GWT

| Topic | libpysal 4.15.1 | Spatex |
|---|---|---|
| GAL weights | not stored; read back as binary | same (documented); a weighted graph writes the binary graph's file |
| GAL header | writes `n`; reads `n` or `0 n shapefile id_var` | same |
| GAL ids containing whitespace | written, then unreadable | `ArgumentError` when writing |
| GWT weight formatting | `%.7f` for float weights, plain `1` for integer-typed weights (binary contiguity) | always `%.7f`; text differs only by `1` vs `1.0000000`, read back identically |
| GWT ids with spaces | space replaced by `_` | same |
| Id typing | integers when every id parses as an int, else strings; GWT may yield float ids | `ids: :auto` (default), `:integer`, `:string`; never floats |
| Input | path only | path (`read_gal/2`) or text (`parse_gal/2`) |

## esda

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| Moran | Spatex.ESDA.moran/3 | M2 | available |
| Moran_BV | Spatex.ESDA.moran_bv/4 | M2 | available |
| Moran_Rate | Spatex.ESDA.moran_rate/4 | M2 | available |
| smoothing.assuncao_rate | Spatex.ESDA.assuncao_rate/2 | M2 | available |
| Geary | Spatex.ESDA.geary/3 | M2 | available |
| G | Spatex.ESDA.getis_ord_g/3 | M2 | available |
| Join_Counts | Spatex.ESDA.join_counts/3 | M2 | available |
| Moran_Local | Spatex.ESDA.moran_local/3 | M3 | available |
| Moran_Local_BV | Spatex.ESDA.moran_local_bv/4 | M3 | available |
| Geary_Local | Spatex.ESDA.geary_local/3 | M3 | available |
| G_Local | Spatex.ESDA.g_local/3 | M3 | available |
| Join_Counts_Local | Spatex.ESDA.join_counts_local/3 | M3 | available |
| Geary_Local_MV | Spatex.ESDA.geary_local_mv/3 | M3 | available |
| Join_Counts_Local_BV | Spatex.ESDA.join_counts_local_bv/4 | M3 | available |
| Join_Counts_Local_MV | Spatex.ESDA.join_counts_local_mv/3 | M3 | available |
| LOSH | Spatex.ESDA.losh/3 | M3 | available |
| fdr | Spatex.ESDA.fdr/2 | M3 | available |

Notes on global Moran (esda 2.10.0 fixtures: columbus queen, 4x4 lattice, graphs with islands,
an asymmetric graph; analytic quantities, bivariate I and EB rates to 1e-9):

- Permutations use a seeded ChaCha8 stream per replicate (`Spatex.ESDA` `seed:`; `nil` draws and
  stores a seed). Results are identical at any thread count but are not numpy's draws, so
  `sim`, `p_sim`, `ei_sim`, `se_i_sim`, `z_sim` and `p_z_sim` are verified statistically
  (binomial bands against PySAL with 9999 permutations), not value by value.
- `p_sim` is the folded pseudo p-value `(larger + 1) / (P + 1)`; `p_z_sim` is one-tailed by
  the sign of `z_sim`; `p_norm` and `p_rand` pick their tail by the sign of `z_norm`, doubled
  when `two_tailed: true`. `Moran_BV` uses sample standard deviations, no `S0` factor, and has
  no analytic moments.
- Non-finite results (for example `vi_rand` for `n < 4`) are the atoms `:nan` / `:infinity`.
- Known deviations (listed in `test/reference/known_deviations.exs`): a constant `y` raises
  `ArgumentError` where PySAL returns NaN; permutation streams differ from numpy's.
- `Moran_Rate` additionally exposes the rates used (`y`); `assuncao_rate/2` validates
  `base > 0` and `events >= 0`, which PySAL does not.

Notes on Geary's C, Getis-Ord G and join counts (same fixtures plus a binary distance band of
columbus centroids with and without islands, a complete graph and binarized CRIME; analytic
quantities, tables and chi-square to 1e-9; permutation quantities statistically):

- `geary/3`: `:transform` defaults to `:r`; inference is one-tailed (tail by the sign of
  `C - 1`, no `two_tailed`); `p_sim` is folded and `p_z_sim` is `sf(|z_sim|)`. Fields use
  `se_c_*` / `vc_*` / `ec_sim` spellings.
- `getis_ord_g/3`: weights are always binarized (no `:transform`); `p_norm` and `p_z_sim` are
  `1 - Phi(|z|)` as in esda; `y` must be non-negative; `z_norm`/`p_norm` are NaN when `vg` is not
  positive (complete graph).
- `join_counts/3`: binary weights forced; every directed row counts, tallied by halves; `drop_islands`
  (default true) removes island rows, and kept island self-loops count as same-colour joins like in
  esda. Yates-corrected `chi2_contingency` of `[[ww, wb], [bw, bb]]` with `bw := bw + wb` reported;
  `p_sim_*` are unfolded upper-tail pseudo p-values; a permutation with a zero expected cell is
  redrawn. esda has no analytic join count moments, so none are exposed.
- Known deviations (see `known_deviations.exs`): constant, negative (G) or non-binary (join counts)
  `y` raise `ArgumentError`; `sim_autocorr_chi2` reports the Pearson statistic instead of esda's
  formula (which divides both terms by the squared mean positive autocorrelation); esda's Geary
  crashes on a `Graph` with islands (Spatex computes them, verified against the `W` object).

Notes on local Moran (`Moran_Local`, `Moran_Local_BV`; fixtures: columbus queen, a 5x5 lattice
with two islands and tiny island graphs; `is`, quadrants `q`, `z`, `lag`, `EI`, `VI`, `EIc`,
`VIc` to 1e-9):

- Inference is esda's conditional randomization (`spatex_core::perm::crand`): one shared
  `P x max_card` index matrix drawn from `seed:` serves all sites (so simulations are correlated
  across sites, as in esda); a site keeps its own value and draws its neighbors without
  replacement from the other `n - 1` sites; self weights stay separate from the neighbors.
  Results are identical at any thread count. Draws are ChaCha8-based, not numba's, so `p_sim`,
  `sim`, `ei_sim`, `se_i_sim`, `z_sim` and `p_z_sim` are verified statistically against PySAL at
  9999 permutations (per-site binomial bands, cluster labels away from the threshold).
- `alternative:` `:directed` (default, esda's `None`), `:two_sided` (percentile mirror, may
  exceed 1), `:greater`, `:lesser`, `:folded`, as `esda.significance.calculate_significance`.
- Islands keep esda's quirk: statistic `0`, `p_sim = 1 / (P + 1)` (the smallest possible), and
  `z_sim`/`p_z_sim` `:nan`; `Moran_Local` never forwards `island_weight`, neither do we.
- Analytic moments follow esda: `ei`, `vi` (total randomization), `ei_c`, `vi_c` (conditional);
  there are no analytic z or p fields, and none for the bivariate statistic. `se_i` is
  `sqrt(vi)`.
- Local statistics default to `keep_simulations: false` (esda always keeps its `n x
  permutations` simulation matrix): `sim` and the simulated moments are `nil` (esda sets them to
  NaN) and only `p_sim` is returned; pass `keep_simulations: true` for them. `p_sim` is f64 (esda stores float32 when simulations are not
  kept).
- `sim` is one list of `P` values per site (esda's `rlisas`, `n x P`; esda's `sim` is the
  transpose). `MoranLocal.cluster_labels/2` replicates `get_cluster_labels` (strict `p < crit`).
- Observed statistics used for the p-value are re-evaluated with the replicate formula so that a
  replicate redrawing the observed neighbors ties exactly; on very small graphs PySAL's `p_sim`
  depends on floating-point ties and is not comparable (flagged `perm_comparable: false`).
- Known deviations (see `test/reference/known_deviations.exs`): constant `y`/`x` raises where
  PySAL returns NaN; fewer than 3 observations raise.

Notes on local Getis-Ord (`G_Local`, `fdr`; fixtures: columbus queen, a binary distance band,
self-weighted queen, tiny island graphs and a lattice with two islands; `gs`, `egs`, `vgs`, `zs`,
`p_norm` to 1e-9; `fdr` bit for bit):

- Same conditional randomization engine as local Moran, drawing from the raw `y`: `G` uses
  `(zrand @ w) / (sum(y) - y_i)`, `G*` uses `(zrand @ w + w_ii y_i) / sum(y)`. `p_sim` comes
  straight from the engine (esda does not recompute it) and, like the other permutation fields,
  is verified statistically against PySAL at 9999 permutations. `:two_sided` and `:folded`
  p-values depend on a sampled median/mean of a reference distribution shared by all sites, so
  they are compared with an absolute bound (0.07) instead of the binomial band.
- The `:star` option replicates `_infer_star_and_structure_w` on the graph as given: `false`
  removes a diagonal, `true` keeps one or assigns self weight `1` (binary graph or `transform:
  :b`) or the row maximum of the input weights (`:r`; esda warns here), `nil` infers from the
  diagonal, a number assigns that self weight (`star: 1` is not `star: true`). The transform
  (`:r` default, or `:b`; anything else raises, as esda asserts) is applied afterwards, so pass
  the graph untransformed. A graph labelled `:b` counts as binary input. Per-node self weight
  arrays are not supported.
- Islands keep esda's quirk: `gs` `0`, `zs`/`p_norm` `:nan`, `p_sim = 1 / (P + 1)`; unlike local
  Moran, `G_Local` forwards `island_weight` to the engine and so do we (`island_weight:`).
- esda has no cluster labels for `G_Local`, so there is no `hot_cold` helper.
- `fdr/2` returns the cut-off exactly as `esda.fdr` (threshold computed as `index * alpha / n`);
  it rejects an empty list, NaN and negative p-values and `alpha` outside `(0, 1]` (esda
  accepts them). Values above 1 (two-sided `p_sim`) are accepted.
- Known deviations (see `test/reference/known_deviations.exs`): negative `y` or a zero sum
  raise where esda returns meaningless values; fewer than 3 observations raise.

Notes on local Geary, local join counts and LOSH (`Geary_Local`, `Join_Counts_Local`, `LOSH`;
fixtures: columbus queen, a 5x5 lattice, lattice with islands, path graphs; `localG`, `LJC`, `Hi`,
`ylag`, `yresid`, `VarHi` and chi-square `pval` to 1e-9, `p_sim` and labels statistically):

- `geary_local/3` always row-standardizes the graph (as esda does); `labels: true` returns esda's
  `1` outlier / `2` cluster / `3` other / `4` not significant codes (`nil` where none applies);
  `join_counts_local/3` treats every non-zero weight as one join and ignores self weights, so no
  `:transform`; `p_sim` is `:nan` where the count is 0. `p_sim` is f64 (esda stores float32).
- `:island_weight` (both): the weight of the single pseudo-neighbor an island is given in the
  simulations; the island's observed statistic stays 0 as in esda (`local_multivariate_islands`).
- `geary_local_mv/3` (`Geary_Local_MV`; fixtures `local_multivariate_*`, `localG` to 1e-9):
  the mean of the per-variable local Geary of z-scores over the row-standardised graph; esda's
  folded p-value (no `alternative`). esda draws from numpy's unseeded global generator, Spatex
  from the seeded crand stream, one draw relabelling the neighbors for every variable. Islands
  get `localG = 0` (esda too on a `Graph`, with either `drop_islands`).
- `join_counts_local_mv/3` and `join_counts_local_bv/4` with `case: :clc` are the univariate
  counts of the product of the variables (esda's MV observation and simulation are exactly that;
  its BV `CLC` simulation `z_i * sum(x_j z_j)` differs only where the count is 0, where `p_sim`
  is `:nan` anyway). `case: :bjc` keeps esda's simulation `x_i * sum(z_j)`, which ignores the
  `1 - z_i` and `1 - x_j` factors of the observed count. `LJC` exact, `p_sim` statistically.
- `losh/3` takes `inference: nil | :chi2`; esda 2.10 documents a permutation method but raises
  `NotImplementedError` for it, and so do we (`ArgumentError`). With `a != 2` the chi-square
  inference logs a warning and leaves `pval` `nil`, as esda does.
- Known deviations (listed in `test/reference/known_deviations.exs`, each with a test that
  documents the PySAL behavior):
  - `Geary_Local` with islands: `localG` has `n - #islands` entries while `p_sim` has `n`
    (misaligned); Spatex returns `n` entries, island `local_g = 0`.
  - `Join_Counts_Local` on a `Graph` with islands raises `IndexError`; on a `W` an island with
    `y = 1` counts itself as a join (`LJC = 1`). Spatex gives islands `ljc = 0`.
    `fit(permutations=)` ignores the constructor's `permutations`; Spatex has the single option.
  - `LOSH` with any island returns all-`NaN` values; Spatex computes the statistic over the sites
    that have neighbors (equal to esda on the island-free graph, `n` in the variance being their
    number) and reports `:nan` for the islands. A site whose neighbor is an island (asymmetric
    graphs) is `:nan` as well.
  - Constant `x` raises where PySAL returns NaN.

## mapclassify

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| EqualInterval | Spatex.Classify.equal_interval/2 | M4 | available |
| Quantiles | Spatex.Classify.quantiles/2 | M4 | available |
| BoxPlot | Spatex.Classify.box_plot/2 | M4 | available |
| StdMean | Spatex.Classify.std_mean/2 | M4 | available |
| Percentiles | Spatex.Classify.percentiles/2 | M4 | available |
| MaximumBreaks | Spatex.Classify.maximum_breaks/2 | M4 | available |
| NaturalBreaks | Spatex.Classify.natural_breaks/2 | M4 | available |
| FisherJenks | Spatex.Classify.fisher_jenks/2 | M4 | available |
| JenksCaspall | Spatex.Classify.jenks_caspall/2 | M4 | available |
| FisherJenksSampled | Spatex.Classify.fisher_jenks_sampled/2 | M4 | available (seeded) |
| JenksCaspallSampled | Spatex.Classify.jenks_caspall_sampled/2 | M4 | available (seeded) |
| JenksCaspallForced | Spatex.Classify.jenks_caspall_forced/2 | M4 | available |
| mapclassify.classify | Spatex.Classify.classify/3 | M4 | available |
| HeadTailBreaks | Spatex.Classify.head_tail_breaks/1 | M4 | available |
| UserDefined | Spatex.Classify.user_defined/3 | M4 | available |
| MapClassifier.find_bin | Spatex.Classify.find_bin/2 | M4 | available |
| PrettyBreaks | Spatex.Classify.pretty_breaks/2 | M4 | available |
| MaxP | Spatex.Classify.max_p/2 | M4 | available (sorted-input parity) |
| gadf | Spatex.Classify.gadf/2 | M4 | available |
| KClassifiers | Spatex.Classify.k_classifiers/2 | M4 | available (best rule fixed) |
| MapClassifier.get_legend_classes | Spatex.Classify.legend/2 (labels), intervals/1 (`{low, high}`) | M4 | available |

Notes on the deterministic classifiers (verified against mapclassify 2.11.0 fixtures: bins to 1e-12,
`yb` and `counts` exactly, `tss`/`adcm`/`gadf` to 1e-9):

- `k` is mapclassify's attribute: the requested `k` for EqualInterval and MaximumBreaks, the
  number of bins otherwise. Quantiles on tied data drops repeated cut points and warns.
- Known deviations (listed in `test/reference/known_deviations.exs`): Quantiles reports
  `k = length(bins)` where mapclassify keeps the requested `k`; and Quantiles forces the top bin
  to `max(y)`, because mapclassify's last percentile can drift below 100 (`1..46`, `k = 23`
  or `36`) and then puts the maximum in class 0.
- A value above the last bin stays in class 0 (mapclassify `bin1d`), which only happens for
  Percentiles bins that stop below 100.
- MaximumBreaks: `numpy.argsort` is unstable, so ties at the cut-off gap are resolved by position
  (later gap first). mapclassify leaves that order unspecified; fixtures skip such cases.
- `mindiff` of MaximumBreaks is accepted and stored, but unused, as in mapclassify.

Notes on the optimal classifiers:

- FisherJenks and JenksCaspall(Forced) match mapclassify exactly (bins, `yb`, `counts`,
  iterations; 40 seeded float32-sensitive samples included). FisherJenks replicates mapclassify's
  `float32` dynamic program; a `float64` port diverges. Cost is `O(n^2 k)` time.
- Documented deviation (JenksCaspallForced): forced moves are by position in `y`, so
  classes are not contiguous in value. mapclassify returns the raw per-class maxima, which can be
  out of order (and `yb`/`counts` then include an empty class). Spatex sorts the cuts so `bins`
  is always ascending; identical to mapclassify whenever mapclassify's cuts are already sorted
  (all committed fixtures).
- NaturalBreaks: mapclassify's sklearn k-means is unseeded, so Spatex uses seeded k-means++
  restarts (`seed:`; `nil`, the default, draws a seed and records it in `info.seed`). Results are compared statistically (within-class variance not
  higher than mapclassify's) and are identical on well separated clusters. Optimal partitions
  need not be unique on tied data, so bins may differ there.
- JenksCaspall on a class that empties: mapclassify's median of an empty class is NaN and the
  run either exhausts memory or raises; Spatex stops and returns the non-empty classes.
- The sampled variants draw from a seeded ChaCha stream, not numpy's global generator, so they
  are reproducible but do not equal mapclassify's draws.
- PrettyBreaks: same numpy `arange` breaks; `k` stays the requested value and `tss`/`adcm`/`gadf`
  cover classes `0..k-1`, as in mapclassify (extra rounded classes are left out of the fit).
  `k < 2` raises (mapclassify divides by `k - 1`); a constant vector raises like the others.
- MaxP: numpy's legacy MT19937 (`np.random.seed` + `permutation`) is reproduced, so growth and
  swap orders match mapclassify for `seed1`/`seed2`. Two mapclassify quirks: it reseeds before
  every one of the `initial` growths (all identical), and it adds moves to `total_moves` only
  after a class stops moving, so exactly one swap round runs; both are kept. Not kept: its
  `_set_bins` takes class positions from the sorted copy but sums squares and reads bins from
  the unsorted `self.y`, so its result depends on the input order. Spatex uses the sorted values
  throughout, which equals mapclassify on sorted input (`max_p_*` fixtures).
- `gadf`: same k-search (`k = 2..maxk`, first `gadf > pct`, else the `maxk` classification);
  `k`, `gadf` and bins match for Quantiles, FisherJenks and MaximumBreaks (`gadf` fixture).
- KClassifiers: mapclassify assigns `pct0 = best[0]` (the k) and `k0 = best[-1]` (the gadf), so
  its best-method comparison mixes the two. Spatex keeps the smallest `k`, ties broken by the
  largest `gadf`, FisherJenks first; NaturalBreaks uses the seeded k-means.

## pointpats

pointpats 2.6.0, modern functional API (the deprecated class API is not ported).

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| mean_center | Spatex.PointPats.mean_center/1 | M6 | available |
| weighted_mean_center | Spatex.PointPats.weighted_mean_center/2 | M6 | available |
| manhattan_median | Spatex.PointPats.manhattan_median/1 | M6 | available |
| euclidean_median | Spatex.PointPats.euclidean_median/1 | M6 | available (Weiszfeld, see notes) |
| std_distance | Spatex.PointPats.std_distance/1 | M6 | available |
| ellipse | Spatex.PointPats.ellipse/2 | M6 | available |
| dtot | Spatex.PointPats.dtot/2 | M6 | available |
| hull | Spatex.PointPats.hull/1 | M6 | available |
| minimum_bounding_rectangle | Spatex.PointPats.mbr/1 | M6 | available |
| minimum_rotated_rectangle | Spatex.PointPats.minimum_rotated_rectangle/2 | M6 | available |
| minimum_bounding_circle (Skyum) | Spatex.PointPats.minimum_bounding_circle/1, skyum/1 | M6 | available (see notes) |
| g / f / j / k / l | Spatex.PointPats.g/2, f/2, j/2, k/2, l/2 | M6 | available |
| g_test / f_test / j_test / k_test / l_test | Spatex.PointPats.g_test/2 ... l_test/2 (`Spatex.PointPats.DistanceTest`, incl. envelope) | M6 | available |
| random.poisson | Spatex.PointPats.poisson/2 | M6 | available |
| random.cluster_poisson | Spatex.PointPats.cluster_poisson/2 | M6 | available |
| random.normal | Spatex.PointPats.normal/2 | M6 | available (statistical, see notes) |
| random.cluster_normal | Spatex.PointPats.cluster_normal/2 | M6 | available (statistical, see notes) |
| random.strauss | Spatex.PointPats.strauss/2 | M6 | available (statistical, see notes) |
| QStatistic / RectangleM / HexagonM | Spatex.PointPats.quadrat_test/2 (`Spatex.PointPats.Quadrat`) | M6 | available |
| alpha_shape (`libpysal.cg.alpha_shape`) | Spatex.PointPats.alpha_shape/2 | v0.2 | available (see alpha-shape notes) |
| alpha_shape_auto (`libpysal.cg.alpha_shape_auto`) | Spatex.PointPats.alpha_shape_auto/2 | v0.2 | available (see alpha-shape notes) |
| other metrics | - | later | not planned for v1 |

### pointpats notes

- Parity: centrography, G, K, L and the quadrat statistics equal pointpats to 1e-9 on the
  committed fixtures; F and J equal it when given the same empty-space points (`empty_points:`).
  Simulations are statistical only: pointpats' generators are unseeded, Spatex uses ChaCha8
  streams per `(seed, simulation)`, deterministic and independent of the thread count.
- Naming: the simulation count is `:permutations` in every test (pointpats' `n_simulations` for
  the distance tests and `realizations` for the quadrat tests). `seed: nil` (default) draws a
  seed recorded in the result; the bare `g/f/j/k/l` functions keep seed 0.
- Replicated quirks: raw-moment `std_distance`; ellipse small-sample correction with `n = len(points)`;
  default support `linspace(0, max nearest-neighbour distance, 20)` also for K and L;
  G/F normalized by the number of in-range distances with a leading 0; strict `<` pair counts and
  the data bounding-box area for K, also inside simulations; empty-space points of the `*_test`
  functions from the observed bounding box; p-value `min(p, 1 - p)`; `int()` truncation of
  `intensity * area`; quadrat realizations gridded on their own bounding box.
- Deviations (all in `test/reference/known_deviations.exs`): `euclidean_median` is the exact
  Weiszfeld optimum (pointpats' BFGS stops at gtol 1e-5, about 1e-4 relative);
  `minimum_bounding_circle` returns the true minimum circle where pointpats' Skyum loop returns a
  larger circle (c200, diag) or raises (r500); rotated-rectangle corners are compared as a set
  (equal-area ties, such as a rhombus, may pick a different rectangle than GEOS); `j_test` returns its
  result where pointpats returns `None` (unreachable with default supports); `HexagonM.num` is
  not exposed (it swaps odd and even rows), use `num_cells`; `poisson` never squeezes a one-point
  pattern; invalid input (NaN, zero weight sum, `n < 3` ellipse, collinear hull) raises
  `ArgumentError`.
- Not ported: edge corrections (pointpats 2.6.0 has none), non-Euclidean metrics, the `"alpha"`
  simulation window of the distance tests (`alpha_shape/2` itself is ported, see below),
  GeoPandas dispatch (shapely outputs).
- `normal`, `cluster_normal` and `strauss` are statistical only: tests compare sample
  moments, nearest-neighbour distances, close-pair counts and window containment with summaries of many pointpats
  realizations (`test/reference/generate_pointpats_processes.py`, `processes.json`). Replicated:
  a scalar `cov` is used as a variance (`cov * I`, so the standard deviation is its square root),
  the default centre is the window centre (vertex mean for a polygon), the default `normal`
  covariance is `(max(width, height) / 2)^2`, and cluster sizes follow `array_split`. Deviations
  (in `known_deviations.exs`): pointpats scales the off-diagonal of a matrix `cov` by the wrong
  standard deviations, so it is only correct for equal variances (Spatex uses the exact
  covariance); `cluster_normal` computes its default covariance once and reuses it in later
  replications (Spatex recomputes it per replication, keeping replications independent); a
  per-replication `(n_simulations, 2, 2)` covariance array is not supported; a centre that almost
  never lands in the window raises `ArgumentError` where pointpats loops forever. `strauss`
  replicates the birth-death sampler (`1.5 n` initial points, `n_iter` steps, first `n` points
  kept, `ArgumentError` where pointpats raises `RuntimeError`) and counts the pairs a moved point
  forms instead of recounting all pairs with a KD-tree (same distribution, no per-step rebuild);
  pointpats seeds it with `int(area * 1.5 n)` points because it passes `1.5 n` as an intensity,
  Spatex seeds it with `1.5 n` points whatever the area. The default `r = 0.05` is as in pointpats
  and only suits unit-scale windows.

<a id="alpha-shape"></a>
### alpha-shape notes

`alpha_shape/2` ports `libpysal.cg.alpha_shape(xys, alpha)` (4.15.1) and returns a list of WKB
polygons instead of a `GeoSeries`. libpysal triangulates with Qhull, keeps the triangles whose
circumradius is below `1 / alpha`, takes the edges that belong to one kept triangle and polygonizes
them with GEOS; Spatex triangulates with spade (the same triangulation for points in general
position, see [triangulation](#deviations-triangulation)) and traces the rings itself, so no GEOS
is involved.

- Equal to libpysal on the committed fixtures (`fixtures/pointpats/alpha_shape.json`: uniform,
  clustered, annulus, island-in-hole, two-blob and C-shaped sets at alphas from empty to the convex
  hull): polygon and hole counts, ring vertices (after normalizing the start vertex and
  orientation) and areas (1e-9 relative) agree. Each alpha sits half-way between two circumradii,
  so no triangle is on the `radius < 1 / alpha` boundary.
- Output form: every polygon has its shell clockwise and its holes counter-clockwise (as GEOS
  returns them), rings are closed in the WKB and start at their smallest vertex, and polygons are
  sorted lexicographically by their vertices. `[]` is an empty shape.
- Replicated on purpose: fewer than four points give their convex hull whatever `alpha` is (empty
  when they are collinear); a polygon inside a hole is its own polygon (`_filter_holes` keeps the
  polygons inside an odd number of shells); a hole face bounded by several polygons that only touch
  at vertices is not a hole to libpysal (the face comes back as an extra polygon) and neither to
  Spatex; two polygons that touch at a vertex are two polygons.
- Deviations (in `test/reference/known_deviations.exs`): `alpha` must be a finite number `>= 0`
  (`0` is the convex hull; libpysal divides by `alpha`, which raises `ZeroDivisionError` for a
  Python float `0.0`, and returns an empty shape for a negative `alpha`); four or more collinear
  points, or fewer than three distinct points, raise `ArgumentError` where Qhull raises
  `QhullError`; duplicate points are ignored; cocircular triangle choices follow spade.
- `alpha_shape_auto/2` (`step:`, `return_radius:`, `return_circles:`; `verbose` is not ported)
  follows libpysal's loop literally: radii sorted descending, `alpha = 1 / r - EPS`, then
  `radius < 1 / alpha`, stopping at the first candidate that is not one polygon with every point.
  libpysal's `polygonize` also returns the face of every hole, so a candidate with a hole is never
  valid and the result has no holes; a point is covered exactly when it is a vertex of a kept
  triangle. The comparison `radius < 1 / (1 / radius - EPS)` is decided by the last bit of the
  radius (it is false for about 90 % of the radii below 0.5), and libpysal's radius depends on the
  vertex order Qhull happens to give a triangle. The oracle (`alpha_shape_auto.json`: 34 cases over
  the same point sets plus small, tiny-scale and large-`step` ones) therefore stores the results of
  libpysal under all six vertex orders of every triangle and Spatex must equal one of them; all
  cases but one have a single result. Polygon vertices, area (1e-9), radius (1e-9) and bounding
  circle centres (1e-8, in ring order from the smallest vertex) agree.
- Deviations of `alpha_shape_auto/2` (in `known_deviations.exs`): fewer than three points or
  collinear points raise `ArgumentError` (libpysal returns an empty polygon for one or two points, a
  `LineString` for three collinear ones, and `UnboundLocalError`/`IndexError` for some option
  combinations); when not even the first candidate is valid (the float comparison above dropped the
  largest triangle and left a point uncovered) Spatex raises `ArgumentError`, where libpysal returns
  the first polygon of that shape, or raises `UnboundLocalError` for the radius; `step` must be a
  positive integer.
- `Spatex.Graph.build_triangulation(method: :voronoi, clip: :alpha_shape)` is still not supported:
  libpysal clips each Voronoi cell to the (generally non-convex, possibly holed) alpha shape, while
  Spatex clips Voronoi edges against a convex region with Cyrus-Beck.

## spreg

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| OLS (+ diagnostics) | Spatex.Regression.ols/3, summary/1 | M7 | available (White/HAC robust errors, LM and Koley-Bera tests, Moran of residuals, VIF, `slx_lags`/`slx_vars`; regimes in `ols_regimes/4`; no WLS; `constant: false` models skip spatial diagnostics) |
| OLS_Regimes | Spatex.Regression.ols_regimes/4 | M7 | available (`constant_regi`, `cols2regi`, `regime_err_sep` separate / group-wise / pooled, White, Chow tests; no spatial diagnostics, HAC or `slx_lags` with regimes) |
| ML_LagFE, ML_ErrorFE (panel.py) | Spatex.Regression.ml_lag_fe/4, ml_error_fe/4 | M7 | available (`time_effects`, lag impacts; no `slx_lags` in panels; dense `n x n`) |
| ML_LagRE (panel.py) | Spatex.Regression.ml_lag_re/4 | M7 | available (alternating `phi` / `rho` bounded searches; spreg's non-symmetric information matrix kept) |
| SUR | Spatex.Regression.sur/3 | M7 | available (two-step and iterated FGLS, LR / LM diagonal-covariance tests, cross-equation Chow, spatial LM error / lag; no regimes; ThreeSLS, SURerrorGM, SURerrorML and SURlagIV are separate rows; the only part of the SUR family not ported is `regimes`) |
| ThreeSLS | Spatex.Regression.three_sls/5 | v0.2.0 | available (2SLS per equation, 3SLS GLS step, `sig`, `corr`, inference, cross-equation Chow; no regimes; see spatial SUR notes) |
| SURerrorGM | Spatex.Regression.sur_error_gm/4 | v0.2.0 | available (lambda per equation by KP98 moments of the OLS residuals to ~5e-6, SUR GLS step on the filtered variables, `sig`, `corr`, inference, cross-equation Chow; no regimes, `spat_diag`, `vm`; see spatial SUR notes) |
| SURerrorML | Spatex.Regression.sur_error_ml/4 | v0.2.0 | available (per-equation ML error starts, iterated SUR residuals, SUR GLS step on the filtered variables and joint lambda search; `sig`, `corr`, inference, cross-equation Chow, LR tests of diagonal sigma and of lambda, `vm`, lambda standard errors and Wald tests of equality and joint zero; no regimes; lambda within 5e-5 of spreg's L-BFGS-B; see spatial SUR notes) |
| SURlagIV | Spatex.Regression.sur_lag_iv/4 | v0.2.0 | available (`W y` as the last endogenous variable, `W X` .. `W^p X` and, with `lag_q`, the lags of the user instruments as instruments, then ThreeSLS; `rho`, `sig`, `corr`, inference, cross-equation Chow, `joinrho` with `spat_diag`; spreg computes no impacts; no regimes; see spatial SUR notes) |
| ML_ErrorRE (panel.py) | Spatex.Regression.ml_error_re/4 | v0.2.0 | available (`time_effects`; formulas match spreg to 1e-9 at a given `(lam, phi)`, optimum from a nested bounded search; symmetric `W` with repeated eigenvalues uses the orthonormal basis, a non-symmetric `W` with repeated or complex eigenvalues raises; no `slx_lags`; dense `n x n`; see [spreg ML_ErrorRE notes](#spreg-ml_errorre-notes)) |
| GM_ErrorRE (panel.py), GM_KKP (sp_panels.py, deprecated alias) | Spatex.Regression.gm_kkp/4 | v0.2.0 | available (`time_effects`, `full_weights`; formulas match spreg to 1e-9 at spreg's own optimizer results, exact minimizer instead of L-BFGS-B (lambda within 1e-5 of spreg's); no `regimes` (spreg itself raises `NotImplementedError`), no `slx_lags`; sparse; see [spreg GM_KKP notes](#spreg-gm_kkp-notes)) |
| PooledOLS, PanelFE, PanelRE | - | - | not ported |
| ML_Lag | Spatex.Regression.ml_lag/4 | M7 | available (`method: :full | :ord | :lu`, `spat_impacts`, `slx_lags` (spatial Durbin, spreg impacts); regimes in `ml_lag_regimes/5`; no common-factor `spat_diag`; see [spreg ML notes](#spreg-ml-notes)) |
| ML_Error | Spatex.Regression.ml_error/4 | M7 | available (`method: :full | :ord | :lu`, `slx_lags` (SLX-error); regimes in `ml_error_regimes/5`; see [spreg ML notes](#spreg-ml-notes)) |
| TSLS | Spatex.Regression.tsls/5 | M7 | available (White/HAC robust errors, `sig2n_k`; no `spat_diag`, `slx_lags`; regimes in `tsls_regimes/6`) |
| TSLS_Regimes | Spatex.Regression.tsls_regimes/6 | M7 | available (`constant_regi`, `cols2regi` over `x` then `yend` with instruments always by regime, `regime_err_sep` separate / pooled, White, HAC (pooled), `sig2n_k`, Chow tests (`spreg_compatible: true` for spreg's mis-laid restrictions); no `spat_diag`, `robust: 'ogmm'` or `slx_lags`; see [spreg regimes notes](#spreg-regimes-notes)) |
| TSLS_Endog_Regimes | - | - | not ported (endogenously determined regimes, `skater_reg`) |
| GM_Lag_Regimes | Spatex.Regression.gm_lag_regimes/5 | M7 | available (`yend`/`q`, `w_lags`, `lag_q`, `constant_regi`, `cols2regi`, `regime_lag_sep` (one lag model per regime on the regime subgraph, or the pooled fit on their union), White (default, as spreg) / HAC (pooled), `sig2n_k`, Anselin-Kelejian test, `pr2_e`, `spat_impacts`, Chow tests (`spreg_compatible: true` for spreg's mis-laid restrictions); no `slx_lags`, `robust: 'ogmm'`; see [spreg regimes notes](#spreg-regimes-notes)) |
| GM_Lag_Endog_Regimes | - | - | not ported (endogenously determined regimes, `skater_reg`) |
| GM_Error_Regimes, GM_Endog_Error_Regimes | Spatex.Regression.gm_error_regimes/5 (`yend`/`q` for the Endog variant) | M7 | available (`estimator: :kp98` (default), `:het`, `:hom`; `constant_regi`, `cols2regi`, `regime_err_sep` pooled / one `gm_error/4` per regime on the regime subgraph, Chow tests (`spreg_compatible: true` for spreg's mis-laid restrictions), lambda to ~5e-6; no `slx_lags`; see [spreg regimes notes](#spreg-regimes-notes)) |
| GM_Error_Het_Regimes, GM_Endog_Error_Het_Regimes | Spatex.Regression.gm_error_regimes/5 (`estimator: :het`, `max_iter`, `epsilon`, `step1c`) | M7 | available (spreg's iteration quirks replicated; the +-0.9 clamp of the separate Endog fits is not; no `slx_lags`, `hard_bound`, `inv_method`) |
| GM_Error_Hom_Regimes, GM_Endog_Error_Hom_Regimes | Spatex.Regression.gm_error_regimes/5 (`estimator: :hom`, `a1: :het \| :hom \| :hom_sc` (default `:het`), `max_iter`, `epsilon`) | M7 | available (spreg's iteration quirks replicated; no `slx_lags`, `hard_bound`) |
| GM_Combo_Regimes, GM_Combo_Het_Regimes, GM_Combo_Hom_Regimes | Spatex.Regression.gm_combo_regimes/5 (`estimator: :kp98 \| :het \| :hom`) | M7 | available (`w_lags`, `lag_q`, `yend`/`q`, `regime_lag_sep` (one `gm_combo/4` per regime), `cols2regi`, `constant_regi`, Chow tests, `pr2_e`; no `slx_lags`; see [spreg regimes notes](#spreg-regimes-notes)) |
| ML_Lag_Regimes | Spatex.Regression.ml_lag_regimes/5 | M7 | available (`constant_regi`, `cols2regi`, `regime_lag_sep` (one `ml_lag/4` per regime on the regime subgraph; spreg requires every column by regime and a constant per regime), `method: :full \| :ord \| :lu`, `epsilon`, `spat_impacts`, `pr2_e`, Chow tests; no `regime_err_sep` (not a parameter of the class), `slx_lags` / `spat_diag`, `cores`, `vm`; see [spreg regimes notes](#spreg-regimes-notes)) |
| ML_Error_Regimes | Spatex.Regression.ml_error_regimes/5 | M7 | available (`constant_regi`, `cols2regi`, `regime_err_sep` (one `ml_error/4` per regime on the regime subgraph; spreg requires every column by regime and a constant per regime), `method: :full \| :ord \| :lu`, `epsilon`, Chow tests; no `regime_lag_sep` (ignored by spreg), `slx_lags` / `slx_vars`, `cores`, `vm`; see [spreg regimes notes](#spreg-regimes-notes)) |
| GM_Lag | Spatex.Regression.gm_lag/4 | M7 | available (`w_lags`, `lag_q`, `yend`/`q`, White/HAC, Anselin-Kelejian test, `spat_impacts`, `pr2_e`, `slx_lags` (spatial Durbin instruments); regimes in `gm_lag_regimes/5`; exact, no optimizer) |
| GM_Error, GM_Endog_Error | Spatex.Regression.gm_error/4 (`estimator: :kp98`, with `yend`/`q` for the Endog variant) | M7 | available (lambda to ~5e-6, `slx_lags`; see [spreg GM notes](#spreg-gm-notes)) |
| GM_Error_Het, GM_Endog_Error_Het | Spatex.Regression.gm_error/4 (`estimator: :het`, `max_iter`, `epsilon`, `step1c`) | M7 | available (the +-0.9 clamp of a non-`rho` endogenous coefficient is not replicated) |
| GM_Error_Hom, GM_Endog_Error_Hom | Spatex.Regression.gm_error/4 (`estimator: :hom`, `a1: :hom_sc \| :hom \| :het`) | M7 | available |
| GM_Combo, GM_Combo_Het, GM_Combo_Hom | Spatex.Regression.gm_combo/4 (`estimator: :kp98 \| :het \| :hom`) | M7 | available (`w_lags`, `lag_q`, `yend`/`q`) |
| GMM_Error | `estimator:` option of gm_error/4 and gm_combo/4 | M7 | not ported as a dispatcher (`add_wy` is `gm_combo/4`) |

### spreg ML notes

Oracle: spreg 1.9.1 fixtures `fixtures/regression/ml_*.json` (Columbus queen, rook, KNN-4 and binary
weights; synthetic n=200 and n=25 lattices with positive and negative dependence).

- **Optimizer.** `rho`/`lambda` come from a literal port of SciPy's bounded Brent search
  (`minimize_scalar(method='bounded')`, `xatol = epsilon = 1e-7`, 500 evaluations), verified
  against SciPy values. The objective is flat at the optimum, so one-ulp differences in a function
  value (BLAS/LAPACK summation order, which no port reproduces bit for bit) can flip a late
  comparison: 13 of 17 `full` fixture cases (all 9 spatial lag fits) reproduce spreg's `rho` to 1e-10 with the same
  evaluation count, the others end within `xatol` (largest observed 6e-8, all spatial *error*
  fits). spreg itself differs by that much between its own `full`, `lu` and `ord` methods.
  Everything else is verified at 1e-9 by evaluating the model at spreg's own `rho` (test hook
  `ml_lag_at/5` and `ml_error_at/5` in `Spatex.Regression`, internal), and `logll` agrees at 1e-9 in every fit.
- **`method: :ord`.** spreg uses `symmetrize(w)` whenever the *binary pattern* of `W` is symmetric
  (it then takes the eigenvalues of `D^{-1/2} B D^{-1/2}`, `B` the binary pattern). That is only
  valid for row-standardized binary weights. Spatex applies the symmetric similarity trick only
  when every nonzero weight is `1 / card(i)` and the pattern is symmetric, and otherwise uses the
  (complex) eigenvalues of `W` itself. spreg additionally substitutes the symmetrized matrix for
  `W` in the variance, so its `ord` standard errors differ from its `full` ones even where `rho` is
  right; Spatex's variance always uses the real `W`. Both are recorded in
  `test/reference/known_deviations.exs` and documented by tests that assert spreg's values
  (`ml_ord_deviation.json`: inverse-distance band, spreg `rho` 0.47861 vs the correct 0.47705).
- **`method: :lu`.** Evaluated with the dense LU of `:full` (spreg uses a sparse LU without the
  determinant sign check); same optimum, within the optimizer noise above.
- **Large `n`.** The log-determinant is a sum of logs (numpy's `det` can overflow for large `n`).
  All methods need dense `n x n` matrices for the variance, so `n > max_dense_n` (default 5000)
  raises `ArgumentError`.
- **Quirks replicated.** `k` counts `rho` for ML lag but not `lambda` for ML error (`aic`,
  `schwarz`); `predy_e` is the power expansion truncated at `epsilon` and is used in the lag
  variance; the ML error coefficient covariance has zero `beta`-`lambda` covariance; the `power`
  impact approximation stops at the first term below `1e-8`, which on bipartite graphs (lattices)
  is after one even term.


### spreg regimes notes

`ols_regimes/4` and `tsls_regimes/6` share the regime design (`regimeX_setup`): the regime-specific
columns of every regime in turn (`<regime>_<name>`), then the global ones (`_Global_<name>`), and
the same Chow Wald tests (`buildR1var`). Regimes are sorted; names use the label's string form.

- **`TSLS_Regimes` defaults.** `regime_err_sep` is `false` (`OLS_Regimes`: `true`) and `sig2n_k`
  is `true`. The separate fits (one `BaseTSLS` per regime) run only with `regime_err_sep: true`,
  every column by regime and `constant_regi: :many`; otherwise the flag is ignored, with no
  group-wise re-weighting (that exists for OLS only). Under HAC it is disabled with a warning.
- **Layout.** In the pooled fit the betas are `x` regime blocks, `x` globals, `yend` regime
  blocks, `yend` globals; `kr` / `kf` count exogenous plus endogenous columns and `kryd` the
  regime-specific endogenous ones. The instruments always vary by regime (`cols2regi: 'all'`); the
  constant of `x` follows `constant_regi`. In the separate fits every regime's block is
  `[constant, x..., yend...]`, columns of `x` constant within any regime are dropped everywhere
  with a warning, and `yend` / `q` are never dropped.
- **Chow layout (deviation, default).** `Chow` / `buildR1var` appends the `kf` zero columns of
  the restrictions at the end of the coefficient vector, which is right only if the globals come
  last. The pooled `TSLS_Regimes` betas are x regime blocks, x globals, yend regime blocks, yend
  globals, so with global exogenous columns and regime-specific endogenous ones (for example
  `constant_regi: 'one'`) spreg's endogenous and joint Chow statistics are not tests of the stated
  hypotheses. `tsls_regimes/6` builds the restrictions on the real layout by default (the project
  policy for upstream defects that yield a wrong statistic, as for `kullback/2`), and
  `spreg_compatible: true` reproduces spreg. Both sets are fixtures: the corrected values come from
  spreg's own `wald_test` on a correctly placed R (the generator asserts they equal spreg's
  whenever the layouts coincide), the flagged cases are `nsa_one_constant`,
  `nsa_one_constant_err_sep`, `nsa_mixed_global_open`, `nsa_mixed_err_sep` and
  `three_one_constant`. Coefficients and covariance are unaffected. `ols_regimes/4` cannot hit the
  difference (no endogenous columns, so the layouts coincide) and is unchanged; the later
  regimes models with endogenous columns will share the helper and the flag.
- **HAC.** The kernel covers all observations (no per-regime trimming) and applies to the pooled
  regime design only; `hac_multi` in spreg is unreachable for `TSLS_Regimes`.
- **Not ported.** `TSLS_Endog_Regimes`, `spat_diag` (Anselin-Kelejian), `robust: 'ogmm'`,
  `slx_lags`, `cores`, and spreg's text summary.

`gm_lag_regimes/5` (`GM_Lag_Regimes`) builds on the same design and Chow helper:

- **Defaults.** `robust` is `'white'` (the other regimes classes default to none; Spatex follows
  spreg here, so `gm_lag/4` and `gm_lag_regimes/5` differ), `sig2n_k` `false`, `spat_diag` and
  `spat_impacts: 'simple'` on, `regime_lag_sep` and `regime_err_sep` `false`.
- **`regime_err_sep` is subordinate to `regime_lag_sep`.** `regime_lag_sep: true` forces
  `regime_err_sep` on (warning); `regime_err_sep: true` alone is switched off (warning "Groupwise
  heteroskedasticity is not currently available"); HAC switches it off (warning) but keeps
  `regime_lag_sep`. Separate regressions therefore need `regime_lag_sep: true`, every column by
  regime, `constant_regi: 'many'` and no HAC; they are one `gm_lag/4` per regime on the regime's
  subgraph (spreg's `BaseTSLS` plus `set_endog_sparse`), with columns of `x` constant within any
  regime dropped everywhere (warning) and `kr = len(cols2regi) + 1` as spreg computes it.
- **Pooled fit.** `W y` and the instruments (`WX`, `W^2 X`, ..., and lags of `q` with `lag_q`) are
  built on the whole graph (`regime_lag_sep: false`) and appended to `yend` / `q`; `W y` is a
  global endogenous column (`cols2regi` gets a trailing `false`), the instruments vary by regime
  (named `<regime>_W_<x>`, `<regime>_W2_<x>`). The coefficient of `W y` is the last beta; `rho`
  and `pr2_e` / `predy_e` use it, the Anselin-Kelejian test uses the regime design (`Z`,
  `varb`) and the whole graph.
- **`regime_lag_sep` mechanics.** The graph is restricted to each regime (`w_subset`: edges
  between regimes are dropped) and re-transformed with the graph's own transform, so a
  row-standardized graph is row-standardized again per regime and a binary one stays binary;
  regime islands get a zero lag and a warning ("resulted in islands for regime r", the last such
  regime). A regime needs at least `len(cols2regi) + 2` observations. With separate regressions each
  regime's fit, diagnostics (`ak_test`, `pr2_e`, `sp_multipliers`, impacts) live in `multi` and
  `rho` is the last regime's. When the separate regressions cannot run (a global column,
  `constant_regi: 'one'` or HAC), `W y` is regime-specific (`cols2regi` gets a trailing `true`) and
  the pooled design is fitted on the union of the regime subgraphs (`w_regimes_union`); `rho` and
  the impact multipliers then use the last regime's coefficient and the union graph, while
  `predy_e` / `e_pred` apply each regime's own coefficient on its subgraph (`sp_att_reg`), and
  `rhos` lists them.
- **Chow layout (deviation, default).** The Chow restrictions of `GM_Lag_Regimes` are the
  `TSLS_Regimes` ones, so the layout defect above also hits it with global exogenous columns
  and a regime-specific `W y` (`regime_lag_sep` with `constant_regi: 'one'` or a global `x`
  column) or regime-specific `yend` (`constant_regi: 'one'`, mixed `cols2regi`). Same policy and
  flag (`spreg_compatible`); the flagged fixture cases are `nsa_endog_one_constant`,
  `nsa_endog_mixed`, `nsa_lagsep_one_constant`, `nsa_lagsep_endog_one_constant`,
  `nsa_lagsep_mixed`, `nsa_lagsep_mixed_rook_binary` and `four_lagsep_mixed`. Separate fits and the
  rest are unaffected (all globals last).
- **Impacts.** `spat_impacts` multiply the non-constant exogenous coefficients (regime-specific
  and global, `Z` order; names ending in `_CONSTANT` are skipped), as spreg's `_summary_impacts`
  does with `regimes=True`. They are omitted when `|rho| >= 1`.
- **Not ported.** `GM_Lag_Endog_Regimes`, `slx_lags` / `slx_vars` (spreg computes `WX` on the
  complete W there), `robust: 'ogmm'`, `cores`, `vm`, the text summary. `name_q` lists the
  spatially lagged instruments of the stale (pre-drop) `x` names, as spreg does.

`gm_error_regimes/5` and `gm_combo_regimes/5` (`GM_Error_Regimes`, `GM_Endog_Error_Regimes`,
`GM_Combo_Regimes`) are the KP98 estimator; the Het / Hom classes (`error_sp_het_regimes.py`,
`error_sp_hom_regimes.py`) are separate spreg files, covered by the Het / Hom bullets below
(`estimator: :het` / `:hom`):

- **Options.** The classes have no `robust`, `sig2n_k` or `spat_diag`; `hard_bound`, `cores`, `vm`
  and the text summary are not ported either. `constant_regi`, `cols2regi` (over `x` then `yend`;
  for the combo `W y` is appended as a global flag), `regime_err_sep` (default `false`), and for
  the combo `w_lags`, `lag_q`, `regime_lag_sep`.
- **Pooled fit.** The first stage is OLS (TSLS with `H = [X | q]` by regime for the endogenous
  models) on the regime design; lambda comes from its residuals through the KP98 moments (the
  `gm_error/4` optimizer, ~5e-6 of spreg's L-BFGS-B result); the base columns (constant, `x`,
  `yend`, including `W y`) and `y` are spatially filtered with it and then put on the regime design
  again (the filter acts on the unsplit columns, not on the regime blocks); the second stage reuses
  the first stage's `H`. `vm = sig2 * varb` with `sig2 = u'u / n` of the filtered fit, `predy` and
  `u` use the unfiltered design with the filtered fit's betas, `pr2` is the squared correlation of
  `y` and `predy`. `vm`, `std_err`, `z_stat` have no `lambda` entry; `kf` counts `lambda` as spreg
  does, and `Chow` trims `lambda` from the betas (so the layout is `{kxr, kxf, kyr, kyf}` of the
  columns without it).
- **`regime_err_sep`.** With every coefficient by regime and `constant_regi: 'many'` it runs one
  `BaseGM_Error` / `BaseGM_Endog_Error` per regime on the regime subgraph (`w_regime`: edges
  between regimes dropped, graph's transform re-applied, island warning, no minimum size), `kr =
  len(cols2regi)`, `kf = 0`, `kryd = 0`; otherwise spreg raises (`ArgumentError` in Spatex).
  Columns of `x` constant within a regime are dropped by `check_const_regi`, but spreg's
  preallocated `vm` / `betas` keep the stale `kr` and numpy raises `ValueError`; Spatex drops them
  with the warning and fits (known deviation).
- **Combo reconciliation.** `regime_lag_sep` and `regime_err_sep` are tied: `regime_lag_sep: true`
  switches `regime_err_sep` on (warning), `regime_err_sep: true` alone is switched off (warning).
  With `regime_err_sep` on, spreg tests `col != True` over `cols2regi`, which is also true for the
  string `'all'`, so the warning "All coefficients must vary ... setting cols2regi = 'all'" is
  emitted even with the default; Spatex reproduces the warning (only an all-true list is silent).
  Separate combos need `constant_regi: 'many'` as well (`'one'` raises). `W y` and its instruments
  are built per regime subgraph with `regime_lag_sep`, on the whole graph otherwise; instrument
  names are `<regime>_W_<x>` (the lags of `q` included with `lag_q`). `rho` is `betas[-2]`
  (pooled) and each regime's `rho` with `regime_lag_sep`; `pr2_e`, `predy_e` and `e_pred` come
  from `sp_att` (omitted for `|rho| >= 1`).
- **Chow layout (deviation, default).** The defect of the `TSLS_Regimes` notes also hits
  `GM_Endog_Error_Regimes` and `GM_Combo_Regimes` with global exogenous columns and a
  regime-specific `yend` (`constant_regi: 'one'` or mixed `cols2regi`); `GM_Error_Regimes`,
  which has no endogenous columns, and the separate regressions cannot hit it. Same policy and
  flag (`spreg_compatible`); the flagged fixture cases are `endog_nsa_one_constant`,
  `endog_nsa_mixed`, `endog_three_one_constant`, `combo_nsa_endog_one_constant` and
  `combo_nsa_mixed`.
- **Optimizer.** As in the [GM notes](#spreg-gm-notes) the objective is flat with binary weights:
  on seven of the 43 fixture fits (`*_rook_binary` on regime subgraphs, three queen fits) spreg's
  lambda is 1e-5 to 0.4 from the minimizer of its own objective. The fixture records spreg's
  lambda, the exact minimizer and both objective values; those cases are tested for lambda and
  layout only, the others at 5e-6 (lambda) and 1e-3 (derived quantities and Chow statistics).
- **Not ported.** `GMM_Error_Regimes` (the dispatcher), `slx_lags` / `slx_vars`, `hard_bound`,
  `cores`, `vm`, the text summary.
- **Het and Hom estimators (`estimator: :het | :hom`).** The pooled fits are the base Het / Hom
  estimators of `gm_error/4` on the regime design; `max_iter` (default 1), `epsilon` (1e-5),
  `step1c` (`:het`) and `a1` (`:hom`; spreg's regimes classes default to `'het'`, the base
  `GM_Error_Hom` to `'hom_sc'`) as in the base models, and `inv_method` is not accepted (the
  power expansion is always used). `vm`, `std_err` and `z_stat` include `lambda` (`k + 1`
  entries), `kf` counts it, `Chow` ignores it, `:het` reports no `sig2` (`:hom`: `u_s'u_s / n`).
  The separate fits (`regime_err_sep`, combo `regime_lag_sep`) are one base fit per regime on the
  regime subgraph, with `kr` counting `lambda` (`kr = len(cols2regi) + 1`) and the Chow tests
  taking `lambda` as one more varying coefficient (so `chow.regi` has a `lambda` row there). The
  reconciliation of `regime_err_sep` / `regime_lag_sep` and the "All coefficients must vary"
  rules are those of the KP98 classes.
- **Het / Hom iteration quirks (replicated).** As in the base models the coefficients come from the
  last second-stage regression, which uses the previous lambda, and `iteration` / `lam` are one
  step ahead. In the pooled regimes classes the second stage filters the base columns
  (constant, `x`, `yend`) and puts them on the regime design afterwards, while the variances
  (`get_vm_het`, `get_Omega_GS2SLS`, `get_omega_hom*`, step 1c and the weighting matrices) filter
  the regime design itself (`reg.x` / `reg.z`), which is not the same matrix. Inside the
  iterations `GM_Error_Het_Regimes` filters with the previous lambda like the base class, but
  `GM_Endog_Error_Het_Regimes`, `GM_Combo_Het_Regimes` and all three `_Hom_` classes filter
  with the first-step lambda (`lambda1`) every time, so a second iteration only moves lambda and
  its weighting, not the coefficients. The regimes classes never clamp the coefficient of the
  last endogenous column to +-0.9 (the base `GM_Combo_Het` does clamp `rho`, replicated there).
- **Het separate fits clamp a coefficient (not replicated).** The separate `GM_Endog_Error_Het`
  fits (`GM_Endog_Error_Het_Regimes` with `regime_err_sep`) are the base class, whose clamp pins
  the last endogenous coefficient at +-0.9 (HOVAL is clamped in four fixture cases); the
  combo's clamp of `rho` is replicated. As for the base model the fixture stores an unclamped
  twin of every clamped case, which Spatex matches.
- **Names.** `GM_Endog_Error_Het_Regimes` and `GM_Combo_Het_Regimes` (separate fits) name the
  lambda of regime `r` `"<r>lambda"` in `name_z`, without the underscore of every other name;
  Spatex uses `"<r>_lambda"`.
- **Het / Hom optimizer.** As in the [GM notes](#spreg-gm-notes) every fixture fit is also run
  with spreg's own lambdas (internal `gm_error_regimes_at/6` / `gm_combo_regimes_at/6` hooks, in
  call order, per regime for the separate fits) and agrees at 1e-8; with the optimizer the fits
  agree at 1e-3 (lambda 5e-6). Seven non-clamped fixture fits are optimizer-limited (spreg's
  lambda 5.1e-6 to 6.6e-5 from the minimizer of its own objective: binary rook weights and
  `het_combo_four_lagsep`); the fixture stores spreg's lambda, the exact minimizer and both
  objectives, and those are checked at the minimizer only.
- **Not ported (Het / Hom).** `slx_lags` / `slx_vars`, `hard_bound`, `inv_method='true_inv'`,
  `cores`, `vm`, the text summary, and the std. errors and z statistics of the separate fits
  exist in Spatex per regime although spreg defines none at the top level.

`ml_lag_regimes/5` (`ML_Lag_Regimes`) reuses the regime design (`regimeX_setup`) and the base `ml_lag/4`:

- **Shapes.** Without `regime_lag_sep` spreg runs one `BaseML_Lag` on the regime design with the
  whole `W`: one `rho`, one likelihood (`k` and the information criteria count `rho`, `kf` counts it
  as one more global column, so the betas are regime blocks, globals, `rho`). With
  `regime_lag_sep: true` it requires every column by regime and `constant_regi: 'many'` (any other
  setting raises "All variables must vary by regimes if regime_lag_sep = True."; the pooled fit on
  the union of the subgraphs is commented out in spreg, unlike `GM_Lag_Regimes`) and fits one
  `BaseML_Lag` per regime on the regime subgraph (`w_regimes`: edges between regimes dropped,
  graph's transform re-applied, island warning, `min_n = kx + 2`). The top-level object then only
  has the stacked betas, the block-diagonal `vm`, `u`, `predy`, `predy_e`, `e_pred`, `utu`,
  `mean_y`, `std_y` and the Chow tests; Spatex additionally concatenates `std_err` / `z_stat` and
  fills `rhos` (a map by regime label; spreg has no top-level `rho` there).
- **`regime_err_sep`.** The docstring says it is "always set to False - kept for compatibility",
  but `__init__` has no such parameter (passing it is a `TypeError`); `ml_lag_regimes/5` raises
  `ArgumentError` for it, so `regime_lag_sep` is the only switch.
- **Chow tests: no layout defect.** `rho` is always the last global column (pooled) or a
  regime-specific column (separate), so spreg's `buildR1var` (globals last) is correct here, also
  with global exogenous columns; the generator asserts that spreg's `Chow` equals one built on the real
  layout in all 29 fixture cases, and `ml_lag_regimes/5` has no `spreg_compatible` option. In the
  separate fits `rho` is one more regime-specific column of the tests (`"W_<y>"`).
- **Impacts.** Multipliers from `rho` and the whole `W` (pooled) or each regime's `rho` and
  subgraph (separate, stored in each `multi` fit); effects are the multipliers times the
  non-constant coefficients (`"_CONSTANT"` names skipped), as in `GM_Lag_Regimes`.
- **Optimizer.** `rho` of every fit comes from the literal Brent port of the [ML notes](#spreg-ml-notes)
  (within `xatol = 1e-7` of spreg's); every fixture case is also evaluated at spreg's own `rho`
  (internal `ml_lag_regimes_at/6`, a number or one `rho` per regime) and agrees at 1e-9 (Chow 1e-8).
- **`method: :ord`.** spreg's variance substitutes the symmetrized matrix for `W` (see the ML notes), also
  in every regime's fit, so for the `ord` cases the covariance-derived quantities (`vm`, `vm1`,
  `std_err`, `z_stat`, Chow) are compared with spreg's `full` run of the same case (stored as
  `full_ref`, 1e-5); everything else is compared with spreg's `ord`. Binary weights are not used with `ord`
  (the symmetrization is wrong there; see the ML notes).
- **Constant columns.** As in the other regimes classes, `check_const_regi` drops a column constant
  within any regime from the separate fits but `kr` and the preallocated `vm` / `betas` keep the
  stale count, so spreg raises `ValueError`; Spatex drops it with spreg's warning
  (`known_deviations.exs`). In the pooled fit such a column is a collinear regime design and both
  fail on the singular `X'X`.
- **Not ported.** `slx_lags` / `slx_vars` (spatial Durbin with regimes, WX on the complete `W`) and the
  common-factor test `spat_diag` that goes with it, `cores`, `vm` and the text summary.

`ml_error_regimes/5` (`ML_Error_Regimes`) reuses the same machinery and the base `ml_error/4`:

- **Shapes.** Without `regime_err_sep` spreg runs one `BaseML_Error` on the regime design with the
  whole `W`: one `lambda`, one likelihood. `k` (and so `aic` / `schwarz`) does not count `lambda`
  (as `ML_Error`), while `kf` counts it as one more global column, so the betas are regime
  blocks, globals, `lambda`. With `regime_err_sep: true` it requires every column by regime and
  `constant_regi: 'many'` (otherwise "All coefficients must vary across regimes if regime_err_sep
  = True."; `regime_lag_sep` is accepted and ignored) and fits one `BaseML_Error` per regime on
  the regime subgraph (`w_regime` with no minimum size, graph's transform re-applied, island
  warning). The top-level object then only has the stacked betas, the block-diagonal `vm`, `u`,
  `predy`, `e_filtered`, `utu`, `mean_y`, `std_y` and the Chow tests; `kr` counts `lambda` and the
  Chow tests include a `"lambda"` row. Spatex additionally concatenates `std_err` / `z_stat` and
  fills `lams` (a map by regime label); the per-regime names are `"<r>_lambda"`.
- **The pooled lagged design.** `BaseML_Error` is fed `regimes_att`: the lag of the design is
  `regimes(W [1 | X])` (the original columns lagged by the whole `W`, then expanded by regime,
  with the same `cols2regi`), not `W regimes([1 | X])`. They differ for every regime-specific
  column of an observation with a neighbour in another regime, so the pooled likelihood is not
  the base `ml_error/4` on the regime design. Spatex reproduces spreg's construction (a Rust
  entry point takes the lagged design; the Elixir side lags the base columns and expands them
  with `Regimes.regimize/3`); the residuals still use the regime design, and `e_filtered`
  filters them with the whole `W`.
- **Chow tests: no layout defect.** `lambda` is the last global column (pooled) or a
  regime-specific one (separate), so the restrictions are on the real layout; the generator
  asserts that spreg's `Chow` equals one built on the real layout in all 27 fixture cases and
  `ml_error_regimes/5` has no `spreg_compatible` option.
- **Optimizer and `method`.** As `ml_lag_regimes/5`: `lambda` comes from the literal Brent port
  (two independent runs on a flat objective end within ~1.5e-7 of each other), every fixture
  case is also evaluated at spreg's own `lambda` (internal `ml_error_regimes_at/6`, a number or
  one per regime) and agrees at 1e-9 (Chow 1e-8). For `method: :ord` the covariance-derived
  quantities are compared with spreg's `full` run (`full_ref`, 1e-5), as the symmetrized `W` of
  spreg's `ord` variance is a known deviation.
- **Constant columns.** A column constant within a regime is dropped from the separate fits by
  `check_const_regi` but `kr` and the preallocated arrays keep the stale count, so spreg raises
  `ValueError`; Spatex drops it with spreg's warning (`known_deviations.exs`). In the pooled fit
  it is an ordinary column (the fixture has one global column constant within each of four
  regimes, with a global constant).
- **Not ported.** `slx_lags` / `slx_vars` (WX on the complete `W`), `cores`, `vm` and the text
  summary. With this class all of the spatial regimes families are available:
  `tsls_regimes/6`, `gm_lag_regimes/5`, `gm_error_regimes/5` and `gm_combo_regimes/5` (KP98, Het,
  Hom), `ml_lag_regimes/5` and `ml_error_regimes/5`; only the endogenously determined regimes
  classes and the `GMM_Error_Regimes` dispatcher stay unported.

### spreg ML_ErrorRE notes

`ml_error_re/4` ports `spreg.ML_ErrorRE` (`spreg/panel.py`). `Panel_RE_Error` in
`panel_re.py` is deprecated in its favor and is not ported (its symmetric branch calls
`symmetrize(w)`, which silently swaps the binary matrix for its row-standardized similar matrix).
`GM_KKP` / `GM_ErrorRE` is a separate row.

- **What depends on LAPACK.** The earlier claim that the likelihood depends on the *signs* of the
  eigenvectors of `numpy.linalg.eig` is wrong. spreg builds `P = diag(c^(-1/2)) V'` with
  `c = t phi^2 + 1 / (1 - lam w)^2` and transforms `ys = y + (P - B) ybar - lam W y`
  (`B = I - lam W`, `ybar` the area means); the within part `(I_t (x) B) ytilde` and the between
  part `(I_t (x) P) ybar` are orthogonal, so the objective only involves `P'P = V diag(1/c) V'`,
  which does not change when a column of `V` changes sign. The generator asserts this on every
  simple-spectrum fixture (random sign flips change the objective by less than 1e-8 relative).
  The final coefficients, residuals and variances use the exact transform
  `P = (t phi I + (B'B)^-1)^(-1/2)`, so they only depend on the optimum `(lam, phi)`.
- **Simple spectrum: reproduced.** For a symmetric `W` the symmetric eigensolver gives the same
  `V diag(1/c) V'` as `numpy.linalg.eig`; for a non-symmetric `W` with a real simple spectrum
  (spreg's own NCOVR test setup, a row-standardized queen `W`) Spatex takes the general
  eigenvectors with unit norm, as LAPACK does, and the objective matches. Fixtures: the NCOVR
  counties of Kansas, Missouri, Oklahoma and Arkansas (`n = 372`, `t = 3`, with and without time
  effects), a binary rook 4 x 6 lattice and the row-standardized Columbus queen graph. Every
  quantity (betas, `vm`, `u`, `sig2`, log-likelihood, AIC, Schwarz) agrees at 1e-9 (`vm` 1e-8) at
  spreg's own `(lam, phi)`, and the objective at three probe points matches to 1e-9.
- **Optimizer.** spreg runs L-BFGS-B from `(0, 0.1)` with `ftol = 1e-9`; Spatex searches `phi` on a
  log scale inside a bounded search of `lam` (bounds `[-0.999, 0.999]` and `[1e-8, 1e9]` for spreg's
  `phi` parameter, the reported `phi` is its square), accurate to about 1e-10. The fixtures store
  spreg's optimum and the exact optimum of spreg's objective (Nelder-Mead polish); Spatex matches
  the exact one to 1e-6 (`lam`) and 1e-5 (`phi`), spreg's to 1e-5 and 1e-4 respectively.
- **Repeated eigenvalues (deviation).** `numpy.linalg.eig` returns a non-orthogonal basis inside a
  degenerate eigenspace, even for a symmetric `W` (a square rook lattice such as 5 x 5: the
  orthogonality error of `V` is 0.9), so spreg's objective there is not the likelihood and depends
  on LAPACK. For a symmetric `W` Spatex uses the orthonormal eigenvectors, i.e. the exact
  likelihood; the fixture `rook5x5` stores spreg's values (its coefficients and variances at its own
  `(lam, phi)` still match, only the objective and hence the optimum differ) and the exact optimum.
  For a non-symmetric `W` with repeated eigenvalues Spatex raises `ArgumentError`: spreg's value
  is basis dependent and no cheap defensible alternative is provided (an exact objective would need
  a dense factorization per evaluation).
- **Complex eigenvalues.** spreg then works in complex arithmetic and returns a meaningless
  result; Spatex raises `ArgumentError`.
- **Information matrix.** `vm` has the slopes' block `sig2 (X*' X*)^-1` and spreg's `lam` / `phi`
  block (`vm1[:2, :2]` of its 3 x 3 information matrix, kept as written, including its
  `t sqrt(phi)` term), with zero cross blocks. spreg can raise `LinAlgError: Singular matrix`
  there for some inputs (seen on a symmetric binary lattice in an earlier spike); Spatex raises
  `ArgumentError` ("the information matrix is singular") instead of fitting.

### spreg GM_KKP notes

`gm_kkp/4` ports the Kapoor-Kelejian-Prucha GM estimator of the random-effects spatial error panel.
In spreg 1.9.1 it lives in two places: `GM_KKP` (`sp_panels.py`, deprecated, its
`__init__` warns "superseded by spreg.GM_ErrorRE in panel.py") and `GM_ErrorRE` (`panel.py`). They
are the same estimator: identical moment equations (one builds `Q0` / `Q1` as `nt x nt` Kronecker
matrices, the other works on the `T x N` array), identical `Tau`, optimizer and GLS stage;
`GM_ErrorRE` adds `time_effects` and `slx_lags`. The fixture generator asserts that both classes
agree on every case without time effects (largest difference 1.7e-6, optimizer noise from the two
orders of floating-point operations). Spatex follows `GM_ErrorRE`. There is no other class with that
name: `GM_ErrorPooled` and `ML_ErrorPooled` in `panel.py` are different (pooled) models.

Oracle: `fixtures/regression/gm_kkp.json` (`generate_spreg_gm_kkp.py`): the NCOVR counties of
Kansas, Missouri, Oklahoma and Arkansas (`n = 372`, `t = 3`, spreg's own test variables HR on RD
and PS, row-standardized queen weights; a subset of the 3085 counties spreg's tests use, to keep
the fixture small), `full_weights` false and true, with and without time effects, plus seeded
synthetic panels on the row-standardized Columbus queen graph and the binary rook 5 x 5 lattice
(`t = 5`). The generator records the result of each of spreg's two `optim_moments` calls (the
first-step `(lambda, sigma_v^2)` and the weighted `(lambda, sigma_v^2, sigma_1^2)`) in call order.

- **Formulas: exact.** `gm_kkp_at/5` (an internal test hook) takes spreg's two optimizer results
  and bypasses the optimizer: coefficients, `vm`, standard errors, z statistics, `pr2`, `u`,
  `e_filtered`, `theta` and the three variance parameters agree with spreg at 1e-9 on every
  fixture case, including `Tau` (`full_weights: true`) and the time-effect dummies.
- **Optimizer: exact minimizer instead of L-BFGS-B.** spreg minimizes `|Ec (g - G v)|^2`,
  `v = [lambda, lambda^2, sigma_v^2, sigma_1^2]`, `Ec = chol(inv(Xi))'`, with `fmin_l_bfgs_b`
  (forward-difference gradient, bounds `lambda` in `[-0.99, 0.99]`, variances `>= 0`), twice: from
  `(0, 1)` for the three within equations and from the first-step result for all six. It stops
  early, as in the [GM notes](#spreg-gm-notes). Spatex profiles the variance parameters out (a
  non-negative least-squares problem in two unknowns, solved over its active sets) and minimizes
  over `lambda` with the same downhill grid walk and bounded Brent search as `gm_error/4`, from
  spreg's start (0, then the first-step `lambda`). On the fixtures `lambda` is within 1e-5 of
  spreg's, the variances within 5e-4 relative and the coefficients within 2e-3 (the worst case is
  the rook lattice, where spreg stops at an objective 1.6e-6 relative above the minimum); the
  generator's own polish of spreg's result (bounded Brent on the same profile) equals Spatex's to
  1e-6 and its global grid search finds no second basin. The `full_weights` NCOVR fits agree with
  spreg to 5e-6 in the coefficients.
- **Reported values.** As spreg: `betas` ends with `lambda`, `sigma2_v`, `sigma2_1`; `vm`,
  `std_err` and `z_stat` cover only the `k` regression coefficients; `vm` is the feasible GLS
  `s2 (X*' X*)^-1` with `s2 = e'e / (nt - k)` (spreg's `BaseOLS` default). `name_x` omits the
  stray leading space spreg puts in front of `sigma2_v`. There is no log-likelihood, AIC or
  iteration count.
- **Degenerate moments.** If the smallest entry of the scaled moment system is exactly zero spreg
  divides by it (its objective is NaN and L-BFGS-B returns the start silently); Spatex raises
  `ArgumentError`.
- **Not ported.** `regimes` (spreg's `GM_ErrorRE` raises `NotImplementedError`; the deprecated
  `GM_KKP` accepts them), `slx_lags` / `slx_vars`, the text summary, and the deprecated `GM_KKP`
  wide / long input detection (only the `n x t` matrices are accepted, as in `ml_error_re/4`).
- **Weights.** `W` is used exactly as given: spreg never transforms it implicitly. The estimator
  never uses the spectrum, so repeated or complex eigenvalues are fine (unlike `ml_error_re/4`).

### spreg spatial SUR notes

Oracle: spreg 1.9.1 fixture `fixtures/regression/three_sls.json` (Columbus: two equations and three
equations of equal and of different size; seeded synthetic systems with two endogenous variables in
one equation and more instruments than endogenous variables), generated by
`test/reference/generate_spreg_three_sls.py`. Everything matches at 1e-9 (inference and Chow tests at
1e-8). `three_sls/5` shares the FGLS machinery of `sur/3`.

- **Estimator.** `BaseThreeSLS` runs 2SLS per equation with `Zhat = [X | H (H'H)^-1 H'Yend]`, takes
  `sig = E'E / n` from the 2SLS residuals computed with the observed `Z = [X | Yend]`, and does one GLS step
  (`varb = (sum_rt sigma^rt Zhat_r'Zhat_t)^-1`) on the instrumented cross products. It does not iterate and
  uses no degrees-of-freedom divisor; the final residuals use the 3SLS coefficients and `Z`.
- **Diagnostics.** spreg reports no non-spatial diagnostic for `ThreeSLS` beyond the cross-equation Chow
  tests (`surchow`, only when every equation has the same number of coefficients); its `nonspat_diag`
  flag only affects the text summary, so Spatex has no such option.
- **Not ported.** `regimes` (spreg builds per-regime blocks per equation; `three_sls/5` raises
  `ArgumentError` for the option), `df`/`dict` input forms (lists of per-equation inputs only), the
  text summary. Every equation needs at least one endogenous variable and as many instruments as
  endogenous variables (spreg raises a singular cross-product error otherwise). Unnamed columns are
  named `var_<i>` as in `sur/3`; spreg names them `var_<i>_<eq>`, `end_<i>_<eq>`, `inst_<i>_<eq>`.
- **`SURerrorGM` (`sur_error_gm/4`).** `BaseSURerrorGM` runs OLS per equation, takes the KP98
  moments (`g = [u'u, (Wu)'(Wu), u'Wu] / n`, the `G` of `GM_Error`, `tr(W'W)` as the last entry) of
  each equation's OLS residuals `u` and minimizes them with `optim_moments` (the `GM_Error`
  optimizer), which yields one lambda per equation; no iteration. Variables (`y`, `X` with the
  constant: `W 1` is filtered too, which only differs from `1 - lambda` for non row-standardized
  `W`) and the OLS residuals are filtered `v - lambda_r W v`; `sig = E_f'E_f / n` comes from the
  filtered OLS residuals, one GLS step on the filtered cross products gives `bSUR` and `varb`
  (the `sur_est` of SUR), and `bigE` is `y - X bSUR` on the unfiltered variables. Diagnostics:
  `sur_inf` and `surchow` only. `spat_diag` and `vm` are documented there as "to be implemented"
  and unused, `nonspat_diag` only affects the text summary; the lambda equality (Wald) test and
  the LR / LM tests are ML-class reports (see `sur_error_ml/4`). Oracle: `fixtures/regression/sur_error_gm.json`
  (Columbus, queen and rook row-standardized and binary rook weights, two equations of equal size,
  three of equal and of different size), generated by
  `test/reference/generate_spreg_sur_error_gm.py`, which records every optimizer call next to the
  exact minimizer of its own objective. With spreg's lambdas (hook `sur_error_gm_at/5`) every
  quantity matches at 1e-9 (inference and Chow at 1e-8). With the optimizer, lambda is within 5e-6
  of spreg and everything else within 5e-4; the two binary-weight cases (flat objective, spreg's
  lambda up to 5e-4 from the minimizer) are tested for lambda against the exact minimizer and
  spreg's objective values, as in the regimes notes. Not ported: `regimes`, `df`/`dict` forms,
  the text summary.
- **`SURerrorML` (`sur_error_ml/4`).** `BaseSURerrorML` (1) fits `ML_Error` per equation (bounded
  Brent, `xatol = epsilon`; `lamols`, `clikerr = -sum fun`), (2) takes the residuals and
  log-likelihood of the iterated classic SUR (`BaseSUR(iter=True)`, its own defaults `maxiter = 5`,
  `epsilon = 1e-5`: `bSUR0`, `llik`), and (3) loops while `|fun0 - fun1| > epsilon` (initially `fun0 = 0`,
  `fun1 = -clikerr`): filter `y`, `X` (constant included) with the current lambdas, filter the
  previous residuals `E - lambda W E`, one `sur_est` GLS step (`b1`, `varb1`, `sig1`), new residuals
  `E = y - X b1`, then `minimize(clik, lam, L-BFGS-B, bounds [-1, 1])` for all lambdas jointly with
  `clik = -(-n/2 ln|sigma(lam)| + sum_r ln|det(I - lam_r W)|)`, `sigma = F'F / n`, `F = E - W E diag(lam)`, and the
  log-Jacobian the sum of `ln|u_ii|` of a sparse LU (absolute values, so it is finite for a negative
  determinant). The reported `bSUR`, `varb`, `sig`, `bigE` come from the last SUR step (the lambdas
  before the last search), `lamsur` and `cliksurerr = -fun1` from the last search. The log-likelihoods
  add `-n/2 n_eq (1 + ln 2 pi)`. Tests: `lrtest = 2 (surerrllik - errllik)` on `n_eq (n_eq - 1) / 2` df
  (`nonspat_diag`); `likrlambda = 2 (surerrllik - llik)` on `n_eq` df (`spat_diag`); with `vm = True`
  the asymptotic covariance `surerrvm` of the lambdas and the upper triangle of sigma (Anselin 1988,
  ch. 10), `lamsetp`, the Wald test of equal lambdas (`lamtest`, `n_eq - 1` df) and, with `spat_diag`,
  of joint zero (`joinlam`); `surchow` for equal-size equations. `nonspat_diag`, `spat_diag` and `vm`
  change numbers, not only the text summary. Oracle: `fixtures/regression/sur_error_ml.json`
  (Columbus, queen and rook row-standardized: two equations, three of equal and of different size),
  generated by `test/reference/generate_spreg_sur_error_ml.py`, which stores spreg's lambda path (the
  start and the output of every search), the same fit with `epsilon = 1e-10` and every reported test.
  With spreg's path (hook `sur_error_ml_at/5`) everything matches at 1e-9 (tests at 1e-8). Spatex
  replaces the L-BFGS-B search with a projected BFGS with analytic gradients (the same objective,
  the exact optimum of each search): both loops take the same number of searches, lambda is
  within 5e-5 of spreg's and of the `epsilon = 1e-10` refit, coefficients, covariances and tests within 5e-4;
  the per-equation starts differ by up to ~5e-8 (Brent's `xatol`, last-digit differences between a
  sparse and a dense LU). Binary weights: spreg raises `Factor is exactly singular`, Spatex returns a
  local optimum. Not ported: `regimes`, `df`/`dict` forms, the text summary, sparse Jacobian.
- **`SURlagIV` (`sur_lag_iv/4`).** `SURlagIV` appends `W y_r` to the user's endogenous variables of
  every equation (`bigyend[r] = [Yend | W y]`, so the spatial coefficient is the last one) and builds
  the instruments `bigq[r] = [Q | W X .. W^p X | W Q .. W^p Q]` (`X` without the constant, `p = w_lags`,
  each lag order as one block of columns; the `W Q` blocks only with `lag_q = True` and user
  instruments; without user instruments just the `W X` blocks), then runs `BaseThreeSLS` unchanged
  (`H = [X | bigq]`). Names follow spreg: `W_<name>`, `WW_<name>`, ... for the lags and `W_<y>` for
  the lag of `y`. Reported: `b2SLS`, `b3SLS`, `varb`, `sig`, `corr`, `bigE`, `tsls_inf`, `surchow`
  (equal-size equations) and, with `spat_diag`, `joinrho`: the Wald test (`sur_joinrho`, `n_eq` df)
  that the last coefficient of every equation is zero. spreg computes no spatial impacts, no
  Anselin-Kelejian or LM diagnostic and no separate `vm` for `SURlagIV`; `nonspat_diag` and `vm`
  only change the text summary. Oracle: `fixtures/regression/sur_lag_iv.json` (Columbus, queen and
  rook row-standardized: two and three equations, `w_lags` 1 and 2, `lag_q` on and off, user
  `yend`/`q`), generated by `test/reference/generate_spreg_sur_lag_iv.py`; everything matches at 1e-9
  (inference, Chow and `joinrho` at 1e-8, residuals at 1e-8: `y - Z b` multiplies the coefficient
  error by `W y`). Not ported: `regimes` (and `regime_lag_sep`), `df`/`dict` forms, the text summary;
  `:yends` and `:qs` are given for every equation or for none, and unnamed user columns are `var_<i>`
  (spreg: `end_<i>_<eq>`, `inst_<i>_<eq>`).

### spreg GM notes

Oracle: spreg 1.9.1 fixtures `fixtures/regression/gm_{columbus,synthetic,clamp,optimizer}.json`
(Columbus queen/rook/binary weights, HAC kernel, `DISCBD` as instrument for `HOVAL`; synthetic
n=200 lattice; 50 spreg fits including the deviation cases below). The generator wraps spreg's
`optim_moments` to store every lambda it returns, in call order.

- **`TSLS`, `GM_Lag`: exact.** No optimizer is involved; betas, `vm`, `varb`, `z_stat`, `sig2`,
  `pr2`, `pr2_e`, the Anselin-Kelejian test and the impact multipliers agree at 1e-9, including the
  White and HAC covariances and `w_lags: 2`, `lag_q: false` and `yend`/`q`.
- **Lambda: exact minimizer instead of L-BFGS-B.** spreg minimizes `|g - G [lam, lam^2, sigma^2]|^2`
  with `fmin_l_bfgs_b` (forward-difference gradient, `factr = 1e7`), which stops early: its lambda is
  up to ~5e-6 from the true minimizer. Reproducing it would mean porting SciPy's L-BFGS-B
  bit for bit. Spatex returns the exact minimizer on `[-0.99, 0.99]`: a 0.01 grid walked downhill
  from 0 (the basin L-BFGS-B starts in), refined by bounded Brent (`xatol = 1e-12`); for the
  three-equation KP98 objective `sigma^2` is profiled out in closed form. Fitted models therefore
  match spreg to 5e-6 in lambda and ~1e-4 relative in derived quantities (compared at 5e-4: betas
  reach 60 and move up to ~100x the lambda gap). The first-step objective is also minimized
  exactly where spreg's search ends early.
- **Every other formula: 1e-9.** `gm_error_at/5` and `gm_combo_at/5` (internal test hooks) take
  spreg's lambdas in call order (first step, then each iteration) and bypass the optimizer; all 35 GMM
  fixture fits then agree with spreg at 1e-9 (the clamped `GM_Endog_Error_Het` run is compared with its clamp-disabled twin), including the heteroskedastic and homoskedastic
  variances (`Omega`, `get_vm_het`, `get_Omega_GS2SLS`, `get_omega_hom*`).
- **Optimizer failures in spreg** (`gm_optimizer.json`, `known_deviations.exs`): with binary rook
  weights the KP98 objective is flat (spreg lambda 0.106667, exact 0.106268; objective 1.858e-6 vs
  1.856e-6) and the Hom weighted step stops 1.6e-5 early; `GM_Endog_Error_Het`'s first step ends at
  0.8227 where the objective is increasing (exact minimum 0.3665, objective 0.028 vs 1.29). Spatex
  returns the minimizer; the tests assert the objective values stored in the fixture.
- **The +-0.9 clamp** (`gm_clamp.json`). `GM_Combo_Het` clamps the coefficient of `Wy` to +-0.9 in
  the first stage and each iteration; Spatex replicates it (reported `rho` can be exactly 0.9; test
  `synthetic200/gm_combo_het/rho_clamp`). `GM_Endog_Error_Het` without `Wy` applies the same
  clamp to its last endogenous regressor, which is a spreg bug (HOVAL's coefficient -1.35 is pinned
  at -0.9); Spatex clamps only when the last column of `Z` is `Wy`. The fixture stores spreg's
  clamped run and a twin run with the clamp disabled, which Spatex matches at 1e-9.
- **Iteration order.** `:het` and `:hom` report the coefficients of the last second-stage
  regression, which uses the previous step's lambda (one iteration behind `lam`), and `:het` has no
  `sig2`, exactly as spreg.
- **Degenerate scale.** If `min(G, g)` is exactly zero spreg divides by it (NaN objective,
  L-BFGS-B returns lambda = 0 silently); Spatex returns 0 with a warning.
- **Not ported.** `regimes` (the base models have none; see `ols_regimes/4`, `tsls_regimes/6` and the regimes notes above), `inv_method='true_inv'` (Spatex always uses the power
  expansion, threshold 1e-10), the `GMM_Error` dispatcher (use `estimator:` and `gm_combo/4`), the
  spreg `h`/`z` matrices on the result, `spat_diag` for `TSLS`.
- **Weights.** As everywhere, `W` is used exactly as given: spreg never transforms it implicitly.

## giddy

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| Markov | Spatex.Dynamics.markov/2 | M8 | available |
| ergodic.steady_state, mfpt; sojourn_time | Spatex.Dynamics.steady_state/2, mfpt/2, sojourn_time/2 | M8 | available |
| Spatial_Markov | Spatex.Dynamics.spatial_markov/3 | M8 | available |
| LISA_Markov | Spatex.Dynamics.lisa_markov/3 | M8 | available |
| LISA_Markov.spillover | Spatex.Dynamics.lisa_spillover/3 | M8 | available |
| rank.Tau | Spatex.Dynamics.kendall_tau/2 | M8 | available |
| rank.SpatialTau | Spatex.Dynamics.spatial_tau/4 | M8 | available |
| rank.Theta | Spatex.Dynamics.theta/3 | M8 | available |
| FullRank_Markov | Spatex.Dynamics.full_rank_markov/2 | M8 | available |
| GeoRank_Markov | Spatex.Dynamics.geo_rank_markov/2 | M8 | available |
| rank.Tau_Local | Spatex.Dynamics.tau_local/2 | M8 | available |
| rank.Tau_Local_Neighbor | Spatex.Dynamics.tau_local_neighbor/4 | M8 | available |
| rank.Tau_Local_Neighborhood | Spatex.Dynamics.tau_local_neighborhood/4 | M8 | available |
| rank.Tau_Regional | Spatex.Dynamics.tau_regional/4 | M8 | available |
| kullback | Spatex.Dynamics.kullback/2 | v0.2.0 | available (corrects giddy's axis slip by default; `giddy_compatible: true` reproduces it) |

Oracle: libpysal's bundled `mexico` example (32 states, 1940-2000, `mexico.gal`) and a seeded synthetic 49 x 10 panel on the Columbus queen graph,
`test/reference/fixtures/dynamics/` (`test/reference/generate_dynamics.py`). Deterministic fields
match to 1e-9; permutation results (`x2_rpvalue`, `tau_spatial_psim`, theta p-values, LISA p-values)
are compared with tolerance bands because the ChaCha8 streams differ from numpy's.

### Deviations: dynamics

- **Mean first passage times of reducible chains.** giddy raises `LinAlgError` for a chain with a
  recurrent cycle outside the destination (e.g. a 2-cycle) and returns numerical garbage (negative or
  beyond 1e12) for some conditional matrices. Spatex returns the exact value where passage is certain
  and `:infinity` elsewhere.
- **Categorical lag ties** (`discrete: true`): libpysal breaks ties that survive the focal-label retry at
  random; Spatex takes the lowest class. The oracle patches libpysal to the same rule.
- **Spatial Markov permutations with `discrete: true`**: giddy shuffles `y` but recomputes the lag from
  the unshuffled classes, so every replicate equals the observation; Spatex shuffles the class rows.
- **Spatial Markov `lclass_ids` after permutations**: giddy leaves the attribute holding the last permuted
  replicate; Spatex keeps the observed lag classes.
- **LISA Markov independence test (`chi_2`, `expected_t`)**: giddy hands `Markov` the `(t, n)` array of
  below-mean indicators, so its marginal chains count moves between neighboring location columns; Spatex
  uses the `(n, t)` orientation (moves between periods) as the method intends. giddy's value is kept in
  the fixture for reference.
- **Kullback conditional homogeneity (`kullback`)**: giddy sums the stratum term `f_si. ln f_si.` over
  axis 1 of the `(s, r, r)` array, which gives the column totals `f_s.j` instead of the row totals
  `f_si.` Kullback's statistic needs. The two agree only when every stratum has equal row and column
  totals (the table in giddy's docstring does), so giddy's value is otherwise wrong: identical strata
  give a negative statistic (-0.517 for two copies of `[[10, 5], [4, 12]]`) and the mexico
  `Spatial_Markov(k=4, m=4).T` gives 63.46 where the row totals give 39.94 (p 0.003 against 0.30).
  Spatex uses the row totals; `giddy_compatible: true` reproduces giddy's value (the fixture stores
  both). Counts must be finite and non-negative (giddy takes logarithms of whatever it gets).
- **Conditional steady states and mfpt** of a matrix with an empty row are `nil` (giddy raises lazily)
  unless `fill_empty_classes: true`.
- **Seeds**: permutation draws come from ChaCha8 streams per replicate; results are reproducible per
  seed at any thread count but not equal to numpy's.
- **Local tau** (`dynamics/tau_local.json`, mexico 1940 vs 2000): statistics exact, p-values
  statistically. `Tau_Local_Neighbor` / `_Neighborhood` permutations give each site distinct random
  other sites as neighbors (giddy relabels all other sites, the same distribution); islands are
  `:nan` (giddy divides 0 by 0). `Tau_Regional` shuffles all sites over the fixed regimes.
- **LISA spillover** (`dynamics/spillover.json`, exact on giddy's `q` and `p_values`): components are
  numbered by their smallest site index; giddy pops them from a Python set, which gives the same
  order for integer ids but an arbitrary one for string ids.

## inequality

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| Gini | Spatex.Inequality.gini/1 | M8 | available |
| Theil | Spatex.Inequality.theil/1 | M8 | available |
| TheilD, TheilDSim | Spatex.Inequality.theil_d/3 | M8 | available |
| Gini_Spatial | Spatex.Inequality.spatial_gini/3 | M8 | available |
| atkinson, Atkinson | Spatex.Inequality.atkinson/2 | v0.2.0 | available (one function returns the index `a` and the `ede`) |
| Schutz | Spatex.Inequality.schutz/1 | v0.2.0 | available (vector input, no DataFrame or `plot`) |
| wolfson, lorenz_curve | Spatex.Inequality.wolfson/1, lorenz_curve/1 | v0.2.0 | available |
| polarization.S | Spatex.Inequality.polarization/3 | v0.2.0 | available (permutation null from ChaCha8 streams; statistic exact) |
| pen, pengram | - | - | not planned (matplotlib charts) |

Gini, Theil and the Theil decomposition are deterministic (1e-9); `bg_pvalue` and the permutation fields of `spatial_gini` are compared with
tolerance bands. `spatial_gini` uses the graph as a binary neighbor structure (the weights'
magnitudes are ignored, `s0 / 2` is half the number of non-zero neighbor entries).

Atkinson, Schutz and Wolfson are deterministic and compared to 1e-9 on the mexico incomes, the
synthetic panel and hand-made cases (`dynamics/inequality_indices.json`, generator step
`inequality_indices`). Notes and deviations:

- `atkinson/2` returns a `Spatex.Inequality.Atkinson` with `a` (PySAL's `A`, also the value of the
  `atkinson()` function), `ede` (`EDE`) and `epsilon`.
- `schutz/1` takes the values themselves; PySAL's `df` / `column_name` pair, the processed
  DataFrame and `plot()` are not ported. `coefficient` keeps PySAL's factor 10.
- `polarization/3` (`dynamics/polarization.json`, generator step `polarization`): the statistic, the
  component counts and the three label vectors are exact on 14 cases (mexico and a synthetic panel
  on libpysal graphs, quantile and explicit bins, collapsed quantile edges, constant data, a lattice
  and a graph with an island). `p_value` and the simulated indices are compared statistically
  because Spatex draws from seeded ChaCha8 streams and PySAL from numpy `SeedSequence` children.
  Only the graph's structure is used (every stored entry is an edge, weights ignored, self-loops
  of islands harmless), as in PySAL. `x` is aligned with the graph's ids by position, where PySAL
  aligns a data frame's index. `bins` follow `pandas.cut` (right-closed, so a value equal to the
  first edge gets no group, and values outside the edges raise); `k` follows
  `pandas.qcut(..., duplicates="drop")` (collapsed edges are merged yet `n_a_components` stays `k`,
  and constant data under `k > 1` leaves every observation without a group, giving `S = 1 + k`).
  `keep_sim` maps to `:keep_simulations`, which defaults to `true` here (a global statistic, see
  `docs/architecture.md` section 9) where PySAL's `keep_sim` is `False`; `n_jobs`, `verbose`, the
  data-frame arguments and the `labels` DataFrame are not ported (the three label vectors are
  fields of the result).
- Inputs PySAL does not guard raise `ArgumentError`: empty vectors (numpy returns NaN) and values
  summing to zero in `schutz/1` (PySAL fails with `IndexError`).

## spopt

spopt 0.8.0 with scikit-learn 1.9.1 (Ward) and scipy 1.18.1 (minimum spanning tree).

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| Skater (`SpanningForest`) | Spatex.Region.skater/3 | M9 | available |
| WardSpatial | Spatex.Region.ward_spatial/3 | M9 | available |
| RegionKMeansHeuristic | Spatex.Region.region_kmeans/3 | M9 | available |
| AZP (basic) | Spatex.Region.azp/3 | M9 | available |
| MaxPHeuristic | Spatex.Region.max_p/3 | M9 | available |
| ObjectiveFunctionPairwise / ObjectiveFunctionCenter | Spatex.Region.objective/3 | M9 | available |
| AZPBasicTabu | Spatex.Region.azp/3 (`method: :tabu`) | M9 | available (defined from the algorithm description: spopt 0.8.0 raises `TypeError`) |
| AZPSimulatedAnnealing | Spatex.Region.azp/3 (`method: :annealing`) | M9 | available (defined from the algorithm description: spopt 0.8.0 raises `TypeError`) |
| AZPReactiveTabu | - | - | not planned (broken in spopt, underspecified) |
| RandomRegion | Spatex.Region.random_region/2 | v0.2.0 | available (distributions, not labels) |
| RandomRegions | Spatex.Region.random_regions/2 | v0.2.0 | available (distributions, not labels) |
| SA3 | Spatex.Region.sa3/3 | v0.2.0 | available (exact labels and linkage; `min_cluster_size >= 2`) |
| Spenc | Spatex.Region.spenc/3 | v0.2.0 | available with notes (exact affinity, Laplacian spectrum and embedding; hierarchical labels exact; discretize / kmeans labels statistical) |

### spopt notes

- Exact parity (labels identical): SKATER (including pruning trace and scores), spatial Ward
  (labels, merge tree, distances 1e-12, disconnected-graph completion, partial-tree numbering),
  Region-K-Means given spopt's seeds (labels, centroids, iterations). Objective helpers match to
  1e-9 for the manhattan, euclidean and squared-euclidean metrics.
- AZP and Max-p are randomized heuristics. spopt uses Python `random` and numpy's legacy
  generator; Spatex uses ChaCha8 streams (`seed:`), so labels are never compared. Tests check
  contiguity, region count, thresholds, objective monotonicity, brute-force optima on tiny
  lattices (AZP reaches the optimum over 30 seeds), p against spopt's, and the Max-p construction
  phase for a fixed visiting order (exact with `top_n: 1`).
- AZP `method: :tabu` / `:annealing` have no PySAL oracle (spopt 0.8.0's classes cannot be
  constructed); they follow the algorithm description summarized in `Spatex.Region.azp/3`, return
  the best partition they visit and end with the same strict local-optimality pass. Tests check validity, monotonicity
  against the start, determinism per seed and the brute-force optimum over 20 seeds.
- `random_region/2` and `random_regions/2` port `RandomRegion` / `RandomRegions` (random partitions
  with optional `num_regions`, `cardinality`, contiguity graph, `compact`, `maxiter`, `max_swaps`;
  `permutations` for the plural). The construction is spopt's (shuffle and cut at random breaks
  without a graph; largest-first region growth with border swaps with one), but spopt draws numpy's
  global generator, so partitions are never compared. Tests check structure (every area once,
  region count, exact cardinalities, contiguity), determinism per seed and thread count, and
  statistical agreement of summaries (number of regions, region sizes in construction order,
  share of feasible runs, compactness) with 600 to 6000 spopt runs per case. Deviations:
  the area ids are positions `0..n-1` (spopt takes arbitrary ids that must equal the weights'
  `id_order`); infeasible requests return `feasible: false` with `labels: nil`, as spopt returns
  `feasible = False` with empty regions; spopt hangs for `num_regions > n - 1` with random
  cardinalities and when the swap search finds no swappable area, Spatex raises
  `ArgumentError` for the former and bounds the latter with `max_swaps`; non-positive
  cardinalities, `num_regions` or `maxiter` raise (spopt treats `0` as unset and builds empty
  regions); `RandomRegions` runs its permutations in parallel from per-permutation ChaCha8 streams.
- `sa3/3` ports `SA3` and `spopt.region.sa3.extract_clusters` with `fast_hdbscan` 0.3.4's
  `condense_tree`, `extract_eom_clusters`, `extract_leaves` and `get_cluster_label_vector`. Each
  connected component with more than `min_cluster_size` areas gets a scikit-learn Ward tree
  (the `ward_spatial/3` core, full tree, `n_clusters = 2`); its distances are made monotone as
  spopt does (a drop becomes the previous distance plus `0.01`, repeated); the condensed tree
  uses lambda = 1 / distance and `float32` lambdas, sizes and stabilities like `fast_hdbscan`;
  clusters are numbered by `fast_hdbscan`'s node ids and then across components; noise is `-1`.
  Labels are identical to spopt's and the linkage matrices agree to 1e-12 on columbus,
  disconnected graphs with a small component and an island, a tied integer lattice and a
  queen lattice, for `min_cluster_size` 2 to 30 and both extractions. Deviation: spopt accepts
  `min_cluster_size` 0 and 1 and returns non-contiguous clusters; Spatex raises for values
  below 2. The result also exposes the per-component linkage matrices (`components`).
- `spenc/3` ports `Spenc` and `spenclib.SPENC.fit`. The affinity (`pairwise_kernels` of the
  attributes times `W` elementwise; `nearest_neighbors` and `precomputed` ignore `W`) matches
  spopt to 1e-12 for rbf, laplacian, linear, polynomial, sigmoid, cosine, nearest neighbors and
  precomputed, with binary and non-binary weights. The normalized Laplacian's `k` smallest
  eigenvalues (1e-8) and the embedding (eigenvectors of `-L` divided by `sqrt(degree)`, sign
  normalized so the largest entry is positive; 1e-6) match spopt's `eigsh`. spopt's ARPACK
  shift-invert is replaced by a dense symmetric eigensolver (`faer`): same eigenpairs, cubic cost,
  at most about 5000 areas; `eigen_solver`, `eigen_tol`, `n_init` and `shift_invert` have no effect
  in spopt either (only its `breakme` path, which is not ported, hands them to scikit-learn).
  Label assignment: `discretize` (default) is a pure function of the embedding row its random
  rotation starts from, and Spatex's labels equal scikit-learn's for that row (checked for every
  row on Columbus); the row comes from ChaCha8 instead of numpy, so the distribution over
  partitions is tested against the exact start-row distribution (total variation < 0.1 over 400
  seeds) and both implementations recover a planted four-block lattice partition in every run.
  `kmeans` is a k-means++ / Lloyd single run like `KMeans(n_clusters)` and is compared by
  adjusted Rand index to spopt's runs. `hierarchical` (Shi and Malik bipartition, grid search for
  the cut) is deterministic and its labels are exact against spopt's code, but spopt 0.8.0
  raises `TypeError` for it (it passes `floor=` to a function that lacks the keyword); the
  oracle patches that keyword. Spectral clusters are **not** guaranteed to be contiguous. A
  non-symmetric affinity raises (spopt returns meaningless eigenvectors); a disconnected affinity
  graph warns, as spopt does, and its zero eigenvalue is degenerate, so the embedding there is
  only defined up to a rotation of that eigenspace.
- Attributes are never scaled implicitly, as in spopt; `Spatex.Region.standardize/1` is provided.
- Deviations (see `test/reference/known_deviations.exs`): SKATER keeps links between areas with
  identical attributes (`zero_dissimilarity: :drop` reproduces spopt); Region-K-Means raises when
  a component has no seed and `seed:` guarantees one seed per component; AZP returns one region for
  `n_clusters: 1` and adds a strict local-optimality pass; Max-p `policy: :keep` raises upfront;
  labels are 0-based integers.
- Kept spopt quirks: `islands: :increase` adds every component (including the main one) to
  `n_clusters`; SKATER's `floor` counts areas; Ward connects components in attribute space and
  numbers clusters by heap layout (full tree) or head rank (`n_clusters >= max(100, 0.02 n)`);
  Max-p frozen failed regions, `top_n` counting areas and fixed annealing constants.
- No dense `n x n` matrix is built (spopt builds them for SKATER and Max-p).

## mgwr

| PySAL name | spatex function | Milestone | Status |
|---|---|---|---|
| GWR (Gaussian) | Spatex.GWR.fit/4 | M9 | available |
| Sel_BW (golden section, interval, scipy), `X_glob` bounds | Spatex.GWR.select_bandwidth/4 (`search: :scipy`, `x_glob:`) | M9 | available |
| GWR `hat_matrix=True` (`S`) | Spatex.GWR.fit/4 (`hat_matrix:`) | M9 | available |
| GWRResults.filter_tvals / critical_tval | Spatex.GWR.filter_tvals/2, critical_tval/2 | M9 | available |
| GWR (Poisson, Binomial), GWRResults deviance/D2/pDev | Spatex.GWR.fit/4, select_bandwidth/4 (`family:`, `offset:`) | M9 | available |
| MGWR (backfitting, multi-bandwidth search, inference) | Spatex.GWR.mgwr/4 | M9 | available |
| GWR.predict, GWRResults.predictions | Spatex.GWR.predict/6 | M9 | available |
| GWRResults.local_collinearity, MGWRResults.local_collinearity | Spatex.GWR.local_collinearity/3 | M9 | available |
| GWRResults.get_bws_intervals, MGWRResults.get_bws_intervals | Spatex.GWR.bandwidth_intervals/2 | M9 | available |
| GWRResults.spatial_variability | Spatex.GWR.spatial_variability/4 | M9 | available |
| MGWRResults.spatial_variability | (not provided) | M9 | not provided: see the notes below |

mgwr 2.2.1 is the oracle (`test/reference/generate_gwr.py`): libpysal's Georgia example (n = 159)
for {adaptive, fixed} x {bisquare, gaussian, exponential}, AIC/BIC/CV, the interval search and the
`bw_max = inf` check, plus a seeded synthetic lon/lat sample (spherical, no constant, constant
column dropped). Adaptive bandwidths and the whole search history match exactly, fixed ones to
1e-6, and every result attribute (`params`, `bse`, `tvalues`, `local_r2`, `influ`, `cooks_d`,
`tr_s`, `tr_sts`, `sigma2`, `aicc`, `r2`, `adj_alpha`, `filter_tvals`, ...) to 1e-8.

Poisson/Binomial GWR and MGWR are checked in the same way (`gwr_glm.json`: seeded n = 80 samples
with an exposure offset, zero counts and 0/1 outcomes; `gwr_mgwr.json`: seeded lattices, a
second n = 100 sample and Georgia, standardized as mgwr requires): selection histories, per-site
`params`/`bse`/`tvalues`, criteria, deviances, `D2`/`pDev`, and for MGWR the bandwidths, the full
backfitting trace, `ENP_j`, `bse`, `aicc` and `filter_tvals` agree to 1e-8 (bandwidths exactly).

### Notes: GWR

- Memory and cost: the hat matrix and the weights are never stored. Each location recomputes its
  weights; `tr(S)`, `tr(S'S)` and the covariance diagonal are accumulated row by row (`O(n k)`
  memory, `O(n^2 k^2)` time, parallel over locations with thread-count independent sums).
  `fit/4` and `select_bandwidth/4` refuse more than `max_n` (default 10,000) observations.
- Replicated from mgwr: the adaptive bandwidth is the `m` nearest points including the focal
  point times `1.0000001` (ties at the m-th distance get a negligible positive bisquare weight),
  fractional adaptive bandwidths are truncated (with a warning), golden section is a literal port
  (`delta = 0.38197`, criterion-difference convergence, half-even rounding of integer probes,
  2-decimal rounding of the reported fixed bandwidth, `<=` tie to the first probe), the interval
  search keeps the earlier bandwidth on ties, `critical_tval` uses `n - 1` degrees of freedom,
  `adj_alpha` is `alpha * k / ENP`, and constant columns of `X` are dropped before the intercept
  is prepended.
- Deviations (see `test/reference/known_deviations.exs`): an adaptive search with fewer than
  `40 + 2 * (columns + 1)` observations and no explicit `bw_min`/`bw_max` raises `ArgumentError`
  (mgwr raises a numpy `kth out of bounds` error at the first evaluation); adaptive bandwidths
  below 2 (mgwr silently wraps or yields NaN weights), above `n`, non-positive fixed bandwidths,
  unsupported kernels and a zero interval step raise `ArgumentError`; `include_global: true`
  replaces `bw_max=np.inf` and cannot be combined with a numeric `bw_max`.
- No local condition-number warning: mgwr 2.2.1 has none. A search of `gwr.py`, `summary.py`,
  `sel_bw.py`, `search.py`, `diagnostics.py` and `utils.py` finds `warnings.warn` only for the
  deprecated `pool` argument, and the summary never reports a condition number. The local
  condition number exists only as an output of `local_collinearity()`, so Spatex adds no warning
  either; callers compare `condition_number` with their own threshold.
- `hat_matrix: true` (`test/reference/generate_gwr_search.py`, `gwr_hat.json`): `fit/4` keeps the
  full `n x n` hat matrix, as a list of rows or (`:tensor`) an `Nx.Tensor`. It agrees with
  `GWR(hat_matrix=True).fit().S` to 1e-8 for Georgia (adaptive, fixed, global), the spherical
  sample, no constant, Poisson (offset) and Binomial. It is opt-in because it costs `O(n^2)`
  memory; `max_cells` (default 4,000,000, `n = 2000`) guards it.
- `search: :scipy` (`gwr_search.json`) ports `scipy.optimize.minimize_scalar(method='bounded')`
  (the same literal Brent port as the spreg ML models) over `_init_section`'s bounds. The result
  `bw` is the optimizer's float, not truncated or rounded, `history` is empty and `nfev`/`converged`
  carry SciPy's result, as in mgwr. Adaptive bandwidths, scores and evaluation counts agree with
  mgwr to 1e-9; fixed ones to 1e-6 relative (see the deviations). Equal bounds raise, as in mgwr.
- `x_glob:` is mgwr's `Sel_BW(X_glob=...)`, which has no semi-parametric fit: it only enters
  `_init_section`, adding its number of columns to the default adaptive lower bound
  `40 + 2 * (columns + 1)`. Spatex reads only that count (the values are never used) and agrees
  with mgwr's search histories (golden section and scipy) to 1e-9; explicit bounds and fixed
  bandwidths are unaffected.
- Deviations of the scipy search (`known_deviations.exs`): mgwr 2.2.1 crashes with NumPy 2.5 for
  the AICc, AIC and BIC criteria (`int(bw)` of a shape-(1,) array), so the fixtures wrap the
  criteria to return floats (the behavior of older NumPy); `include_global` is not available
  (mgwr ignores `bw_max=np.inf` there).

### Notes: bandwidth intervals and spatial variability

- `Spatex.GWR.bandwidth_intervals/2` is `get_bws_intervals`: Akaike weights over the search history
  (`sel_hist`), cumulated in order of AICc, and the bounds of the first `len(cum_w < level) + 1`
  entries. It takes a `Bandwidth` (GWR) or an `MGWRResult` (MGWR, whose new `sel_hist` field keeps
  the `{bw, score}` history of every backfitting search; the last `k` searches are used, as in
  mgwr's `sel_hist[-k + j]`). The histories equal mgwr's to 1e-9 and adaptive intervals agree
  exactly. `search: :scipy` has no history (as in mgwr) and raises.
- Deviation (`known_deviations.exs`): a fixed search records each probe and its two-decimal
  rounding with the same score; pandas breaks such ties with numpy's unstable quicksort, Spatex
  by search order, so fixed bounds can differ by the rounding step (0.005).
- `Spatex.GWR.spatial_variability/4` is `GWRResults.spatial_variability`: the standard deviation
  of each coefficient surface against the standard deviations of surfaces refitted on randomly
  re-ordered coordinates, with the bandwidth searched again each time. Options follow
  `select_bandwidth/4` plus `permutations:` (mgwr's `n_iters`, default 1000), `seed:` (ChaCha8,
  `nil` draws and records one; mgwr's default is the fixed 5536) and `keep_simulations:`
  (`true`, `false`, `:tensor`). It is expensive (a search per permutation), so `max_n` defaults to
  2000. The observed standard deviations and bandwidth agree with mgwr to 1e-8; p-values and the
  mean simulated standard deviation are compared within Monte Carlo bands because numpy's stream
  differs (`test/reference/generate_gwr_inference.py`, `gwr_bws.json`).
- `MGWRResults.spatial_variability` is not provided: it repeats the multi-bandwidth backfitting
  (a full `mgwr/4` search) for every draw, which costs a multiple of the GWR test's search per
  permutation; it is left out of this release (not implemented, by choice). All other mgwr 2.2.1 GWR/MGWR result methods and `Sel_BW` options are now available.

### Notes: prediction and local collinearity

`test/reference/generate_gwr_predict.py` writes `gwr_predict.json` and `gwr_collinearity.json`
(mgwr 2.2.1). Every value agrees to 1e-8: `params`, `bse`, `tvalues`, `predictions` and, for the
Gaussian family, `local_r2` of `GWR.predict` on Georgia (mgwr's own test setup, whose ten
counties are also calibration points, and a 139/20 held-out split), a seeded spherical sample,
a model without a constant, seeded Poisson (with an offset) and Binomial samples and the
`exog_scale`/`exog_resid` arguments; and the correlations, VIFs, condition numbers and
variance-decomposition proportions of `GWRResults.local_collinearity()` (adaptive, fixed,
Gaussian and exponential kernels, global bandwidth, no constant) and of
`MGWRResults.local_collinearity()` (a lattice and Georgia).

- `predict/6` takes the calibration data and the new points together (the result structs hold no
  data): weights come from the distances to the calibration points, an adaptive bandwidth is the
  distance to the `bw`-th nearest calibration point, and the standard errors use the calibration
  `sigma2` (a fit at the calibration points runs first, as in mgwr) or `exog_scale`/`exog_resid`.
  Nothing `m x n` is stored. For Poisson and Binomial the predictions are the linear predictor,
  as in mgwr, and there is no local R-squared (mgwr raises).
- Deviations (see `test/reference/known_deviations.exs`): mgwr fails with an `IndexError` when
  there are more prediction points than calibration points because its local fit reads the
  calibration row `X[i]`; its `predy`, `influ` and residuals at prediction points mix prediction
  and calibration rows, so Spatex exposes only `params`, `bse`, `tvalues`, `predictions` and
  `local_r2`. Spatex drops, from `x_new` as well, the constant columns it drops from `x`
  (mgwr would fail on the column mismatch). `local_collinearity/3` raises `ArgumentError` with
  fewer than two explanatory variables (mgwr crashes inside `np.cov`/`corr`) and for an exactly
  singular local correlation matrix (mgwr raises `LinAlgError`).
- The decomposition uses a thin SVD of the `n x k` weighted design per location (`O(n k^2)`, no
  `n x n` matrix, parallel and thread-count independent); the correlations use the weighted
  covariance with the normalizing factor of `np.cov(aweights=...)` cancelled, which does not
  change the correlation.

### Notes: GLM GWR and MGWR

- Poisson (log link, optional offset) and Binomial (logit, 0/1) are fitted by IWLS at every
  location, a literal port of `spglm.iwls`: zero start coefficients, the odd Poisson starts
  (`y / offset` for the linear predictor, raw `y` for the mean), the *minimum* coefficient change
  against `iwls_tol` (`1e-5`, mgwr's `fit` default) with at most `iwls_max_iter` = 20 iterations,
  and a hat matrix from the weights that produced the final coefficients. `influ` carries the
  extra IRLS weight factor, `scale` is 1, `aic = deviance + 2 tr(S)`, `aicc` adds
  `2 tr(S) (tr(S) + 1) / (n - tr(S) - 1)`, and the Poisson null deviance omits `-(y - mu)` as in
  spglm. `r2`, `adj_r2` and `local_r2` are `nil` for GLM; `d2`, `adj_d2` and `p_dev` carry the
  deviance-based fit.
- Deviation: a zero Poisson count gives `NaN` local `deviance`/`pDev` in mgwr (`0 * ln 0`);
  Spatex uses `0 ln 0 = 0`. The IWLS settings are `iwls_tol`/`iwls_max_iter` because `tol` and
  `max_iter` already name the bandwidth-search settings.
- MGWR (`Spatex.GWR.mgwr/4`, Gaussian only) ports the backfitting of `search.multi_bw` literally:
  Gauss-Seidel residual updates, the SOC score (or `rss_score`), the `bws_same_times` freeze,
  per-column `multi_bw_min`/`multi_bw_max` (the outer `bw_min`/`bw_max` do not exist, mgwr ignores
  them too), the initial one-bandwidth search or `init_bw`, then the inference of
  `MGWR._chunk_compute_R`. The same default-bracket restriction as GWR applies (adaptive searches
  need `n >= 40 + 2 * columns` for the initial search and `n >= 42` per covariate).
- Memory guard: inference needs `R_j` for every covariate. They are never stored in full: the
  columns are processed in blocks (`chunk_size`, default 256), so memory is about
  `8 (k + 2) n chunk_size` bytes while time is `O(iterations * k * n^3)`. `mgwr/4` raises above
  `max_n` observations (default 5,000) unless `inference: false` (then the cost is `O(n^2)` per
  univariate fit and there is no guard). Results do not depend on the chunk size or the thread
  count.
- `standardize: true` standardizes `y` and the non-intercept columns (ddof 0) first; the default
  follows mgwr, which only documents that this should be done. Standard errors divide by the
  regressor values as in mgwr, so exact zeros give non-finite values (a warning names the
  columns). `influ`, `std_res`, `cooks_d`, `cv` are `nil` (mgwr raises), `MGWR.exact_fit` is not
  ported (it ignores kernel settings).
