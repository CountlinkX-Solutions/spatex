# Roadmap

The milestones below record how Spatex was built and what each one delivered. Every deliverable ships with oracle tests against PySAL.

| # | Milestone | Focus |
|---|---|---|
| M0 | Foundation | Scaffold, Graph core, input normalization |
| M1 | Weights | libpysal builders and I/O |
| M2 | Global ESDA | Moran, Geary, G, join counts |
| M3 | Local ESDA | LISA family, LOSH, FDR |
| M4 | Classification | mapclassify |
| M5 | Integration | GeoFrame / Explorer, Livebook, docs |
| M6 | Point patterns | pointpats |
| M7 | Spatial regression | spreg |
| M8 | Dynamics and inequality | giddy, inequality |
| M9 | Regionalization and GWR | spopt, mgwr |
| M10 | Release | Precompiled NIFs, Hex, benchmarks |
| v0.2.0 | Deferred PySAL features | Items left planned or not yet implemented in 0.1.x |

## M0 Foundation

Goal: a buildable repo and a usable immutable graph.

- [x] Scaffold: mix project, Cargo workspace (`spatex_core`, `spatex_native`), `.tool-versions`, CI (Elixir matrix + Rust job), lint (format, credo, clippy)
- [x] Oracle infra: `test/reference/*.py`, fixtures dir, the test-only reference helper, `known_deviations.exs`
- [x] `Spatex.Graph` core: CSR `ResourceArc`, `from_arrays`, `from_adjacency`, `transform` (R/B/D/V), `lag`, `cardinalities`, `isolates`, `s0/s1/s2`, `symmetric?`, `to_adjacency`
- [x] Input normalization: lists, `Nx.Tensor`, `Explorer.Series`

Exit: CI green; graph transforms and lag match libpysal.graph fixtures at 1e-9.

## M1 Weights

Goal: all common libpysal graph builders.

- [x] `build_knn` (`Graph.build_knn`)
- [x] `build_distance_band` (binary / alpha)
- [x] `build_kernel`: gaussian, triangular, bisquare, tricube, epanechnikov, uniform; fixed and adaptive (`:epanechnikov` and `:uniform` are aliases of `:parabolic` and `:boxcar`)
- [x] `build_contiguity` queen / rook from WKB
- [x] `build_block_contiguity`
- [x] `build_triangulation`: delaunay, gabriel, relative_neighborhood
- [x] GAL / GWT read and write

Exit: adjacency tables equal libpysal on fixture datasets (including islands, ties, multipolygons).

## M2 Global ESDA

- [x] `moran` (esda.Moran), `moran_bv` (Moran_BV), `moran_rate` (Moran_Rate)
- [x] `geary` (Geary), `getis_ord_g` (G), `join_counts` (Join_Counts)

Exit: analytic statistics at 1e-9; p_sim statistically consistent; seed determinism across thread counts.

## M3 Local ESDA

- [x] `moran_local` (Moran_Local) with quadrants, `geoda_quads`, p_sim alternatives, conditional randomization
- [x] `moran_local_bv`, `geary_local`, `g_local` (G_Local, star true/false), `join_counts_local`, `losh`
- [x] `fdr` (esda.fdr)

Exit: local statistics and quadrants equal esda fixtures; p-values within statistical tolerance.

## M4 Classification

- [x] equal_interval, quantiles, box_plot, std_mean, percentiles, maximum_breaks, natural_breaks, fisher_jenks, jenks_caspall, head_tail_breaks, user_defined
- [x] `%Spatex.Classify{bins, counts, yb, k, method, adcm, gadf, tss}`

Exit: bins/counts/yb equal mapclassify fixtures (stochastic methods compared by fit metrics).

## M5 Integration

- [x] `Spatex.GeoFrame` adapter (WKB / coords in, Explorer columns out)
- [x] Livebook notebooks, guides, ExDoc

Exit: end-to-end notebook from GeoDataFrame to LISA map data runs in CI. The offline tour and PySAL comparison notebooks run in CI; the GeoDataFrame notebook needs a local GeoFrame checkout (GeoFrame is not on Hex yet), so it runs locally (`SPATEX_GEOFRAME_PATH`).

## M6 Point patterns

- [x] Centrography: mean_center, weighted_mean_center, manhattan_median, euclidean_median, std_distance, ellipse, hull, mbr, minimum rotated rectangle, minimum bounding circle
- [x] Distance statistics G, F, J, K, L with simulation envelopes; `poisson` and `cluster_poisson`
- [x] Quadrat test (rectangle and hexagon grids, chi-square and Monte Carlo p-values)

Exit: pointpats fixtures match; envelopes deterministic per seed.

## M7 Spatial regression

- [x] OLS with diagnostics: R2, adj R2, F, t-stats, Jarque-Bera, Breusch-Pagan, Koenker-Bassett, White, condition number, Moran on residuals, LM-lag / LM-error / robust / SARMA
- [x] `ml_lag`, `ml_error` (spreg.ML_Lag, ML_Error)
- [x] `tsls`, `gm_lag` (2SLS), `gm_error` (KP98, Het, Hom; Endog variants), `gm_combo`

Exit: coefficients and diagnostics match spreg at 1e-6 or better (ML tolerance documented).

## M8 Dynamics and inequality

- [x] giddy: `markov`, `spatial_markov`, `lisa_markov`, rank mobility (`kendall_tau`, `spatial_tau`, `theta`)
- [x] inequality: `gini`, `theil`, `theil_d`, `spatial_gini`

Exit: transition matrices and indices match fixtures.

## M9 Regionalization and GWR

- [x] spopt: `skater`, `ward_spatial`, `region_kmeans`, `azp` (basic), `max_p`, objective helpers
- [x] `Spatex.GWR`: GWR with bandwidth selection (mgwr) for the Gaussian, Poisson and Binomial families, and multiscale GWR (`mgwr/4`)

Exit: deterministic methods match; heuristic methods validated by contiguity and objective constraints.

## M10 Release

- [x] `rustler_precompiled` targets and release workflow (the checksum file is generated from the published release artifacts and ships in the Hex package)
- [x] Hex package and ExDoc build (first published as `0.2.1`)
- [x] Benchmarks vs PySAL ([docs/benchmarks.md](benchmarks.md))

Exit: `mix deps.get` works without a Rust toolchain on the precompiled targets (met with the public `0.2.1` release on Hex; earlier versions were not publicly released and built from source); published benchmarks (met).

## v0.2.0 Deferred PySAL features

Shipped as `0.2.0` (see the [changelog](../CHANGELOG.md)), in planned order:

- [x] `keep_simulations: :tensor`: kept simulations as an `Nx` tensor, without list decoding
- [x] `Graph.to_sparse/1` (libpysal `Graph.sparse`)
- [x] GWR: `predict`, hat matrix, local collinearity, condition-number warning, bandwidth intervals, scipy search, semi-parametric `X_glob` (mgwr)
- [x] Fuzzy contiguity (libpysal `build_fuzzy_contiguity`)
- [x] Regimes for the spatial regression models (TSLS, GM, ML)
- [x] Spatial SUR and three-stage least squares (ThreeSLS, SURerrorGM, SURerrorML, SURlagIV)
- [x] Panel random-effects error models (ML_ErrorRE, GM_KKP), starting with a feasibility spike
- [x] Atkinson, Schutz, Wolfson, spatial polarization (inequality) and Kullback (giddy)
- [x] Point processes: `normal`, `cluster_normal`, `strauss` (pointpats)
- [x] Alpha-shape hull (libpysal `alpha_shape`, `alpha_shape_auto`)
- [x] Random regions (spopt `RandomRegion`, `RandomRegions`)
- [x] Spenc and SA3 (spopt)

Not scheduled (see [docs/parity.md](parity.md)): AZP reactive tabu (broken upstream in spopt and
underspecified, so there is no oracle) and inequality's `pen`/`pengram` (matplotlib charts).

Exit: every item above is available with oracle tests against PySAL, or documented as a deviation.

## Out of scope

splot, tobler, momepy, spaghetti, access, segregation (revisit later).
