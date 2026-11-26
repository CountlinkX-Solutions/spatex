# Changelog

All notable changes to Spatex are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/). Versions before 0.2.1 were not publicly released;
their entries are kept as the project's history.

## [Unreleased]

## [0.2.1] - 2026-11-26

First public release, and the first one published to Hex. The library code and its API are the
same as in 0.2.0; this release changes packaging, dependency metadata and documentation.

### Changed

- Spatex is installed from Hex (`{:spatex, "~> 0.2"}`) and downloads a precompiled NIF for its six
  targets (NIF versions 2.16 and 2.17), so those targets need no Rust toolchain. `SPATEX_BUILD=1`
  still builds the NIF from source.
- `rustler` is an optional dependency, as `rustler_precompiled` recommends: projects that use the
  precompiled NIF no longer fetch it. A source build, which includes using Spatex as a GitHub or
  path dependency, needs `{:rustler, "~> 0.38", runtime: false}` in the project's own
  dependencies; the notebooks' `Mix.install` lists it.
- `jason` is an optional dependency. The library never used it (only the oracle-test helper and
  the benchmarks do), so projects no longer fetch it through Spatex.
- The NIF checksum file is generated from the release artifacts when a release is published and
  ships only inside the Hex package; it is git-ignored, so a git checkout always builds the NIF
  from source.
- The Hex package also ships `.cargo/config.toml` (the target-specific link settings a source
  build needs on macOS and musl) and the third-party license texts in `LICENSES/`.
- The documentation links the sources of the release tag instead of `main`.
- The release workflow attaches the NIF artifacts from a separate job that runs on tag pushes
  only and is the only one with write access to the repository contents. It attaches all 12
  artifacts, including the Windows ones, which are named without the `lib` prefix, and fails
  instead of publishing an empty release. Every workflow action is pinned to a commit SHA.

### Fixed

- `NOTICE` said that Spatex contains no PySAL code and reimplements everything from the
  literature. It now lists the files that port or adapt upstream code (mapclassify, esda, SciPy,
  scikit-learn, the Python `heapq` module, spopt, fast_hdbscan, mgwr, spglm, spreg, libpysal and
  NumPy), separates them from the implementations validated against upstream output, names the
  published algorithms the special functions follow, and states that the test fixtures embed
  libpysal example datasets as well as upstream output. The upstream license texts
  are in `LICENSES/`.
- The README, the `Spatex.GeoFrame` documentation and the roadmap describe the public release:
  installation from Hex, the source-build opt-in, the optional dependencies and the stability of
  the 0.x API. The README and the notebooks say that the notebooks run from a git checkout.

## [0.2.0] - 2026-11-13

Milestone v0.2.0 of the PySAL features deferred from 0.1.x: simulations as `Nx` tensors,
`Graph.to_sparse`, GWR extras, fuzzy contiguity, spatial regimes, spatial SUR and 3SLS, panel
random-effects errors, more inequality indices and Kullback, point processes, alpha shapes,
random regions, Spenc and SA3. The Rust crates stay at 0.1.0.

### Added

- `Spatex.Regression.gm_kkp/4` ports spreg's `GM_ErrorRE` (`GM_KKP` in the deprecated
  `sp_panels.py` is the same estimator): the Kapoor-Kelejian-Prucha GM
  estimator of the spatial error panel with random effects, with `time_effects` and
  `full_weights`, returning the coefficients, `lam`, `sigma2_v`, `sigma2_1`, `vm` and the feasible
  GLS residuals. The formulas match spreg at 1e-9 at spreg's own optimizer results (NCOVR Kansas /
  Missouri / Oklahoma / Arkansas, a Columbus panel and a binary rook lattice); the two minimizations
  find the exact minimum instead of stopping early like L-BFGS-B (`lambda` within 1e-5 of spreg's).
  `regimes` and `slx_lags` are not supported.
- `Spatex.Regression.ml_error_re/4` ports spreg's `ML_ErrorRE`: the spatial
  error panel with random effects, with `time_effects`, coefficients, `lam`, the variance ratio
  `phi` and `sigma2_u`, `vm` with spreg's `lam` / `phi` information block, log-likelihood, AIC and
  Schwarz. The formulas match spreg at 1e-9 at a given `(lam, phi)` (NCOVR Kansas / Missouri /
  Oklahoma / Arkansas, a symmetric rook lattice and a non-symmetric Columbus panel); the optimum
  comes from a nested bounded search instead of L-BFGS-B. The old parity note was wrong: spreg's
  likelihood does not depend on LAPACK's eigenvector signs. A symmetric `W` with repeated
  eigenvalues uses the orthonormal eigenvectors (spreg's `numpy.linalg.eig` basis is then
  non-orthogonal and its objective is not a likelihood); a non-symmetric `W` with repeated or
  complex eigenvalues raises `ArgumentError`.
- `Spatex.Region.spenc/3` ports spopt's `Spenc`: spatially encouraged
  spectral clustering with the `pairwise_kernels` affinities, `nearest_neighbors` and
  `precomputed`, and `assign_labels: :discretize | :kmeans | :hierarchical`. The affinity matrix,
  normalized Laplacian eigenvalues and embedding (up to eigenvector sign) match spopt, with a
  dense symmetric eigensolver in place of ARPACK; the randomized assignment uses seeded ChaCha8
  and is validated statistically, `:hierarchical` is exact (spopt 0.8.0 itself raises
  `TypeError` there). The result exposes the embedding and eigenvalues, optionally as `Nx`
  tensors.
- `Spatex.Regression.sur_lag_iv/4` ports spreg's `SURlagIV`: seemingly
  unrelated regressions with a spatial lag of the dependent variable in every equation, estimated
  by three-stage least squares with `W X` ... `W^p X` (`:w_lags`) and, with `:lag_q`, the lags of
  the user instruments as instruments, plus optional user endogenous variables per equation. It
  reports the coefficients (`rho` per equation), `varb`, standard errors and z statistics, the
  cross-equation covariance and correlation, the Chow tests and, with `spat_diag: true`, the Wald
  test that the spatial coefficients are jointly zero. Everything matches spreg at 1e-9. spreg
  computes no spatial impacts for this class. `regimes` is not ported, which completes the
  spatial SUR family.
- `Spatex.Regression.sur_error_ml/4` ports spreg's `SURerrorML`: seemingly
  unrelated regressions with a spatial error parameter per equation, estimated by maximum
  likelihood (per-equation ML error starts, iterated SUR residuals, then SUR GLS steps alternated
  with a joint search of the lambdas), with `vm`, standard errors and z statistics, the
  likelihood-ratio tests of a diagonal covariance and of the spatial parameters, the asymptotic
  covariance of the lambdas and the error covariance, the Wald tests of equal and of jointly zero
  lambdas, and the cross-equation Chow tests. Everything matches spreg at 1e-9 given its lambda
  path; the lambdas are the tighter optimum (within 5e-5 of spreg's L-BFGS-B). `regimes` is not
  ported.
- `Spatex.Regression.sur_error_gm/4` ports spreg's `SURerrorGM`: seemingly
  unrelated regressions with a spatial error parameter per equation, estimated by the
  Kelejian-Prucha moments of each equation's OLS residuals and one SUR GLS step on the spatially
  filtered variables, with `vm`, standard errors, z statistics, the lambdas, the error covariance
  and correlation and the cross-equation Chow tests. Everything but lambda matches spreg at 1e-9
  given its lambdas; lambda is the exact moment minimizer (within ~5e-6 of spreg's L-BFGS-B).
  `regimes` is not ported.
- `Spatex.Regression.three_sls/5` ports spreg's `ThreeSLS`: three-stage least
  squares for a system of equations with per-equation endogenous variables and instruments (2SLS per
  equation, cross-equation covariance, one GLS step on the instrumented system), with `vm`,
  standard errors, z statistics, the error covariance and correlation and the cross-equation Chow
  tests, matching spreg at 1e-9. `regimes` is not ported.
- `Spatex.Region.sa3/3` ports spopt's `SA3`: spatially constrained Ward
  clustering per connected component whose merge tree is cut with HDBSCAN-style extraction
  (`:min_cluster_size`, `extraction: :eom | :leaf`), returning contiguous multiscale clusters and
  noise (`-1`). Labels equal spopt's and the per-component linkage matrices (exposed in
  `components`) agree to 1e-12; `min_cluster_size` below 2 raises (spopt returns non-contiguous
  clusters). The extraction is a Rust port of `fast_hdbscan`'s condensed-tree functions with
  its `float32` arithmetic.
- `Spatex.Regression.ml_lag_regimes/5` ports spreg's `ML_Lag_Regimes`:
  the maximum-likelihood spatial lag model with coefficients by regime (`constant_regi`,
  `cols2regi`), one `rho` on the regime design or, with `regime_lag_sep`, one `ml_lag/4` fit per
  regime on its subgraph; `method: :full | :ord | :lu`, `epsilon`, `spat_impacts`, `pr2_e` and
  Chow tests. Matches spreg on 29 committed fixture cases (at spreg's `rho` to 1e-9; the Brent
  search to its `xatol`); the Chow restrictions need no `spreg_compatible` switch because `rho`
  never precedes regime-specific columns. `slx_lags`, `cores` and `vm` are not ported and spreg's
  shape error for columns constant within a regime is replaced by dropping them with a warning.
- `Spatex.Regression.ml_error_regimes/5` ports spreg's `ML_Error_Regimes` (the last
  of the regimes models): the maximum-likelihood spatial error model with coefficients by
  regime (`constant_regi`, `cols2regi`), one `lambda` on the regime design or, with
  `regime_err_sep`, one `ml_error/4` fit per regime on its subgraph; `method: :full | :ord |
  :lu`, `epsilon` and Chow tests. Matches spreg on 27 committed fixture cases (at spreg's
  `lambda` to 1e-9; the Brent search to its `xatol`), including spreg's construction of the
  pooled lagged design (`regimes(W X)`, not `W regimes(X)`), for which the native ML error fit
  gained an entry point taking the lagged design. `slx_lags`, `cores` and `vm` are not ported and
  spreg's shape error for columns constant within a regime is replaced by dropping them with a
  warning.
- `Spatex.Region.random_region/2` and `random_regions/2` port spopt's `RandomRegion` and
  `RandomRegions`: random partitions of the areas with optional `:num_regions`,
  `:cardinality`, contiguity (pass a `Spatex.Graph`), `:compact`, `:maxiter` and `:max_swaps`, and
  `:permutations` for the plural, seeded with ChaCha8 (`:seed`, recorded; permutation `i` uses
  stream `i`, so results are independent of the thread count). Infeasible requests return
  `feasible: false` as in spopt. numpy's generator is not reproduced: tests check structure and
  compare summary distributions with spopt references. spopt's endless loops (`num_regions`
  above `n - 1`, no possible swap) raise or are bounded by `:max_swaps`.
- `Spatex.PointPats.alpha_shape/2` ports `libpysal.cg.alpha_shape`: the alpha shape of a
  point set as a list of WKB polygons (shells clockwise, holes counter-clockwise), assembled from
  the Delaunay boundary edges without GEOS. Polygon and hole counts, ring vertices and areas equal
  libpysal on committed fixtures (uniform, clustered, annulus, island-in-hole, two-blob and
  C-shaped sets); `alpha = 0` is the convex hull and degenerate input raises `ArgumentError`
  (documented deviations).
- `Spatex.PointPats.alpha_shape_auto/2` ports `libpysal.cg.alpha_shape_auto`: the
  tightest alpha shape that is one polygon containing every point, with `:step`, `:return_radius`
  and `:return_circles` (bounding-circle centres). It follows libpysal's float comparison literally,
  so the oracle accepts any of libpysal's results under the six vertex orders of a triangle.
- `Spatex.Graph.build_fuzzy_contiguity/2` ports libpysal's fuzzy contiguity on
  `Polygon` / `MultiPolygon` WKB: `:tolerance` (share of half the shorter side of the total
  bounds) or `:buffer`, and `:predicate` (`:intersects`, `nil` for bounding boxes, and without a
  buffer `:touches`, `:overlaps`, `:within`, `:contains`, `:covers`, `:covered_by`,
  `:contains_properly`, `:crosses`). Without GEOS, a buffer is handled as an exact
  polygon-to-polygon distance test (`distance <= 2 * buffer`); neighbor sets match libpysal on
  columbus, sids2 and synthetic fixtures, except pairs within about 0.15 % below `2 * buffer`,
  where GEOS's chord-approximated buffers are smaller (documented deviation).
- `Spatex.PointPats.normal/2` and `cluster_normal/2` port pointpats' `random.normal` and
  `random.cluster_normal`: bivariate normal points truncated to a box or convex
  window (`:center`, `:cov` as a variance or a 2 x 2 matrix, `:size`, `:n_seeds`, `:seed`), with
  seeded ChaCha8 streams per `(seed, replication)` independent of the thread count. pointpats
  is unseeded, so parity is statistical (sample moments and nearest-neighbour distances against
  summaries of pointpats realizations).
- `Spatex.PointPats.strauss/2` ports pointpats' `random.strauss`: the birth-death
  Metropolis-Hastings sampler with `:gamma`, `:r`, `:n_iter` and `:max_iter`, `:size`,
  `:intensity` and `:seed`, ChaCha8 streams per `(seed, replication)`. Parity is statistical:
  close-pair counts and nearest-neighbour distances match summaries of pointpats realizations.
- `Spatex.Regression.tsls_regimes/6` ports spreg's `TSLS_Regimes`:
  two-stage least squares with coefficients that vary by regime (`constant_regi`, `cols2regi` over
  the exogenous then endogenous columns, instruments always by regime, `regime_err_sep` with one
  2SLS per regime, White and HAC errors, `sig2n_k`) and the global and per-variable Chow tests,
  returned as `Spatex.Regression.TSLSRegimes`. It matches spreg to 1e-9 on committed Columbus
  fixtures; `spat_diag`, `robust: 'ogmm'`, `slx_lags` and `TSLS_Endog_Regimes` are not ported.
  spreg's Chow restrictions assume global columns come last and compare the wrong coefficients
  when exogenous columns are global and endogenous ones vary by regime; Spatex tests the right
  ones by default and `spreg_compatible: true` reproduces spreg's values.
- `Spatex.Regression.gm_lag_regimes/5` ports spreg's `GM_Lag_Regimes`: the
  spatial lag model with coefficients that vary by regime (`yend` / `q`, `w_lags`, `lag_q`,
  `constant_regi`, `cols2regi`, `regime_lag_sep` with one lag model per regime on the regime's
  subgraph or the pooled fit on the union of the subgraphs, White (default) and HAC errors,
  `sig2n_k`, Anselin-Kelejian test, `pr2_e`, `spat_impacts`, global and per-variable Chow tests),
  returned as `Spatex.Regression.GMLagRegimes`. It matches spreg to 1e-9 on 33 committed Columbus
  fixture cases; `slx_lags`, `robust: 'ogmm'` and `GM_Lag_Endog_Regimes` are not ported. The Chow
  layout defect of `tsls_regimes/6` also applies (global exogenous columns with a
  regime-specific `W y`); the same `spreg_compatible` flag reproduces spreg.
- `Spatex.Regression.gm_error_regimes/5` and `gm_combo_regimes/5` port spreg's
  `GM_Error_Regimes`, `GM_Endog_Error_Regimes` (with `yend` / `q`) and `GM_Combo_Regimes` for the
  KP98 estimator: the spatial error and combo models with coefficients that
  vary by regime (`constant_regi`, `cols2regi`, `regime_err_sep` with one error model per regime on
  the regime's subgraph, `w_lags`, `lag_q`, `regime_lag_sep`, `pr2_e`, global and per-variable
  Chow tests), returned as `Spatex.Regression.GMErrorRegimes` and `GMComboRegimes`. Lambda is the
  exact minimizer of the KP98 moment objective, so fits agree with spreg to ~1e-3 on 36 of 43
  committed Columbus fixture cases; on seven, where spreg's L-BFGS-B stops away from its own
  minimizer (flat objective with binary weights), lambda is checked against the minimizer.
  `slx_lags` and `GMM_Error_Regimes` are not ported. The Chow layout defect of `tsls_regimes/6` also applies to the endogenous and combo
  models (same `spreg_compatible` flag); spreg fails with a shape error when a column is constant
  within a regime under `regime_err_sep`, Spatex drops it with a warning.
- `gm_error_regimes/5` and `gm_combo_regimes/5` accept `estimator: :het` and `:hom`, porting
  spreg's `GM_Error_Het_Regimes`, `GM_Endog_Error_Het_Regimes`, `GM_Combo_Het_Regimes` and the
  three `_Hom_` classes with `:max_iter`, `:epsilon`, `:step1c` (`:het`) and `:a1` (`:hom`, default `:het` as in spreg's regimes classes). The pooled fits reuse
  the base Het / Hom estimators on the regime design (second stage on the filtered base columns,
  spreg's iteration order and first-step-lambda quirks included) and the separate regressions
  fit `gm_error/4` / `gm_combo/4` per regime; covariances include `lambda`, which the separate
  Chow tests treat as one more coefficient. 108 Columbus fixture fits agree with spreg at 1e-8
  when fed spreg's lambdas and at 1e-3 with the optimizer; spreg's +-0.9 clamp of the separate
  `GM_Endog_Error_Het` fits (a defect) and its `<r>lambda` coefficient names are not reproduced.
- `Spatex.GWR.bandwidth_intervals/2` ports mgwr's `get_bws_intervals` (Akaike-weight confidence
  intervals for the selected bandwidth) for GWR (`Bandwidth`) and MGWR (`MGWRResult`, which now
  carries `sel_hist`, the history of every backfitting search); and `Spatex.GWR.spatial_variability/4`
  ports the Monte Carlo test of spatial variability (`:permutations`, `:seed`, `:keep_simulations`
  with `true`, `false` or `:tensor`). Intervals match mgwr to 1e-9 on committed fixtures (fixed
  bounds to the 0.005 rounding step); the observed statistic matches exactly and p-values agree
  within Monte Carlo bands (ChaCha8 streams). This completes the remaining mgwr features.
- `Spatex.Inequality.atkinson/2`, `schutz/1`, `wolfson/1` and `lorenz_curve/1` port the inequality
  package's `Atkinson`, `Schutz`, `wolfson` and `lorenz_curve`. `atkinson/2` and
  `schutz/1` return `Spatex.Inequality.Atkinson` (`a`, `ede`, `epsilon`) and
  `Spatex.Inequality.Schutz` (`distance`, `intersection_point`, `coefficient`) structs. All match
  PySAL to 1e-9 on committed fixtures; `pen` and `pengram` (matplotlib charts) are not ported.
- `Spatex.Inequality.polarization/3` ports the spatial polarization index `inequality.polarization.S`:
  quantile (`:k`) or explicit (`:bins`) groups, the number of connected same-group
  components of the graph, and a seeded permutation null (`:permutations`, default 999, `:seed`,
  `:keep_simulations` with `true`, `false` or `:tensor`, default `true`). The statistic, component
  counts and labels match PySAL exactly; the p-value is statistical (ChaCha8 streams).
- `Spatex.Dynamics.kullback/2` ports giddy's Kullback information test of Markov homogeneity. It
  takes a list of `r x r` count matrices, a rank-3 tensor or a `SpatialMarkov` result. giddy sums
  one term over the wrong axis, so its value is wrong unless every stratum has equal row and column
  totals; Spatex computes Kullback's formula and `giddy_compatible: true` reproduces giddy's value.
- `Spatex.GWR.predict/6` predicts at new locations (`mgwr` `GWR.predict`): local coefficients,
  standard errors, t-values, predictions and, for the Gaussian family, local R-squared, for the
  Gaussian, Poisson and binomial families, with the calibration scale and residuals taken from a
  fit or passed as `exog_scale`/`exog_resid`. It supports more prediction points than
  observations, which mgwr does not.
- `Spatex.GWR.local_collinearity/3` returns the local correlations, variance inflation factors,
  condition numbers and variance-decomposition proportions of the geographically weighted design
  (`GWRResults.local_collinearity`), and with `bws:` the condition numbers and proportions of an
  MGWR design (`MGWRResults.local_collinearity`). mgwr has no local condition-number warning, so
  none is added.
- `Spatex.GWR.fit/4` takes `hat_matrix: true` or `:tensor` to keep the full `n x n` hat matrix `S`
  in `result.hat_matrix` (`GWR(hat_matrix=True)`), guarded by `max_cells` (default 4,000,000). It
  matches mgwr to 1e-8 for the Gaussian, Poisson and binomial families and is off by default, so a
  normal fit keeps its `O(n k)` memory.
- `Spatex.GWR.select_bandwidth/4` takes `search: :scipy`, mgwr's `search_method='scipy'`
  (SciPy's bounded Brent minimizer, a literal port shared with the spreg ML models); the result
  carries SciPy's float `bw`, `nfev` and `converged`. mgwr 2.2.1 crashes with NumPy 2.5 for the
  AICc, AIC and BIC criteria here; Spatex follows mgwr's algorithm.
- `Spatex.GWR.select_bandwidth/4` takes `x_glob:`, mgwr's `Sel_BW(X_glob=...)`: its columns only
  raise the default adaptive lower bound `40 + 2 * (columns + 1)`, as in mgwr, which has no
  semi-parametric fit.
- `Spatex.Graph.to_sparse/1` exports the graph as the CSR structure of libpysal's `Graph.sparse`:
  `%{indptr: _, indices: _, data: _, shape: {n, n}}` with `Nx` tensors (`indptr` `{:s, 64}`,
  `indices` `{:u, 32}`, `data` `{:f, 64}`). It stores exactly libpysal's entries (island
  self-loops and explicit zero weights are explicit zeros, indices sorted within rows) and
  reflects the current transform. Needs the optional `:nx` dependency and raises an
  `ArgumentError` naming `{:nx, "~> 0.10"}` without it.
- `keep_simulations: :tensor` returns the kept simulations as an `Nx.Tensor` of type `{:f, 64}`
  built straight from the native binary, with no list decoding and the same shape as the list form.
  It is accepted wherever `:keep_simulations` was: the `Spatex.ESDA` global and local statistics,
  `Spatex.Dynamics` local and regional tau, `Spatex.Inequality.spatial_gini/3` and the
  `Spatex.PointPats` distance tests. `true` and `false` and the defaults are unchanged. At
  n=10,000 with 999 permutations, local Moran with `:tensor` takes 134 ms against 3.03 s for
  `true`. Needs the optional `:nx` dependency.

## [0.1.1] - 2026-11-12

Release-tooling fix. The library code is the same as in 0.1.0.

### Fixed

- The "Release NIFs" workflow now builds every precompiled NIF artifact (six targets, NIF 2.16 and
  2.17) and attaches it to the GitHub release. The 0.1.0 release has no NIF artifacts because the
  workflow failed on its first run:
  - `spatex_native` now declares the `nif_version_2_15`, `nif_version_2_16` and `nif_version_2_17`
    features, which `rustler_precompiled` has needed since Rustler 0.30;
  - non-cross jobs no longer pass `use-cross` as the string `"false"`, which made the action install
    the Linux `cross` binary on Windows and macOS;
  - macOS jobs moved from the retired `macos-13` and `macos-14` images to `macos-15-intel` and
    `macos-15`.

### Changed

- The release workflow also runs as a dry run, without touching a release, on pull requests that
  change the build inputs and on demand (`workflow_dispatch`).

## [0.1.0] - 2026-11-12

First release: a PySAL-style spatial analysis toolbox for Elixir with a pure-Rust core
(`spatex_core`) behind a thin Rustler NIF layer. Statistics are verified against PySAL fixtures
(to 1e-9 where deterministic; permutation inference statistically); permutations are seeded and
identical at any thread count. See [docs/parity.md](docs/parity.md) for the function-level mapping
and every known deviation, and [docs/benchmarks.md](docs/benchmarks.md) for timings.

### Added

- **Spatial graphs** (`Spatex.Graph`, libpysal): construction from arrays and adjacency lists,
  R/B/D/V transforms, spatial lag, cardinalities, isolates, S0-S2, symmetry checks; queen/rook
  and block contiguity from WKB polygons, k-nearest neighbors, distance bands, kernel weights,
  Delaunay/Gabriel/relative-neighborhood triangulation; haversine (great-circle, km) KNN and
  distance bands; seeded jitter for coincident points; Voronoi weights (bounding-box, convex-hull
  or extended clipping); strict (exact-predicate) and perimeter-weighted contiguity; vertex
  contiguity for lines; GAL and GWT read and write.
- **Global ESDA** (`Spatex.ESDA`, esda): Moran's I (univariate, bivariate, rates), Geary's C,
  Getis-Ord G and join counts with analytic and seeded permutation inference.
- **Local ESDA**: local Moran (univariate and bivariate, cluster labels), local Geary, G and G*,
  local join counts (univariate, bivariate and multivariate), multivariate local Geary, LOSH,
  `island_weight` for islands in conditional permutations and false discovery rate cut-offs.
- **Map classification** (`Spatex.Classify`, mapclassify): equal interval, quantiles, box plot,
  standard deviation, percentiles, maximum breaks, head/tail breaks, user defined, Fisher-Jenks
  (exact and sampled), Jenks-Caspall variants, natural breaks, Max-P (numpy's MT19937
  reproduced) and pretty breaks, plus `gadf/2` and `k_classifiers/2`.
- **Point patterns** (`Spatex.PointPats`, pointpats): centrography, G/F/J/K/L functions with seeded
  simulation envelopes, Poisson and cluster Poisson processes, quadrat tests.
- **Spatial regression** (`Spatex.Regression`, spreg): OLS with non-spatial and spatial diagnostics,
  White and HAC robust errors, LM tests; maximum-likelihood spatial lag and error models (full and
  Ord log-determinant methods, spatial impacts); GM estimators (`gm_lag`, `gm_error` with KP98,
  heteroskedastic and homoskedastic variants, `gm_combo`, 2SLS with endogenous regressors);
  SLX / spatial Durbin terms (`slx_lags`, `slx_vars`); OLS with regimes and Chow tests; panel
  ML lag and error models with fixed effects and the lag model with random effects; seemingly
  unrelated regressions (SUR) with LR, Breusch-Pagan LM, Chow (Wald) and spatial LM tests.
- **Geographically weighted regression** (`Spatex.GWR`, mgwr): Gaussian, Poisson and binomial
  GWR and MGWR with bandwidth selection (bisquare, Gaussian, exponential kernels; fixed or
  adaptive; AICc, AIC, BIC, CV).
- **Regionalization** (`Spatex.Region`, spopt): SKATER, spatial Ward, Region-K-Means, AZP (basic,
  tabu and simulated annealing) and Max-p, with objective and contiguity helpers.
- **Dynamics and inequality** (`Spatex.Dynamics`, `Spatex.Inequality`, giddy, inequality): Markov
  chains, spatial Markov with homogeneity tests, LISA Markov, steady state and first passage
  times, full-rank and geographic-rank Markov, Kendall (O(n log n)), spatial, local, neighbor,
  neighborhood and regional tau, LISA spillover, theta, Gini, Theil with decomposition, spatial
  Gini.
- **Integration**: optional `Spatex.GeoFrame` adapter, `Spatex.Explorer.to_dataframe/2`, Nx and
  Explorer inputs (series, label series and data frames for panels), and Livebook notebooks
  (`notebooks/`).
- **API conventions** (see [docs/architecture.md](docs/architecture.md#9-api-conventions)): every
  entry point takes an optional keyword list with unknown keys rejected; `seed: nil` draws a seed
  that is recorded in the result; `:permutations` is the one simulation-count name; local ESDA
  defaults to `keep_simulations: false`.
- **Native guards**: `max_cells` / `max_edges` limits on dense and graph allocations, and
  long-running NIFs scheduled on dirty CPU schedulers.
- **Release tooling**: Benchee benchmarks against PySAL (`bench/`, results in
  [docs/benchmarks.md](docs/benchmarks.md)), `rustler_precompiled` scaffolding with a tag-triggered
  release workflow, and a Hex package definition. Spatex compiles the NIF from source until a
  checksum file for the release artifacts is published (Rust toolchain required).

### Requirements

Elixir ~> 1.17, Erlang/OTP 27+, and (until precompiled NIFs are usable) Rust 1.91 or newer.

### Known limitations

- Local ESDA with `keep_simulations: true` decodes the full simulation matrix into nested Elixir
  lists, which is slow and memory-heavy for large n (about 3 s at n=10,000 with 999 permutations).
  The default is `keep_simulations: false`.
- Precompiled NIFs could not be downloaded yet, so a Rust toolchain was still required. Spatex was
  not published to Hex.

### Not included yet

Fuzzy contiguity, regimes for the spatial regression models, ML_ErrorRE, spatial SUR and
three-stage least squares, reactive tabu AZP, and the other PySAL variants listed as planned or
not ported in the [parity matrix](docs/parity.md).

[Unreleased]: https://github.com/CountlinkX-Solutions/spatex/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/CountlinkX-Solutions/spatex/releases/tag/v0.2.1
