# Spatex

[![Hex.pm](https://img.shields.io/hexpm/v/spatex.svg)](https://hex.pm/packages/spatex)
[![HexDocs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/spatex)
[![CI](https://github.com/CountlinkX-Solutions/spatex/actions/workflows/ci.yml/badge.svg)](https://github.com/CountlinkX-Solutions/spatex/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

PySAL-style spatial analysis for Elixir, backed by a Rust core (Rustler).

Version `0.2.1` is the first public release. It covers spatial weights, ESDA, map
classification, point patterns, spatial regression, GWR, spatial dynamics, inequality and
regionalization (milestones M0-M10 and the v0.2.0 milestone of deferred PySAL features). See the
[changelog](CHANGELOG.md), the [roadmap](docs/roadmap.md) and the [parity matrix](docs/parity.md)
for what is and is not covered.

## What it is

spatex brings the PySAL toolbox to the BEAM: spatial weights (graphs), exploratory spatial data analysis (ESDA), map classification, point patterns, spatial regression, spatial dynamics, inequality, and regionalization. Algorithms follow PySAL and are verified against it numerically.

No mature ESDA library exists in Elixir or Rust today; spatex fills that gap.

## Relation to GeoFrame

[GeoFrame](https://github.com/CountlinkX-Solutions/geoframe) provides GeoPandas-style dataframes (GeoArrow + Rust). spatex is to GeoFrame what PySAL is to GeoPandas: **complementary and independent**.

- spatex does not depend on GeoFrame. Geometry input is generic: WKB binaries, `{x, y}` lists, `{n, 2}` Nx tensors.
- An optional adapter, `Spatex.GeoFrame`, calls GeoFrame's public API at runtime (no compile-time dependency; it raises a clear error when GeoFrame is absent). It builds graphs from a `GeoDataFrame` or `GeoSeries` and writes ESDA and classification results back as columns. `Spatex.Explorer.to_dataframe/2` gives the same per-site results as a plain `Explorer.DataFrame`, without GeoFrame.

```elixir
frame = GeoFrame.GeoDataFrame.from_wkb!(df, "geometry")        # polygons + attributes
graph = Spatex.GeoFrame.graph(frame, :queen)                    # or :rook, :knn (k:), :distance_band (threshold:), :kernel, :triangulation
crime = Spatex.GeoFrame.values(frame, "crime")                  # Explorer.Series
lisa  = Spatex.ESDA.moran_local(crime, graph, permutations: 999, seed: 1)
frame = Spatex.GeoFrame.put_results(frame, lisa)                # adds lisa_is, lisa_q, lisa_p_sim, lisa_cluster, ...
frame = Spatex.GeoFrame.put_results(frame, Spatex.Classify.quantiles(crime, k: 5))  # adds class_yb
```

Distance methods use polygon centroids and refuse geographic (lon/lat) CRSs unless `allow_geographic: true` (or `:warn`); project the frame first.

GeoFrame is an optional integration and is not on Hex yet, so using the adapter currently needs a local GeoFrame checkout in your own dependencies (for example `{:geoframe, path: "../geoframe"}`). Spatex's own GeoFrame tests and notebooks find that checkout through `SPATEX_GEOFRAME_PATH`: `SPATEX_GEOFRAME_PATH=/path/to/geoframe mix test --only geoframe`.

## Features

| Area | PySAL package | Planned scope | Milestone | Status |
|---|---|---|---|---|
| Spatial weights / graphs | libpysal | Contiguity, KNN, distance band, kernel, triangulation, transforms, lag | M0-M1 | available: `Spatex.Graph` core (from arrays/adjacency, R/B/D/V transforms, lag, S0-S2); `build_knn`, `build_distance_band`, `build_kernel` (also haversine great-circle distances), `build_triangulation` (Delaunay, Gabriel, relative neighborhood, Voronoi with bounding-box, convex-hull or extended clipping), queen/rook and block contiguity (vertex and strict methods, perimeter weights, lines), fuzzy contiguity (`build_fuzzy_contiguity`), seeded jitter, GAL/GWT read/write, `Graph.to_sparse/1` (the CSR arrays as `Nx` tensors) |
| Global ESDA | esda | Moran (+BV, Rate), Geary, Getis-Ord G, join counts | M2 | available: Moran, bivariate Moran, Moran for rates, Geary's C, Getis-Ord G and join counts with analytic and seeded permutation inference |
| Local ESDA | esda | Local Moran, Geary, G/G*, join counts, LOSH, FDR | M3 | available: local Moran and bivariate local Moran (conditional randomization, cluster labels), local Geary (also multivariate), G_Local / G*, local join counts (univariate, bivariate and multivariate), LOSH and FDR; `keep_simulations` defaults to `false`; `keep_simulations: :tensor` returns the simulations as an `Nx.Tensor` (no list decoding) |
| Classification | mapclassify | Equal interval, quantiles, Fisher-Jenks, natural breaks, ... | M4 | available: equal interval, quantiles, box plot, std-mean, percentiles, maximum breaks, head/tail breaks, user-defined, Fisher-Jenks, Jenks-Caspall (plain, forced, sampled), natural breaks, Max-P, pretty breaks, `gadf/2` and `k_classifiers/2` |
| Integration | - | GeoFrame / Explorer adapter, Livebook notebooks | M5 | available: `Spatex.GeoFrame` adapter, `Spatex.Explorer.to_dataframe/2`, and Livebook notebooks (`notebooks/`) |
| Point patterns | pointpats | Centrography, G/F/J/K/L, envelopes, quadrat test | M6 | available: `Spatex.PointPats` centrography (centers, medians, standard distance, ellipse, hull, bounding rectangle/rotated rectangle/circle), G/F/J/K/L with seeded simulation envelopes (`*_test`), `poisson`/`cluster_poisson`/`normal`/`cluster_normal`/`strauss`, alpha shapes (`alpha_shape/2`, `alpha_shape_auto/2`), rectangle and hexagon quadrat tests |
| Spatial regression | spreg | OLS + diagnostics, ML lag/error, GM lag/error/combo | M7 | available: `Spatex.Regression.ols/3` (spreg diagnostics, White/HAC robust errors, LM tests, Moran of residuals) and `summary/1` available; `ml_lag/4` and `ml_error/4` (spreg ML, full/Ord methods, spatial impacts) available; `tsls/5`, `gm_lag/4` (Anselin-Kelejian, White/HAC, impacts), `gm_error/4` and `gm_combo/4` (KP98, Het, Hom GMM estimators, endogenous variants) available; SLX / spatial Durbin terms, OLS with regimes and Chow tests, panel ML lag/error models (fixed effects; lag and error with random effects, `ml_error_re/4`, and the Kapoor-Kelejian-Prucha GM estimator `gm_kkp/4`), spatial regimes (`ols_regimes`, `tsls_regimes`, `gm_lag_regimes`, `gm_error_regimes`, `gm_combo_regimes`, `ml_lag_regimes`, `ml_error_regimes`) and seemingly unrelated regressions (SUR; `three_sls/5`, error GM and ML, and the spatial-lag 3SLS `sur_lag_iv/4`) available |
| Dynamics / inequality | giddy, inequality | Markov variants, rank mobility, Gini, Theil | M8 | available: Markov, spatial Markov (homogeneity tests), LISA Markov, steady state / first passage times, full-rank and geographic-rank Markov, Kendall / spatial / local / neighbor / neighborhood / regional tau, LISA spillover, theta, Gini, Theil (+ decomposition), spatial Gini, Atkinson, Schutz, Wolfson, polarization, and Kullback's information test of Markov homogeneity (`Spatex.Dynamics.kullback/2`) |
| Regionalization | spopt, mgwr | SKATER, Max-p, AZP, Ward, GWR | M9 | available: `Spatex.Region` SKATER, spatial Ward, Region-K-Means, AZP (basic, tabu and simulated annealing), Max-p (with `objective/3`, `contiguous?/2`), SA3, Spenc (spatially encouraged spectral clustering) and `random_region/2` / `random_regions/2`; `Spatex.GWR` (`fit/4`, `select_bandwidth/4` for Gaussian, Poisson and Binomial families; `mgwr/4` multiscale GWR; `predict/6` at new locations; `local_collinearity/3`; the hat matrix (`hat_matrix:` option of `fit/4`); `bandwidth_intervals/2`; bisquare/Gaussian/exponential kernels, fixed or adaptive, spherical distances, AICc/AIC/BIC/CV) |
| Release | - | Precompiled NIFs, Hex package, benchmarks | M10 | available: Hex package with precompiled NIFs for six targets, release workflow, [benchmarks](docs/benchmarks.md) |

Function-level mapping: [docs/parity.md](docs/parity.md).

## API sketch

```elixir
# Build a spatial graph from polygon WKB (queen contiguity), row-standardize
g =
  wkbs
  |> Spatex.Graph.build_contiguity(rook: false)
  |> Spatex.Graph.transform(:r)

# Global Moran's I with 999 permutations, deterministic
m = Spatex.ESDA.moran(y, g, permutations: 999, seed: 0)
m.i       #=> 0.42
m.p_sim   #=> 0.001

# Local Moran (LISA) with quadrants and pseudo p-values
lisa = Spatex.ESDA.moran_local(y, g, permutations: 999, seed: 0)

# Local Getis-Ord G* hot/cold spots and the FDR p-value cut-off
gi = Spatex.ESDA.g_local(y, g, star: true, permutations: 999, seed: 0)
cutoff = Spatex.ESDA.fdr(gi.p_sim, 0.05)

# Classification
c = Spatex.Classify.fisher_jenks(y, k: 5)
c.bins
```

## Design principles

- **Rust core.** All algorithms live in a pure-Rust crate (`spatex_core`), reusable outside Elixir; a thin Rustler layer exposes NIFs.
- **Determinism contract.** Same seed gives identical results across runs and thread counts. Bit-for-bit equality with numpy's RNG is not a goal.
- **Oracle-tested against PySAL.** Deterministic statistics are compared with PySAL fixtures, mostly to 1e-9, with looser tolerances (for example 1e-7 or 5e-6) and a list of documented skips and deviations where algorithms legitimately differ (`test/reference/known_deviations.exs`, [parity matrix](docs/parity.md)); permutation p-values match statistically. Fixtures are generated from real PySAL and committed, so CI needs no Python.
- **Strict TDD.** ExUnit, StreamData properties, Rust unit tests.
- **PySAL names, Elixir idioms.** Immutable structs, pipelines, options as keyword lists.

## Stability

Spatex follows [Semantic Versioning](https://semver.org/). Until 1.0, breaking changes may land
in minor versions (0.3, 0.4, ...); each one is listed in the [changelog](CHANGELOG.md). Results
are checked against PySAL, but function names, options and result structs are not frozen yet.

## Installation

Add `spatex` to your dependencies:

```elixir
def deps do
  [
    {:spatex, "~> 0.2"}
  ]
end
```

Requirements: Elixir ~> 1.17 and Erlang/OTP 27 or newer. On the precompiled targets no Rust
toolchain is needed: the package downloads the NIF of its release and verifies it against the
checksum file shipped in the package. Precompiled targets: `aarch64-apple-darwin`,
`x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`,
`aarch64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`, for NIF versions 2.16 and 2.17.

Optional dependencies, each enabling the parts that need it:

- `{:nx, "~> 0.10"}`: `Nx.Tensor` inputs, and tensor outputs such as
  `keep_simulations: :tensor` and `Spatex.Graph.to_sparse/1`.
- `{:explorer, "~> 0.13"}`: `Explorer.Series` and `Explorer.DataFrame` inputs, and
  `Spatex.Explorer.to_dataframe/2`.
- GeoFrame: the `Spatex.GeoFrame` adapter. GeoFrame is not on Hex yet; see
  [Relation to GeoFrame](#relation-to-geoframe).

### Building the NIF from source

Set `SPATEX_BUILD=1` (or `true`) when compiling Spatex to build the NIF instead of downloading
it, for example on a target without a precompiled NIF. A source build needs a Rust toolchain
(stable, edition 2021, Rust 1.91 or newer: the MSRV set by Rustler 0.39, `rust-version` in
`Cargo.toml`) and Rustler, which Spatex declares as an optional dependency, so add it to your
own dependencies:

```elixir
{:spatex, "~> 0.2"},
{:rustler, "~> 0.38", runtime: false}
```

Then rebuild the dependency with `SPATEX_BUILD=1 mix deps.compile spatex --force`. A git
checkout of Spatex, a GitHub or path dependency (as in the notebooks' `Mix.install`) and a
pre-release version (a `-dev` or `-rc` suffix) always build from source, because the checksum
file only exists inside a published Hex package; as dependencies they need the same `rustler`
entry.

### Releasing (maintainers)

The workflow `.github/workflows/release.yml` ("Release NIFs") builds the NIF matrix: six targets
times two NIF versions, 12 artifacts. On pull requests that change the build inputs, and on
demand (Actions, "Release NIFs", "Run workflow"), it is a dry run that only uploads workflow
artifacts, so a broken matrix shows up before a tag is pushed. On a pushed `v*` tag it also
attaches the artifacts to the GitHub release of that tag.

1. Set the release version in `mix.exs` (`@version`) and add its `CHANGELOG.md` section; merge to
   `main`.
2. Tag the merge commit `v<version>` and push the tag. The workflow checks that the tag equals
   `v` plus the `mix.exs` version, builds the 12 artifacts and attaches them to the GitHub release
   `v<version>`. They are named `libspatex_native-v<version>-nif-<nif>-<target>.so.tar.gz` on
   Linux and macOS and `spatex_native-v<version>-nif-<nif>-x86_64-pc-windows-msvc.dll.tar.gz` on
   Windows (no `lib` prefix).
3. Once all 12 artifacts are attached, in a checkout of the tag run
   `mix rustler_precompiled.download Spatex.Native --all --print`. It compiles the project (so
   this checkout builds the NIF from source once), downloads every artifact and writes
   `checksum-Elixir.Spatex.Native.exs` with their SHA-256 checksums. The file is git-ignored: do
   not commit it.
4. `mix hex.build` and inspect the result: the file list must include
   `checksum-Elixir.Spatex.Native.exs` (`mix.exs` packages it only when it exists) and no build
   output (`priv/native`, `_build`, `target`).
5. `mix hex.publish`.
6. Delete the checksum file afterwards: while it exists, compiling the checkout downloads the
   precompiled NIF instead of building it from source.

## Documentation

- [ESDA guide](docs/guide_esda.md): choosing weights, transforms, permutation inference, FDR, reading LISA maps
- Notebooks (Livebook, offline): [tour](notebooks/tour.livemd), [PySAL comparison](notebooks/pysal_comparison.livemd), [GeoFrame](notebooks/geoframe.livemd), and use cases on real data: [North Carolina SIDS](notebooks/nc_sids_case.livemd), [US state income 1929-2009](notebooks/us_income_case.livemd) [Baltimore house prices](notebooks/baltimore_case.livemd), [Georgia education with GWR](notebooks/georgia_case.livemd), [John Snow's cholera map](notebooks/snow_case.livemd) and [Mexico maps and regions](notebooks/mexico_case.livemd), also in Spanish ([SIDS](notebooks/es/caso_sids_carolina_del_norte.livemd), [renta](notebooks/es/caso_ingresos_eeuu.livemd), [vivienda](notebooks/es/caso_vivienda_baltimore.livemd), [educación](notebooks/es/caso_educacion_georgia.livemd), [cólera](notebooks/es/caso_colera_snow.livemd), [regiones](notebooks/es/caso_regiones_mexico.livemd)). The notebooks run from a git checkout of Spatex, not from the Hex package: they install Spatex from the checkout with `path:`, and the tour, PySAL comparison and GeoFrame notebooks read `test/reference/fixtures`. The GeoFrame notebooks also need a local GeoFrame checkout
- [Research](docs/research.md): PySAL ecosystem, algorithm details, crate choices
- [Architecture](docs/architecture.md): crate layout, NIF boundary, testing strategy
- [Roadmap](docs/roadmap.md): milestones M0-M10
- [Parity matrix](docs/parity.md): PySAL to spatex mapping

## Contributing

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/CountlinkX-Solutions/spatex). See
[CONTRIBUTING.md](https://github.com/CountlinkX-Solutions/spatex/blob/main/CONTRIBUTING.md) for
the development setup and the checks CI runs, and
[SECURITY.md](https://github.com/CountlinkX-Solutions/spatex/blob/main/SECURITY.md) for reporting
vulnerabilities. Participation is governed by the
[Code of Conduct](https://github.com/CountlinkX-Solutions/spatex/blob/main/CODE_OF_CONDUCT.md).

## License

Apache-2.0, see [LICENSE](LICENSE). Spatex follows PySAL closely and ports parts of PySAL
packages, SciPy, scikit-learn and other projects; [NOTICE](NOTICE) lists them, and
[LICENSES](https://github.com/CountlinkX-Solutions/spatex/tree/main/LICENSES) holds their
license texts.
