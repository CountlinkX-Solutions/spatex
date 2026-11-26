# Architecture

## 1. Layout

```
spatex/
├── mix.exs                  # app :spatex, elixir ~> 1.17, rustler_precompiled, optional rustler
├── Cargo.toml               # workspace, resolver 2, edition 2021
├── native/
│   ├── spatex_core/         # pure Rust rlib: all algorithms + unit tests (no rustler)
│   │   └── src/{graph,esda,classify,pointpats,regression,dynamics,inequality,region,rng}/
│   └── spatex_native/       # cdylib: thin Rustler layer (NIFs, resources, encoding)
├── lib/spatex/
│   ├── native.ex            # use RustlerPrecompiled, crate: "spatex_native" (source build fallback)
│   ├── graph.ex  esda.ex  classify.ex  point_pats.ex  regression.ex
│   ├── dynamics.ex  inequality.ex  region.ex  geo_frame.ex (optional)
│   └── input.ex             # input normalization
├── test/
│   ├── reference/*.py       # dev-only PySAL fixture generators
│   ├── reference/fixtures/  # committed JSON oracles
│   ├── reference/known_deviations.exs
│   └── support/reference.ex # Spatex.Reference comparison helper
├── docs/
└── .tool-versions           # erlang 28.5, elixir 1.19.5-otp-28
```

Rule: `spatex_core` never depends on rustler; `spatex_native` contains no algorithms.

## 2. NIF boundary

| Concern | Decision |
|---|---|
| Numeric vectors | native-endian f64 binaries (`<<x::float-64-native>>`); avoids list decoding cost |
| Index vectors | u32 / i64 binaries |
| Return type | Rust `Result<_, String>`; Rustler maps to `{:ok, _}` / `{:error, reason}` |
| Scheduling | every NIF whose work grows with the input (O(n) and up) uses `schedule = "DirtyCpu"`; only O(1) accessors run on normal schedulers. A dirty NIF cannot be cancelled: killing the caller does not stop it, the scheduler stays busy until it returns |
| Allocation guards | sizes a caller drives (grid cells, edges, classes, permutation and DP tables, simulated points) are checked with overflow-safe arithmetic against `spatex_core::limits` before allocating, since a failed Rust allocation aborts the BEAM; `max_cells` / `max_edges` options raise the defaults |
| Graph | CSR (offsets, neighbor indices, weights) in a Rust `ResourceArc`; immutable |
| Parallelism | rayon inside a NIF; per-site / per-chunk derived RNG streams |

Elixir `Spatex.Native` holds raw NIF stubs only; public modules never expose its return shapes.

## 3. Elixir module map

| Module | Responsibility | PySAL |
|---|---|---|
| `Spatex.Graph` | CSR graph struct `%Spatex.Graph{ref, n, ids, transform}`, builders, transforms, lag, stats | libpysal.graph |
| `Spatex.ESDA` | Global and local statistics, FDR | esda |
| `Spatex.Classify` | Map classification | mapclassify |
| `Spatex.PointPats` | Centrography, distance statistics, quadrat | pointpats |
| `Spatex.Regression` | OLS + diagnostics, ML, GM | spreg |
| `Spatex.Dynamics` | Markov, spatial Markov, LISA Markov, rank mobility | giddy |
| `Spatex.Inequality` | Gini, Theil, spatial Gini | inequality |
| `Spatex.Region` | SKATER, Max-p, AZP, Ward | spopt |
| `Spatex.GWR` | Geographically weighted regression and MGWR | mgwr |
| `Spatex.GeoFrame` | Optional adapter, calls `GeoFrame` at runtime (graphs from geometry, results as columns) | - |
| `Spatex.Explorer` | Per-site results as `Explorer.DataFrame` | - |
| `Spatex.Native` | NIF stubs | - |

`Spatex.Graph` is immutable: transforms return new graphs. `ids` is the user id list (default `0..n-1`). Interop: `to_adjacency/1` (`{focal, neighbor, weight}` list), `to_sparse/1` (libpysal's CSR arrays as `Nx` tensors) and GAL / GWT read and write (`read_gal/2`, `write_gal/2`, `read_gwt/2`, `write_gwt/2`).

Result types are structs named after PySAL attributes, e.g. `%Spatex.ESDA.Moran{i, ei, vi_norm, seI_norm, z_norm, p_norm, vi_rand, seI_rand, z_rand, p_rand, sim, ei_sim, seI_sim, z_sim, p_sim}`.

## 4. Input normalization

`Spatex.Input` converts user data to the binaries NIFs expect.

| Input | Support |
|---|---|
| Lists of numbers | core |
| `Nx.Tensor` | optional dep (`{:nx, optional: true}`) |
| `Explorer.Series` | optional dep |
| Points | `{x, y}` tuples / `[x, y]` lists, `{n, 2}` tensors, or a `{xs, ys}` pair |
| Index vectors | non-negative integers packed as `u32` |
| Polygons | WKB binaries |
| Labels (regimes, partitions) | lists of any terms, rank-1 tensors, `Explorer.Series` of any dtype (`to_labels/2`) |
| `n x t` matrices (panels) | lists of rows, rank-2 tensors, `Explorer.DataFrame` with one column per period (`to_matrix/2`) |

Validation happens in Elixir before the NIF call (lengths, finite values, graph size matches data). NaN/infinity are rejected unless `allow_nan: true`; `nil` in an `Explorer.Series` maps to NaN only with `allow_nil: true`. Decoding maps non-finite bit patterns to `:nan`, `:infinity`, `:neg_infinity`.

## 5. Determinism contract

- Same seed gives identical results across runs and thread counts.
- RNG: `rand_chacha` via `seed_from_u64`; each permutation site (or fixed-size chunk) derives its own stream from `(seed, index)`, so rayon scheduling cannot change results.
- Not a goal: bit-for-bit equality with numpy's RNG.
- Statistics are deterministic and must match PySAL to 1e-9.

## 6. Error handling

- Public functions return the value directly (no `!` variants), like GeoFrame.
- Invalid arguments raise `ArgumentError` with an actionable message.
- NIF `{:error, reason}` is converted to a raised error in the public layer.
- Mathematically degenerate cases follow PySAL where defined (e.g. islands give lag 0); otherwise raise.

## 7. Testing strategy

| Layer | Tooling | Purpose |
|---|---|---|
| Rust unit tests | `cargo test` in `spatex_core` | Algorithm correctness, edge cases |
| ExUnit | `mix test` | Public API, normalization, errors |
| Properties | StreamData | Invariants (row sums, symmetry, permutation invariance, determinism across seeds/threads) |
| Oracle | JSON fixtures from real PySAL | Parity with `Spatex.Reference` helper |

- Strict TDD: observed RED, then GREEN, then REFACTOR.
- Tolerances: deterministic statistics 1e-9; permutation p-values compared statistically (tolerance bands and distribution tests), never exact.
- Fixtures come from `test/reference/*.py`, committed as JSON, so CI needs no Python. Accepted differences go in `test/reference/known_deviations.exs`.
- CI: Elixir 1.17/OTP 27 and 1.19/OTP 28 matrix, plus a Rust job (fmt, clippy `-D warnings`, test).

## 8. GeoFrame integration contract

- spatex does not depend on GeoFrame; no shared types.
- NIF resources cannot be shared across NIFs, so the interchange is **WKB binaries** (polygons) and **coordinate binaries / lists** (points).
- `Spatex.GeoFrame` has no compile-time dependency on GeoFrame (not on Hex yet, so CI does not fetch it). It calls the public API through module atoms with `apply/3`, guarded by `Code.ensure_loaded?/1`, and raises an `ArgumentError` when GeoFrame is absent. Used API: `GeoDataFrame.geometry/1`, `data/1`, `n_rows/1`, `pull/2`, `put/3`; `GeoSeries.to_wkb/2`, `centroid/1`, `x/1`, `y/1`, `is_geographic/1`.
- Contiguity graphs come from WKB (`Spatex.Graph.build_contiguity/2`); point graphs from centroid coordinates. Distance methods refuse geographic CRSs unless `allow_geographic:` is given.
- Results return as Explorer columns: `Spatex.Explorer.to_dataframe/2` (no GeoFrame needed) and `Spatex.GeoFrame.put_results/3` (columns prefixed, default `lisa_`, `g_`, `class_`).
- Tests tagged `:geoframe` need a local GeoFrame checkout: `mix.exs` adds `{:geoframe, path: ..., only: :test, optional: true}` only when `SPATEX_GEOFRAME_PATH` is set, so `mix.lock` never references it (path deps are not locked) and CI excludes the tag.

## 9. API conventions

Every public entry point follows the same rules, so options read the same across modules.

- **Options.** The last argument is a keyword list defaulting to `[]`. Unknown keys raise
  `ArgumentError` naming the key; boolean flags must be `true` or `false`. Defaults follow the
  PySAL function being ported.
- **Seeds.** `seed: nil` (the default where a result struct exists) draws a random seed and records
  it in the result (`seed` field or `info.seed`), so any run can be reproduced. Functions whose
  output is a bare value (e.g. `PointPats.g/2`, `PointPats.f/2`) keep a fixed default seed of 0.
- **Simulations.** `:permutations` is the only name for a simulation count. Global statistics keep
  their simulated values by default (`keep_simulations: true`); local statistics, which hold
  `n x permutations` values, default to `keep_simulations: false`. Callers that need the simulated
  moments (e.g. `Dynamics.lisa_markov/3`) ask for them.
- **Weights transform.** ESDA and dynamics statistics take `:transform` (default `:r`, as in esda)
  and transform the graph themselves, because their PySAL counterparts do. Spatial regression
  uses the graph exactly as given, as spreg does: row-standardize it first with
  `Spatex.Graph.transform/2`. Regionalization only reads the graph's neighbors, so its weights do
  not matter. Silently re-transforming a graph inside a model would change the
  estimates of anyone who passes binary weights on purpose.
- **Inputs.** Numeric vectors, points, labels and matrices go through `Spatex.Input`, so lists,
  `Nx` tensors and `Explorer` series/data frames are accepted wherever data is.
