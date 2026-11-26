# Oracle fixtures

Spatex is checked against real PySAL output. The fixtures are produced by dev-only Python
generators and committed as JSON, so **CI and normal `mix test` runs need no Python**.

## Layout

| Path | Purpose |
|---|---|
| `common.py` | Deterministic JSON dumping, version `meta`, graph export |
| `generate_graph.py` | `libpysal.graph.Graph` fixtures |
| `generate_contiguity.py` | Queen/rook contiguity (WKB in, adjacency out) and block contiguity fixtures, `fixtures/weights/` |
| `generate_weights_kernel.py` | Kernel weights (`build_kernel`, distance band with `kernel=`); `fixtures/weights/kernel_*.json`; run under `prlimit --as=2147483648` |
| `generate_weights_triangulation.py` | Delaunay, Gabriel and relative neighborhood graphs (`build_triangulation`); `fixtures/weights/triangulation_*.json`; run under `prlimit --as=2147483648` |
| `generate_weights_fuzzy_contiguity.py` | libpysal `build_fuzzy_contiguity` (`fixtures/weights/contiguity_fuzzy.json`): columbus, sids2 and synthetic sets over tolerance, buffer and predicate, plus near-threshold polygon pairs measuring GEOS buffer agreement; run under `prlimit --as=2147483648` |
| `generate_weights_knn.py` | KNN and distance-band builders; `fixtures/weights/{knn,distance_band}_*.json` (points plus positional adjacency per case) |
| `generate_weights_io.py` | GAL/GWT files written by libpysal plus the adjacency it reads back; `fixtures/io/*.{gal,gwt,json}`; run under `prlimit --as=2147483648` |
| `generate_classify.py` | mapclassify fixtures (`fixtures/classify/*.json`): bins, yb, counts, k, tss, adcm, gadf, legend, find_bin |
| `fixtures/graph/*.json` | Graph adjacency, transforms, cardinalities, s0/s1/s2, lag |
| `fixtures/datasets/columbus.json` | Columbus polygons (WKB hex) and CRIME/INC/HOVAL, shared by several generators, tests and notebooks |
| `generate_esda_local.py` | esda local Moran fixtures (`fixtures/esda/local_moran_*.json`): analytic arrays and 9999-permutation PySAL estimates (`seed=12345`) |
| `generate_esda_g_local.py` | esda G_Local / G* fixtures (`fixtures/esda/g_local_*.json`) and `esda.fdr` vectors (`fixtures/esda/fdr.json`); graphs are stored untransformed with an `input_transform` label |
| `generate_esda_global.py` | esda global Moran fixtures (`fixtures/esda/moran_*.json`): analytic quantities and 9999-permutation PySAL estimates |
| `generate_pointpats.py` | pointpats fixtures (`fixtures/pointpats/*.json`): centrography, G/F/J/K/L with stored empty-space points, simulation summaries (`*_test`, statistical only) and quadrat statistics; run under `prlimit --as=2147483648` |
| `generate_spreg_ols.py` | spreg `OLS` fixtures (`fixtures/regression/ols_*.json`): Columbus (queen/rook, binary and row-standardized W, White, HAC kernel, dummy, collinear and no-constant designs), a synthetic n=200 case and a 3x3 lattice; stores the exact sparse W spreg used; run under `prlimit --as=2147483648` |
| `generate_spopt_region.py` | spopt fixtures (`fixtures/region/*.json`): SKATER (traces, floors, islands, duplicates), spatial Ward (full, disconnected and partial trees), Region-K-Means with numpy's seeds, objective functions, AZP / Max-p objective distributions with brute-force optima and Max-p construction orders; run under `prlimit --as=2147483648` (about one minute) |
| `generate_spreg_ml.py` | spreg `ML_Lag` / `ML_Error` fixtures (`fixtures/regression/ml_*.json`): Columbus (queen, rook, KNN-4, binary W; full, ord, lu), an inverse-distance case documenting the `ord` bugs, synthetic n=200 and n=25 (negative dependence) lattices; stores the exact sparse W, the optimizer's `nfev` and whether spreg took the `symmetrize` path; run under `prlimit --as=2147483648` |
| `generate_spreg_gm.py` | spreg `TSLS`, `GM_Lag`, `GM_Error`, `GM_Endog_Error`, `GM_Error_Het`, `GM_Endog_Error_Het`, `GM_Error_Hom`, `GM_Endog_Error_Hom`, `GM_Combo`, `GM_Combo_Het`, `GM_Combo_Hom` fixtures (`fixtures/regression/gm_{columbus,synthetic,clamp,optimizer}.json`): Columbus queen/rook/binary W with `DISCBD` as the instrument of `HOVAL` and a triangular HAC kernel, a synthetic n=200 lattice, the +-0.9 clamp cases (with a clamp-disabled twin) and the cases where L-BFGS-B stops away from the minimizer; stores every lambda spreg's `optim_moments` returned so the `*_at` test hooks can be fed spreg's own values; run under `prlimit --as=2147483648` |
| `generate_spreg_tsls_regimes.py` | spreg `TSLS_Regimes` fixture (`fixtures/regression/tsls_regimes.json`): Columbus CRIME ~ INC + OPEN with HOVAL instrumented by DISCBD / PLUMB, regimes NSA (2), NSA x EW (4) and X terciles (3); `constant_regi` both values, `cols2regi` all / mixed, `regime_err_sep` true / false (separate vs pooled), White, HAC (triangular kernel), `sig2n_k`; stores layout, betas, vm, z, Chow tests and every regime's fit; run under `prlimit --as=2147483648` |
| `generate_region_spenc.py` | spopt `Spenc` fixtures (`fixtures/region/spenc.json`): the affinity matrix, embedding and `eigsh` eigenvalues of rbf / laplacian / linear / polynomial / sigmoid / cosine / nearest-neighbors / precomputed affinities on Columbus, deterministic hierarchical bipartition labels (spopt 0.8.0 passes `floor=` to a function that does not accept it, so the generator accepts and ignores it), sklearn `discretize` labels for every possible start row, and label sets over many `random_state` values on Columbus and a planted-partition lattice; run under `prlimit --as=2147483648` |
| `generate_alpha_shapes.py` | libpysal `alpha_shape` / `alpha_shape_auto` fixtures (`fixtures/pointpats/alpha_shape*.json`): polygons (clockwise shells, counter-clockwise holes, start vertex normalized) with areas for random, clustered, annulus, island-in-hole, two-blob and C-shaped sets at alphas half-way between circumradii, and the automatic search (`step`, radius, bounding circles) with libpysal's result under all six triangle vertex orders; run under `prlimit --as=2147483648` |
| `known_deviations.exs` | Accepted differences from PySAL and mapclassify (see file) |
| `../support/spatex/reference.ex` | `Spatex.Reference`: `load!/1`, `fixture_path/1`, `assert_close/3` |

## Regenerating

The committed fixtures were produced with Python 3.12 and libpysal 4.15.1, esda 2.10.0,
spreg 1.9.1, mapclassify 2.11.0, pointpats 2.6.0, giddy 2.3.9, inequality 1.2.0, mgwr 2.2.1,
spglm 1.1.0, spopt 0.8.0, fast_hdbscan 0.3.4, numpy 2.5.4, scipy 1.18.1 and shapely 2.2.0, plus
geopandas.

```
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install libpysal==4.15.1 esda==2.10.0 spreg==1.9.1 mapclassify==2.11.0 \
  pointpats==2.6.0 giddy==2.3.9 inequality==1.2.0 mgwr==2.2.1 spglm==1.1.0 spopt==0.8.0 \
  fast_hdbscan==0.3.4 numpy==2.5.4 scipy==1.18.1 shapely==2.2.0 geopandas
.venv/bin/python -I test/reference/generate_graph.py
```

Generation is deterministic: rerunning with the same versions leaves `git diff` empty. Every
fixture records the producing versions in its `meta` block. Commit changed JSON together with
the reason (version bump or new case).

## Fixture conventions

* JSON, sorted keys, floats at `repr` precision. Non-finite floats are the strings `"nan"`,
  `"inf"`, `"-inf"`; `Spatex.Reference.load!/1` decodes them to `:nan`, `:inf`, `:neg_inf`.
  Fixture ids must therefore never equal those strings.
* Graphs use positional indices `0..n-1` into `ids` (the graph's `unique_ids` order).
  `focal`/`neighbor`/`weight` mirror `Graph.adjacency` row order. An island is a single
  zero-weight self-loop `(i, i, 0.0)`, exactly as `Graph.adjacency` reports it, and is also
  listed in `isolates`.
* Transforms: `O` (original), `R` (row), `B` (binary), `D` (double), `V` (variance stabilising).
  Each carries the transformed adjacency, `lag` of the fixture's variable on the transformed
  graph, and `s0`, `s1`, `s2`.
* `s0 = sum w_ij`, `s1 = 1/2 sum (w_ij + w_ji)^2`, `s2 = sum_i (sum_j w_ij + sum_j w_ji)^2`,
  computed densely on the transformed weights and asserted equal to `Graph.to_W().s0/s1/s2` at
  generation time.
* `symmetric` is true when `Graph.asymmetry()` is empty.

## Tolerance policy

`Spatex.Reference.assert_close(actual, expected, tol)` uses `abs(a - e) <= tol * max(1, abs(e))`.

* `1.0e-12` for graph algebra (weights, transforms, lag, s0/s1/s2).
* `1.0e-9` for statistics (Moran's I, z-scores, ...).
* Permutation p-values are tested statistically, not against fixtures.

Anything looser must be recorded in `known_deviations.exs` with a reason.

## Tests

Oracle tests carry `@moduletag :oracle`; run only them with `mix test --only oracle`.
