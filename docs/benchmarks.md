# Benchmarks

Spatex against PySAL on the same problem shapes. Scripts: Benchee in `bench/*.exs` (run with
`mix run bench/<name>.exs`, Benchee is a dev-only dependency) and `bench/pysal_bench.py`. Raw
results are committed in `bench/results/spatex.json` and `bench/results/pysal.json`.

## Method and caveats

- **Machine:** AMD Ryzen 7 8700F (8 cores, 16 hardware threads), WSL2 on Linux 6.18, 15 GiB RAM,
  shared with other build jobs while the numbers were taken (the run started at a load average of
  about 2; the figures below are one sequential pass, not a quiet-machine lab result, and repeated
  runs differ by several percent, sometimes more).
- **Versions:** Spatex 0.1.0 (release-mode NIF, rustc stable, Elixir 1.19.5, OTP 28.5), libpysal
  4.15.1, esda 2.10.0, spreg 1.9.1, mapclassify 2.11.0, mgwr 2.2.1,
  numpy 2.5.4, shapely 2.2.0, Python 3.12.14.
- **Timing:** median of the timed iterations. Benchee: 2 s warm-up, 8 s of runs. PySAL: one
  untimed-for-the-table first call (it includes numba JIT compilation, recorded as `first_call_ms`
  in the JSON; for Moran Local and GWR that is 6 to 13 s), then 3 to 20 timed calls.
- **Data:** the same sizes and structure in both (columbus; a 100x100 unit-square lattice; seeded
  uniform points; normal covariates). Random values come from `:rand` and numpy respectively, so the
  numbers differ; that does not change timings. Spatex receives plain Elixir lists and WKB binaries
  and pays for validating and encoding them; PySAL receives numpy arrays and parsed geometries.
  Results are not compared numerically here (that is what the oracle tests are for).
- **Threads:** Spatex uses rayon with all 16 threads by default (`RAYON_NUM_THREADS` overrides it;
  the "1 thread" rows set it to 1). PySAL was not pinned: numpy/scipy BLAS (OpenBLAS) use their
  defaults, numba kernels run with the default threading layer, and esda's `n_jobs` is 1 (its
  default; `Moran` has no such option). This is the usual way both are called, not an equal-thread
  comparison: read the "1 Spatex thread" rows for a per-core view.
- **Memory cap:** PySAL ran under `prlimit --as=2147483648` (2 GiB address space, repository
  convention). Importing the PySAL stack alone reaches 1.8 GiB peak address space (measured `VmPeak`,
  apparently per-thread BLAS arenas), and two cases did not fit with default BLAS threading: `Moran_Local` with the
  simulation matrix kept and `ML_Lag` at n=2,500 (both MemoryError). Those two rows use
  `OPENBLAS_NUM_THREADS=1` for PySAL (both fit that way; the other PySAL rows use the default).
  Rows run both ways are in `bench/results/pysal.json` (e.g. `ols`, `gwr`: no measurable difference).
  Sizes were not reduced.

## Results

| Benchmark | Spatex | PySAL | Spatex vs PySAL | Notes |
|---|---|---|---|---|
| build_contiguity, columbus (n=49, rook) | 0.09 ms | 2.90 ms | 30.8x faster | Spatex parses WKB; PySAL gets parsed shapely geometries |
| build_contiguity, 100x100 lattice (n=10,000, rook) | 6.53 ms | 114.14 ms | 17.5x faster | same |
| build_knn (n=10,000, k=8) | 19.63 ms | 59.44 ms | 3.0x faster |  |
| build_knn (n=100,000, k=8) | 220.61 ms | 648.91 ms | 2.9x faster |  |
| Moran, 999 permutations (n=10,000) | 9.25 ms | 118.17 ms | 12.8x faster | Spatex 16 threads; PySAL single-threaded (no `n_jobs` in `Moran`) |
| Moran, 999 permutations, 1 Spatex thread | 55.71 ms | 118.17 ms | 2.1x faster | `RAYON_NUM_THREADS=1` |
| Moran Local, 999 perms, default `keep_simulations: false` (n=10,000) | 42.47 ms | 228.11 ms | 5.4x faster | Spatex 16 threads; PySAL `n_jobs=1` (numba-compiled) |
| Moran Local, 999 perms, 1 Spatex thread, default `keep_simulations: false` | 95.92 ms | 228.11 ms | 2.4x faster | `RAYON_NUM_THREADS=1` |
| Moran Local, 999 perms, `keep_simulations: true` (opt-in) | 3.03 s | 310.54 ms | 9.7x slower | PySAL (which always keeps them) needs `OPENBLAS_NUM_THREADS=1` to fit the cap; see below |
| Moran Local, 999 perms, `keep_simulations: :tensor` (n=10,000) | 133.85 ms | 310.54 ms | 2.3x faster | simulations as an `Nx.Tensor` (shape `{10000, 999}`) built from the native binary; PySAL row as above; 22.6x faster than `keep_simulations: true` |
| Fisher-Jenks (n=5,000, k=5) | 34.88 ms | 47.59 ms | 1.4x faster |  |
| OLS + LM diagnostics + Moran of residuals (n=10,000, k=4) | 8.24 ms | 8.64 ms | 1.0x faster |  |
| ML spatial lag, Ord (n=2,500) | 2.49 s | 2.90 s | 1.2x faster | PySAL needs `OPENBLAS_NUM_THREADS=1` to fit the cap |
| GWR `select_bandwidth` (n=1,000, adaptive bisquare, AICc) | 14.88 ms | 1.25 s | 83.7x faster | PySAL median excludes a 6 s numba warm-up |

## Reading the numbers

- Contiguity (17-31x), the permutation tests (5-13x with the default options) and GWR bandwidth search
  (about 84x) are where Spatex is clearly ahead; KNN is about 3x; Fisher-Jenks, OLS and the dense
  ML lag are near parity (1.0-1.4x), which is expected: they are dominated by dense linear algebra
  and single passes that PySAL also runs in compiled code.
- **Moran Local with the default options is about 5.4x faster than PySAL** (42 ms against 228 ms),
  because local statistics default to `keep_simulations: false` and return only the p-values.
  **Only the explicit opt-in `keep_simulations: true` is slower in Spatex** (about 10x at n=10,000).
  The permutations themselves take about 40 ms; the rest is decoding the 10,000 x 999 simulation
  matrix into nested Elixir lists (`Input.from_f64_binary/1` and `Enum.chunk_every/2`).
  `keep_simulations: :tensor` returns the same matrix as an `Nx.Tensor` built straight from the
  native binary: 134 ms (median of 58 runs on the same machine, Spatex side only, PySAL not
  rerun), 2.3x faster than PySAL's 311 ms and about 23x faster than the list form. The matrix is
  80 MB of f64 values, which accounts for the extra time over the 42 ms without simulations
  (not profiled separately).
- Permutation results are deterministic per seed and identical at any thread count; the 1-thread rows
  show the parallel scaling (about 6.0x for Moran on 16 threads).
- `ml_lag` and `moran_local` use the n=10,000 / 2,500 sizes of the task; `ml_lag` is dense (`O(n^3)`
  eigendecomposition), so it is the practical size limit (`max_dense_n` defaults to 5,000).

## Reproducing

```sh
# Spatex (one script per benchmark; RAYON_NUM_THREADS=1 for the single-thread rows)
mix run bench/build_contiguity.exs   # also build_knn, moran, moran_local, fisher_jenks, ols, ml_lag, gwr

# PySAL, under the 2 GiB cap; one process per case avoids address-space fragmentation
for c in contiguity knn moran moran_local fisher_jenks ols ml_lag gwr; do
  prlimit --as=2147483648 <venv>/bin/python -I bench/pysal_bench.py $c
done
OPENBLAS_NUM_THREADS=1 prlimit --as=2147483648 <venv>/bin/python -I bench/pysal_bench.py moran_local ml_lag
```
