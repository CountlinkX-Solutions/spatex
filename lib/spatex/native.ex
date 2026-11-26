defmodule Spatex.Native do
  @moduledoc false
  # NIF bindings to the `spatex_native` Rust crate. Each function below is replaced
  # when the NIF library loads; the bodies only run if loading failed.
  #
  # The library comes from one of two places (see "Installation" in the README):
  #
  #   * downloaded by rustler_precompiled from the GitHub release `v<version>` of this exact
  #     version and verified against `checksum-Elixir.Spatex.Native.exs` (the default for the
  #     Hex package on a precompiled target), or
  #   * built from source with Rustler (always in a git checkout and in CI).
  #
  # Building from source is forced when SPATEX_BUILD is "1"/"true", when the version is a
  # pre-release, or when the checksum file is absent. That file is generated from the release
  # artifacts when a release is published, ships inside the Hex package and is never committed
  # (it is git-ignored), so a checkout, a GitHub dependency and CI always build locally.
  version = Mix.Project.config()[:version]

  force_build? =
    System.get_env("SPATEX_BUILD") in ["1", "true"] or
      Version.parse!(version).pre != [] or
      not File.exists?("checksum-Elixir.Spatex.Native.exs")

  use RustlerPrecompiled,
    otp_app: :spatex,
    crate: "spatex_native",
    version: version,
    base_url: "https://github.com/CountlinkX-Solutions/spatex/releases/download/v#{version}",
    force_build: force_build?,
    nif_versions: ["2.16", "2.17"],
    targets: ~w(
      aarch64-apple-darwin
      aarch64-unknown-linux-gnu
      x86_64-apple-darwin
      x86_64-pc-windows-msvc
      x86_64-unknown-linux-gnu
      x86_64-unknown-linux-musl
    )

  def nif_version, do: :erlang.nif_error(:nif_not_loaded)
  def sum_f64(_binary), do: :erlang.nif_error(:nif_not_loaded)

  # Graph (CSR resource). Index vectors are u32 binaries, numeric vectors f64 binaries.
  def graph_from_arrays(_n, _focal, _neighbor, _weight), do: :erlang.nif_error(:nif_not_loaded)
  def graph_nnz(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_to_arrays(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_to_csr(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_transform(_graph, _kind), do: :erlang.nif_error(:nif_not_loaded)
  def graph_lag(_graph, _y), do: :erlang.nif_error(:nif_not_loaded)
  def graph_cardinalities(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_isolates(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_stats(_graph), do: :erlang.nif_error(:nif_not_loaded)
  def graph_symmetric(_graph, _tol), do: :erlang.nif_error(:nif_not_loaded)

  # Global ESDA. Scalars come back as one f64 binary (NaN-safe), simulations as another.
  def esda_moran(_graph, _y, _permutations, _seed, _two_tailed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_moran_bv(_graph, _x, _y, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_rate(_events, _base, _adjusted), do: :erlang.nif_error(:nif_not_loaded)

  # Local ESDA. Per-site blocks come back as one f64 binary, simulations (n x P) as another.
  def esda_moran_local(_graph, _y, _geoda, _permutations, _seed, _alternative, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_moran_local_bv(_graph, _x, _y, _geoda, _permutations, _seed, _alternative, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_g_local(_graph, _y, _prep, _permutations, _seed, _alternative, _island_weight, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_fdr(_p_values, _alpha), do: :erlang.nif_error(:nif_not_loaded)

  def esda_geary_local(
        _graph,
        _x,
        _permutations,
        _seed,
        _alternative,
        _keep,
        _sig,
        _island_weight
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  def esda_geary_local_mv(_graph, _vars, _k, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_join_counts_local_bjc(_graph, _x, _z, _perms, _seed, _alternative, _keep, _island),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_join_counts_local(_graph, _y, _permutations, _seed, _alternative, _keep, _island),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_losh(_graph, _y, _a, _chi2), do: :erlang.nif_error(:nif_not_loaded)

  def esda_geary(_graph, _y, _permutations, _seed, _keep), do: :erlang.nif_error(:nif_not_loaded)

  def esda_getis_ord_g(_graph, _y, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def esda_join_counts(_graph, _y, _drop_islands, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  # Dynamics (giddy). Class panels are u32 binaries (n x t, row-major), numeric blocks f64.
  def dynamics_transitions(_ids, _n, _t, _k), do: :erlang.nif_error(:nif_not_loaded)
  def dynamics_chain(_p, _k), do: :erlang.nif_error(:nif_not_loaded)
  def dynamics_chi2_test(_t1, _t2, _k), do: :erlang.nif_error(:nif_not_loaded)
  def dynamics_chi2_sf(_x, _dof), do: :erlang.nif_error(:nif_not_loaded)
  def dynamics_homogeneity(_counts, _m, _k), do: :erlang.nif_error(:nif_not_loaded)
  def dynamics_kullback(_counts, _dims, _giddy_totals), do: :erlang.nif_error(:nif_not_loaded)

  # Grouped arguments: dims {n, t}, classes {k, m, fixed, discrete}, cutoffs {cutoffs, lag_cutoffs}
  # (binaries or nil), draws {permutations, seed}.
  def dynamics_spatial_markov(_graph, _y, _dims, _classes, _cutoffs, _draws),
    do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_tau(_x, _y), do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_tau_local(_x, _y), do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_tau_local_neighbor(_graph, _x, _y, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_tau_local_neighborhood(_graph, _x, _y, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_tau_regional(_x, _y, _regime, _permutations, _seed, _keep),
    do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_spatial_tau(_graph, _x, _y, _permutations, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def dynamics_theta(_y, _n, _t, _regime, _permutations, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  # GWR (Gaussian, Poisson, Binomial) and MGWR. x holds k columns of n values; per-location
  # results are row-major n x k.
  def gwr_fit(_coords, _y, _x, _model, _glm, _offset, _bw, _flags),
    do: :erlang.nif_error(:nif_not_loaded)

  def gwr_select(_coords, _y, _x, _model, _glm, _offset, _search, _shape),
    do: :erlang.nif_error(:nif_not_loaded)

  def mgwr_fit(_coords, _y, _x, _model, _search, _bounds, _outer, _inference),
    do: :erlang.nif_error(:nif_not_loaded)

  def gwr_permutation(_n, _seed, _replicate), do: :erlang.nif_error(:nif_not_loaded)

  def gwr_t_critical(_alpha, _df), do: :erlang.nif_error(:nif_not_loaded)

  def gwr_predict(_coords, _y, _x, _model, _glm, _offset, _bw, _new),
    do: :erlang.nif_error(:nif_not_loaded)

  def gwr_collinearity(_coords, _x, _model, _bw, _constant),
    do: :erlang.nif_error(:nif_not_loaded)

  def mgwr_collinearity(_coords, _x, _model, _bws), do: :erlang.nif_error(:nif_not_loaded)

  # Inequality. Results are f64 binaries; partitions are u32 binaries.
  def inequality_gini(_x), do: :erlang.nif_error(:nif_not_loaded)
  def inequality_theil(_y, _n, _t), do: :erlang.nif_error(:nif_not_loaded)

  def inequality_polarization(_graph, _x, _grouping, _draws),
    do: :erlang.nif_error(:nif_not_loaded)

  def inequality_atkinson(_y, _epsilon), do: :erlang.nif_error(:nif_not_loaded)
  def inequality_schutz(_y), do: :erlang.nif_error(:nif_not_loaded)
  def inequality_lorenz_curve(_y), do: :erlang.nif_error(:nif_not_loaded)
  def inequality_wolfson(_y), do: :erlang.nif_error(:nif_not_loaded)
  def inequality_theil_d(_y, _dims, _partition, _draws), do: :erlang.nif_error(:nif_not_loaded)

  def inequality_spatial_gini(_graph, _x, _permutations, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  # Regression (spreg). Returns `[{name, f64_binary}]`; matrices are row-major.
  def regression_ols(_y, _x, _k, _robust, _sig2n_k, _flags, _graph, _gwk),
    do: :erlang.nif_error(:nif_not_loaded)

  # ML lag/error (spreg ML_Lag, ML_Error); `opts` is `{method, epsilon, impacts, at, max_dense_n}`.
  def regression_ml_lag(_y, _x, _k, _graph, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def regression_panel_fe(_y, _x, _k, _graph, _settings), do: :erlang.nif_error(:nif_not_loaded)

  def regression_panel_error_re(_y, _x, _k, _graph, _settings),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_panel_gm_kkp(_y, _x, _k, _graph, _settings),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_ml_error(_y, _x, _k, _graph, _opts), do: :erlang.nif_error(:nif_not_loaded)

  def regression_ml_error_xlag(_y, _x, _xlag, _k, _graph, _opts),
    do: :erlang.nif_error(:nif_not_loaded)

  # 2SLS and GM models: `x`, `yend`, `q` are `{f64 column-major binary, column count}`.
  def regression_tsls(_y, _x, _yend, _q, _robust, _sig2n_k, _gwk),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_gm_lag(_y, _x, _yend, _q, _graph, _gwk, _settings),
    do: :erlang.nif_error(:nif_not_loaded)

  # Spatial lag helpers of the regimes models: instruments, reduced-form prediction, diagnostics.
  def regression_lag_endog(_y, _x, _yend, _q, _graph, _w_lags, _lag_q),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_sp_att(_y, _predy, _wy, _graph, _rho, _hard_bound),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_lag_diag(_y, _predy_e, _z, _varb, _u, _graph, _rho, _settings),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_gm_error(_y, _x, _yend, _q, _graph, _settings),
    do: :erlang.nif_error(:nif_not_loaded)

  # Pieces of the regimes error models: lambda from first-stage residuals, spatial filter of a
  # column block, 2SLS with an explicit instrument matrix.
  def regression_error_lambda(_u, _graph), do: :erlang.nif_error(:nif_not_loaded)

  def regression_sur_error_search(_e, _k, _graph, _lam0, _opts),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_sur_error_traces(_graph, _lam, _max_dense_n),
    do: :erlang.nif_error(:nif_not_loaded)

  def regression_sp_filter(_cols, _graph, _lam), do: :erlang.nif_error(:nif_not_loaded)
  def regression_tsls_h(_y, _x, _yend, _h), do: :erlang.nif_error(:nif_not_loaded)

  # Contiguity (queen/rook from WKB, block). Return {focal_u32_binary, neighbor_u32_binary}.
  def contiguity_edges(_wkbs, _rook, _strict, _by_perimeter),
    do: :erlang.nif_error(:nif_not_loaded)

  def fuzzy_contiguity_edges(_wkbs, _tolerance, _buffer, _predicate),
    do: :erlang.nif_error(:nif_not_loaded)

  def block_edges(_codes), do: :erlang.nif_error(:nif_not_loaded)

  # Weights builders (return a graph resource). Points are an interleaved f64 binary.
  def weights_knn(_points, _k, _coincident, _metric, _max_edges),
    do: :erlang.nif_error(:nif_not_loaded)

  def weights_distance_band(_points, _threshold, _binary, _alpha, _metric, _max_edges),
    do: :erlang.nif_error(:nif_not_loaded)

  def weights_kernel(
        _points,
        _kernel,
        _bandwidth,
        _k,
        _taper_decay,
        _coincident,
        _metric,
        _max_edges
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  def weights_distance_band_kernel(
        _points,
        _threshold,
        _kernel,
        _bandwidth,
        _taper_decay,
        _metric,
        _max_edges
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  def weights_triangulation(
        _points,
        _method,
        _kernel,
        _bandwidth,
        _taper,
        _decay,
        _coincident,
        _clip
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  # Classification (mapclassify). `scalar` is k or hinge, `list` multiples/percentiles/bins
  # (`[initial]` for natural breaks, `[pct]` for the sampled classifiers).
  def classify_run(_method, _y, _scalar, _list, _flag, _lowest, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def classify_find_bin(_bins, _x), do: :erlang.nif_error(:nif_not_loaded)

  # Point patterns. Points are interleaved x, y f64 binaries; results with possible NaN come
  # back as f64 binaries.
  def pp_centrography(_op, _points, _aux, _flags), do: :erlang.nif_error(:nif_not_loaded)

  def pp_alpha_shape(_points, _alpha), do: :erlang.nif_error(:nif_not_loaded)
  def pp_alpha_shape_auto(_points, _step, _circles), do: :erlang.nif_error(:nif_not_loaded)

  def pp_distance(_kind, _points, _spec_kind, _spec, _empty, _truncate, _linearized),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_ripley_test(_kind, _points, _spec_kind, _spec, _window_kind, _window, _options),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_poisson(_window_kind, _window, _intensity, _sizes, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_cluster_poisson(_window_kind, _window, _intensity, _sizes, _cluster, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_normal(_window_kind, _window, _sizes, _center, _cov, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_cluster_normal(_window_kind, _window, _sizes, _n_seeds, _cov, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_strauss(_window_kind, _window, _intensity, _sizes, _params, _seed),
    do: :erlang.nif_error(:nif_not_loaded)

  def pp_quadrat(_points, _shape, _geometry, _realizations, _seed, _max_cells),
    do: :erlang.nif_error(:nif_not_loaded)

  # Regionalization (spopt). Attribute matrices are column-major f64 binaries; labels come back
  # as lists. Scalar settings travel as tuples to keep the arities small.
  def region_skater(_graph, _x, _p, _k, _floor, _metric, _flags),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_ward(_graph, _x, _p, _k), do: :erlang.nif_error(:nif_not_loaded)

  def region_sa3(_graph, _x, _p, _min_cluster_size, _extraction),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_spenc(_graph, _x, _p, _k, _modes, _params, _seed, _flags),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_kmeans(_graph, _x, _p, _k, _seeds, _seed, _max_iterations, _symmetrize),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_objective(_x, _p, _labels, _kind, _metric), do: :erlang.nif_error(:nif_not_loaded)
  def region_non_contiguous(_graph, _labels), do: :erlang.nif_error(:nif_not_loaded)

  def region_azp(_graph, _x, _p, _k_seed_moves, _initial, _metric, _symmetrize, _method),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_random(_graph, _n, _num_regions, _cardinality, _settings, _seed, _permutations),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_max_p(_graph, _x, _p, _t, _settings, _policy, _coords),
    do: :erlang.nif_error(:nif_not_loaded)

  def region_max_p_construct(_graph, _x, _p, _t, _threshold, _order, _top_n, _seed),
    do: :erlang.nif_error(:nif_not_loaded)
end
