defmodule Spatex.ESDA do
  @moduledoc """
  Exploratory spatial data analysis, mirroring `esda` 2.10.0.

  ## Global Moran's I

    * `moran/3` - univariate Moran's I (`esda.Moran`)
    * `moran_bv/4` - bivariate Moran's I (`esda.Moran_BV`)
    * `moran_rate/4` - Moran's I of rates with empirical-Bayes adjustment (`esda.Moran_Rate`)
    * `assuncao_rate/2` - the empirical-Bayes standardized rates on their own

  ## Geary, Getis-Ord G and join counts

    * `geary/3` - Geary's C (`esda.Geary`), one-tailed analytic inference
    * `getis_ord_g/3` - general G (`esda.G`); weights are always binarized, so there is no
      `:transform` option, and `y` must be non-negative
    * `join_counts/3` - binary join counts (`esda.Join_Counts`); takes `drop_islands: true`
      (default) and no `:transform` (binary weights are forced)

  They take the options below (except where noted) and reuse the same seeded permutation
  engine as Moran.

  ## Local Moran's I

    * `moran_local/3` - local Moran's I / LISA (`esda.Moran_Local`)
    * `moran_local_bv/4` - bivariate local Moran's I (`esda.Moran_Local_BV`)

  Both use conditional randomization and return a `Spatex.ESDA.MoranLocal` whose per-site
  fields are lists aligned with the graph's ids. See their documentation for the extra options.

  ## Local Geary, local join counts and LOSH

    * `geary_local/3` - local Geary (`esda.Geary_Local`); the graph is row-standardized
    * `geary_local_mv/3` - multivariate local Geary (`esda.Geary_Local_MV`)
    * `join_counts_local/3` - univariate local join counts (`esda.Join_Counts_Local`) of a
      binary variable; weights are used as binary, self weights are ignored
    * `join_counts_local_bv/4`, `join_counts_local_mv/3` - bivariate (`CLC`, `BJC`) and
      multivariate local join counts (`esda.Join_Counts_Local_BV`, `Join_Counts_Local_MV`)
    * `losh/3` - local spatial heteroscedasticity (`esda.LOSH`); weights are used as given

  The first two take `:permutations`, `:seed`, `:keep_simulations` and `:alternative` as
  `moran_local/3` does (no `:transform`); `losh/3` takes `:a` and `:inference`.

  ## Local Getis-Ord G and multiple testing

    * `g_local/3` - local `G_i` and `G*_i` (`esda.G_Local`), returning a `Spatex.ESDA.GLocal`;
      takes `:star`, `:alternative` and `:island_weight` besides the common options, and only
      the `:r` and `:b` transforms
    * `fdr/2` - the false-discovery-rate p-value cut-off (`esda.fdr`)

  ## Options

  The global statistics accept:

    * `:permutations` - number of random permutations, default `999`; `0` skips the
      permutation test and leaves the simulation fields `nil`.
    * `:seed` - non-negative integer below 2^64. Default `nil`: a random seed is drawn and
      stored in the result's `seed` field, so any run can be reproduced. The same seed gives
      bit-for-bit identical simulations regardless of how many threads run them. The stream
      is ChaCha8-based, so the simulated values differ from numpy's; they agree with PySAL
      statistically, not draw by draw.
    * `:transform` - weights transformation applied to the graph first (`:r`, `:b`, `:d`,
      `:v`, `:o`), default `:r` as in PySAL. A graph already carrying that transform is used
      as is.
    * `:keep_simulations` - keep the simulations (`sim`) in the result: `true` (nested lists),
      `false` or `:tensor`. Default `true` for the global statistics and `false` for the local
      ones, whose `n x permutations` matrix can be large (`n = 50_000` with 999 permutations is
      about 2 GB of boxed floats). `:tensor` returns the same draws as an `Nx.Tensor` of type
      `{:f, 64}` built straight from the native binary, with no list decoding: shape
      `{permutations}` for a global statistic and `{n, permutations}` for a local one (join
      counts give one such tensor per `sim_*` vector). It needs the optional `:nx` dependency;
      with `permutations: 0` the simulations are `nil` whatever the value.

  `moran/3` and `moran_rate/4` also take `:two_tailed` (default `true`), which doubles the
  analytic `p_norm` and `p_rand`. The permutation p-value `p_sim` is the folded
  pseudo p-value `(larger + 1) / (P + 1)`, and `p_z_sim` is always one-tailed, as in esda.

  ## Input

  `y`, `x`, `events` and `base` are anything `Spatex.Input.to_f64_binary/2` accepts, aligned
  with the graph's node order, and must be finite. Invalid input raises `ArgumentError`.
  Constant variables raise (PySAL returns NaN), as does a graph without weights.
  """

  alias Spatex.ESDA.{
    G,
    Geary,
    GearyLocal,
    GearyLocalMV,
    GLocal,
    JoinCounts,
    JoinCountsLocal,
    Losh
  }

  alias Spatex.ESDA.{Moran, MoranBV, MoranLocal, MoranRate}
  alias Spatex.{Graph, Input, Native}

  require Logger

  @moran_keys ~w(i ei vi_norm se_i_norm z_norm p_norm vi_rand se_i_rand z_rand p_rand)a
  @perm_keys ~w(p_sim ei_sim se_i_sim vi_sim z_sim p_z_sim)a
  @geary_keys ~w(c ec vc_norm se_c_norm z_norm p_norm vc_rand se_c_rand z_rand p_rand)a
  @geary_perm_keys ~w(p_sim ec_sim se_c_sim vc_sim z_sim p_z_sim)a
  @g_keys ~w(g eg vg se_g z_norm p_norm den_sum)a
  @g_perm_keys ~w(p_sim eg_sim se_g_sim vg_sim z_sim p_z_sim)a
  @jc_keys ~w(j bb ww bw chi2 chi2_p chi2_dof)a
  @jc_perm_keys ~w(min_bb mean_bb max_bb min_bw mean_bw max_bw p_sim_bb p_sim_bw p_sim_chi2
                   p_sim_autocorr_pos p_sim_autocorr_neg sim_autocorr_chi2)a
  @max_seed 18_446_744_073_709_551_615
  @common_opts [permutations: 999, seed: nil, transform: :r, keep_simulations: true]

  @doc """
  Global Moran's I of `y` over `graph`.

  See the module documentation for the options. Returns a `Spatex.ESDA.Moran`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> m = Spatex.ESDA.moran([1, 2, 4], g, permutations: 0)
      iex> Float.round(m.i, 6)
      -0.035714
      iex> m.ei
      -0.5
      iex> m.sim
      nil
  """
  @spec moran(term(), Graph.t(), keyword()) :: Moran.t()
  def moran(y, graph, opts \\ []) do
    opts = validate!(opts, @common_opts ++ [two_tailed: true])
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    struct!(Moran, moran_fields(graph, y, opts))
  end

  @doc """
  Global Geary's C of `y` over `graph` (`esda.Geary`).

  Takes the options of the module documentation (`:transform` defaults to `:r`, as in PySAL).
  The analytic p-values are one-tailed (the tail follows the sign of `C - 1`), there is no
  `:two_tailed` option, and `p_z_sim` is the survival function of `|z_sim|`. Returns a
  `Spatex.ESDA.Geary`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.geary([1, 2, 4, 3], g, permutations: 0)
      iex> {Float.round(m.c, 4), m.ec}
      {0.525, 1.0}
  """
  @spec geary(term(), Graph.t(), keyword()) :: Geary.t()
  def geary(y, graph, opts \\ []) do
    opts = validate!(opts, @common_opts)
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {scalars, sim} =
      graph.ref
      |> Native.esda_geary(y, perms, seed || 0, Input.keep_flag(opts[:keep_simulations]))
      |> unwrap()

    {analytic, perm} = scalars |> Input.from_f64_binary() |> Enum.split(length(@geary_keys))

    struct!(
      Geary,
      Enum.zip(@geary_keys, analytic) ++
        perm_fields(perm, sim, opts, @geary_perm_keys) ++
        [n: graph.n, permutations: perms, seed: seed]
    )
  end

  @doc """
  Global Getis-Ord general G of `y` over `graph` (`esda.G`).

  Weights are binarized whatever transformation the graph carries (PySAL does the same and
  offers no choice), so `:transform` is not accepted. `y` must be non-negative and have at
  least two non-zero values. `z_norm` and `p_norm` are `:nan` when the variance is not
  positive; `p_norm` and `p_z_sim` are `1 - Phi(|z|)` as in esda. Returns a `Spatex.ESDA.G`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.getis_ord_g([1, 2, 4, 3], g, permutations: 0)
      iex> {Float.round(m.g, 4), Float.round(m.eg, 4)}
      {0.6286, 0.5}
  """
  @spec getis_ord_g(term(), Graph.t(), keyword()) :: G.t()
  def getis_ord_g(y, graph, opts \\ []) do
    opts = validate!(opts, Keyword.delete(@common_opts, :transform))
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {scalars, sim} =
      graph.ref
      |> Native.esda_getis_ord_g(y, perms, seed || 0, Input.keep_flag(opts[:keep_simulations]))
      |> unwrap()

    {analytic, perm} = scalars |> Input.from_f64_binary() |> Enum.split(length(@g_keys))

    struct!(
      G,
      Enum.zip(@g_keys, analytic) ++
        perm_fields(perm, sim, opts, @g_perm_keys) ++
        [n: graph.n, permutations: perms, seed: seed]
    )
  end

  @doc """
  Binary join counts of `y` (0 or 1) over `graph` (`esda.Join_Counts`).

  The graph is binarized and every directed edge counts, tallied by halves, so a symmetric
  graph yields `j` undirected joins. With `drop_islands: true` (default) island nodes are
  ignored; with `false` an island's zero-weight self-loop counts as a same-colour join, as in
  esda. `y` must be strictly 0/1 with both values present. Returns a
  `Spatex.ESDA.JoinCounts`; see it for the field list and the `sim_autocorr_chi2` deviation.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.join_counts([1, 1, 0, 0], g, permutations: 0)
      iex> {m.j, m.bb, m.ww, m.bw}
      {3.0, 1.0, 1.0, 1.0}
  """
  @spec join_counts(term(), Graph.t(), keyword()) :: JoinCounts.t()
  def join_counts(y, graph, opts \\ []) do
    opts = validate!(opts, Keyword.delete(@common_opts, :transform) ++ [drop_islands: true])
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {scalars, sims} =
      graph.ref
      |> Native.esda_join_counts(
        y,
        opts[:drop_islands],
        perms,
        seed || 0,
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    values = Input.from_f64_binary(scalars)
    {head, rest} = Enum.split(values, length(@jc_keys))
    {crosstab, rest} = Enum.split(rest, 4)
    {expected, rest} = Enum.split(rest, 4)
    {[pos, neg], perm} = Enum.split(rest, 2)

    fields =
      Enum.zip(@jc_keys, head) ++
        [
          crosstab: Enum.chunk_every(crosstab, 2),
          expected: Enum.chunk_every(expected, 2),
          autocorr_pos: pos,
          autocorr_neg: neg,
          n: graph.n,
          permutations: perms,
          seed: seed
        ]

    struct!(JoinCounts, fields ++ join_perm_fields(perm, sims, perms, opts))
  end

  @doc """
  Bivariate Moran's I between `x` and the spatial lag of `y` (`esda.Moran_BV`).

  Both variables are standardized with the sample standard deviation and the statistic is
  `sum(zx * W zy) / (n - 1)`; permutations shuffle `y`. Takes the options of the module
  documentation except `:two_tailed`. Returns a `Spatex.ESDA.MoranBV`.
  """
  @spec moran_bv(term(), term(), Graph.t(), keyword()) :: MoranBV.t()
  def moran_bv(x, y, graph, opts \\ []) do
    opts = validate!(opts, @common_opts)
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    x = vector!(x, "x", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {scalars, sim} =
      graph.ref
      |> Native.esda_moran_bv(x, y, perms, seed || 0, Input.keep_flag(opts[:keep_simulations]))
      |> unwrap()

    [i | perm] = Input.from_f64_binary(scalars)

    struct!(
      MoranBV,
      [i: i, n: graph.n, permutations: perms, seed: seed] ++ perm_fields(perm, sim, opts)
    )
  end

  @doc """
  Moran's I of rates `events / base` (`esda.Moran_Rate`).

  With `adjusted: true` (default) the rates are the empirical-Bayes standardized rates of
  `assuncao_rate/2`; with `adjusted: false` they are plain ratios. Takes the options of
  `moran/3` plus `:adjusted`. `base` must be positive and `events` non-negative. Returns a
  `Spatex.ESDA.MoranRate` whose `y` holds the rates used.
  """
  @spec moran_rate(term(), term(), Graph.t(), keyword()) :: MoranRate.t()
  def moran_rate(events, base, graph, opts \\ []) do
    opts = validate!(opts, @common_opts ++ [two_tailed: true, adjusted: true])
    graph = prepare_graph!(graph, opts)
    e = vector!(events, "events", graph)
    b = vector!(base, "base", graph)
    rates = e |> Native.esda_rate(b, opts[:adjusted]) |> unwrap()

    struct!(
      MoranRate,
      [adjusted: opts[:adjusted], y: Input.from_f64_binary(rates)] ++
        moran_fields(graph, rates, opts)
    )
  end

  @local_opts [
    permutations: 999,
    seed: nil,
    transform: :r,
    keep_simulations: false,
    geoda_quads: false,
    alternative: :directed
  ]
  @alternatives ~w(directed two_sided greater lesser folded)a

  @doc """
  Local Moran's I (LISA) of `y` over `graph` (`esda.Moran_Local`).

  Returns a `Spatex.ESDA.MoranLocal`; use `Spatex.ESDA.MoranLocal.cluster_labels/2` for the
  High-High / Low-Low / ... map classes. Takes `:permutations`, `:seed`, `:transform` and
  `:keep_simulations` as described in the module documentation, plus:

    * `:geoda_quads` - number the quadrants as GeoDa does (Low-High 3, Low-Low 2), default
      `false`.
    * `:alternative` - `:directed` (default, esda's `None`), `:two_sided`, `:greater`,
      `:lesser` or `:folded`: the tail(s) of the pseudo p-value, replicating
      `esda.significance.calculate_significance` (`:two_sided` can exceed 1, as in esda).

  Inference is by conditional randomization: the site's value stays fixed and its neighbors'
  values are drawn without replacement from the other sites. As in esda, one set of
  permutation draws is shared by all sites, which makes simulations correlated across sites.
  Islands get the smallest possible `p_sim`, `1 / (permutations + 1)` (an esda quirk kept for
  parity). Constant `y` raises (PySAL returns NaN).

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.moran_local([1, 5, 2, 8], g, permutations: 0)
      iex> m.q
      [2, 4, 2, 4]
      iex> Enum.map(m.is, &Float.round(&1, 4))
      [-0.3, -0.25, -0.5, -0.8]
  """
  @spec moran_local(term(), Graph.t(), keyword()) :: MoranLocal.t()
  def moran_local(y, graph, opts \\ []) do
    opts = validate_local!(opts)
    graph = prepare_graph!(graph, opts)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {blocks, sim} =
      graph.ref
      |> Native.esda_moran_local(
        y,
        opts[:geoda_quads],
        perms,
        seed || 0,
        Atom.to_string(opts[:alternative]),
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    [is, q, z, lag, ei, vi, ei_c, vi_c | perm] = blocks(blocks, graph.n)

    local_struct(graph, opts, seed, perm, sim,
      is: is,
      q: Enum.map(q, &trunc/1),
      z: z,
      lag: lag,
      ei: ei,
      vi: vi,
      se_i: Enum.map(vi, &sqrt/1),
      ei_c: ei_c,
      vi_c: vi_c
    )
  end

  @doc """
  Bivariate local Moran's I (`esda.Moran_Local_BV`): the standardized `x` at each site against
  the spatial lag of the standardized `y`.

  Takes the options of `moran_local/3`; permutations draw the neighbors' `y` values. esda has
  no analytic moments for this statistic, so `ei`, `vi`, `se_i`, `ei_c` and `vi_c` are `nil`
  and `lag` is the lag of `z_y`. Returns a `Spatex.ESDA.MoranLocal`.
  """
  @spec moran_local_bv(term(), term(), Graph.t(), keyword()) :: MoranLocal.t()
  def moran_local_bv(x, y, graph, opts \\ []) do
    opts = validate_local!(opts)
    graph = prepare_graph!(graph, opts)
    x = vector!(x, "x", graph)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {blocks, sim} =
      graph.ref
      |> Native.esda_moran_local_bv(
        x,
        y,
        opts[:geoda_quads],
        perms,
        seed || 0,
        Atom.to_string(opts[:alternative]),
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    [is, q, z, lag, z_y | perm] = blocks(blocks, graph.n)

    local_struct(graph, opts, seed, perm, sim,
      is: is,
      q: Enum.map(q, &trunc/1),
      z: z,
      z_y: z_y,
      lag: lag
    )
  end

  @g_local_opts [
    permutations: 999,
    seed: nil,
    transform: :r,
    keep_simulations: false,
    star: false,
    alternative: :directed,
    island_weight: 0.0
  ]

  @doc """
  Local Getis-Ord `G_i` / `G*_i` of `y` over `graph` (`esda.G_Local`).

  `y` must be finite and non-negative with a positive sum (esda does not check). Returns a
  `Spatex.ESDA.GLocal`. Takes `:permutations`, `:seed` and `:keep_simulations` as described in
  the module documentation, plus:

    * `:transform` - `:r` (default, esda's `"R"`) or `:b`. Unlike the other statistics the
      graph is transformed only after the diagonal has been decided, so pass the graph as
      built, not pre-transformed (a graph already labelled `:b` is treated as binary input).
    * `:star` - the self weight rule of esda: `false` (default) computes `G_i`, removing any
      diagonal; `true` computes `G*_i`, keeping a nonzero diagonal or else assigning self weight
      `1` (binary graph or `transform: :b`) or the largest weight of the node's row (`:r`);
      `nil` infers it from the diagonal of the graph; a number assigns that self weight to every
      node (so `star: 1` is a unit self weight, not `true`).
    * `:alternative` - `:directed` (default), `:two_sided`, `:greater`, `:lesser` or `:folded`,
      as for `moran_local/3`.
    * `:island_weight` - weight of the single pseudo-neighbor an island is given in the
      simulations, default `0.0` (esda's `island_weight`).

  Inference is by conditional randomization of the raw `y` values (see `moran_local/3`); islands
  get the smallest possible `p_sim`, an esda quirk kept for parity.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.g_local([1, 2, 3, 4], g, permutations: 0)
      iex> Enum.map(m.gs, &Float.round(&1, 4))
      [0.2222, 0.25, 0.4286, 0.5]
      iex> m = Spatex.ESDA.g_local([1, 2, 3, 4], g, permutations: 0, star: true)
      iex> {m.star, Enum.map(m.gs, &Float.round(&1, 4))}
      {true, [0.15, 0.2, 0.3, 0.35]}
  """
  @spec g_local(term(), Graph.t(), keyword()) :: GLocal.t()
  def g_local(y, graph, opts \\ []) do
    opts = validate_g_local!(opts)
    validate_perms!(opts[:permutations])
    graph = ensure_graph!(graph)
    y = vector!(y, "y", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)
    {star_kind, star_weight} = star_args(opts[:star])
    transform = opts[:transform]

    {blocks, sim, star, y_sum} =
      graph.ref
      |> Native.esda_g_local(
        y,
        {graph.transform == :b, star_kind, star_weight, Atom.to_string(transform)},
        perms,
        seed || 0,
        Atom.to_string(opts[:alternative]),
        opts[:island_weight] * 1.0,
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    [gs, egs, vgs, zs, p_norm | perm] = blocks(blocks, graph.n)

    perm_fields =
      case perm do
        [] ->
          []

        [p_sim, eg_sim, se_g_sim, vg_sim, z_sim, p_z_sim] ->
          if opts[:keep_simulations] do
            [
              p_sim: p_sim,
              sim: Input.simulations(sim, opts[:keep_simulations], {graph.n, perms}),
              eg_sim: eg_sim,
              se_g_sim: se_g_sim,
              vg_sim: vg_sim,
              z_sim: z_sim,
              p_z_sim: p_z_sim
            ]
          else
            [p_sim: p_sim]
          end
      end

    struct!(
      GLocal,
      [gs: gs, egs: egs, vgs: vgs, zs: zs, p_norm: p_norm] ++
        perm_fields ++
        [
          star: star,
          star_arg: opts[:star],
          w_transform: transform,
          y_sum: y_sum,
          n: graph.n,
          permutations: perms,
          seed: seed,
          ids: graph.ids,
          alternative: opts[:alternative],
          island_weight: opts[:island_weight]
        ]
    )
  end

  @doc """
  The false discovery rate cut-off of `p_values` (`esda.fdr`).

  A Benjamini-Hochberg style threshold, not a vector of adjusted p-values: reject the sites
  whose p-value is below the returned number. The p-values are sorted in descending order and
  compared with `index * alpha / n`, `index = n, ..., 1`; the first threshold strictly greater
  than its p-value is returned. If none qualifies, the conservative Bonferroni bound
  `alpha / n` is returned. `alpha` is in `(0, 1]` (default `0.05`).

  Raises `ArgumentError` for an empty list, non-numeric or NaN p-values and negative p-values.
  (esda would sort NaN first; `p_sim` with `alternative: :two_sided` may exceed 1, which is
  accepted.)

  ## Examples

      iex> Spatex.ESDA.fdr([0.001, 0.01, 0.02, 0.04, 0.3, 0.5])
      0.025000000000000005
      iex> Spatex.ESDA.fdr([0.5, 0.6])
      0.025
      iex> Spatex.ESDA.fdr([0.5, 0.6, 0.7, 0.8], 0.1)
      0.025
  """
  @spec fdr(term(), number()) :: float()
  def fdr(p_values, alpha \\ 0.05)

  def fdr(p_values, alpha) when is_number(alpha) do
    {bin, _n} = Input.to_f64_binary(p_values, name: "p_values")
    bin |> Native.esda_fdr(alpha * 1.0) |> unwrap()
  end

  def fdr(_p_values, alpha) do
    raise ArgumentError, "alpha must be a number in (0, 1], got #{inspect(alpha)}"
  end

  @doc """
  Local Geary (`esda.Geary_Local`) of `x` over `graph`, which is always row-standardized.

  Returns a `Spatex.ESDA.GearyLocal`. Takes `:permutations` (default `999`), `:seed`,
  `:keep_simulations` and `:alternative` as in `moran_local/3`, plus:

    * `:labels` - also return esda's cluster/outlier labels in `labs` (`1` outlier, `2`
      cluster, `3` other, `4` not significant), default `false`; needs `permutations > 0`.
    * `:sig` - significance level of the labels, in `(0, 1]`, default `0.05`.
    * `:island_weight` - weight of the single pseudo-neighbor an island is given in the
      simulations, default `0.0` (esda's `island_weight`).

  Inference is by conditional randomization (see `moran_local/3`). Islands get `local_g` 0 and
  the smallest possible `p_sim`. Constant `x` raises (PySAL returns NaN).

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> m = Spatex.ESDA.geary_local([1, 2, 3], g, permutations: 0)
      iex> Enum.map(m.local_g, &Float.round(&1, 4))
      [1.5, 1.5, 1.5]
  """
  @spec geary_local(term(), Graph.t(), keyword()) :: GearyLocal.t()
  def geary_local(x, graph, opts \\ []) do
    opts = validate_local_ext!(opts, labels: false, sig: 0.05, island_weight: 0.0)
    graph = graph |> ensure_graph!() |> Graph.transform(:r)
    x = vector!(x, "x", graph)
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)
    sig = opts[:sig]

    unless is_boolean(opts[:labels]), do: raise(ArgumentError, ":labels must be a boolean")

    unless is_number(sig) and sig > 0 and sig <= 1,
      do: raise(ArgumentError, ":sig must be a number in (0, 1], got #{inspect(sig)}")

    if opts[:labels] and perms == 0,
      do: raise(ArgumentError, ":labels needs p_sim: use permutations > 0")

    {blocks, sim} =
      graph.ref
      |> Native.esda_geary_local(
        x,
        perms,
        seed || 0,
        Atom.to_string(opts[:alternative]),
        Input.keep_flag(opts[:keep_simulations]),
        if(opts[:labels], do: sig / 1, else: 0.0),
        opts[:island_weight] * 1.0
      )
      |> unwrap()

    [local_g | rest] = blocks(blocks, graph.n)
    {p_sim, labs} = geary_rest(rest)

    struct!(GearyLocal,
      local_g: local_g,
      p_sim: p_sim,
      sim: local_sims(sim, graph, perms, opts),
      labs: labs,
      n: graph.n,
      permutations: perms,
      seed: seed,
      ids: graph.ids,
      sig: sig / 1,
      alternative: opts[:alternative]
    )
  end

  defp geary_rest([]), do: {nil, nil}
  defp geary_rest([p_sim]), do: {p_sim, nil}

  defp geary_rest([p_sim, labs]),
    do: {p_sim, Enum.map(labs, fn l -> if is_float(l), do: trunc(l) end)}

  defp local_sims(sim, graph, perms, opts) do
    if perms > 0 and opts[:keep_simulations],
      do: Input.simulations(sim, opts[:keep_simulations], {graph.n, perms})
  end

  @doc """
  Multivariate local Geary (`esda.Geary_Local_MV`) of `variables`, a list of at least two
  vectors aligned with `graph`.

  Returns a `Spatex.ESDA.GearyLocalMV`. The graph is row-standardised, every variable is
  standardised (ddof 0) and `local_g` averages the per-variable local Geary statistics. Options:
  `:permutations` (default `999`), `:seed` and `:keep_simulations` as in `moran_local/3`. The
  pseudo p-value is esda's folded one (there is no `:alternative`); one conditional draw
  relabels the neighbors for every variable at once. esda's draws are unseeded, Spatex's are
  seeded. Islands get `local_g` 0 and the smallest possible `p_sim`; a constant variable raises.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> m = Spatex.ESDA.geary_local_mv([[1, 2, 3], [3, 1, 2]], g, permutations: 0)
      iex> Enum.map(m.local_g, &Float.round(&1, 4))
      [3.75, 2.625, 1.5]
  """
  @spec geary_local_mv([term()], Graph.t(), keyword()) :: GearyLocalMV.t()
  def geary_local_mv(variables, graph, opts \\ []) do
    opts = validate!(opts, permutations: 999, seed: nil, keep_simulations: false)
    validate_perms!(opts[:permutations])
    graph = graph |> ensure_graph!() |> Graph.transform(:r)

    unless is_list(variables) and length(variables) >= 2,
      do: raise(ArgumentError, "geary_local_mv needs at least two variables (a list of vectors)")

    k = length(variables)

    vars =
      variables
      |> Enum.with_index()
      |> Enum.map(fn {v, i} -> vector!(v, "variable #{i}", graph) end)
      |> IO.iodata_to_binary()

    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {blocks, sim} =
      graph.ref
      |> Native.esda_geary_local_mv(
        vars,
        k,
        perms,
        seed || 0,
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    {local_g, p_sim} =
      case blocks(blocks, graph.n) do
        [local_g] -> {local_g, nil}
        [local_g, p_sim] -> {local_g, p_sim}
      end

    struct!(GearyLocalMV,
      local_g: local_g,
      p_sim: p_sim,
      sim: local_sims(sim, graph, perms, opts),
      n: graph.n,
      k: k,
      permutations: perms,
      seed: seed,
      ids: graph.ids
    )
  end

  @doc """
  Univariate local join counts (`esda.Join_Counts_Local`) of the binary variable `y`.

  Returns a `Spatex.ESDA.JoinCountsLocal`. Takes `:permutations` (default `999`), `:seed`,
  `:keep_simulations` and `:alternative` as in `moran_local/3`, and `:island_weight` as in
  `geary_local/3`. Self weights are ignored and
  every non-zero weight is one join, so the graph needs no `:transform`. `y` must contain
  only 0 and 1 (`ArgumentError` otherwise); islands count 0, and `p_sim` is `:nan` wherever the
  count is 0, as in esda.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> Spatex.ESDA.join_counts_local([1, 1, 0], g, permutations: 0).ljc
      [1.0, 1.0, 0.0]
  """
  @spec join_counts_local(term(), Graph.t(), keyword()) :: JoinCountsLocal.t()
  def join_counts_local(y, graph, opts \\ []) do
    opts = validate_local_ext!(opts, island_weight: 0.0)
    graph = ensure_graph!(graph)
    y = vector!(y, "y", graph)

    local_joins(graph, opts, fn perms, seed, alt, keep, iw ->
      Native.esda_join_counts_local(graph.ref, y, perms, seed, alt, keep, iw)
    end)
  end

  @doc """
  Bivariate local join counts (`esda.Join_Counts_Local_BV`) of the binary variables `x` and `z`.

  `:case` selects the statistic:

    * `:clc` (default) - co-location: at a site with `x = z = 1`, the neighbors with
      `x = z = 1`. This is `join_counts_local/3` of `x * z` (esda simulates `z_i` times the drawn
      products, which differs only where the count, and so `p_sim`, is 0 / `:nan`).
    * `:bjc` - bivariate joins: at a site with `x = 1, z = 0`, the neighbors with `x = 0, z = 1`.
      As in esda the simulation draws `z` only (`x_i` times the drawn `z` joins), so it ignores
      the `1 - z_i` and `1 - x_j` factors of the observation.

  Other options and the returned `Spatex.ESDA.JoinCountsLocal` are those of
  `join_counts_local/3`.
  """
  @spec join_counts_local_bv(term(), term(), Graph.t(), keyword()) :: JoinCountsLocal.t()
  def join_counts_local_bv(x, z, graph, opts \\ []) do
    {kind, opts} = Keyword.pop(opts, :case, :clc)
    opts = validate_local_ext!(opts, island_weight: 0.0)
    graph = ensure_graph!(graph)
    [xs, zs] = binary_vectors!([{x, "x"}, {z, "z"}], graph)

    case kind do
      :clc ->
        prod = co_location([xs, zs])

        local_joins(graph, opts, fn perms, seed, alt, keep, iw ->
          Native.esda_join_counts_local(graph.ref, prod, perms, seed, alt, keep, iw)
        end)

      :bjc ->
        {xb, zb} = {pack(xs), pack(zs)}

        local_joins(graph, opts, fn perms, seed, alt, keep, iw ->
          Native.esda_join_counts_local_bjc(graph.ref, xb, zb, perms, seed, alt, keep, iw)
        end)

      other ->
        raise ArgumentError, ":case must be :clc or :bjc, got #{inspect(other)}"
    end
  end

  @doc """
  Multivariate local join counts (`esda.Join_Counts_Local_MV`): at a site where every binary
  variable is 1, the neighbors where every variable is 1.

  `variables` is a list of at least two binary vectors. The statistic and its simulation are
  those of `join_counts_local/3` on the product of the variables, as in esda. Options and the
  returned `Spatex.ESDA.JoinCountsLocal` are those of `join_counts_local/3`.
  """
  @spec join_counts_local_mv([term()], Graph.t(), keyword()) :: JoinCountsLocal.t()
  def join_counts_local_mv(variables, graph, opts \\ []) do
    opts = validate_local_ext!(opts, island_weight: 0.0)
    graph = ensure_graph!(graph)

    unless is_list(variables) and length(variables) >= 2,
      do: raise(ArgumentError, "join_counts_local_mv needs at least two variables (a list)")

    prod =
      variables
      |> Enum.with_index()
      |> Enum.map(fn {v, i} -> {v, "variable #{i}"} end)
      |> binary_vectors!(graph)
      |> co_location()

    local_joins(graph, opts, fn perms, seed, alt, keep, iw ->
      Native.esda_join_counts_local(graph.ref, prod, perms, seed, alt, keep, iw)
    end)
  end

  # Decoded binary vectors (each value 0 or 1), aligned with `graph`.
  defp binary_vectors!(named, graph) do
    for {v, name} <- named do
      values = v |> vector!(name, graph) |> Input.from_f64_binary()

      unless Enum.all?(values, &(&1 in [0.0, 1.0])),
        do: raise(ArgumentError, "#{name} must be binary (every value 0 or 1)")

      values
    end
  end

  defp co_location(vectors), do: vectors |> Enum.zip_with(&Enum.product/1) |> pack()

  defp pack(values), do: values |> Input.to_f64_binary() |> elem(0)

  # Runs a local join count NIF `fun.(perms, seed, alternative, keep, island_weight)` and builds
  # the result.
  defp local_joins(graph, opts, fun) do
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {blocks, sim} =
      perms
      |> fun.(
        seed || 0,
        Atom.to_string(opts[:alternative]),
        Input.keep_flag(opts[:keep_simulations]),
        opts[:island_weight] * 1.0
      )
      |> unwrap()

    {ljc, p_sim} =
      case blocks(blocks, graph.n) do
        [ljc] -> {ljc, nil}
        [ljc, p_sim] -> {ljc, p_sim}
      end

    struct!(JoinCountsLocal,
      ljc: ljc,
      p_sim: p_sim,
      sim: local_sims(sim, graph, perms, opts),
      n: graph.n,
      permutations: perms,
      seed: seed,
      ids: graph.ids,
      alternative: opts[:alternative]
    )
  end

  @doc """
  Local spatial heteroscedasticity (`esda.LOSH`, Ord and Getis 2012) of `y` over `graph`.

  Returns a `Spatex.ESDA.Losh`. The graph is used as given (the statistic does not depend on
  row scaling). Options:

    * `:a` - residual exponent, default `2` (a variance measure); `1` gives absolute
      deviations.
    * `:inference` - `nil` (default) or `:chi2` (`:chi_square` is accepted too): chi-square
      p-values `pval`, valid for `a == 2` only (with another `a` a warning is logged and
      `pval` stays `nil`, as in esda). esda has no permutation inference, so there is none.

  Islands: esda returns all-`NaN` values as soon as one site has no neighbors; Spatex computes
  the statistic over the sites that have neighbors and reports `:nan` for the islands.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> m = Spatex.ESDA.losh([1, 2, 4], g)
      iex> Enum.map(m.hi, &Float.round(&1, 4))
      [0.1429, 1.4286, 0.1429]
  """
  @spec losh(term(), Graph.t(), keyword()) :: Losh.t()
  def losh(y, graph, opts \\ []) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    opts = Keyword.validate!(opts, a: 2, inference: nil)
    graph = ensure_graph!(graph)
    a = opts[:a]

    validate_losh_a!(a)
    inference = losh_inference!(opts[:inference])
    y = vector!(y, "y", graph)
    chi2 = inference == :chi2 and a == 2

    if inference == :chi2 and not chi2,
      do: Logger.warning("LOSH chi-square inference assumes a = 2, got a = #{a}: no p-values")

    [hi, ylag, yresid, var_hi | pval] =
      graph.ref |> Native.esda_losh(y, a / 1, chi2) |> unwrap() |> blocks(graph.n)

    struct!(Losh,
      hi: hi,
      ylag: ylag,
      yresid: yresid,
      var_hi: var_hi,
      pval: List.first(pval),
      n: graph.n,
      a: a / 1,
      ids: graph.ids,
      inference: inference
    )
  end

  defp validate_losh_a!(a) when is_number(a) and a > 0, do: :ok

  defp validate_losh_a!(a),
    do: raise(ArgumentError, ":a must be a positive number, got #{inspect(a)}")

  defp losh_inference!(nil), do: nil
  defp losh_inference!(i) when i in [:chi2, :chi_square], do: :chi2

  defp losh_inference!(other),
    do: raise(ArgumentError, ":inference must be nil or :chi2, got #{inspect(other)}")

  defp validate_local_ext!(opts, extra) do
    opts =
      validate!(
        opts,
        [permutations: 999, seed: nil, keep_simulations: false, alternative: :directed] ++ extra
      )

    validate_perms!(opts[:permutations])

    if Keyword.has_key?(opts, :island_weight) and not is_number(opts[:island_weight]),
      do: raise(ArgumentError, ":island_weight must be a number")

    unless opts[:alternative] in @alternatives do
      raise ArgumentError,
            ":alternative must be one of #{inspect(@alternatives)}, got #{inspect(opts[:alternative])}"
    end

    opts
  end

  @doc """
  Empirical-Bayes standardized rates of Assuncao and Reis (1999), `esda.smoothing.assuncao_rate`.

  ## Examples

      iex> [a, b] = Spatex.ESDA.assuncao_rate([1, 4], [10, 20])
      iex> {Float.round(a, 4), Float.round(b, 4)}
      {-0.7559, 0.3651}
  """
  @spec assuncao_rate(term(), term()) :: [float()]
  def assuncao_rate(events, base) do
    {e, n} = Input.to_f64_binary(events, name: "events")
    {b, _} = Input.to_f64_binary(base, name: "base", length: n)
    e |> Native.esda_rate(b, true) |> unwrap() |> Input.from_f64_binary()
  end

  # -- shared plumbing -------------------------------------------------------------------

  defp moran_fields(graph, y, opts) do
    perms = opts[:permutations]
    seed = resolve_seed(opts[:seed], perms)

    {scalars, sim} =
      graph.ref
      |> Native.esda_moran(
        y,
        perms,
        seed || 0,
        opts[:two_tailed],
        Input.keep_flag(opts[:keep_simulations])
      )
      |> unwrap()

    {analytic, perm} = scalars |> Input.from_f64_binary() |> Enum.split(length(@moran_keys))

    Enum.zip(@moran_keys, analytic) ++
      perm_fields(perm, sim, opts) ++ [n: graph.n, permutations: perms, seed: seed]
  end

  # Permutation scalars arrive only when permutations > 0; otherwise the struct defaults (nil) stay.
  defp perm_fields(perm, sim, opts, keys \\ @perm_keys)
  defp perm_fields([], _sim, _opts, _keys), do: []

  defp perm_fields(perm, sim, opts, keys) do
    sim =
      if opts[:keep_simulations],
        do: Input.simulations(sim, opts[:keep_simulations], {div(byte_size(sim), 8)})

    [sim: sim] ++ Enum.zip(keys, perm)
  end

  defp join_perm_fields([], _sims, _perms, _opts), do: []

  defp join_perm_fields(perm, sims, perms, opts) do
    sim_fields =
      if opts[:keep_simulations] do
        keep = opts[:keep_simulations]
        [bb, ww, bw, chi2] = Input.simulation_vectors(sims, keep, 4, perms)

        [
          sim_bb: bb,
          sim_bw: bw,
          sim_chi2: chi2,
          sim_autocurr_pos: Input.add_simulations(bb, ww),
          sim_autocurr_neg: bw
        ]
      else
        []
      end

    sim_fields ++ Enum.zip(@jc_perm_keys, perm)
  end

  defp blocks(binary, n), do: binary |> Input.from_f64_binary() |> Enum.chunk_every(n)

  defp sqrt(v) when is_float(v), do: if(v < 0.0, do: :nan, else: :math.sqrt(v))
  defp sqrt(other), do: other

  defp local_struct(graph, opts, seed, perm, sim, fields) do
    perm_fields =
      case perm do
        [] ->
          []

        [p_sim, ei_sim, se_i_sim, vi_sim, z_sim, p_z_sim] ->
          if opts[:keep_simulations] do
            [
              p_sim: p_sim,
              sim:
                Input.simulations(
                  sim,
                  opts[:keep_simulations],
                  {graph.n, opts[:permutations]}
                ),
              ei_sim: ei_sim,
              se_i_sim: se_i_sim,
              vi_sim: vi_sim,
              z_sim: z_sim,
              p_z_sim: p_z_sim
            ]
          else
            [p_sim: p_sim]
          end
      end

    struct!(
      MoranLocal,
      fields ++
        perm_fields ++
        [
          n: graph.n,
          permutations: opts[:permutations],
          seed: seed,
          ids: graph.ids,
          geoda_quads: opts[:geoda_quads],
          alternative: opts[:alternative]
        ]
    )
  end

  defp validate_local!(opts) do
    opts = validate!(opts, @local_opts)

    unless is_boolean(opts[:geoda_quads]),
      do: raise(ArgumentError, ":geoda_quads must be a boolean")

    unless opts[:alternative] in @alternatives do
      raise ArgumentError,
            ":alternative must be one of #{inspect(@alternatives)}, got #{inspect(opts[:alternative])}"
    end

    opts
  end

  defp validate_g_local!(opts) do
    opts = validate!(opts, @g_local_opts)

    unless opts[:alternative] in @alternatives do
      raise ArgumentError,
            ":alternative must be one of #{inspect(@alternatives)}, got #{inspect(opts[:alternative])}"
    end

    unless is_number(opts[:island_weight]),
      do: raise(ArgumentError, ":island_weight must be a number")

    unless opts[:star] in [true, false, nil] or is_number(opts[:star]) do
      raise ArgumentError,
            ":star must be true, false, nil or a self weight, got #{inspect(opts[:star])}"
    end

    Keyword.put(opts, :transform, g_transform!(opts[:transform]))
  end

  defp g_transform!(t) when t in [:r, :b], do: t

  defp g_transform!(t) when t in [:R, :B],
    do: t |> Atom.to_string() |> String.downcase() |> String.to_atom()

  defp g_transform!(t) when is_binary(t) do
    case String.downcase(t) do
      "r" -> :r
      "b" -> :b
      _ -> g_transform!(nil)
    end
  end

  defp g_transform!(t) do
    raise ArgumentError,
          "g_local supports only the :transform values :r and :b, got #{inspect(t)}"
  end

  defp star_args(nil), do: {"infer", 0.0}
  defp star_args(false), do: {"off", 0.0}
  defp star_args(true), do: {"on", 0.0}
  defp star_args(weight), do: {"weight", weight * 1.0}

  defp prepare_graph!(graph, opts) do
    validate_perms!(opts[:permutations])
    graph = ensure_graph!(graph)
    # Statistics that force binary weights (G, join counts) take no :transform.
    if Keyword.has_key?(opts, :transform),
      do: Graph.transform(graph, opts[:transform]),
      else: graph
  end

  defp vector!(values, name, graph) do
    {bin, _n} = Input.to_f64_binary(values, name: name, length: graph.n)
    bin
  end

  defp ensure_graph!(%Graph{} = graph), do: graph

  defp ensure_graph!(other) do
    raise ArgumentError, "expected a Spatex.Graph, got #{inspect(other, limit: 3)}"
  end

  defp validate_perms!(p) when is_integer(p) and p >= 0, do: :ok

  defp validate_perms!(p),
    do: raise(ArgumentError, ":permutations must be a non-negative integer, got #{inspect(p)}")

  defp resolve_seed(_seed, 0), do: nil
  defp resolve_seed(nil, _perms), do: :rand.uniform(4_294_967_296) - 1
  defp resolve_seed(seed, _perms) when is_integer(seed) and seed in 0..@max_seed//1, do: seed

  defp resolve_seed(seed, _perms) do
    raise ArgumentError, ":seed must be an integer in 0..2^64-1 or nil, got #{inspect(seed)}"
  end

  defp validate!(opts, defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    opts = Keyword.validate!(opts, defaults)

    Input.keep_simulations!(opts[:keep_simulations])

    for key <- [:two_tailed, :adjusted, :drop_islands], Keyword.has_key?(opts, key) do
      unless is_boolean(opts[key]), do: raise(ArgumentError, ":#{key} must be a boolean")
    end

    opts
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
