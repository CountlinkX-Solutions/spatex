defmodule Spatex.Inequality do
  @moduledoc """
  Inequality measures, mirroring the `inequality` package (1.2.0).

    * `gini/1` - Gini coefficient (`inequality.gini.Gini`)
    * `theil/1` - Theil's T (`inequality.theil.Theil`)
    * `theil_d/3` - between / within decomposition of Theil's T by regions, with an optional
      permutation test (`inequality.theil.TheilD` / `TheilDSim`)
    * `spatial_gini/3` - how much inequality lies between neighbors (`inequality.gini.Gini_Spatial`)
    * `polarization/3` - spatial polarization index with a permutation test
      (`inequality.polarization.S`)
    * `atkinson/2` - Atkinson index and equally distributed equivalent
      (`inequality.atkinson.Atkinson`)
    * `schutz/1` - Schutz distance, intersection point and coefficient (`inequality.schutz.Schutz`)
    * `lorenz_curve/1`, `wolfson/1` - Lorenz curve and Wolfson bipolarization index
      (`inequality.wolfson`)

  Values are lists of numbers (or anything `Spatex.Input.to_f64_binary/2` accepts); `theil/1` and
  `theil_d/3` also take an `n x t` matrix (a list of rows) and then return one value per column.
  Invalid input raises `ArgumentError`. Permutation tests take `:permutations` and `:seed`;
  without a seed a random one is drawn and stored in the result, and the same seed gives
  bit-identical results at any thread count.
  """

  alias Spatex.{Graph, Input, Native}
  alias Spatex.Inequality.{Atkinson, Polarization, Schutz, SpatialGini, TheilD}

  @max_seed 18_446_744_073_709_551_615

  @doc """
  Gini coefficient of non-negative values, `(2 sum_i i x_(i) - (n + 1) sum x) / (n sum x)`.

  `:nan` when the values sum to zero.

  ## Examples

      iex> Spatex.Inequality.gini([1, 1, 1, 1])
      0.0
      iex> Spatex.Inequality.gini([0, 0, 0, 1])
      0.75
  """
  @spec gini(term()) :: float() | :nan
  def gini(x) do
    {bin, n} = Input.to_f64_binary(x, name: "x")
    if n == 0, do: raise(ArgumentError, "x must not be empty")
    [g] = bin |> Native.inequality_gini() |> unwrap() |> Input.from_f64_binary()
    g
  end

  @doc """
  Theil's T, `sum_i s_i ln(n s_i)` with `s_i = y_i / sum y` (zeros are lifted to the smallest
  positive float, like the Python package).

  A vector gives a float, an `n x t` matrix a list of `t` floats.

  ## Examples

      iex> Spatex.Inequality.theil([2, 2, 2])
      0.0
  """
  @spec theil(term()) :: float() | [float()]
  def theil(y) do
    {bin, n, t, shape} = values!(y, "y")
    result = bin |> Native.inequality_theil(n, t) |> unwrap() |> Input.from_f64_binary()
    shaped(result, shape)
  end

  @doc """
  Decomposition of Theil's T by a partition of the observations into regions
  (`inequality.theil.TheilD`).

  `partition` has one region label per observation (a list, `Nx` vector or `Explorer.Series`; see `Spatex.Input.to_labels/2`). Returns a
  `Spatex.Inequality.TheilD`.

  ## Options

    * `:permutations` - shuffle the rows of `y` across observations this many times to test the
      between-group term (default `0`, `inequality.TheilDSim`), `:seed`.
  """
  @spec theil_d(term(), [term()], keyword()) :: TheilD.t()
  def theil_d(y, partition, opts \\ []) do
    opts = opts!(opts, permutations: 0, seed: nil)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    {bin, n, t, shape} = values!(y, "y")

    partition = Input.to_labels(partition, name: "partition", length: n)

    codes = partition |> Enum.uniq() |> Enum.sort() |> Enum.with_index() |> Map.new()
    part_bin = for label <- partition, into: <<>>, do: <<Map.fetch!(codes, label)::32-native>>

    blocks =
      bin
      |> Native.inequality_theil_d({n, t}, part_bin, {perms, seed || 0})
      |> unwrap()
      |> Input.from_f64_binary()

    {total, rest} = Enum.split(blocks, t)
    {bg, rest} = Enum.split(rest, t)
    {wg, pvalues} = Enum.split(rest, t)

    %TheilD{
      t: shaped(total, shape),
      bg: shaped(bg, shape),
      wg: shaped(wg, shape),
      bg_pvalue: if(perms > 0, do: shaped(pvalues, shape)),
      seed: seed,
      permutations: perms
    }
  end

  @doc """
  Spatial Gini (`inequality.gini.Gini_Spatial`): splits the total absolute difference between
  observations into pairs of neighbors and non-neighbors and tests the non-neighbor part by
  permutation.

  `x` is aligned with the graph's ids. The graph is read as binary: every non-zero edge makes two
  observations neighbors, whatever its weight.

  ## Options

    * `:permutations` - default `99` (as in PySAL); `0` skips the test.
    * `:seed` - see the module documentation.
    * `:keep_simulations` - keep `wcgp` and `polarization_sim` in the result (default `true`).
      `true` keeps them as lists, `:tensor` as `{permutations}` `Nx.Tensor`s of type `{:f, 64}`
      (optional `:nx` dependency), `false` drops them.
  """
  @spec spatial_gini(term(), Graph.t(), keyword()) :: SpatialGini.t()
  def spatial_gini(x, graph, opts \\ []) do
    opts = opts!(opts, permutations: 99, seed: nil, keep_simulations: true)
    graph = ensure_graph!(graph)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    keep = Input.keep_simulations!(opts[:keep_simulations])
    {bin, _n} = Input.to_f64_binary(x, name: "x", length: graph.n)

    {scalars, wcgp, pol_sim} =
      graph.ref |> Native.inequality_spatial_gini(bin, perms, seed || 0) |> unwrap()

    [g, wg, wcg, share, dtotal, den, pol | perm] = Input.from_f64_binary(scalars)

    base = %SpatialGini{
      g: g,
      wg: wg,
      wcg: wcg,
      wcg_share: share,
      dtotal: dtotal,
      den: den,
      polarization: pol,
      seed: seed,
      permutations: perms
    }

    if perms == 0 do
      base
    else
      [p_sim, e_wcg, s_wcg, z_wcg, p_z_sim, pol_p] = perm

      %{
        base
        | p_sim: p_sim,
          e_wcg: e_wcg,
          s_wcg: s_wcg,
          z_wcg: z_wcg,
          p_z_sim: p_z_sim,
          polarization_p_sim: pol_p,
          wcgp: if(keep, do: Input.simulations(wcgp, keep, {perms})),
          polarization_sim: if(keep, do: Input.simulations(pol_sim, keep, {perms}))
      }
    end
  end

  @doc """
  Spatial polarization index of a variable over a graph (`inequality.polarization.S`).

  The values are split into groups (`:k` quantile classes, or explicit `:bins`) and the index
  measures how far like groups form connected blocks of the graph: 1 when each group is one
  connected block, towards 0 when the groups are scattered. Only the graph's structure is used:
  every stored entry is an edge whatever its weight, and islands are components of their own.
  `x` is aligned with the graph's ids (PySAL aligns a data frame's index).

  Observations the grouping leaves without a group take no part: with `:bins` (right-closed,
  `(a, b]`) a value equal to the first edge, and with constant data under `k > 1`, every
  observation. Quantile edges that coincide are merged but the number of groups stays `k`, as in
  `pandas.qcut(..., duplicates: "drop")`.

  ## Options

    * `:k` - number of quantile classes, an integer in `1..n` (default `2`); ignored with `:bins`.
    * `:bins` - at least two strictly increasing edges covering the data (default `nil`).
    * `:permutations` - shuffle the group labels over the nodes this many times (default `999`, as
      in PySAL); `0` skips the test. `:seed` - see the module documentation.
    * `:keep_simulations` - keep the simulated indices in `sim` (default `true`, the global
      statistic convention; PySAL's `keep_sim` defaults to `False`). `true` keeps a list, `:tensor`
      a `{permutations}` `Nx.Tensor` of type `{:f, 64}` (optional `:nx` dependency), `false` drops
      them.

  Returns a `Spatex.Inequality.Polarization`. Permutation draws come from Spatex's seeded ChaCha8
  streams, so `p_value` agrees with PySAL's statistically, not bit for bit.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1], nil, ids: [0, 1, 2])
      iex> s = Spatex.Inequality.polarization([1, 2, 9], g, permutations: 0)
      iex> {s.statistic, s.n_i_components}
      {1.0, 2}
  """
  @spec polarization(term(), Graph.t(), keyword()) :: Polarization.t()
  def polarization(x, graph, opts \\ []) do
    opts =
      opts!(opts, k: 2, bins: nil, permutations: 999, seed: nil, keep_simulations: true)

    graph = ensure_graph!(graph)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    keep = Input.keep_simulations!(opts[:keep_simulations])
    {bin, n} = Input.to_f64_binary(x, name: "x", length: graph.n)
    grouping = grouping!(opts[:k], opts[:bins], n)

    {scalars, labels, sim} =
      graph.ref
      |> Native.inequality_polarization(bin, grouping, {perms, seed || 0, Input.keep_flag(keep)})
      |> unwrap()

    [statistic, p_value, n_a, n_g, n_i] = Input.from_f64_binary(scalars)
    {i_labels, rest} = labels |> Input.from_f64_binary() |> Enum.map(&label/1) |> Enum.split(n)
    {a_labels, g_labels} = Enum.split(rest, n)

    %Polarization{
      statistic: statistic,
      n: n,
      n_a_components: round(n_a),
      n_g_components: round(n_g),
      n_i_components: round(n_i),
      i_labels: i_labels,
      a_labels: a_labels,
      g_labels: g_labels,
      p_value: if(perms > 0, do: p_value),
      sim: if(perms > 0 and keep, do: Input.simulations(sim, keep, {perms})),
      seed: seed,
      permutations: perms
    }
  end

  defp label(value) when value < 0, do: nil
  defp label(value), do: round(value)

  defp grouping!(k, nil, n) do
    unless is_integer(k) and k >= 1 and k <= n,
      do: raise(ArgumentError, ":k must be an integer in 1..#{n}, got #{inspect(k)}")

    {k, <<>>}
  end

  defp grouping!(_k, bins, _n) do
    unless is_list(bins) and length(bins) >= 2,
      do: raise(ArgumentError, ":bins must be a list of at least two edges, got #{inspect(bins)}")

    {bin, _} = Input.to_f64_binary(bins, name: "bins")
    {1, bin}
  end

  @doc """
  Atkinson index of strictly positive values (`inequality.atkinson.Atkinson`).

  `epsilon >= 0` is the inequality aversion: larger values weigh the lower tail more. `0` gives
  an index of 0 and `1` uses the geometric mean. Returns a `Spatex.Inequality.Atkinson` with the
  index `a` and the equally distributed equivalent `ede` (PySAL's `A` and `EDE`; the
  `atkinson()` function returns only `A`).

  Raises `ArgumentError` for empty input, a value that is not positive, or a negative `epsilon`.

  ## Examples

      iex> a = Spatex.Inequality.atkinson([10, 20, 30, 40, 50], 0.5)
      iex> Float.round(a.a, 5)
      0.06315
      iex> Float.round(a.ede, 4)
      28.1054
  """
  @spec atkinson(term(), number()) :: Atkinson.t()
  def atkinson(y, epsilon) do
    unless is_number(epsilon) and epsilon >= 0,
      do: raise(ArgumentError, "epsilon must be a non-negative number, got #{inspect(epsilon)}")

    bin = vector!(y, "y")

    [a, ede] =
      bin |> Native.inequality_atkinson(epsilon * 1.0) |> unwrap() |> Input.from_f64_binary()

    %Atkinson{a: a, ede: ede, epsilon: epsilon}
  end

  @doc """
  Schutz measures of a distribution (`inequality.schutz.Schutz`).

  `y` is a vector of values (PySAL takes a data frame and a column name). Returns a
  `Spatex.Inequality.Schutz`: the Schutz `distance` between the Lorenz curve and the line of
  equality, the cumulative population share (`intersection_point`) where it is first reached
  and the original Schutz `coefficient`. Raises `ArgumentError` for empty input or values that
  sum to zero (PySAL fails with an `IndexError` there).

  ## Examples

      iex> s = Spatex.Inequality.schutz([1000, 2000, 1500, 3000, 2500])
      iex> {Float.round(s.distance, 2), Float.round(s.intersection_point, 1)}
      {0.15, 0.6}
      iex> Float.round(s.coefficient, 1)
      7.5
  """
  @spec schutz(term()) :: Schutz.t()
  def schutz(y) do
    [distance, point, coefficient] =
      y |> vector!("y") |> Native.inequality_schutz() |> unwrap() |> Input.from_f64_binary()

    %Schutz{distance: distance, intersection_point: point, coefficient: coefficient}
  end

  @doc """
  Lorenz curve of a distribution (`inequality.wolfson.lorenz_curve`).

  Returns `{population, share}`: the cumulative population share (`n + 1` evenly spaced points
  from 0 to 1) and the cumulative share of the sorted values, both starting at 0.

  ## Examples

      iex> {pop, share} = Spatex.Inequality.lorenz_curve([1, 1, 2])
      iex> pop
      [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
      iex> share
      [0.0, 0.25, 0.5, 1.0]
  """
  @spec lorenz_curve(term()) :: {[float()], [float() | :nan]}
  def lorenz_curve(y) do
    {pop, share} = y |> vector!("y") |> Native.inequality_lorenz_curve() |> unwrap()
    {Input.from_f64_binary(pop), Input.from_f64_binary(share)}
  end

  @doc """
  Wolfson bipolarization index (`inequality.wolfson.wolfson`),
  `W = (2 D50 - G) mean / median` with `D50 = 0.5 - L(0.5)`, `L` the Lorenz curve and `G` the
  Gini coefficient.

  ## Examples

      iex> Spatex.Inequality.wolfson([6, 6, 8, 8, 10, 10, 12, 12]) |> Float.round(4)
      0.0833
  """
  @spec wolfson(term()) :: float() | :nan | :infinity | :neg_infinity
  def wolfson(y) do
    [w] = y |> vector!("y") |> Native.inequality_wolfson() |> unwrap() |> Input.from_f64_binary()
    w
  end

  # A vector (flat list) or an n x t matrix (list of rows), as {binary, n, t, :vector | :matrix}.
  defp values!([first | _] = rows, name) when is_list(first) do
    t = length(first)

    unless t > 0 and Enum.all?(rows, &(is_list(&1) and length(&1) == t)),
      do: raise(ArgumentError, "every row of #{name} must have the same length")

    {bin, n} = Input.to_f64_binary(List.flatten(rows), name: name)
    {bin, div(n, t), t, :matrix}
  end

  defp values!(y, name) do
    {bin, n} = Input.to_f64_binary(y, name: name)
    if n == 0, do: raise(ArgumentError, "#{name} must not be empty")
    {bin, n, 1, :vector}
  end

  defp vector!(y, name) do
    {bin, n} = Input.to_f64_binary(y, name: name)
    if n == 0, do: raise(ArgumentError, "#{name} must not be empty")
    bin
  end

  defp shaped([value], :vector), do: value
  defp shaped(values, _shape), do: values

  defp ensure_graph!(%Graph{} = graph), do: graph

  defp ensure_graph!(other),
    do: raise(ArgumentError, "expected a Spatex.Graph, got #{inspect(other, limit: 3)}")

  defp opts!(opts, defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    Keyword.validate!(opts, defaults)
  end

  defp permutations!(p) when is_integer(p) and p >= 0, do: p

  defp permutations!(other),
    do:
      raise(ArgumentError, ":permutations must be a non-negative integer, got #{inspect(other)}")

  defp resolve_seed(_seed, 0), do: nil
  defp resolve_seed(nil, _perms), do: :rand.uniform(4_294_967_296) - 1
  defp resolve_seed(seed, _perms) when is_integer(seed) and seed in 0..@max_seed//1, do: seed

  defp resolve_seed(seed, _perms) do
    raise ArgumentError, ":seed must be an integer in 0..2^64-1 or nil, got #{inspect(seed)}"
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
  defp unwrap(other), do: other
end
