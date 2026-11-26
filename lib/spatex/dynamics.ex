defmodule Spatex.Dynamics do
  @moduledoc """
  Spatial dynamics, mirroring `giddy` 2.3.9.

  ## Markov chains

    * `markov/2` - classic Markov chain on a panel of classes (`giddy.markov.Markov`)
    * `spatial_markov/3` - transitions conditioned on the class of the spatial lag
      (`giddy.markov.Spatial_Markov`) with the homogeneity tests
    * `lisa_markov/3` - Markov chain on local Moran quadrants (`giddy.markov.LISA_Markov`)
    * `kullback/2` - Kullback's information test of Markov homogeneity across strata
      (`giddy.markov.kullback`)
    * `steady_state/2`, `mfpt/2`, `sojourn_time/2` - the ergodic summaries of any transition
      probability matrix (`giddy.ergodic`)

  Panels are `n x t` matrices: one row per location, one column per period, given as a list of
  lists or a rank-2 `Nx.Tensor`. `markov/2` takes class ids (integers) or, with `k:`, continuous
  values that are classified into quantiles period by period. Matrices come back as lists of rows.
  Mean first passage and sojourn times can be `:infinity`.

  ## Rank mobility

    * `kendall_tau/2` - Kendall's tau-b (`giddy.rank.Tau`)
    * `spatial_tau/4` - tau over neighbor pairs with a permutation test (`giddy.rank.SpatialTau`)
    * `theta/3` - regime-wise direction of rank changes (`giddy.rank.Theta`)

  ## Permutations and seeds

  Permutation tests take `:permutations` and `:seed`. Without a seed a random one is drawn and
  stored in the result, so a run can be reproduced; the same seed gives bit-identical results
  at any thread count (ChaCha8 streams, so draws differ from numpy's while the distribution
  agrees).

  Invalid input raises `ArgumentError`.
  """

  alias Spatex.{Classify, ESDA, Graph, Input, Native}
  alias Spatex.Dynamics.{Kullback, LisaMarkov, Markov, SpatialMarkov, SpatialTau, Tau, Theta}
  alias Spatex.Dynamics.{TauLocal, TauLocalNeighbor, TauLocalNeighborhood, TauRegional}

  @max_seed 18_446_744_073_709_551_615
  # Quadrant order of the Kronecker product of two binary chains (A in giddy).
  @kron_to_quadrant [0, 2, 3, 1]

  # ---------------------------------------------------------------------------------------------
  # Markov
  # ---------------------------------------------------------------------------------------------

  @doc """
  Markov chain of a panel of classes (`giddy.markov.Markov`).

  `data` is an `n x t` matrix of integer class ids, or of numbers when `:k` is given (each
  period is then classified into `k` quantile classes, like `mapclassify.Quantiles(y_s, k)`).

  ## Options

    * `:k` - number of quantile classes for continuous values (default `nil`: `data` holds
      class ids).
    * `:classes` - the class labels in row order (default: the sorted distinct labels).
    * `:fill_empty_classes` - give rows without observed transitions a self-transition of 1
      (default `false`; `steady_state`, `mfpt` and `sojourn_time` always do, as in giddy).

  ## Examples

      iex> m = Spatex.Dynamics.markov([[0, 1, 1], [1, 0, 0]])
      iex> m.transitions
      [[1.0, 1.0], [1.0, 1.0]]
      iex> m.steady_state
      [0.5, 0.5]
  """
  @spec markov(term(), keyword()) :: Markov.t()
  def markov(data, opts \\ []) do
    opts = opts!(opts, k: nil, classes: nil, fill_empty_classes: false)
    fill = boolean!(opts, :fill_empty_classes)
    k = opt_k!(opts[:k], :k, nil)
    {rows, n, t} = panel!(data, "data")
    rows = if k, do: classify_periods(rows, k), else: rows
    classes = classes!(rows, opts[:classes])
    {ids, _} = encode_ids(rows, classes)
    build_markov(ids, n, t, classes, fill)
  end

  @doc """
  Markov chain of the rank of every area (`giddy.markov.FullRank_Markov`).

  `y` is an `n x t` matrix of numbers. In each period the areas are ranked from `1` (largest)
  to `n` (smallest); ties keep their order of appearance (scipy's `rankdata(method="ordinal")`
  on the ascending order). The ranks are the states of `markov/2`, kept in `ranks`. Options:
  `:fill_empty_classes` as in `markov/2`. At most 1000 areas (the class limit of `markov/2`).

  ## Examples

      iex> m = Spatex.Dynamics.full_rank_markov([[1, 3], [2, 1], [3, 2]])
      iex> m.ranks
      [[3, 1], [2, 3], [1, 2]]
  """
  @spec full_rank_markov(term(), keyword()) :: Markov.t()
  def full_rank_markov(y, opts \\ []) do
    rank_markov(y, opts, fn order, n ->
      # order[p] is the area at ascending position p; its full rank is n - p.
      order |> Enum.with_index() |> Enum.sort() |> Enum.map(fn {_area, p} -> n - p end)
    end)
  end

  @doc """
  Markov chain of the area holding every rank (`giddy.markov.GeoRank_Markov`).

  `y` is an `n x t` matrix of numbers. In each period, state `r` (1-based, ascending ranks as in
  giddy) is the 1-based index of the area holding rank `r`, with ties in order of appearance;
  these states are kept in `ranks` and fed to `markov/2`. Options: `:fill_empty_classes`.

  ## Examples

      iex> Spatex.Dynamics.geo_rank_markov([[1, 3], [2, 1], [3, 2]]).ranks
      [[1, 2], [2, 3], [3, 1]]
  """
  @spec geo_rank_markov(term(), keyword()) :: Markov.t()
  def geo_rank_markov(y, opts \\ []) do
    rank_markov(y, opts, fn order, _n -> Enum.map(order, &(&1 + 1)) end)
  end

  # Applies `states.(order, n)` to every period, where `order` lists the area indices in
  # ascending value order (ties by index), and builds the Markov chain of the states.
  defp rank_markov(y, opts, states) do
    opts = opts!(opts, fill_empty_classes: false)
    fill = boolean!(opts, :fill_empty_classes)
    {rows, n, t} = panel!(y, "y")

    unless Enum.all?(rows, fn row -> Enum.all?(row, &is_number/1) end),
      do: raise(ArgumentError, "y must contain only numbers")

    ranks =
      rows
      |> Enum.zip_with(& &1)
      |> Enum.map(fn col ->
        col
        |> Enum.with_index()
        |> Enum.sort_by(fn {v, i} -> {v, i} end)
        |> Enum.map(&elem(&1, 1))
        |> states.(n)
      end)
      |> Enum.zip_with(& &1)

    classes = classes!(ranks, nil)
    {ids, _} = encode_ids(ranks, classes)
    %{build_markov(ids, n, t, classes, fill) | ranks: ranks}
  end

  @doc """
  Steady-state distribution of a transition probability matrix (`giddy.ergodic.steady_state`).

  A list of `k` floats for an irreducible chain, otherwise one list per recurrent class. A
  matrix with all-zero rows raises unless `fill_empty_classes: true`, which makes those states
  absorbing.

  ## Examples

      iex> Spatex.Dynamics.steady_state([[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.5]])
      ...> |> Enum.map(&Float.round(&1, 6))
      [0.4, 0.2, 0.4]
  """
  @spec steady_state([[number()]], keyword()) :: [float()] | [[float()]]
  def steady_state(p, opts \\ []), do: p |> ergodic!(opts) |> Map.fetch!(:steady_state)

  @doc """
  Mean first passage times of a transition probability matrix (`giddy.ergodic.mfpt`).

  `result[i][j]` is the expected number of steps from `i` to first reach `j` (the recurrence
  time when `i == j`), `:infinity` where passage is not certain. Same options as `steady_state/2`.
  """
  @spec mfpt([[number()]], keyword()) :: [[float() | :infinity]]
  def mfpt(p, opts \\ []), do: p |> ergodic!(opts) |> Map.fetch!(:mfpt)

  @doc """
  Sojourn time `1 / (1 - p_ii)` of every state (`giddy.markov.sojourn_time`); `:infinity` for
  absorbing states. Empty rows are treated as absorbing.
  """
  @spec sojourn_time([[number()]], keyword()) :: [float() | :infinity]
  def sojourn_time(p, opts \\ []) do
    opts = opts!(opts, fill_empty_classes: true)
    p |> ergodic!(opts) |> Map.fetch!(:sojourn_time)
  end

  defp ergodic!(p, opts) do
    opts = opts!(opts, fill_empty_classes: false)
    fill = boolean!(opts, :fill_empty_classes)
    {matrix, k} = square!(p)

    if not fill and Enum.any?(matrix, fn row -> Enum.sum(row) == 0.0 end) do
      empty = Enum.count(matrix, fn row -> Enum.sum(row) == 0.0 end)

      raise ArgumentError,
            "input transition probability matrix has #{empty} rows full of 0s; " <>
              "pass fill_empty_classes: true to make those states absorbing"
    end

    chain!(matrix, k)
  end

  @doc """
  Spatial Markov (`giddy.markov.Spatial_Markov`): the transition probabilities of `y` conditioned
  on the class of each location's spatial lag in the initial period, with tests of whether the
  conditional matrices differ from the overall one.

  `y` is an `n x t` matrix aligned with the graph's ids. The lag `W y` uses the graph's current
  weights (row-standardize it first, as in PySAL practice).

  ## Options

    * `:k` / `:m` - number of classes of `y` / of its lag (default `4` each).
    * `:fixed` - classify all periods together with one set of quantile cut points (default
      `true`) or each period on its own.
    * `:discrete` - `y` already holds class labels; the lag is the weighted mode of the
      neighbors' labels (ties go to the focal label, then the lowest class). `:k` and `:m`
      then equal the number of labels.
    * `:cutoffs` / `:lag_cutoffs` - ascending class upper bounds (without the final infinity).
    * `:permutations` - rows of `y` shuffled to test `x2` (default `0`), `:seed`.
    * `:fill_empty_classes` - make rows without transitions absorbing in `p` and `p_cond`
      (default `false`).

  Returns a `Spatex.Dynamics.SpatialMarkov`.
  """
  @spec spatial_markov(term(), Graph.t(), keyword()) :: SpatialMarkov.t()
  def spatial_markov(y, graph, opts \\ []) do
    opts =
      opts!(opts,
        k: 4,
        m: 4,
        fixed: true,
        discrete: false,
        cutoffs: nil,
        lag_cutoffs: nil,
        permutations: 0,
        seed: nil,
        fill_empty_classes: false
      )

    graph = ensure_graph!(graph)
    {rows, n, t} = panel!(y, "y")
    check_rows!(n, graph, "y")
    fill = boolean!(opts, :fill_empty_classes)
    fixed = boolean!(opts, :fixed)
    discrete = boolean!(opts, :discrete)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    cutoffs = cutoffs!(opts[:cutoffs], :cutoffs)
    lag_cutoffs = cutoffs!(opts[:lag_cutoffs], :lag_cutoffs)
    k = opt_k!(opts[:k], :k, 4)
    m = opt_k!(opts[:m], :m, 4)

    {ybin, classes, k, m} =
      if discrete do
        classes = rows |> List.flatten() |> labels!() |> Enum.uniq() |> Enum.sort()
        {ids, _} = encode_ids(rows, classes)
        {pack(List.flatten(ids), "y"), classes, length(classes), length(classes)}
      else
        {pack(List.flatten(rows), "y"), nil, k, m}
      end

    {meta, blocks} =
      graph.ref
      |> Native.dynamics_spatial_markov(
        ybin,
        {n, t},
        {k, m, fixed, discrete},
        {cutoffs && pack(cutoffs, "cutoffs"), lag_cutoffs && pack(lag_cutoffs, "lag_cutoffs")},
        {perms, seed || 0}
      )
      |> unwrap()

    parsed = parse_spatial(meta, Input.from_f64_binary(blocks), n, t)
    classes = classes || Enum.to_list(0..(parsed.k - 1)//1)

    build_spatial(parsed, classes, fill,
      discrete: discrete,
      permutations: perms,
      seed: seed
    )
  end

  defp parse_spatial(meta, vals, n, t) do
    [k, m, has_cut, n_cut, has_lag, n_lag, n_sims, has_p] = meta
    {class_ids, vals} = Enum.split(vals, n * t)
    {lclass_ids, vals} = Enum.split(vals, n * t)
    {cutoffs, vals} = Enum.split(vals, n_cut)
    {lag_cutoffs, vals} = Enum.split(vals, n_lag)
    {transitions, vals} = Enum.split(vals, k * k)
    {t_cond, [_x2 | vals]} = Enum.split(vals, m * k * k)
    {sims, vals} = Enum.split(vals, n_sims)

    %{
      k: k,
      m: m,
      class_ids: class_ids |> Enum.map(&trunc/1) |> Enum.chunk_every(t),
      lclass_ids: lclass_ids |> Enum.map(&trunc/1) |> Enum.chunk_every(t),
      cutoffs: if(has_cut == 1, do: cutoffs),
      lag_cutoffs: if(has_lag == 1, do: lag_cutoffs),
      transitions: Enum.chunk_every(transitions, k),
      t: t_cond |> Enum.chunk_every(k * k) |> Enum.map(&Enum.chunk_every(&1, k)),
      sims: if(n_sims > 0, do: sims),
      rpvalue: if(has_p == 1, do: hd(vals))
    }
  end

  defp build_spatial(r, classes, fill, extra) do
    %{k: k, m: m, t: t_cond, transitions: transitions} = r
    p = probabilities(transitions, fill)
    p_cond = Enum.map(t_cond, &probabilities(&1, fill))
    overall = chain_or_nil(p, k, fill)
    conditional = Enum.map(p_cond, &chain_or_nil(&1, k, fill))
    s = overall && overall.steady_state
    s_cond = Enum.map(conditional, &(&1 && &1.steady_state))

    [q, q_p, lr, lr_p, dof] =
      t_cond
      |> List.flatten()
      |> pack("t")
      |> Native.dynamics_homogeneity(m, k)
      |> unwrap()
      |> Input.from_f64_binary()

    transitions_bin = pack(List.flatten(transitions), "transitions")

    chi2 =
      for tm <- t_cond do
        {stat, p_value, dof} = chi2_test(tm, transitions_bin, k)
        {stat, p_value, dof}
      end

    x2 = chi2 |> Enum.map(&elem(&1, 0)) |> Enum.sum()
    x2_dof = k * (k - 1) * (k - 1)

    %SpatialMarkov{
      k: k,
      m: m,
      classes: classes,
      cutoffs: r.cutoffs,
      lag_cutoffs: r.lag_cutoffs,
      class_ids: r.class_ids,
      lclass_ids: r.lclass_ids,
      transitions: transitions,
      p: p,
      t: t_cond,
      p_cond: p_cond,
      s: s,
      s_cond: s_cond,
      f: overall && overall.mfpt,
      f_cond: Enum.map(conditional, &(&1 && &1.mfpt)),
      q: q,
      q_p_value: q_p,
      lr: lr,
      lr_p_value: lr_p,
      dof_hom: trunc(dof),
      shtest: shtest(s, s_cond, t_cond, k),
      chi2: chi2,
      x2: x2,
      x2_pvalue: chi2_sf(x2, x2_dof),
      x2_dof: x2_dof,
      x2_rpvalue: r.rpvalue,
      x2_realizations: r.sims,
      seed: extra[:seed],
      discrete: extra[:discrete],
      permutations: extra[:permutations],
      fill_empty_classes: fill
    }
  end

  # Difference between each conditional steady state and the overall one (multinomial chi2).
  # Steady states of reducible chains are matrices; giddy broadcasts them against each other and
  # sums everything, which this reproduces (rows must match, or one side must be a single row).
  defp shtest(s, s_cond, t_cond, k) do
    for {sl, tm} <- Enum.zip(s_cond, t_cond) do
      if s != nil and sl != nil, do: steady_difference(s, sl, tm, k)
    end
  end

  defp steady_difference(s, sl, tm, k) do
    case broadcast_rows(rows_of(s), rows_of(sl)) do
      {:ok, pairs} ->
        nt = tm |> List.flatten() |> Enum.sum()
        terms = for {r1, r2} <- pairs, {a, b} <- Enum.zip(r1, r2), do: {nt * b, nt * a}
        multinomial_chi2(terms, k)

      :error ->
        nil
    end
  end

  defp multinomial_chi2(terms, k) do
    if Enum.any?(terms, fn {_o, e} -> e == 0.0 end) do
      {:nan, :nan, k - 1}
    else
      stat = terms |> Enum.map(fn {o, e} -> (o - e) * (o - e) / e end) |> Enum.sum()
      {stat, chi2_sf(stat, k - 1), k - 1}
    end
  end

  defp rows_of([first | _] = s) when is_list(first), do: s
  defp rows_of(s), do: [s]

  defp broadcast_rows(a, b) when length(a) == length(b), do: {:ok, Enum.zip(a, b)}
  defp broadcast_rows([a], b), do: {:ok, Enum.map(b, &{a, &1})}
  defp broadcast_rows(a, [b]), do: {:ok, Enum.map(a, &{&1, b})}
  defp broadcast_rows(_, _), do: :error

  @doc """
  Kullback's information test of Markov homogeneity across strata (`giddy.markov.kullback`).

  `source` holds the transition *counts* (not probabilities) of `s` strata over the same `r`
  classes: a list of `s` square `r x r` matrices, a rank-3 `Nx.Tensor` of shape `{s, r, r}`, or a
  `Spatex.Dynamics.SpatialMarkov` (its conditional count matrices `t`, one per spatial-lag class).
  It tests whether the transition probabilities are the same in every stratum; a large
  `statistic` (small `p_value`) says they differ. Returns a `Spatex.Dynamics.Kullback`.

  ## Options

    * `:giddy_compatible` - reproduce giddy's arithmetic (default `false`). Kullback's statistic is
      `2 [sum f_sij ln f_sij - sum f_si. ln f_si. - sum f_.ij ln f_.ij + sum f_.i. ln f_.i.]`, with
      `f_si.` the row totals of each stratum. giddy sums that term over the wrong axis (the column
      totals `f_s.j`), which gives the same value only when every stratum has equal row and column
      totals (as in the table of its docstring) and otherwise a wrong one, even a negative
      statistic for identical strata. `true` reproduces giddy's value.

  ## Examples

      iex> k = Spatex.Dynamics.kullback([[[10, 5], [4, 12]], [[10, 5], [4, 12]]])
      iex> {k.dof, Float.round(abs(k.statistic), 9)}
      {2, 0.0}
  """
  @spec kullback(term(), keyword()) :: Kullback.t()
  def kullback(source, opts \\ []) do
    opts = opts!(opts, giddy_compatible: false)
    giddy? = boolean!(opts, :giddy_compatible)
    {matrices, s, r} = strata!(source)

    [statistic, p_value, dof] =
      matrices
      |> List.flatten()
      |> pack("counts")
      |> Native.dynamics_kullback({s, r}, giddy?)
      |> unwrap()
      |> Input.from_f64_binary()

    %Kullback{statistic: statistic, dof: round(dof), p_value: p_value}
  end

  defp strata!(%SpatialMarkov{t: t}), do: strata!(t)

  defp strata!(%{__struct__: Nx.Tensor} = tensor) do
    case call(Nx, :shape, [tensor]) do
      {_, _, _} -> tensor |> then(&call(Nx, :to_list, [&1])) |> strata!()
      shape -> raise ArgumentError, "expected a {s, r, r} tensor, got shape #{inspect(shape)}"
    end
  end

  defp strata!([first | _] = strata) do
    r = if is_list(first), do: length(first), else: 0

    unless r > 0 and Enum.all?(strata, &square_matrix?(&1, r)) do
      raise ArgumentError,
            "every stratum must be a square matrix of counts, all of the same size (got #{inspect(strata, limit: 3)})"
    end

    {strata, length(strata), r}
  end

  defp strata!(other) do
    raise ArgumentError,
          "kullback needs a non-empty list of strata (r x r count matrices), got #{inspect(other, limit: 3)}"
  end

  defp square_matrix?(m, r),
    do: is_list(m) and length(m) == r and Enum.all?(m, &(is_list(&1) and length(&1) == r))

  # ---------------------------------------------------------------------------------------------
  # LISA Markov
  # ---------------------------------------------------------------------------------------------

  @doc """
  Markov chain of local Moran quadrants (`giddy.markov.LISA_Markov`).

  `y` is an `n x t` matrix aligned with the graph's ids. Each period gets a
  `Spatex.ESDA.moran_local/3`; the quadrants (`1..4`) are the states.

  ## Options

    * `:permutations` - conditional randomizations per period (default `0`: no p-values), `:seed`
      (period `s` uses `seed + s`).
    * `:significance_level` - threshold for `significant_moves` (default `0.05`).
    * `:geoda_quads` - GeoDa's quadrant numbering (default `false`), `:transform` - the weights
      transform of the local statistics (default `:r`).

  The independence test (`chi_2`) lags `y` with the graph as given. Returns a
  `Spatex.Dynamics.LisaMarkov`.
  """
  @spec lisa_markov(term(), Graph.t(), keyword()) :: LisaMarkov.t()
  def lisa_markov(y, graph, opts \\ []) do
    opts =
      opts!(opts,
        permutations: 0,
        seed: nil,
        significance_level: 0.05,
        geoda_quads: false,
        transform: :r
      )

    graph = ensure_graph!(graph)
    {rows, n, t} = panel!(y, "y")
    check_rows!(n, graph, "y")
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    level = significance_level!(opts[:significance_level])
    cols = transpose(rows)

    locals =
      for {col, s} <- Enum.with_index(cols) do
        ESDA.moran_local(col, graph,
          permutations: perms,
          seed: seed && rem(seed + s, @max_seed + 1),
          geoda_quads: boolean!(opts, :geoda_quads),
          transform: opts[:transform],
          # p_z_sim needs the simulated moments, which local ESDA only returns with them.
          keep_simulations: true
        )
      end

    q = locals |> Enum.map(& &1.q) |> transpose()
    classes = [1, 2, 3, 4]
    ids = Enum.map(q, fn row -> Enum.map(row, &(&1 - 1)) end)
    markov = build_markov(ids, n, t, classes, false)

    move_types = Enum.map(q, &Enum.zip_with(&1, tl(&1), fn a, b -> 4 * (a - 1) + b end))

    {p_values, significant} =
      if perms > 0 do
        pv = locals |> Enum.map(& &1.p_z_sim) |> transpose()
        {pv, significant_moves(q, pv, level)}
      else
        {nil, nil}
      end

    {chi_2, expected} = independence_test(rows, cols, graph, markov.transitions)

    fields =
      markov
      |> Map.from_struct()
      |> Map.delete(:ranks)
      |> Map.merge(%{
        q: q,
        move_types: move_types,
        significant_moves: significant,
        p_values: p_values,
        chi_2: chi_2,
        expected_t: expected,
        seed: seed,
        significance_level: level,
        permutations: perms
      })

    struct!(LisaMarkov, fields)
  end

  defp significant_moves(q, pv, level) do
    Enum.zip_with(q, pv, fn qs, ps ->
      sig = Enum.map(ps, &(is_number(&1) and &1 <= level))
      moves = Enum.zip_with(qs, tl(qs), fn a, b -> 4 * (a - 1) + b end)

      Enum.zip_with([moves, sig, tl(sig)], fn [move, s0, s1] ->
        16 * significance_index(s0, s1) + move
      end)
    end)
  end

  defp significance_index(true, true), do: 0
  defp significance_index(true, false), do: 1
  defp significance_index(false, true), do: 2
  defp significance_index(false, false), do: 3

  # Null of independent own and lag moves: Kronecker product of the two binary "below own mean"
  # chains, reordered to quadrant order, scaled by the observed origin counts.
  defp independence_test(rows, cols, graph, transitions) do
    means = Enum.map(rows, fn row -> Enum.sum(row) / length(row) end)
    lag_cols = Enum.map(cols, &Graph.lag(graph, &1))
    lag_rows = transpose(lag_cols)
    p_own = binary_chain(rows, means)
    p_lag = binary_chain(lag_rows, means)

    kron =
      for a <- 0..1, b <- 0..1 do
        for c <- 0..1,
            d <- 0..1,
            do: Enum.at(Enum.at(p_own, a), c) * Enum.at(Enum.at(p_lag, b), d)
      end

    kp =
      for i <- @kron_to_quadrant do
        for j <- @kron_to_quadrant, do: kron |> Enum.at(i) |> Enum.at(j)
      end

    expected =
      for {row, origin} <- Enum.zip(kp, transitions) do
        total = Enum.sum(origin)
        Enum.map(row, &(total * &1))
      end

    {stat, p_value, dof} =
      chi2_test(List.flatten(transitions), pack(List.flatten(expected), "expected"), 4)

    {{stat, p_value, dof}, expected}
  end

  # 2 x 2 transition probabilities of "value below the location's mean" across periods.
  defp binary_chain(rows, means) do
    ids = Enum.zip_with(rows, means, fn row, mean -> Enum.map(row, &below_mean(&1, mean)) end)

    ids
    |> pack_ids()
    |> Native.dynamics_transitions(length(rows), length(hd(rows)), 2)
    |> unwrap()
    |> Input.from_f64_binary()
    |> Enum.chunk_every(2)
    |> probabilities(false)
  end

  defp below_mean(v, mean), do: if(mean > 0.0 and v / mean < 1.0, do: 1, else: 0)

  # ---------------------------------------------------------------------------------------------
  # Rank mobility
  # ---------------------------------------------------------------------------------------------

  @doc """
  Spill-over of significant LISA clusters between consecutive periods
  (`giddy.markov.LISA_Markov.spillover`).

  For every period, the significant sites of `:quadrant` (`q == quadrant` and
  `p_values <= significance_level`) are joined to their neighbors in `graph` (and, with
  `neighbors_on: true`, to their neighbors' neighbors), and the connected components of those
  links are taken. A site that belongs to a component in period `t + 1`, to none in period `t`,
  and whose `t + 1` component meets a period-`t` component spills over.

  Returns `%{components: n x t, spill_over: n x (t - 1)}` (lists of rows): `components[i][t]`
  is the 1-based component of site `i` in period `t` (0 outside every component; the last
  period stays 0, as in giddy) and `spill_over[i][t]` is `1` when site `i` spills over between
  `t` and `t + 1`. Components are numbered by their smallest site index; giddy follows Python's
  set order, which agrees for integer ids. `nil` when `lisa_markov` ran without permutations.
  Options: `:quadrant` (`1..4`, default `1`) and `:neighbors_on` (default `false`).
  """
  @spec lisa_spillover(LisaMarkov.t(), Graph.t(), keyword()) ::
          %{components: [[non_neg_integer()]], spill_over: [[0 | 1]]} | nil
  def lisa_spillover(%LisaMarkov{} = lm, graph, opts \\ []) do
    opts = opts!(opts, quadrant: 1, neighbors_on: false)
    graph = ensure_graph!(graph)
    quadrant = opts[:quadrant]
    on = boolean!(opts, :neighbors_on)

    unless quadrant in 1..4,
      do: raise(ArgumentError, ":quadrant must be 1, 2, 3 or 4, got #{inspect(quadrant)}")

    if lm.p_values == nil or lm.permutations == 0,
      do: nil,
      else: spillover(lm, graph, quadrant, on)
  end

  defp spillover(lm, graph, quadrant, on) do
    n = length(lm.q)

    if n != graph.n,
      do: raise(ArgumentError, "the LISA Markov has #{n} sites but the graph has #{graph.n}")

    neighbors = neighbor_map(graph)
    sig = significant_sites(lm, quadrant)
    t = length(sig)
    comps = Enum.map(sig, &spill_components(&1, neighbors, on))

    components =
      comps
      |> Enum.take(t - 1)
      |> Enum.with_index()
      |> Enum.reduce(zeros(n, t), fn {cs, p}, acc -> label_components(acc, cs, p) end)

    spill =
      comps
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.reduce(zeros(n, t - 1), fn {[c1, c2], p}, acc ->
        c1 |> spilled(c2) |> Enum.reduce(acc, &put_cell(&2, &1, p, 1))
      end)

    %{components: components, spill_over: spill}
  end

  # Sites of the period-2 components `c2` that are in no period-1 component of `c1` while their
  # component meets one.
  defp spilled(c1, c2) do
    old = c1 |> List.flatten() |> MapSet.new()

    c2
    |> Enum.filter(fn comp -> Enum.any?(comp, &MapSet.member?(old, &1)) end)
    |> List.flatten()
    |> Enum.reject(&MapSet.member?(old, &1))
  end

  defp neighbor_map(graph) do
    index = graph.ids |> Enum.with_index() |> Map.new()
    empty = Map.new(0..(graph.n - 1)//1, &{&1, []})

    graph
    |> Graph.to_adjacency()
    |> Enum.reject(fn {a, b, w} -> a == b and w == 0.0 end)
    |> Enum.reduce(empty, fn {a, b, _w}, acc ->
      Map.update!(acc, Map.fetch!(index, a), &[Map.fetch!(index, b) | &1])
    end)
    |> Map.new(fn {i, nb} -> {i, Enum.reverse(nb)} end)
  end

  # Per period, the indices of the significant sites of `quadrant`.
  defp significant_sites(lm, quadrant) do
    lm.q
    |> Enum.zip(lm.p_values)
    |> Enum.map(fn {qs, ps} -> Enum.zip(qs, ps) end)
    |> Enum.zip_with(& &1)
    |> Enum.map(fn period ->
      for {{q, p}, i} <- Enum.with_index(period),
          q == quadrant and is_number(p) and p <= lm.significance_level,
          do: i
    end)
  end

  # Connected components (sorted lists, ordered by their smallest member) of the links from the
  # significant sites to their neighbors (and the neighbors' neighbors with `on`).
  defp spill_components(sites, neighbors, on) do
    edges =
      for i <- sites, j <- neighbors[i], e <- [{i, j} | second(j, neighbors, on)], do: e

    adj =
      Enum.reduce(edges, %{}, fn {a, b}, acc ->
        acc |> Map.update(a, [b], &[b | &1]) |> Map.update(b, [a], &[a | &1])
      end)

    adj
    |> Map.keys()
    |> Enum.sort()
    |> Enum.reduce({[], MapSet.new()}, fn node, {comps, seen} ->
      if MapSet.member?(seen, node) do
        {comps, seen}
      else
        comp = reach([node], adj, MapSet.new([node]))
        {[Enum.sort(comp) | comps], MapSet.union(seen, comp)}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp second(_j, _neighbors, false), do: []
  defp second(j, neighbors, true), do: Enum.map(neighbors[j], &{j, &1})

  defp reach([], _adj, seen), do: seen

  defp reach([v | rest], adj, seen) do
    new = Enum.reject(Map.get(adj, v, []), &MapSet.member?(seen, &1))
    reach(new ++ rest, adj, Enum.reduce(new, seen, &MapSet.put(&2, &1)))
  end

  defp label_components(matrix, comps, p) do
    comps
    |> Enum.with_index(1)
    |> Enum.reduce(matrix, fn {comp, c}, acc ->
      Enum.reduce(comp, acc, &put_cell(&2, &1, p, c))
    end)
  end

  defp zeros(n, t), do: List.duplicate(List.duplicate(0, t), n)

  defp put_cell(matrix, i, p, value),
    do: List.update_at(matrix, i, &List.replace_at(&1, p, value))

  @doc """
  Kendall's tau-b of two variables with its asymptotic p-value (`giddy.rank.Tau`).

  ## Examples

      iex> Spatex.Dynamics.kendall_tau([1, 2, 3, 4], [2, 4, 6, 8]).tau
      1.0
  """
  @spec kendall_tau(term(), term()) :: Tau.t()
  def kendall_tau(x, y) do
    {xb, n} = Input.to_f64_binary(x, name: "x")
    {yb, _} = Input.to_f64_binary(y, name: "y", length: n)
    [tau, p, c, d, ex, ey] = xb |> Native.dynamics_tau(yb) |> unwrap() |> Input.from_f64_binary()
    %Tau{tau: tau, p: p, concordant: c, discordant: d, extra_x: ex, extra_y: ey}
  end

  @doc """
  Spatial tau (`giddy.rank.SpatialTau`): Kendall's tau restricted to pairs of neighbors, with a
  permutation test of its value under random relabelling of locations.

  `x` and `y` are aligned with the graph's ids; edge weights are ignored (binary). Options:
  `:permutations` (default `0`) and `:seed`.
  """
  @spec spatial_tau(term(), term(), Graph.t(), keyword()) :: SpatialTau.t()
  def spatial_tau(x, y, graph, opts \\ []) do
    opts = opts!(opts, permutations: 0, seed: nil)
    graph = ensure_graph!(graph)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    {xb, _} = Input.to_f64_binary(x, name: "x", length: graph.n)
    {yb, _} = Input.to_f64_binary(y, name: "y", length: graph.n)

    {scalars, taus} =
      graph.ref |> Native.dynamics_spatial_tau(xb, yb, perms, seed || 0) |> unwrap()

    [tau, tau_p, c, d, ex, ey, tau_s, pairs, cs, ds, psim] = Input.from_f64_binary(scalars)

    %SpatialTau{
      tau: tau,
      tau_p: tau_p,
      concordant: c,
      discordant: d,
      extra_x: ex,
      extra_y: ey,
      tau_spatial: tau_s,
      pairs_spatial: trunc(pairs),
      concordant_spatial: cs,
      discordant_spatial: ds,
      taus: if(perms > 0, do: Input.from_f64_binary(taus)),
      tau_spatial_psim: if(perms > 0, do: psim),
      seed: seed,
      permutations: perms
    }
  end

  @doc """
  Local Kendall tau (`giddy.rank.Tau_Local`) of `x` and `y`.

  `tau_local_i` is the mean concordance `sign((x_i - x_j)(y_i - y_j))` of site `i` with every
  other site and `tau` the mean over all ordered pairs. O(n^2) time, O(n) memory.

  ## Examples

      iex> t = Spatex.Dynamics.tau_local([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
      iex> {t.tau, t.tau_local}
      {0.6, [0.5, 0.5, 0.5, 0.5, 1.0]}
  """
  @spec tau_local(term(), term()) :: TauLocal.t()
  def tau_local(x, y) do
    {xb, n} = Input.to_f64_binary(x, name: "x")
    {yb, _} = Input.to_f64_binary(y, name: "y", length: n)
    {tau, local} = xb |> Native.dynamics_tau_local(yb) |> unwrap()
    %TauLocal{tau: tau, tau_local: Input.from_f64_binary(local), n: n}
  end

  @doc """
  Neighbor-set local Kendall tau (`giddy.rank.Tau_Local_Neighbor`) over `graph`.

  `tau_ln_i` is the mean concordance of site `i` with its neighbors (binary weights, self pairs
  ignored; `:nan` for islands). Options: `:permutations` (default `0`, as in giddy), `:seed`
  and `:keep_simulations` (default `true`; `false` or `:tensor`, an `Nx.Tensor` of shape
  `{n, permutations}`, are also accepted). A permutation gives every site distinct random other
  sites as neighbors (conditional randomization); `p_sim` is giddy's
  `(min(#sim >= obs, #sim <= obs) + 1) / (P + 1)`. Returns a `Spatex.Dynamics.TauLocalNeighbor`.
  """
  @spec tau_local_neighbor(term(), term(), Graph.t(), keyword()) :: TauLocalNeighbor.t()
  def tau_local_neighbor(x, y, graph, opts \\ []) do
    {graph, xb, yb, perms, seed, keep} = local_tau_args!(x, y, graph, opts)

    {blocks, sim} =
      graph.ref
      |> Native.dynamics_tau_local_neighbor(xb, yb, perms, seed || 0, Input.keep_flag(keep))
      |> unwrap()

    [weights, tau_ln | rest] = blocks |> Input.from_f64_binary() |> Enum.chunk_every(graph.n)

    %TauLocalNeighbor{
      tau_ln: tau_ln,
      weights: weights,
      sign: signs(tau_ln),
      p_sim: List.first(rest),
      sim: local_tau_sims(sim, keep, graph.n, perms),
      n: graph.n,
      permutations: perms,
      seed: seed,
      ids: graph.ids
    }
  end

  @doc """
  Neighborhood local Kendall tau (`giddy.rank.Tau_Local_Neighborhood`) over `graph`.

  `tau_lnhood_i` is the tau of the neighborhood `{i} + N(i)`: the sum of the concordances of all
  its ordered pairs over `m (m - 1)` (`:nan` for islands). Options and inference as in
  `tau_local_neighbor/4`. Returns a `Spatex.Dynamics.TauLocalNeighborhood`.
  """
  @spec tau_local_neighborhood(term(), term(), Graph.t(), keyword()) ::
          TauLocalNeighborhood.t()
  def tau_local_neighborhood(x, y, graph, opts \\ []) do
    {graph, xb, yb, perms, seed, keep} = local_tau_args!(x, y, graph, opts)

    {blocks, sim} =
      graph.ref
      |> Native.dynamics_tau_local_neighborhood(xb, yb, perms, seed || 0, Input.keep_flag(keep))
      |> unwrap()

    [tau_lnhood | rest] = blocks |> Input.from_f64_binary() |> Enum.chunk_every(graph.n)

    %TauLocalNeighborhood{
      tau_lnhood: tau_lnhood,
      sign: signs(tau_lnhood),
      p_sim: List.first(rest),
      sim: local_tau_sims(sim, keep, graph.n, perms),
      n: graph.n,
      permutations: perms,
      seed: seed,
      ids: graph.ids
    }
  end

  @doc """
  Regional Kendall tau (`giddy.rank.Tau_Regional`): the mean concordance between every pair of
  regimes (and within each, on the diagonal).

  `regime` holds one label per observation (a list, `Nx` vector or `Explorer.Series`; see `Spatex.Input.to_labels/2`), sorted for the row order. Options:
  `:permutations` (default `0`), `:seed` and `:keep_simulations` (default `true`; `false` or
  `:tensor`, a `{permutations, k, k}` `Nx.Tensor`, are also accepted); permutations
  shuffle all observations over the fixed regimes. Returns a `Spatex.Dynamics.TauRegional`.

  ## Examples

      iex> Spatex.Dynamics.tau_regional([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], [1, 1, 2, 2, 2]).tau_reg
      [[-1.0, 1.0], [1.0, 0.3333333333333333]]
  """
  @spec tau_regional(term(), term(), [term()], keyword()) :: TauRegional.t()
  def tau_regional(x, y, regime, opts \\ []) do
    opts = opts!(opts, permutations: 0, seed: nil, keep_simulations: true)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    keep = Input.keep_simulations!(opts[:keep_simulations])
    {xb, n} = Input.to_f64_binary(x, name: "x")
    {yb, _} = Input.to_f64_binary(y, name: "y", length: n)

    regime = Input.to_labels(regime, name: "regime", length: n)

    labels = regime |> Enum.uniq() |> Enum.sort()
    index = labels |> Enum.with_index() |> Map.new()
    codes = Enum.map(regime, &Map.fetch!(index, &1))
    k = length(labels)

    {_codes, tau_reg, p, sim} =
      xb
      |> Native.dynamics_tau_regional(
        yb,
        codes,
        perms,
        seed || 0,
        Input.keep_flag(keep)
      )
      |> unwrap()

    matrix = fn bin -> bin |> Input.from_f64_binary() |> Enum.chunk_every(k) end

    %TauRegional{
      regimes: labels,
      tau_reg: matrix.(tau_reg),
      p_sim: if(perms > 0, do: matrix.(p)),
      sim: if(sim != <<>>, do: Input.simulations(sim, keep, {perms, k, k})),
      permutations: perms,
      seed: seed
    }
  end

  defp local_tau_args!(x, y, graph, opts) do
    opts = opts!(opts, permutations: 0, seed: nil, keep_simulations: true)
    graph = ensure_graph!(graph)
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)
    {xb, _} = Input.to_f64_binary(x, name: "x", length: graph.n)
    {yb, _} = Input.to_f64_binary(y, name: "y", length: graph.n)
    {graph, xb, yb, perms, seed, Input.keep_simulations!(opts[:keep_simulations])}
  end

  defp signs(values), do: Enum.map(values, &if(is_number(&1) and &1 < 0, do: -1, else: 1))

  defp local_tau_sims(<<>>, _keep, _n, _perms), do: nil
  defp local_tau_sims(sim, keep, n, perms), do: Input.simulations(sim, keep, {n, perms})

  @doc """
  Regime mobility measure theta (`giddy.rank.Theta`).

  `y` is an `n x t` matrix (`t >= 2`), `regime` a list of `n` labels. Options: `:permutations`
  (default `999`, as in giddy; `0` skips the test) and `:seed`.

  ## Examples

      iex> Spatex.Dynamics.theta([[1, 4], [2, 3], [3, 2], [4, 1]], [0, 0, 1, 1], permutations: 0).theta
      [1.0]
  """
  @spec theta(term(), [term()], keyword()) :: Theta.t()
  def theta(y, regime, opts \\ []) do
    opts = opts!(opts, permutations: 999, seed: nil)
    {rows, n, t} = panel!(y, "y")
    perms = permutations!(opts[:permutations])
    seed = resolve_seed(opts[:seed], perms)

    unless is_list(regime) and length(regime) == n do
      raise ArgumentError, "regime must be a list with one label per row of y (#{n})"
    end

    regimes = regime |> Enum.uniq() |> Enum.sort()
    index = regimes |> Enum.with_index() |> Map.new()
    regime_bin = regime |> Enum.map(&Map.fetch!(index, &1)) |> then(&pack_ids([&1]))

    {blocks, ranks} =
      Native.dynamics_theta(pack(List.flatten(rows), "y"), n, t, regime_bin, perms, seed || 0)
      |> unwrap()

    q = t - 1
    vals = Input.from_f64_binary(blocks)
    {theta, vals} = Enum.split(vals, q)
    {total, [max_total | vals]} = Enum.split(vals, q)
    {left, vals} = Enum.split(vals, if(perms > 0, do: q, else: 0))
    {right, z} = Enum.split(vals, if(perms > 0, do: q, else: 0))

    %Theta{
      theta: theta,
      total: total,
      max_total: max_total,
      ranks: ranks |> Input.from_f64_binary() |> Enum.chunk_every(t),
      regimes: regimes,
      pvalue_left: if(perms > 0, do: left),
      pvalue_right: if(perms > 0, do: right),
      z: if(perms > 0, do: z),
      seed: seed,
      permutations: perms
    }
  end

  # ---------------------------------------------------------------------------------------------
  # Shared helpers
  # ---------------------------------------------------------------------------------------------

  # Builds a Markov struct from an n x t matrix of class indices (list of lists or u32 binary).
  defp build_markov(ids, n, t, classes, fill) do
    k = length(classes)
    ids_bin = if is_binary(ids), do: ids, else: pack_ids(ids)

    transitions =
      ids_bin
      |> Native.dynamics_transitions(n, t, k)
      |> unwrap()
      |> Input.from_f64_binary()
      |> Enum.chunk_every(k)

    p = probabilities(transitions, fill)
    c = chain!(p, k)

    %Markov{
      classes: classes,
      k: k,
      transitions: transitions,
      p: p,
      steady_state: c.steady_state,
      mfpt: c.mfpt,
      sojourn_time: c.sojourn_time,
      communication_classes: c.communication,
      recurrent_classes: c.recurrent,
      transient_classes: c.transient,
      absorbing_states: c.absorbing,
      fill_empty_classes: fill
    }
  end

  defp chain_or_nil(p, k, fill) do
    if not fill and Enum.any?(p, fn row -> Enum.sum(row) == 0.0 end), do: nil, else: chain!(p, k)
  end

  defp chain!(p, k) do
    {steady, mfpt, sojourn, {comm, rec, trans, absorbing}} =
      p |> List.flatten() |> pack("p") |> Native.dynamics_chain(k) |> unwrap()

    rows = steady |> Input.from_f64_binary() |> Enum.chunk_every(k)

    %{
      steady_state: if(length(comm) == 1, do: hd(rows), else: rows),
      mfpt: mfpt |> Input.from_f64_binary() |> Enum.chunk_every(k),
      sojourn_time: Input.from_f64_binary(sojourn),
      communication: comm,
      recurrent: rec,
      transient: trans,
      absorbing: absorbing
    }
  end

  defp chi2_test(counts, expected_bin, k) do
    counts_bin = if is_binary(counts), do: counts, else: pack(List.flatten(counts), "counts")

    [stat, p, dof] =
      counts_bin
      |> Native.dynamics_chi2_test(expected_bin, k)
      |> unwrap()
      |> Input.from_f64_binary()

    {stat, p, trunc(dof)}
  end

  defp chi2_sf(x, dof) do
    [p] = x |> Native.dynamics_chi2_sf(dof * 1.0) |> Input.from_f64_binary()
    p
  end

  # Row-normalizes counts; an all-zero row stays zero, or gets 1 on the diagonal with `fill`.
  defp probabilities(counts, fill) do
    counts
    |> Enum.with_index()
    |> Enum.map(fn {row, i} -> normalize_row(row, i, fill) end)
  end

  defp normalize_row(row, i, fill) do
    total = Enum.sum(row)

    cond do
      total > 0.0 ->
        Enum.map(row, &(&1 * (1.0 / total)))

      fill ->
        unit_row(length(row), i)

      true ->
        row
    end
  end

  defp unit_row(size, i), do: for(j <- 0..(size - 1), do: if(i == j, do: 1.0, else: 0.0))

  defp transpose(rows), do: Enum.zip_with(rows, & &1)

  defp classify_periods(rows, k) do
    rows
    |> transpose()
    |> Enum.map(fn col -> Classify.quantiles(col, k: k).yb end)
    |> transpose()
  end

  defp classes!(rows, nil) do
    rows |> List.flatten() |> labels!() |> Enum.uniq() |> Enum.sort()
  end

  defp classes!(_rows, classes) when is_list(classes) and classes != [] do
    if length(Enum.uniq(classes)) != length(classes),
      do: raise(ArgumentError, ":classes must not repeat a label")

    classes
  end

  defp classes!(_rows, other) do
    raise ArgumentError, ":classes must be a non-empty list of labels, got #{inspect(other)}"
  end

  defp labels!(flat) do
    unless Enum.all?(flat, &is_integer/1) do
      raise ArgumentError,
            "data must hold integer class ids (pass k: to classify continuous values)"
    end

    flat
  end

  defp encode_ids(rows, classes) do
    index = classes |> Enum.with_index() |> Map.new()

    {Enum.map(rows, fn row -> Enum.map(row, &class_index!(index, &1, classes)) end),
     length(classes)}
  end

  defp class_index!(index, label, classes) do
    case Map.fetch(index, label) do
      {:ok, i} ->
        i

      :error ->
        raise ArgumentError, "class #{inspect(label)} is not in classes #{inspect(classes)}"
    end
  end

  defp pack_ids(rows) do
    for row <- rows, v <- row, into: <<>>, do: <<v::unsigned-integer-32-native>>
  end

  defp pack(values, name) do
    {bin, _} = Input.to_f64_binary(values, name: name)
    bin
  end

  defp panel!(data, name) do
    rows = to_rows(data)

    unless is_list(rows) and rows != [] and Enum.all?(rows, &is_list/1) do
      raise ArgumentError,
            "#{name} must be an n x t matrix (a list of lists or a rank-2 Nx.Tensor)"
    end

    t = length(hd(rows))

    unless Enum.all?(rows, &(length(&1) == t)),
      do: raise(ArgumentError, "every row of #{name} must have the same length")

    if t < 2, do: raise(ArgumentError, "#{name} needs at least two time periods (columns)")
    {rows, length(rows), t}
  end

  defp to_rows(%{__struct__: Nx.Tensor} = tensor) do
    case call(Nx, :shape, [tensor]) do
      {_, _} -> call(Nx, :to_list, [tensor])
      shape -> raise ArgumentError, "expected a rank-2 tensor, got shape #{inspect(shape)}"
    end
  end

  defp to_rows(other), do: other

  # Optional dependencies are called dynamically so Spatex compiles without them.
  defp call(module, fun, args), do: apply(module, fun, args)

  defp square!(p) do
    rows = to_rows(p)

    unless is_list(rows) and rows != [] and
             Enum.all?(rows, &(is_list(&1) and length(&1) == length(rows))) do
      raise ArgumentError, "the transition matrix must be square (a list of k rows of k numbers)"
    end

    {Enum.map(rows, fn row -> Enum.map(row, &(&1 * 1.0)) end), length(rows)}
  end

  defp check_rows!(n, graph, name) do
    if n != graph.n,
      do: raise(ArgumentError, "#{name} has #{n} rows but the graph has #{graph.n} nodes")
  end

  defp ensure_graph!(%Graph{} = graph), do: graph

  defp ensure_graph!(other),
    do: raise(ArgumentError, "expected a Spatex.Graph, got #{inspect(other, limit: 3)}")

  defp opts!(opts, defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    Keyword.validate!(opts, defaults)
  end

  defp boolean!(opts, key) do
    value = opts[key]
    unless is_boolean(value), do: raise(ArgumentError, ":#{key} must be a boolean")
    value
  end

  defp opt_k!(nil, _key, nil), do: nil
  defp opt_k!(k, _key, _default) when is_integer(k) and k >= 1, do: k

  defp opt_k!(other, key, _default),
    do: raise(ArgumentError, ":#{key} must be a positive integer, got #{inspect(other)}")

  defp cutoffs!(nil, _key), do: nil

  defp cutoffs!(list, key) when is_list(list) and list != [] do
    ok = Enum.all?(list, &is_number/1) and list == Enum.sort(Enum.uniq(list))

    unless ok,
      do: raise(ArgumentError, ":#{key} must be a strictly increasing list of numbers")

    list
  end

  defp cutoffs!(other, key),
    do:
      raise(
        ArgumentError,
        ":#{key} must be a strictly increasing list of numbers, got #{inspect(other)}"
      )

  defp significance_level!(level) when is_number(level) and level > 0 and level <= 1,
    do: level * 1.0

  defp significance_level!(other),
    do: raise(ArgumentError, ":significance_level must be in (0, 1], got #{inspect(other)}")

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
