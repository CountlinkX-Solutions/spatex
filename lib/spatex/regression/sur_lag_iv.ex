defmodule Spatex.Regression.SURlagIV do
  @moduledoc """
  Result of `Spatex.Regression.sur_lag_iv/4` (`spreg.SURlagIV`): seemingly unrelated regressions
  with a spatial lag of the dependent variable in every equation, estimated by three-stage least
  squares with the spatially lagged exogenous variables as instruments.

  Per-equation fields are lists with one entry per equation, in order.

    * `n`, `n_eq`, `k` - observations, equations and, per equation, the number of coefficients
      (constant, exogenous, user endogenous and, last, the spatial lag).
    * `w_lags` - the order of the spatial lags used as instruments.
    * `b_2sls`, `residuals_2sls` - the equation-by-equation 2SLS coefficients and residuals that
      start the estimation.
    * `b_3sls` - the 3SLS coefficients (constant, exogenous, user endogenous, then the spatial
      lag coefficient); `rho` - the spatial lag coefficient of every equation (the last of each);
      `varb` - covariance of all coefficients stacked; `tsls_inf` - per equation, one
      `{std_err, z, p}` per coefficient.
    * `sig`, `corr` - the cross-equation error covariance (`E'E / n` of the 2SLS residuals) and its
      correlation matrix.
    * `chow` - with the same number of coefficients in every equation, one `{stat, df, p}` Wald
      test per coefficient of its equality across equations; `nil` otherwise.
    * `joinrho` - with `spat_diag: true`, the `{stat, df, p}` Wald test that the spatial lag
      coefficients of all equations are jointly zero; `nil` otherwise.
    * `residuals`, `predicted` - one list of `n` values per equation (from the 3SLS coefficients
      and the observed `W y`).
    * `name_y`, `name_x`, `name_yend`, `name_q` - names, per equation (`name_x` starts with
      `"Constant_<eq>"`, `name_yend` ends with `"W_<name_y>"`, `name_q` lists the user
      instruments, then the lags of `X` and, with `lag_q`, of the user instruments).
  """

  @enforce_keys [:n, :n_eq]
  defstruct [
    :n,
    :n_eq,
    :k,
    :w_lags,
    :b_2sls,
    :residuals_2sls,
    :b_3sls,
    :rho,
    :varb,
    :tsls_inf,
    :sig,
    :corr,
    :chow,
    :joinrho,
    :residuals,
    :predicted,
    :name_y,
    :name_x,
    :name_yend,
    :name_q,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.SurLagIv do
  @moduledoc false
  # `Spatex.Regression.sur_lag_iv/4`: spreg's SURlagIV. Each equation gets `W y` as its last
  # endogenous variable (after the user's) and `[Q, W X, ..., W^p X, W Q, ..., W^p Q]` as
  # instruments; the estimation is `ThreeSls` (spreg's BaseThreeSLS).

  alias Spatex.{Graph, Regression}
  alias Spatex.Regression.{Sur, SURlagIV, ThreeSls}

  @allowed [
    :yends,
    :qs,
    :w_lags,
    :lag_q,
    :spat_diag,
    :name_bigy,
    :name_bigx,
    :name_bigyend,
    :name_bigq
  ]

  @spec fit([term()], [term()], term(), keyword()) :: SURlagIV.t()
  def fit(ys, xs, graph, opts) do
    opts = validate!(opts)

    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "the graph must be a Spatex.Graph")

    {eqs, n, names_y} = Sur.equations("sur_lag_iv", ys, xs, opts[:name_bigy], opts[:name_bigx])

    unless graph.n == n,
      do: raise(ArgumentError, "the graph has #{graph.n} nodes but the equations have #{n} rows")

    names_y = Enum.map(names_y, &to_string/1)
    {yends, qs} = user_blocks!(opts, length(eqs))

    built =
      for {e, r} <- Enum.with_index(eqs) do
        equation(
          e,
          r,
          Enum.at(names_y, r),
          graph,
          yends && Enum.at(yends, r),
          qs && Enum.at(qs, r),
          opts
        )
      end

    res =
      ThreeSls.fit(
        ys,
        Enum.map(built, & &1.x),
        Enum.map(built, & &1.yend),
        Enum.map(built, & &1.q),
        name_bigy: names_y,
        name_bigx: Enum.map(built, & &1.x_names),
        name_bigyend: Enum.map(built, & &1.yend_names),
        name_bigq: Enum.map(built, & &1.q_names)
      )

    wrap(res, opts)
  end

  # `:yends` and `:qs` come together (spreg needs instruments when there are endogenous variables).
  defp user_blocks!(opts, n_eq) do
    yends = per_equation!(opts[:yends], n_eq, :yends)
    qs = per_equation!(opts[:qs], n_eq, :qs)

    if yends == nil != (qs == nil),
      do: raise(ArgumentError, "instruments (:qs) are needed exactly when :yends are given")

    {yends, qs}
  end

  defp equation(e, r, name_y, graph, yend, q, opts) do
    w_lags = opts[:w_lags]
    x_cols = tl(e.x)
    x_names = tl(e.names)
    {yend_cols, yend_names} = user(e.y, yend, opts[:name_bigyend], r, "end")
    {q_cols, q_names} = user(e.y, q, opts[:name_bigq], r, "inst")

    {wx, wx_names} = lags(graph, x_cols, x_names, w_lags)

    {wq, wq_names} =
      if opts[:lag_q] and q_cols != [], do: lags(graph, q_cols, q_names, w_lags), else: {[], []}

    %{
      x: x_cols,
      x_names: x_names,
      yend: yend_cols ++ [Graph.lag(graph, e.y)],
      yend_names: yend_names ++ ["W_" <> name_y],
      q: q_cols ++ wx ++ wq,
      q_names: q_names ++ wx_names ++ wq_names
    }
  end

  # The columns and names of a user-supplied block; unnamed columns follow three_sls (`var_<i>`).
  defp user(_y, nil, _names, _r, _kind), do: {[], []}

  defp user(y, block, names, r, _kind) do
    {_, _, cols_names, bin, _, _} =
      Regression.prepare_design(y, block, constant: false, name_x: nth(names, r))

    n = length(y)
    {bin |> Spatex.Input.from_f64_binary() |> Enum.chunk_every(n), cols_names}
  end

  # W v, W^2 v, ... W^p v for every column (all columns of one lag, then the next), named
  # `W_<name>`, `WW_<name>`, ...
  defp lags(graph, cols, names, w_lags) do
    {blocks, _} =
      Enum.map_reduce(1..w_lags, cols, fn j, prev ->
        lagged = Enum.map(prev, &Graph.lag(graph, &1))
        {{lagged, Enum.map(names, &(String.duplicate("W", j) <> "_" <> &1))}, lagged}
      end)

    {Enum.flat_map(blocks, &elem(&1, 0)), Enum.flat_map(blocks, &elem(&1, 1))}
  end

  defp nth(nil, _r), do: nil
  defp nth(list, r), do: Enum.at(list, r)

  defp per_equation!(nil, _n_eq, _key), do: nil

  defp per_equation!(list, n_eq, key) do
    unless is_list(list) and length(list) == n_eq,
      do: raise(ArgumentError, "#{inspect(key)} needs one entry per equation (#{n_eq})")

    list
  end

  defp wrap(r, opts) do
    ks = r.k
    rho = for b <- r.b_3sls, do: List.last(b)

    %SURlagIV{
      n: r.n,
      n_eq: r.n_eq,
      k: ks,
      w_lags: opts[:w_lags],
      b_2sls: r.b_2sls,
      residuals_2sls: r.residuals_2sls,
      b_3sls: r.b_3sls,
      rho: rho,
      varb: r.varb,
      tsls_inf: r.tsls_inf,
      sig: r.sig,
      corr: r.corr,
      chow: r.chow,
      joinrho: if(opts[:spat_diag], do: joinrho(ks, r.b_3sls, r.varb)),
      residuals: r.residuals,
      predicted: r.predicted,
      name_y: r.name_y,
      name_x: r.name_x,
      name_yend: r.name_yend,
      name_q: r.name_q,
      warnings: r.warnings
    }
  end

  # spreg's sur_joinrho: Wald test that the last coefficient of every equation is zero.
  defp joinrho(ks, b_3sls, varb) do
    total = Enum.sum(ks)
    last = ks |> Enum.scan(&+/2) |> Enum.map(&(&1 - 1))
    rows = for c <- last, do: selector(total, c)
    {stat, df} = Sur.wald(List.flatten(b_3sls), varb, rows)
    {stat, df, Sur.chi2_sf(stat, df)}
  end

  defp selector(total, c), do: for(i <- 0..(total - 1), do: if(i == c, do: 1.0, else: 0.0))

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError,
              "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)} " <>
                "(regimes are not supported by sur_lag_iv)"
    end

    opts = Keyword.merge([w_lags: 1, lag_q: true, spat_diag: false], opts)

    unless is_integer(opts[:w_lags]) and opts[:w_lags] >= 1,
      do: raise(ArgumentError, "w_lags must be an integer >= 1")

    opts
  end
end
