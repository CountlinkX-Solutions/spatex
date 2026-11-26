defmodule Spatex.Regression.ML do
  @moduledoc false
  # Option handling and result assembly for `Spatex.Regression.ml_lag/4` and `ml_error/4`.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{MLError, MLLag}

  @max_dense_n 5_000
  @methods %{full: 0, ord: 1, lu: 2}
  @impacts %{nil => 0, simple: 1, full: 2, power: 3, all: 4}
  @common_opts [
    :constant,
    :name_x,
    :name_y,
    :method,
    :epsilon,
    :max_dense_n,
    :slx_lags,
    :slx_vars
  ]

  @lag_keys ~w(rho vm vm1 std_err z_stat logll aic schwarz sig2 utu pr2 pr2_e mean_y std_y
               iterations converged u predy predy_e e_pred mult_simple mult_full mult_power
               power_terms betas)a
  @error_keys ~w(lam betas vm vm1 std_err z_stat logll aic schwarz sig2 pr2 mean_y std_y
                 iterations converged u predy e_filtered)a

  @spec lag(term(), term(), term(), keyword(), float() | nil) :: MLLag.t()
  def lag(y, x, graph, opts, at \\ nil) do
    opts = validate!(opts, @common_opts ++ [:spat_impacts], spat_impacts: :simple)
    impacts = opts[:spat_impacts]

    unless Map.has_key?(@impacts, impacts) do
      raise ArgumentError,
            "spat_impacts must be :simple, :full, :power, :all or nil, got #{inspect(impacts)}"
    end

    {y_bin, x_bin, k, n, names, warnings} = prepare(y, x, graph, opts)

    e =
      Native.regression_ml_lag(y_bin, x_bin, k, graph.ref, settings(opts, impacts, at))
      |> entries(@lag_keys)

    name_y = to_string(opts[:name_y])
    names = names ++ ["W_" <> name_y]
    kc = k + 1

    {slopes, slope_betas, thetas} = durbin_terms(names, Enum.take(e.betas, k), opts)
    {multipliers, effects} = impacts_of(e, slopes, slope_betas, thetas)

    %MLLag{
      n: n,
      k: kc,
      betas: e.betas,
      rho: hd(e.rho),
      vm: rows(e.vm, kc),
      vm1: rows(e.vm1, kc + 1),
      std_err: e.std_err,
      z_stat: pairs(e.z_stat),
      logll: hd(e.logll),
      aic: hd(e.aic),
      schwarz: hd(e.schwarz),
      sig2: hd(e.sig2),
      utu: hd(e.utu),
      u: e.u,
      predy: e.predy,
      predy_e: e.predy_e,
      e_pred: e.e_pred,
      pr2: hd(e.pr2),
      pr2_e: e[:pr2_e] && hd(e.pr2_e),
      mean_y: hd(e.mean_y),
      std_y: hd(e.std_y),
      sp_multipliers: multipliers,
      impacts: effects,
      power_terms: e[:power_terms] && round(hd(e.power_terms)),
      method: opts[:method],
      epsilon: opts[:epsilon] * 1.0,
      iterations: round(hd(e.iterations)),
      converged: hd(e.converged) == 1.0,
      name_x: names,
      name_y: name_y,
      warnings: warnings ++ convergence_warning(e)
    }
  end

  # `xlag` (internal, `ml_error_regimes/5`): the spatially lagged design as `k` columns, used in
  # place of `W x`.
  @spec error(term(), term(), term(), keyword(), float() | nil, [[float()]] | nil) :: MLError.t()
  def error(y, x, graph, opts, at \\ nil, xlag \\ nil) do
    opts = validate!(opts, @common_opts, [])
    {y_bin, x_bin, k, n, names, warnings} = prepare(y, x, graph, opts)

    e =
      error_native(y_bin, x_bin, xlag, k, graph, settings(opts, nil, at))
      |> entries(@error_keys)

    kc = k + 1

    %MLError{
      n: n,
      k: k,
      betas: e.betas,
      lam: hd(e.lam),
      vm: rows(e.vm, kc),
      vm1: rows(e.vm1, 2),
      std_err: e.std_err,
      z_stat: pairs(e.z_stat),
      logll: hd(e.logll),
      aic: hd(e.aic),
      schwarz: hd(e.schwarz),
      sig2: hd(e.sig2),
      u: e.u,
      predy: e.predy,
      e_filtered: e.e_filtered,
      pr2: hd(e.pr2),
      mean_y: hd(e.mean_y),
      std_y: hd(e.std_y),
      method: opts[:method],
      epsilon: opts[:epsilon] * 1.0,
      iterations: round(hd(e.iterations)),
      converged: hd(e.converged) == 1.0,
      name_x: names ++ ["lambda"],
      name_y: to_string(opts[:name_y]),
      warnings: warnings ++ convergence_warning(e)
    }
  end

  defp error_native(y_bin, x_bin, nil, k, graph, settings),
    do: Native.regression_ml_error(y_bin, x_bin, k, graph.ref, settings)

  defp error_native(y_bin, x_bin, xlag, k, graph, settings) do
    lagged = for col <- xlag, v <- col, into: <<>>, do: <<v * 1.0::float-native-64>>
    Native.regression_ml_error_xlag(y_bin, x_bin, lagged, k, graph.ref, settings)
  end

  defp settings(opts, impacts, at) do
    {Map.fetch!(@methods, opts[:method]), opts[:epsilon] * 1.0, Map.fetch!(@impacts, impacts),
     at && at * 1.0, opts[:max_dense_n]}
  end

  # -- input ---------------------------------------------------------------------------

  defp validate!(opts, allowed, extra_defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    opts =
      Keyword.merge(
        [
          constant: true,
          name_x: nil,
          name_y: "dep_var",
          method: :full,
          epsilon: 1.0e-7,
          max_dense_n: @max_dense_n,
          slx_lags: 0,
          slx_vars: :all
        ] ++ extra_defaults,
        opts
      )

    check_values!(opts)
    opts
  end

  defp check_values!(opts) do
    unless is_boolean(opts[:constant]), do: raise(ArgumentError, "constant must be a boolean")

    unless Map.has_key?(@methods, opts[:method]) do
      raise ArgumentError, "method must be :full, :ord or :lu, got #{inspect(opts[:method])}"
    end

    unless is_number(opts[:epsilon]) and opts[:epsilon] > 0 do
      raise ArgumentError, "epsilon must be a positive number, got #{inspect(opts[:epsilon])}"
    end

    unless is_integer(opts[:max_dense_n]) and opts[:max_dense_n] > 0 do
      raise ArgumentError, "max_dense_n must be a positive integer"
    end
  end

  defp prepare(y, x, graph, opts) do
    unless match?(%Graph{}, graph) do
      raise ArgumentError,
            "graph must be a Spatex.Graph, got #{inspect(graph, limit: 3)}"
    end

    {y_bin, n, names, x_bin, k, warnings} = Regression.prepare_design(y, x, opts, graph)

    if graph.n != n do
      raise ArgumentError, "graph has #{graph.n} nodes but y has #{n} rows"
    end

    if n > opts[:max_dense_n] do
      raise ArgumentError,
            "n = #{n} exceeds max_dense_n = #{opts[:max_dense_n]}: the ML estimators build dense " <>
              "n x n matrices (about #{div(n * n * 8, 1_048_576)} MiB each). Raise :max_dense_n " <>
              "if you have the memory, or use a GM estimator for large n"
    end

    {y_bin, x_bin, k, n, names, warnings}
  end

  # -- results -------------------------------------------------------------------------

  defp entries(result, keys) do
    case result do
      {:error, reason} ->
        raise ArgumentError, to_string(reason)

      {:ok, list} ->
        Map.new(list, fn {name, bin} ->
          {Enum.find(keys, &(Atom.to_string(&1) == name)) ||
             raise(ArgumentError, "unexpected native result #{inspect(name)}"),
           Input.from_f64_binary(bin)}
        end)
    end
  end

  defp rows(flat, k), do: Enum.chunk_every(flat, k)
  defp pairs(flat), do: flat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)

  defp convergence_warning(e) do
    if hd(e.converged) == 1.0 do
      []
    else
      ["The optimizer did not converge (evaluation limit reached or NaN objective)."]
    end
  end

  defp durbin_effect(name, b, theta, {adi, aii, ati}) when theta == 0.0,
    do: %{name: name, direct: times(adi, b), indirect: times(aii, b), total: times(ati, b)}

  defp durbin_effect(name, b, theta, {adi, _aii, ati}) do
    total = times(ati, b + theta)
    direct = times(adi, b)
    %{name: name, direct: direct, indirect: minus(total, direct), total: total}
  end

  defp minus(a, b) when is_float(a) and is_float(b), do: a - b
  defp minus(_, _), do: :nan

  defp times(a, b) when is_float(a) and is_float(b), do: a * b
  defp times(_, _), do: :nan

  @doc false
  # The original slopes, their betas and, per slope, the sum of its SLX coefficients (0 when not
  # lagged): spreg's Durbin impacts add the latter to the total effect only.
  def durbin_terms(names, betas, opts) do
    {names, betas} = if opts[:constant], do: {tl(names), tl(betas)}, else: {names, betas}
    lags = opts[:slx_lags]

    if lags == 0 do
      {names, betas, List.duplicate(0.0, length(names))}
    else
      kx = base_count(length(names), opts)
      flags = if opts[:slx_vars] == :all, do: List.duplicate(true, kx), else: opts[:slx_vars]
      {x_names, wx_betas} = {Enum.take(names, kx), Enum.drop(betas, kx)}
      lagged = for {name, true} <- Enum.zip(x_names, flags), do: name
      per_lag = wx_betas |> Enum.chunk_every(length(lagged)) |> Enum.zip_with(&Enum.sum/1)
      theta = Map.new(Enum.zip(lagged, per_lag))
      {x_names, Enum.take(betas, kx), Enum.map(x_names, &Map.get(theta, &1, 0.0))}
    end
  end

  @doc false
  # Number of original columns among `total` non-constant regressors with SLX terms.
  def base_count(total, opts),
    do: div(total + opts[:slx_lags] * count_dropped(opts), opts[:slx_lags] + 1)

  # Columns left out of the SLX lags by `slx_vars`.
  defp count_dropped(opts) do
    case opts[:slx_vars] do
      :all -> 0
      flags -> Enum.count(flags, &(&1 == false))
    end
  end

  @doc false
  def impacts_of(e, slopes, slope_betas, thetas \\ nil) do
    thetas = thetas || List.duplicate(0.0, length(slopes))

    found =
      for method <- [:simple, :full, :power], Map.has_key?(e, :"mult_#{method}") do
        [adi, aii, ati] = Map.fetch!(e, :"mult_#{method}")

        effects =
          for {{name, b}, theta} <- Enum.zip(Enum.zip(slopes, slope_betas), thetas) do
            durbin_effect(name, b, theta, {adi, aii, ati})
          end

        {method, {adi, aii, ati}, effects}
      end

    case found do
      [] ->
        {nil, nil}

      _ ->
        {Map.new(found, fn {m, mult, _} -> {m, mult} end),
         Map.new(found, fn {m, _, eff} -> {m, eff} end)}
    end
  end
end
