defmodule Spatex.Regression.MLLagRegimes do
  @moduledoc """
  Result of `Spatex.Regression.ml_lag_regimes/5` (`spreg.ML_Lag_Regimes`).

    * `betas`, `vm`, `std_err`, `z_stat` (`[{z, p}]`) - coefficients in `name_x` order. In
      `:pooled` mode this is spreg's layout: the regime-specific columns of every regime in turn
      (`"<regime>_<name>"`), the global ones (`"_Global_<name>"`) and `rho` last
      (`"_Global_W_<y>"`). In `:separate` mode every regime's `[constant, x..., rho]` block
      follows the previous one and `vm` is block diagonal.
    * `kr`, `kf`, `nr` - regime-specific columns per regime (`rho` included in `:separate` mode),
      global columns (`rho` included in `:pooled` mode, as spreg counts it) and regimes;
      `regimes` - the labels in order.
    * `rho` - the spatial lag coefficient (`:pooled` mode, `nil` otherwise); `rhos` - a map of the
      coefficient by regime label (`:separate` mode, `nil` otherwise).
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      are equal across regimes, jointly and per regime-specific column (`rho` is one more column
      in `:separate` mode).
    * `vm1` (covariance of the coefficients and `sigma^2`), `logll`, `aic`, `schwarz`, `sig2`,
      `utu`, `pr2`, `pr2_e` (`nil` when `|rho| >= 1`), `k` (coefficients including `rho`),
      `iterations` and `converged` - as in `Spatex.Regression.MLLag`, for the `:pooled` fit;
      `nil` in `:separate` mode, where every regime's `Spatex.Regression.MLLag` in `multi` holds
      them.
    * `mean_y`, `std_y`, `utu`, `u`, `predy`, `predy_e` (reduced form), `e_pred` - for every
      observation (`utu` is the sum of the squared residuals in both modes).
    * `sp_multipliers`, `impacts`, `power_terms` - as in `Spatex.Regression.MLLag`, with the
      impacts of the non-constant coefficients (`:pooled` mode; per regime in `multi`).
    * `mode` - `:pooled` (one ML fit on the regime design, `ml` holds the
      `Spatex.Regression.MLLag`) or `:separate` (`regime_lag_sep`: one `Spatex.Regression.MLLag`
      per regime on the regime's subgraph, in `multi` by regime label).
    * `method`, `epsilon`, `constant_regi`, `cols2regi`, `regime_lag_sep`, `name_x`, `name_y`,
      `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :kr,
    :kf,
    :nr,
    :regimes,
    :betas,
    :vm,
    :vm1,
    :std_err,
    :z_stat,
    :rho,
    :rhos,
    :chow,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :utu,
    :pr2,
    :pr2_e,
    :mean_y,
    :std_y,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :sp_multipliers,
    :impacts,
    :power_terms,
    :mode,
    :ml,
    :multi,
    :method,
    :epsilon,
    :iterations,
    :converged,
    :constant_regi,
    :cols2regi,
    :regime_lag_sep,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.RegimesML do
  @moduledoc false
  # `Spatex.Regression.ml_lag_regimes/5`: spreg's ML_Lag_Regimes on top of the regime design of
  # `Spatex.Regression.Regimes` and the ML lag fit of `ml_lag/4`.

  alias Spatex.{Graph, Input, Regression}
  alias Spatex.Regression.{ML, MLLagRegimes, Regimes}

  @allowed [
    :constant_regi,
    :cols2regi,
    :regime_lag_sep,
    :method,
    :epsilon,
    :spat_impacts,
    :max_dense_n,
    :name_x,
    :name_y
  ]
  @defaults [
    constant_regi: :many,
    cols2regi: :all,
    regime_lag_sep: false,
    method: :full,
    epsilon: 1.0e-7,
    spat_impacts: :simple,
    max_dense_n: 5_000,
    name_x: nil,
    name_y: "dep_var"
  ]

  @spec fit(term(), term(), [term()], term(), keyword(), term()) :: MLLagRegimes.t()
  def fit(y, x, regimes, graph, opts, at \\ nil) do
    opts = validate!(opts)
    b = prepare(y, x, regimes, graph, opts, at)
    %{cols2regi: cols2regi, regimes: regimes, set: set} = b

    cond do
      not opts[:regime_lag_sep] ->
        pooled(b)

      Enum.all?(cols2regi) and opts[:constant_regi] == :many ->
        subs = Regimes.regime_graphs(b.graph, regimes, set, length(cols2regi) + 2)
        separate(b, subs)

      true ->
        raise ArgumentError, "All variables must vary by regimes if regime_lag_sep = true."
    end
  end

  # The validated inputs shared by `ml_lag_regimes/5` and `ml_error_regimes/5`: the design
  # without the constant (spreg adds it per `constant_regi`), the regimes and the checks.
  @doc false
  def prepare(y, x, regimes, graph, opts, at) do
    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "graph must be a Spatex.Graph, got #{inspect(graph, limit: 3)}")

    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: opts[:name_x])

    if graph.n != n, do: raise(ArgumentError, "graph has #{graph.n} nodes but y has #{n} rows")
    regimes = Input.to_labels(regimes, name: "regimes", length: n)
    xcols = x_bin |> Regimes.bin_cols(n) |> tl()
    names = tl(names)
    set = Regimes.check_regimes!(regimes, n, length(names))

    %{
      n: n,
      y: Input.from_f64_binary(y_bin),
      xcols: xcols,
      names: names,
      regimes: regimes,
      set: set,
      cols2regi: Regimes.cols2regi!(opts[:cols2regi], length(names)),
      graph: graph,
      name_y: to_string(opts[:name_y]),
      warnings: warnings,
      opts: opts,
      at: at
    }
  end

  # spreg's check_const_regi for the separate fits: a column constant within any regime is
  # dropped everywhere. Returns `{row ids by regime, kept {name, column}, dropped names}`.
  @doc false
  def constant_split(b) do
    ids = Map.new(b.set, fn r -> {r, Regimes.indices(b.regimes, r)} end)

    keep =
      for {name, col} <- Enum.zip(b.names, b.xcols),
          Enum.all?(b.set, &Regimes.varies?(Regimes.take(col, ids[&1]))),
          do: {name, col}

    {ids, keep, b.names -- Enum.map(keep, &elem(&1, 0))}
  end

  @doc false
  def ml_opts(b, extra) do
    [
      method: b.opts[:method],
      epsilon: b.opts[:epsilon],
      max_dense_n: b.opts[:max_dense_n],
      spat_impacts: b.opts[:spat_impacts],
      name_y: b.name_y
    ] ++ extra
  end

  # -- one ML fit on the regime design -----------------------------------------------------

  defp pooled(b) do
    const_regi = b.opts[:constant_regi] == :many
    ones = List.duplicate(1.0, b.n)

    {cols, dnames, kr, kf, regi_names} =
      Regimes.regimize(
        [
          {"CONSTANT", ones, const_regi}
          | Enum.zip_with([b.names, b.xcols, b.cols2regi], &List.to_tuple/1)
        ],
        b.regimes,
        b.set
      )

    if kr == 0, do: raise(ArgumentError, "at least one column must vary by regime")

    ml = lag_fit(b, b.y, cols, b.graph, ml_opts(b, constant: false, name_x: dnames), b.at)
    nr = length(b.set)
    names = dnames ++ ["_Global_W_" <> b.name_y]

    xs =
      for {name, i} <- Enum.with_index(dnames),
          not String.ends_with?(name, "_CONSTANT"),
          do: {name, Enum.at(ml.betas, i)}

    {multipliers, effects} = effects(ml, xs)

    %MLLagRegimes{
      n: b.n,
      k: ml.k,
      kr: kr,
      kf: kf + 1,
      nr: nr,
      regimes: b.set,
      betas: ml.betas,
      vm: ml.vm,
      vm1: ml.vm1,
      std_err: ml.std_err,
      z_stat: ml.z_stat,
      rho: ml.rho,
      chow: Regimes.chow(ml.betas, ml.vm, {kr, kf + 1, 0, 0}, nr, regi_names),
      logll: ml.logll,
      aic: ml.aic,
      schwarz: ml.schwarz,
      sig2: ml.sig2,
      utu: ml.utu,
      pr2: ml.pr2,
      pr2_e: ml.pr2_e,
      mean_y: ml.mean_y,
      std_y: ml.std_y,
      u: ml.u,
      predy: ml.predy,
      predy_e: ml.predy_e,
      e_pred: ml.e_pred,
      sp_multipliers: multipliers,
      impacts: effects,
      power_terms: ml.power_terms,
      mode: :pooled,
      ml: ml,
      method: ml.method,
      epsilon: ml.epsilon,
      iterations: ml.iterations,
      converged: ml.converged,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_lag_sep: false,
      name_x: names,
      name_y: b.name_y,
      warnings: b.warnings ++ ml.warnings
    }
  end

  # The multipliers of the fit and the effects of the given `{name, beta}` slopes.
  defp effects(%{sp_multipliers: nil}, _slopes), do: {nil, nil}

  defp effects(ml, slopes) do
    e = for {m, {a, i, t}} <- ml.sp_multipliers, into: %{}, do: {:"mult_#{m}", [a, i, t]}
    ML.impacts_of(e, Enum.map(slopes, &elem(&1, 0)), Enum.map(slopes, &elem(&1, 1)))
  end

  defp lag_fit(_b, y, cols, graph, opts, nil), do: Regression.ml_lag(y, cols, graph, opts)
  defp lag_fit(_b, y, cols, graph, opts, at), do: Regression.ml_lag_at(y, cols, graph, at, opts)

  # -- one ML fit per regime (regime_lag_sep) ------------------------------------------------

  defp separate(b, subs) do
    {ids, keep, dropped} = constant_split(b)
    ats = at_by_regime(b)

    fits =
      Map.new(b.set, fn r -> {r, regime_fit(b, ids[r], keep, subs[r], r, ats[r])} end)

    fit_list = Enum.map(b.set, &fits[&1])
    nr = length(b.set)
    kr = length(keep) + 2
    betas = Enum.flat_map(fit_list, & &1.betas)
    vm = Regimes.block_diagonal(Enum.map(fit_list, & &1.vm))
    u = Regimes.scatter(b.n, ids, b.set, &fits[&1].u)
    chow_names = ["CONSTANT" | Enum.map(keep, &elem(&1, 0))] ++ ["W_" <> b.name_y]

    %MLLagRegimes{
      n: b.n,
      kr: kr,
      kf: 0,
      nr: nr,
      regimes: b.set,
      betas: betas,
      vm: vm,
      std_err: Enum.flat_map(fit_list, & &1.std_err),
      z_stat: Enum.flat_map(fit_list, & &1.z_stat),
      rhos: Map.new(b.set, &{&1, fits[&1].rho}),
      chow: Regimes.chow(betas, vm, {kr, 0, 0, 0}, nr, chow_names),
      utu: Enum.sum(Enum.map(u, &(&1 * &1))),
      mean_y: mean(b.y),
      std_y: std(b.y),
      u: u,
      predy: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy),
      predy_e: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy_e),
      e_pred: Regimes.scatter(b.n, ids, b.set, &fits[&1].e_pred),
      mode: :separate,
      multi: fits,
      method: b.opts[:method],
      epsilon: b.opts[:epsilon] * 1.0,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_lag_sep: true,
      name_x: Enum.flat_map(fit_list, & &1.name_x),
      name_y: b.name_y,
      warnings: b.warnings ++ dropped_warning(dropped) ++ Regimes.island_warning(b.set, subs)
    }
  end

  @doc false
  def at_by_regime(%{at: nil}), do: %{}

  def at_by_regime(%{at: at, set: set}) when is_list(at) and length(at) == length(set),
    do: Map.new(Enum.zip(set, at))

  def at_by_regime(%{set: _}), do: raise(ArgumentError, "at must hold one value per regime")

  @doc false
  def dropped_warning([]), do: []

  def dropped_warning(names),
    do: [
      "Variable(s) #{inspect(names)} removed for being constant for at least one of the regimes."
    ]

  defp regime_fit(b, rows, keep, sub, r, at) do
    cols = Enum.map(keep, fn {_, col} -> Regimes.take(col, rows) end)
    opts = ml_opts(b, name_x: Enum.map(keep, &elem(&1, 0)))
    fit = lag_fit(b, Regimes.take(b.y, rows), cols, sub, opts, at)
    prefix = fn names -> Enum.map(names, &"#{r}_#{&1}") end

    %{
      fit
      | name_x: prefix.(fit.name_x),
        name_y: "#{r}_#{b.name_y}",
        impacts:
          fit.impacts &&
            Map.new(fit.impacts, fn {m, list} ->
              {m, Enum.map(list, &%{&1 | name: "#{r}_#{&1.name}"})}
            end)
    }
  end

  @doc false
  def mean(v), do: Enum.sum(v) / length(v)

  @doc false
  def std(v) do
    m = mean(v)
    :math.sqrt(Enum.sum(Enum.map(v, &((&1 - m) * (&1 - m)))) / (length(v) - 1))
  end

  # -- options ---------------------------------------------------------------------------------

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    if Keyword.has_key?(opts, :regime_err_sep),
      do:
        raise(
          ArgumentError,
          "ml_lag_regimes has no :regime_err_sep (spreg's ML_Lag_Regimes fixes it to false); " <>
            "use :regime_lag_sep for one lag model per regime"
        )

    case Keyword.keys(opts) -- @allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)}"
    end

    opts = Keyword.merge(@defaults, opts)

    unless opts[:constant_regi] in [:many, :one],
      do:
        raise(
          ArgumentError,
          ":constant_regi must be :many or :one, got #{inspect(opts[:constant_regi])}"
        )

    unless is_boolean(opts[:regime_lag_sep]),
      do: raise(ArgumentError, ":regime_lag_sep must be a boolean")

    opts
  end
end
