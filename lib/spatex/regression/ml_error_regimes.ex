defmodule Spatex.Regression.MLErrorRegimes do
  @moduledoc """
  Result of `Spatex.Regression.ml_error_regimes/5` (`spreg.ML_Error_Regimes`).

    * `betas`, `vm`, `std_err`, `z_stat` (`[{z, p}]`) - coefficients in `name_x` order. In
      `:pooled` mode this is spreg's layout: the regime-specific columns of every regime in turn
      (`"<regime>_<name>"`), the global ones (`"_Global_<name>"`) and `lambda` last. In
      `:separate` mode every regime's `[constant, x..., lambda]` block follows the previous one
      and `vm` is block diagonal.
    * `kr`, `kf`, `nr` - regime-specific columns per regime (`lambda` included in `:separate`
      mode), global columns (`lambda` included in `:pooled` mode, as spreg counts it) and
      regimes; `regimes` - the labels in order.
    * `lam` - the spatial error coefficient (`:pooled` mode, `nil` otherwise); `lams` - a map of
      the coefficient by regime label (`:separate` mode, `nil` otherwise).
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      are equal across regimes, jointly and per regime-specific column (`lambda` is one more
      column in `:separate` mode).
    * `vm1` (covariance of `lambda` and `sigma^2`), `logll`, `aic`, `schwarz`, `sig2`, `utu`,
      `pr2`, `k` (design columns, not counting `lambda`, as `Spatex.Regression.MLError`),
      `iterations` and `converged` - for the `:pooled` fit; `nil` in `:separate` mode, where
      every regime's `Spatex.Regression.MLError` in `multi` holds them.
    * `mean_y`, `std_y`, `utu`, `u`, `predy`, `e_filtered` - for every observation (`utu` is the
      sum of the squared residuals in both modes).
    * `mode` - `:pooled` (one ML fit on the regime design, `ml` holds the
      `Spatex.Regression.MLError`) or `:separate` (`regime_err_sep`: one
      `Spatex.Regression.MLError` per regime on the regime's subgraph, in `multi` by label).
    * `method`, `epsilon`, `constant_regi`, `cols2regi`, `regime_err_sep`, `name_x`, `name_y`,
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
    :lam,
    :lams,
    :chow,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :utu,
    :pr2,
    :mean_y,
    :std_y,
    :u,
    :predy,
    :e_filtered,
    :mode,
    :ml,
    :multi,
    :method,
    :epsilon,
    :iterations,
    :converged,
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.RegimesMLError do
  @moduledoc false
  # `Spatex.Regression.ml_error_regimes/5`: spreg's ML_Error_Regimes on top of the regime design
  # of `Spatex.Regression.Regimes`, the base ML error fit of `ml_error/4` and the shared setup of
  # `Spatex.Regression.RegimesML`.

  alias Spatex.Graph
  alias Spatex.Regression.{ML, MLErrorRegimes, Regimes, RegimesML}

  @allowed [
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :method,
    :epsilon,
    :max_dense_n,
    :name_x,
    :name_y
  ]
  @defaults [
    constant_regi: :many,
    cols2regi: :all,
    regime_err_sep: false,
    method: :full,
    epsilon: 1.0e-7,
    max_dense_n: 5_000,
    name_x: nil,
    name_y: "dep_var"
  ]

  @spec fit(term(), term(), [term()], term(), keyword(), term()) :: MLErrorRegimes.t()
  def fit(y, x, regimes, graph, opts, at \\ nil) do
    opts = validate!(opts)
    b = RegimesML.prepare(y, x, regimes, graph, opts, at)

    cond do
      not opts[:regime_err_sep] ->
        pooled(b)

      Enum.all?(b.cols2regi) and opts[:constant_regi] == :many ->
        separate(b, Regimes.regime_graphs(graph, b.regimes, b.set, 0))

      true ->
        raise ArgumentError, "All coefficients must vary across regimes if regime_err_sep = true."
    end
  end

  defp ml_opts(b, extra) do
    [
      method: b.opts[:method],
      epsilon: b.opts[:epsilon],
      max_dense_n: b.opts[:max_dense_n],
      name_y: b.name_y
    ] ++ extra
  end

  # -- one ML fit on the regime design -----------------------------------------------------

  defp pooled(b) do
    const_regi = b.opts[:constant_regi] == :many
    ones = List.duplicate(1.0, b.n)

    vars = [
      {"CONSTANT", ones, const_regi}
      | Enum.zip_with([b.names, b.xcols, b.cols2regi], &List.to_tuple/1)
    ]

    {cols, dnames, kr, kf, regi_names} = Regimes.regimize(vars, b.regimes, b.set)
    if kr == 0, do: raise(ArgumentError, "at least one column must vary by regime")

    # spreg lags the ORIGINAL columns and then expands them by regime (`regimes(W [1 | X])`),
    # which is not the lag of the regime design `W regimes([1 | X])` across regime borders.
    lagged = for {name, col, flag} <- vars, do: {name, Graph.lag(b.graph, col), flag}
    {xlag, _, _, _, _} = Regimes.regimize(lagged, b.regimes, b.set)

    ml = ML.error(b.y, cols, b.graph, ml_opts(b, constant: false, name_x: dnames), b.at, xlag)
    nr = length(b.set)

    %MLErrorRegimes{
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
      lam: ml.lam,
      chow: Regimes.chow(ml.betas, ml.vm, {kr, kf + 1, 0, 0}, nr, regi_names),
      logll: ml.logll,
      aic: ml.aic,
      schwarz: ml.schwarz,
      sig2: ml.sig2,
      utu: Enum.sum(Enum.map(ml.u, &(&1 * &1))),
      pr2: ml.pr2,
      mean_y: ml.mean_y,
      std_y: ml.std_y,
      u: ml.u,
      predy: ml.predy,
      e_filtered: ml.e_filtered,
      mode: :pooled,
      ml: ml,
      method: ml.method,
      epsilon: ml.epsilon,
      iterations: ml.iterations,
      converged: ml.converged,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_err_sep: false,
      name_x: dnames ++ ["lambda"],
      name_y: b.name_y,
      warnings: b.warnings ++ ml.warnings
    }
  end

  # -- one ML fit per regime (regime_err_sep) ------------------------------------------------

  defp separate(b, subs) do
    {ids, keep, dropped} = RegimesML.constant_split(b)
    ats = RegimesML.at_by_regime(b)
    fits = Map.new(b.set, fn r -> {r, regime_fit(b, ids[r], keep, subs[r], r, ats[r])} end)
    fit_list = Enum.map(b.set, &fits[&1])
    kr = length(keep) + 2
    betas = Enum.flat_map(fit_list, & &1.betas)
    vm = Regimes.block_diagonal(Enum.map(fit_list, & &1.vm))
    u = Regimes.scatter(b.n, ids, b.set, &fits[&1].u)
    chow_names = ["CONSTANT" | Enum.map(keep, &elem(&1, 0))] ++ ["lambda"]

    %MLErrorRegimes{
      n: b.n,
      kr: kr,
      kf: 0,
      nr: length(b.set),
      regimes: b.set,
      betas: betas,
      vm: vm,
      std_err: Enum.flat_map(fit_list, & &1.std_err),
      z_stat: Enum.flat_map(fit_list, & &1.z_stat),
      lams: Map.new(b.set, &{&1, fits[&1].lam}),
      chow: Regimes.chow(betas, vm, {kr, 0, 0, 0}, length(b.set), chow_names),
      utu: Enum.sum(Enum.map(u, &(&1 * &1))),
      mean_y: RegimesML.mean(b.y),
      std_y: RegimesML.std(b.y),
      u: u,
      predy: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy),
      e_filtered: Regimes.scatter(b.n, ids, b.set, &fits[&1].e_filtered),
      mode: :separate,
      multi: fits,
      method: b.opts[:method],
      epsilon: b.opts[:epsilon] * 1.0,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_err_sep: true,
      name_x: Enum.flat_map(fit_list, & &1.name_x),
      name_y: b.name_y,
      warnings:
        b.warnings ++ RegimesML.dropped_warning(dropped) ++ Regimes.island_warning(b.set, subs)
    }
  end

  defp regime_fit(b, rows, keep, sub, r, at) do
    cols = Enum.map(keep, fn {_, col} -> Regimes.take(col, rows) end)
    opts = ml_opts(b, name_x: Enum.map(keep, &elem(&1, 0)))
    fit = ML.error(Regimes.take(b.y, rows), cols, sub, opts, at)
    prefix = fn names -> Enum.map(names, &"#{r}_#{&1}") end
    %{fit | name_x: prefix.(fit.name_x), name_y: "#{r}_#{b.name_y}"}
  end

  # -- options ---------------------------------------------------------------------------------

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

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

    unless is_boolean(opts[:regime_err_sep]),
      do: raise(ArgumentError, ":regime_err_sep must be a boolean")

    opts
  end
end
