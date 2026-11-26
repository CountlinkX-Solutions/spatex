defmodule Spatex.Regression.GMLagRegimes do
  @moduledoc """
  Result of `Spatex.Regression.gm_lag_regimes/5` (`spreg.GM_Lag_Regimes`).

    * `betas`, `vm`, `std_err`, `z_stat` (`[{z, p}]`) - coefficients in `names` order. In
      `:pooled` mode this is spreg's layout: the regime-specific exogenous columns of every
      regime in turn (`"<regime>_<name>"`), the global exogenous ones (`"_Global_<name>"`), then
      the same for the endogenous columns with `W y` last (`"_Global_W_<y>"`, or one
      `"<regime>_W_<y>"` per regime with `regime_lag_sep`). In `:separate` mode every regime's
      `[constant, x..., yend..., W y]` block follows the previous one.
    * `name_x`, `name_yend`, `name_q` - names as spreg lists them (`name_yend` ends with the
      `W y` name, `name_q` includes the spatially lagged instruments). `kr`, `kf`, `kryd`, `nr` -
      regime-specific columns per regime (exogenous plus endogenous, `W y` included under
      `regime_lag_sep`), global columns, regime-specific endogenous columns, and regimes;
      `regimes` - the labels in order.
    * `rho` - the spatial lag coefficient: the single `W y` coefficient, or with `regime_lag_sep`
      the last regime's (as spreg's `rho` attribute); `rhos` - with `regime_lag_sep`, a map of the
      coefficient by regime label, otherwise `nil`.
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      are equal across regimes, jointly and per regime-specific column.
    * `u`, `predy` - residuals and fitted values; `predy_e`, `e_pred` - reduced-form prediction
      `(I - rho W)^-1 X beta` and its residual (zeros for a regime whose `|rho| >= 1`).
    * `pr2`, `pr2_e` - squared correlations of `y` with `predy` and `predy_e` (`:pooled` mode;
      `pr2_e` is `nil` when `|rho| >= 1`); `ak_test` - Anselin-Kelejian test `{statistic, p}`
      (`:pooled` mode; `nil` with `spat_diag: false` or `robust: :hac`); `sp_multipliers`,
      `impacts`, `power_terms` - as in `Spatex.Regression.GMLag`, for the non-constant exogenous
      coefficients (`:pooled` mode). In `:separate` mode these live in the per-regime fits.
    * `mode` - `:separate` (one `Spatex.Regression.GMLag` per regime on the regime's subgraph,
      in `multi` by regime label) or `:pooled` (one 2SLS on the regime design, `tsls` holds the
      `Spatex.Regression.TSLS` fit).
    * `cols2regi` - the flags over the `x` then `yend` columns (without the `W y` flag);
      `constant_regi`, `regime_lag_sep`, `regime_err_sep` (as used after spreg's reconciliation of
      the two), `robust`, `w_lags`, `lag_q`, `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :kr,
    :kf,
    :kryd,
    :nr,
    :regimes,
    :betas,
    :vm,
    :std_err,
    :z_stat,
    :rho,
    :rhos,
    :chow,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :pr2,
    :pr2_e,
    :ak_test,
    :sp_multipliers,
    :impacts,
    :power_terms,
    :mode,
    :multi,
    :tsls,
    :constant_regi,
    :cols2regi,
    :regime_lag_sep,
    :regime_err_sep,
    :robust,
    :w_lags,
    :lag_q,
    :names,
    :name_x,
    :name_yend,
    :name_q,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.RegimesLag do
  @moduledoc false
  # `Spatex.Regression.gm_lag_regimes/5`: spreg's GM_Lag_Regimes on top of the 2SLS regime
  # machinery of `Spatex.Regression.Regimes` and the spatial lag fit of `gm_lag/4`.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{GM, GMLagRegimes, ML, Regimes}

  @allowed [
    :yend,
    :q,
    :w_lags,
    :lag_q,
    :robust,
    :gwk,
    :sig2n_k,
    :spat_diag,
    :spat_impacts,
    :constant_regi,
    :cols2regi,
    :regime_lag_sep,
    :regime_err_sep,
    :hard_bound,
    :spreg_compatible,
    :name_x,
    :name_y,
    :name_yend,
    :name_q
  ]
  @impacts %{nil => 0, simple: 1, full: 2, power: 3, all: 4}
  @booleans [
    :lag_q,
    :sig2n_k,
    :spat_diag,
    :regime_lag_sep,
    :regime_err_sep,
    :hard_bound,
    :spreg_compatible
  ]
  @groupwise "Groupwise heteroskedasticity is not currently available for this method, so regime_err_sep has been set to False."
  @hac_err_sep "Error by regimes is not available for HAC estimation. The error by regimes has been disabled for this model."
  @hac_diag "Spatial diagnostics are not available for HAC estimation. Hence, spatial diagnostics have been disabled for this model."

  @spec fit(term(), term(), [term()], term(), keyword()) :: GMLagRegimes.t()
  def fit(y, x, regimes, graph, opts) do
    opts = validate!(opts)
    check_graph!(graph, opts[:gwk])
    b = prepare(y, x, regimes, graph, opts)
    {opts, reconciled} = reconcile(opts)
    b = %{b | opts: opts, warnings: b.warnings ++ reconciled}
    lag_q_names = if opts[:lag_q], do: b.q_names, else: []
    lags = Regimes.lag_names(b.names ++ lag_q_names, opts[:w_lags])
    b = Map.put(b, :q_all, b.q_names ++ lags)

    cond do
      not opts[:regime_lag_sep] ->
        pooled(b, graph, nil)

      opts[:regime_err_sep] and Enum.all?(b.cols2regi) and opts[:constant_regi] == :many ->
        separate(b, Regimes.regime_graphs(b.graph, b.regimes, b.set, length(b.cols2regi) + 2))

      true ->
        subs = Regimes.regime_graphs(b.graph, b.regimes, b.set, length(b.cols2regi) + 2)
        pooled(b, Regimes.union_graph(b.graph, b.set, subs), subs)
    end
  end

  defp prepare(y, x, regimes, graph, opts) do
    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: opts[:name_x])

    check_sizes!(graph, opts[:gwk], n)
    regimes = Input.to_labels(regimes, name: "regimes", length: n)
    {yend_names, yend_bin, ky} = GM.extra(opts[:yend], n, opts[:name_yend], "yend")
    {q_names, q_bin, kq} = GM.extra(opts[:q], n, opts[:name_q], "q")
    if ky == 0 and kq > 0, do: raise(ArgumentError, "q needs a matching yend")

    xcols = x_bin |> Regimes.bin_cols(n) |> tl()
    names = tl(names)
    set = Regimes.check_regimes!(regimes, n, length(names))

    %{
      n: n,
      y: Input.from_f64_binary(y_bin),
      y_bin: y_bin,
      xcols: xcols,
      x_bin: IO.iodata_to_binary(Enum.map(xcols, &Regimes.bin/1)),
      names: names,
      ycols: Regimes.bin_cols(yend_bin, n),
      yend_bin: yend_bin,
      ky: ky,
      yend_names: yend_names,
      qcols: Regimes.bin_cols(q_bin, n),
      q_bin: q_bin,
      q_names: q_names,
      kq: kq,
      regimes: regimes,
      set: set,
      cols2regi: Regimes.cols2regi!(opts[:cols2regi], length(names) + ky),
      graph: graph,
      name_y: to_string(opts[:name_y]),
      warnings: warnings,
      opts: opts
    }
  end

  defp check_sizes!(graph, gwk, n) do
    if graph.n != n, do: raise(ArgumentError, "graph has #{graph.n} nodes but y has #{n} rows")

    if gwk && gwk.n != n,
      do: raise(ArgumentError, "gwk has #{gwk.n} nodes but y has #{n} rows")
  end

  # -- reconciling the options as spreg's __init__ does ---------------------------------------

  defp reconcile(opts) do
    {err_sep, n1} = lag_err_sep(opts[:regime_lag_sep], opts[:regime_err_sep])
    hac? = opts[:robust] == :hac
    {err_sep, n2} = if hac? and err_sep, do: {false, [@hac_err_sep]}, else: {err_sep, []}

    {spat_diag, n3} =
      if hac? and opts[:spat_diag], do: {false, [@hac_diag]}, else: {opts[:spat_diag], []}

    {opts |> Keyword.put(:regime_err_sep, err_sep) |> Keyword.put(:spat_diag, spat_diag),
     n1 ++ n2 ++ n3}
  end

  defp lag_err_sep(true, false),
    do: {true, ["regime_err_sep set to True when regime_lag_sep=True."]}

  defp lag_err_sep(false, true), do: {false, [@groupwise]}
  defp lag_err_sep(_lag_sep, err_sep), do: {err_sep, []}

  # -- one lag model per regime ---------------------------------------------------------------

  defp separate(b, subs) do
    ids = Map.new(b.set, fn r -> {r, Regimes.indices(b.regimes, r)} end)

    # spreg's check_const_regi: a column constant within any regime is dropped everywhere.
    keep =
      for {name, col} <- Enum.zip(b.names, b.xcols),
          Enum.all?(b.set, &Regimes.varies?(Regimes.take(col, ids[&1]))),
          do: {name, col}

    dropped = b.names -- Enum.map(keep, &elem(&1, 0))
    fits = Map.new(b.set, fn r -> {r, regime_fit(b, ids[r], keep, subs[r], r)} end)
    fit_list = Enum.map(b.set, &fits[&1])
    nr = length(b.set)
    kr = length(keep) + 1 + b.ky + 1
    betas = Enum.flat_map(fit_list, & &1.betas)
    vm = Regimes.block_diagonal(Enum.map(fit_list, & &1.vm))
    w_name = "W_" <> b.name_y
    x_names = fn r -> Enum.map(["CONSTANT" | Enum.map(keep, &elem(&1, 0))], &"#{r}_#{&1}") end
    prefixed = fn r, list -> Enum.map(list, &"#{r}_#{&1}") end

    chow_names = ["CONSTANT" | Enum.map(keep, &elem(&1, 0))] ++ b.yend_names ++ [w_name]
    rhos = Map.new(b.set, &{&1, fits[&1].rho})

    %GMLagRegimes{
      n: b.n,
      kr: kr,
      kf: 0,
      kryd: 0,
      nr: nr,
      regimes: b.set,
      betas: betas,
      vm: vm,
      std_err: Enum.flat_map(fit_list, & &1.std_err),
      z_stat: Enum.flat_map(fit_list, & &1.z_stat),
      rho: rhos[List.last(b.set)],
      rhos: rhos,
      chow: Regimes.chow(betas, vm, {kr, 0, 0, 0}, nr, chow_names),
      u: Regimes.scatter(b.n, ids, b.set, &fits[&1].u),
      predy: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy),
      predy_e: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy_e),
      e_pred: Regimes.scatter(b.n, ids, b.set, &fits[&1].e_pred),
      mode: :separate,
      multi: fits,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_lag_sep: true,
      regime_err_sep: true,
      robust: b.opts[:robust],
      w_lags: b.opts[:w_lags],
      lag_q: b.opts[:lag_q],
      names: Enum.flat_map(fit_list, & &1.name_x),
      name_x: Enum.flat_map(b.set, x_names),
      name_yend: Enum.flat_map(b.set, &prefixed.(&1, b.yend_names ++ [w_name])),
      name_q: Enum.flat_map(b.set, &prefixed.(&1, b.q_all)),
      name_y: b.name_y,
      warnings: b.warnings ++ dropped_warning(dropped) ++ Regimes.island_warning(b.set, subs)
    }
  end

  defp dropped_warning([]), do: []

  defp dropped_warning(names),
    do: [
      "Variable(s) #{inspect(names)} removed for being constant for at least one of the regimes."
    ]

  defp regime_fit(b, rows, keep, sub, r) do
    pick = fn cols -> Enum.map(cols, &Regimes.take(&1, rows)) end

    extra = fn
      [] -> nil
      cols -> pick.(cols)
    end

    fit =
      Regression.gm_lag(
        Regimes.take(b.y, rows),
        pick.(Enum.map(keep, &elem(&1, 1))),
        sub,
        yend: extra.(b.ycols),
        q: extra.(b.qcols),
        w_lags: b.opts[:w_lags],
        lag_q: b.opts[:lag_q],
        robust: b.opts[:robust],
        sig2n_k: b.opts[:sig2n_k],
        spat_diag: b.opts[:spat_diag],
        spat_impacts: b.opts[:spat_impacts],
        hard_bound: b.opts[:hard_bound],
        name_x: Enum.map(keep, &elem(&1, 0)),
        name_y: b.name_y,
        name_yend: if(b.ky == 0, do: nil, else: b.yend_names),
        name_q: if(b.kq == 0, do: nil, else: b.q_names)
      )

    prefix = fn names -> Enum.map(names, &"#{r}_#{&1}") end

    %{
      fit
      | name_x: prefix.(fit.name_x),
        name_y: "#{r}_#{b.name_y}",
        name_yend: prefix.(fit.name_yend),
        name_q: prefix.(fit.name_q),
        impacts:
          fit.impacts &&
            Map.new(fit.impacts, fn {m, list} ->
              {m, Enum.map(list, &%{&1 | name: "#{r}_#{&1.name}"})}
            end)
    }
  end

  # -- one 2SLS on the regime design ----------------------------------------------------------

  # `w` is the graph that builds `W y` and the instruments: the graph itself, or with
  # `regime_lag_sep` the union of the regime subgraphs (`subs`).
  defp pooled(b, w, subs) do
    lag_sep = b.opts[:regime_lag_sep]
    {ycols, qcols} = Regimes.lag_endog(b, w)
    wy = List.last(ycols)

    tb = %{
      n: b.n,
      y: b.y,
      xcols: b.xcols,
      names: b.names,
      ycols: ycols,
      yend_names: b.yend_names ++ ["W_" <> b.name_y],
      qcols: qcols,
      q_names: b.q_all,
      regimes: b.regimes,
      set: b.set,
      cols2regi: b.cols2regi ++ [lag_sep],
      warnings: b.warnings,
      opts: b.opts
    }

    {t, z} = Regimes.tsls_pooled_design(tb)
    fit = t.tsls
    nr = length(b.set)

    x_flags = [b.opts[:constant_regi] == :many | Enum.take(b.cols2regi, length(b.names))]
    krx = Enum.count(x_flags, & &1)
    kfx = length(x_flags) - krx

    rhos =
      if lag_sep,
        do:
          Map.new(Enum.with_index(b.set), fn {r, i} ->
            {r, Enum.at(fit.betas, nr * krx + kfx + i * t.kryd + t.kryd - 1)}
          end)

    rho = if lag_sep, do: rhos[List.last(b.set)], else: List.last(fit.betas)

    {predy_e, e_pred, warn} =
      if lag_sep,
        do: regime_prediction(b, subs, fit.predy, wy, rhos),
        else: Regimes.prediction(b.graph, b.y, fit.predy, wy, rho, b.opts[:hard_bound])

    diag = diagnostics(b, w, predy_e, z, fit, rho)

    xs =
      for {name, i} <- Enum.with_index(t.name_x),
          not String.ends_with?(name, "_CONSTANT"),
          do: {name, Enum.at(fit.betas, i)}

    {multipliers, effects} =
      ML.impacts_of(diag, Enum.map(xs, &elem(&1, 0)), Enum.map(xs, &elem(&1, 1)))

    %GMLagRegimes{
      n: b.n,
      kr: t.kr,
      kf: t.kf,
      kryd: t.kryd,
      nr: nr,
      regimes: b.set,
      betas: t.betas,
      vm: t.vm,
      std_err: t.std_err,
      z_stat: t.z_stat,
      rho: rho,
      rhos: rhos,
      chow: t.chow,
      u: t.u,
      predy: t.predy,
      predy_e: predy_e,
      e_pred: e_pred,
      pr2: fit.pr2,
      pr2_e: diag[:pr2_e] && hd(diag.pr2_e),
      ak_test: diag[:ak_test] && List.to_tuple(diag.ak_test),
      sp_multipliers: multipliers,
      impacts: effects,
      power_terms: diag[:power_terms] && round(hd(diag.power_terms)),
      mode: :pooled,
      tsls: fit,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_lag_sep: lag_sep,
      regime_err_sep: b.opts[:regime_err_sep],
      robust: b.opts[:robust],
      w_lags: b.opts[:w_lags],
      lag_q: b.opts[:lag_q],
      names: t.names,
      name_x: t.name_x,
      name_yend: t.name_yend,
      name_q: t.name_q,
      name_y: b.name_y,
      warnings: t.warnings ++ warn ++ if(subs, do: Regimes.island_warning(b.set, subs), else: [])
    }
  end

  # spreg's sp_att_reg: each regime's reduced form on its own subgraph and coefficient.
  defp regime_prediction(b, subs, predy, wy, rhos) do
    ids = Map.new(b.set, fn r -> {r, Regimes.indices(b.regimes, r)} end)

    parts =
      Map.new(b.set, fn r ->
        rows = ids[r]

        {predy_e, e_pred, _warn} =
          Regimes.prediction(
            subs[r],
            Regimes.take(b.y, rows),
            Regimes.take(predy, rows),
            Regimes.take(wy, rows),
            rhos[r],
            b.opts[:hard_bound]
          )

        {r, {predy_e, e_pred}}
      end)

    {Regimes.scatter(b.n, ids, b.set, &elem(parts[&1], 0)),
     Regimes.scatter(b.n, ids, b.set, &elem(parts[&1], 1)), []}
  end

  defp diagnostics(b, w, predy_e, z, fit, rho) do
    {:ok, list} =
      Native.regression_lag_diag(
        b.y_bin,
        Regimes.bin(predy_e),
        {IO.iodata_to_binary(Enum.map(z, &Regimes.bin/1)), length(z)},
        fit.varb |> List.flatten() |> Regimes.bin(),
        Regimes.bin(fit.u),
        w.ref,
        rho * 1.0,
        %{spat_diag: b.opts[:spat_diag], impacts: Map.fetch!(@impacts, b.opts[:spat_impacts])}
      )
      |> Regimes.ok!()

    Map.new(list, fn {k, v} -> {String.to_existing_atom(k), Input.from_f64_binary(v)} end)
  end

  # -- options ---------------------------------------------------------------------------------

  defp check_graph!(graph, gwk) do
    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "graph must be a Spatex.Graph, got #{inspect(graph, limit: 3)}")

    unless gwk == nil or match?(%Graph{}, gwk),
      do: raise(ArgumentError, "gwk must be a Spatex.Graph")
  end

  @defaults [
    yend: nil,
    q: nil,
    w_lags: 1,
    lag_q: true,
    robust: :white,
    gwk: nil,
    sig2n_k: false,
    spat_diag: true,
    spat_impacts: :simple,
    constant_regi: :many,
    cols2regi: :all,
    regime_lag_sep: false,
    regime_err_sep: false,
    hard_bound: false,
    spreg_compatible: false,
    name_x: nil,
    name_y: "dep_var",
    name_yend: nil,
    name_q: nil
  ]

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)}"
    end

    opts = Keyword.merge(@defaults, opts)
    check_choices!(opts)

    for key <- @booleans,
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    opts
  end

  defp check_choices!(opts) do
    unless opts[:constant_regi] in [:many, :one],
      do:
        raise(
          ArgumentError,
          ":constant_regi must be :many or :one, got #{inspect(opts[:constant_regi])}"
        )

    unless opts[:robust] in [nil, :white, :hac],
      do:
        raise(
          ArgumentError,
          ":robust must be nil, :white or :hac, got #{inspect(opts[:robust])}"
        )

    if opts[:robust] == :hac and opts[:gwk] == nil,
      do: raise(ArgumentError, "robust: :hac needs :gwk, a kernel Spatex.Graph")

    unless Map.has_key?(@impacts, opts[:spat_impacts]),
      do:
        raise(
          ArgumentError,
          "spat_impacts must be :simple, :full, :power, :all or nil, got #{inspect(opts[:spat_impacts])}"
        )

    unless is_integer(opts[:w_lags]) and opts[:w_lags] >= 1,
      do: raise(ArgumentError, "w_lags must be a positive integer, got #{inspect(opts[:w_lags])}")
  end
end
