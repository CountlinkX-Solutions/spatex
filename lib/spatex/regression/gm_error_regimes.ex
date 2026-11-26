defmodule Spatex.Regression.GMErrorRegimes do
  @moduledoc """
  Result of `Spatex.Regression.gm_error_regimes/5` (`spreg.GM_Error_Regimes` and, with `:yend`,
  `spreg.GM_Endog_Error_Regimes`; the `_Het_` and `_Hom_` classes with `estimator: :het` /
  `:hom`).

    * `betas`, `vm`, `std_err`, `z_stat` (`[{z, p}]`) - in `:pooled` mode spreg's layout: the
      regime-specific exogenous columns of every regime in turn (`"<regime>_<name>"`), the global
      ones (`"_Global_<name>"`), the same for the endogenous columns, then `lambda` last. `vm`,
      `std_err` and `z_stat` have no entry for `lambda` (KP98 reports none). In `:separate` mode
      every regime's `[constant, x..., yend..., lambda]` block follows the previous one and `vm`
      is block diagonal. With `:het` / `:hom` the covariance, `std_err` and `z_stat` also have
      the `lambda` entry (`:separate`: one per block), and `kr` counts `lambda` in `:separate`
      mode, as spreg does.
    * `names` - the coefficient names in `betas` order (`lambda` included); `name_x`,
      `name_yend`, `name_q` - names as spreg lists them (`name_x` ends with `"lambda"` for a
      model without `:yend`, as spreg's does).
    * `kr`, `kf`, `kryd`, `nr` - regime-specific columns per regime (exogenous plus endogenous),
      global columns (`lambda` counted, as spreg does), regime-specific endogenous columns and
      regimes; `regimes` - the labels in order.
    * `lam` - the spatial error coefficient (`:pooled`); `lams` - `:separate`, a map by regime
      label.
    * `sig2`, `pr2` - error variance (`u'u / n` of the filtered fit; `nil` for `:het`, which
      reports none) and squared correlation of `y` with `predy` (`:pooled`; spreg reports none for
      the regimes of `:separate`, which live in the per-regime fits).
    * `estimator`, `iteration`, `iter_stop`, `step1c` (`:het`), `max_iter`, `epsilon`, `a1`
      (`:hom`) and `lambdas` (every lambda the moment optimizer returned, in call order; a map
      by regime in `:separate` mode) - `nil` for `:kp98` (`:pooled`; in `:separate` mode the
      iterations live in the per-regime fits).
    * `u`, `predy` (`Z betas`), `e_filtered` (`u - lambda W u`).
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      (`lambda` excluded, except in the `:separate` Het / Hom fits, where spreg tests it too) are
      equal across regimes, jointly and per regime-specific column.
    * `mode` - `:separate` (one `Spatex.Regression.GMError` per regime on the regime's subgraph,
      in `multi` by label) or `:pooled` (one KP98 fit on the regime design).
    * `cols2regi` - the flags over the `x` then `yend` columns; `constant_regi`,
      `regime_err_sep`, `warnings`.
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
    :lam,
    :lams,
    :sig2,
    :pr2,
    :u,
    :predy,
    :e_filtered,
    :chow,
    :mode,
    :multi,
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :names,
    :name_x,
    :name_yend,
    :name_q,
    :name_y,
    :iteration,
    :iter_stop,
    :step1c,
    :max_iter,
    :epsilon,
    :a1,
    :lambdas,
    estimator: :kp98,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.GMComboRegimes do
  @moduledoc """
  Result of `Spatex.Regression.gm_combo_regimes/5` (`spreg.GM_Combo_Regimes`, and the `_Het_` /
  `_Hom_` classes with `estimator: :het` / `:hom`).

  As `Spatex.Regression.GMErrorRegimes`, with `W y` among the endogenous columns:

    * `betas` - the `rho` coefficient (`"_Global_W_<y>"`, or one `"<regime>_W_<y>"` per regime
      with `regime_lag_sep`) follows the endogenous columns, `lambda` is last. `name_yend` ends
      with the `W y` name(s), `name_q` includes the spatially lagged instruments.
    * `rho` - the `W y` coefficient (`:pooled`; `nil` in `:separate` mode); `rhos` - with
      `regime_lag_sep`, a map by regime label, otherwise `nil`. `lam`, `lams` as in the error
      model.
    * `predy_e`, `e_pred` - reduced-form prediction `(I - rho W)^-1 X beta` and its residual
      (zeros for a regime whose `|rho| >= 1`); `pr2_e` - squared correlation of `y` with `predy_e`
      (`:pooled`; `nil` when `|rho| >= 1`).
    * `regime_lag_sep` - one coefficient of `W y` per regime, every regime fitted on its own
      subgraph (`:separate`, `multi` holds the `Spatex.Regression.GMCombo` fits). spreg ties
      `regime_err_sep` to it: either one switches the other on, and with
      `regime_lag_sep: false` `regime_err_sep` is switched off with a warning.
    * `w_lags`, `lag_q`, `warnings`; `estimator`, `iteration`, `iter_stop`, `step1c`, `max_iter`,
      `epsilon`, `a1`, `lambdas` as in `Spatex.Regression.GMErrorRegimes`.
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
    :lam,
    :lams,
    :rho,
    :rhos,
    :sig2,
    :pr2,
    :pr2_e,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :e_filtered,
    :chow,
    :mode,
    :multi,
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :regime_lag_sep,
    :w_lags,
    :lag_q,
    :names,
    :name_x,
    :name_yend,
    :name_q,
    :name_y,
    :iteration,
    :iter_stop,
    :step1c,
    :max_iter,
    :epsilon,
    :a1,
    :lambdas,
    estimator: :kp98,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.RegimesError do
  @moduledoc false
  # `Spatex.Regression.gm_error_regimes/5` and `gm_combo_regimes/5`: spreg's regimes models
  # (`GM_Error_Regimes`, `GM_Endog_Error_Regimes`, `GM_Combo_Regimes` and their Het / Hom
  # classes) on top of the regime design of `Spatex.Regression.Regimes`. KP98 pooled fits use
  # three native pieces (lambda from first-stage residuals, the spatial filter, 2SLS with an
  # explicit instrument matrix); Het and Hom pooled fits call the base native estimator with
  # the regime design (`regimes:` setting); the separate regressions are `gm_error/4` and
  # `gm_combo/4` on the regime subgraphs for every estimator.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{GM, GMComboRegimes, GMErrorRegimes, Regimes}

  @common [
    :yend,
    :q,
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :estimator,
    :max_iter,
    :epsilon,
    :step1c,
    :a1,
    :spreg_compatible,
    :name_x,
    :name_y,
    :name_yend,
    :name_q
  ]
  @combo [:w_lags, :lag_q, :regime_lag_sep]
  @booleans [:regime_err_sep, :regime_lag_sep, :lag_q, :spreg_compatible, :step1c]
  @iteration_opts [:max_iter, :epsilon, :step1c, :a1]
  @err_sep_all "All coefficients must vary across regimes if regime_err_sep = True, so setting cols2regi = 'all'."
  @degenerate "The moment equations have a zero scale (spreg's optimizer fails here); lambda was set to 0."

  # `at`: spreg's lambdas to consume instead of the optimizer (Het and Hom only; internal test
  # hook): a list for the pooled fit, one list per regime (in regime order) for the separate fits.
  @spec fit(:error | :combo, term(), term(), [term()], term(), keyword(), [term()] | nil) ::
          struct()
  def fit(kind, y, x, regimes, graph, opts, at \\ nil) do
    opts = validate!(kind, opts)

    if at != nil and opts[:estimator] == :kp98,
      do: raise(ArgumentError, "lambdas can only be given for the :het and :hom estimators")

    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "graph must be a Spatex.Graph, got #{inspect(graph, limit: 3)}")

    {b, notes} = kind |> prepare(y, x, regimes, graph, opts) |> reconcile()
    b = %{b | warnings: b.warnings ++ notes, at: at}

    cond do
      not b.opts[:regime_err_sep] ->
        pooled(b)

      Enum.all?(b.flags) ->
        separate(b)

      true ->
        raise ArgumentError, "All coefficients must vary across regimes if regime_err_sep = true."
    end
  end

  # -- input -----------------------------------------------------------------------------------

  defp prepare(kind, y, x, regimes, graph, opts) do
    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: opts[:name_x])

    if graph.n != n, do: raise(ArgumentError, "graph has #{graph.n} nodes but y has #{n} rows")
    regimes = Input.to_labels(regimes, name: "regimes", length: n)
    {yend_names, yend_bin, ky} = GM.extra(opts[:yend], n, opts[:name_yend], "yend")
    {q_names, q_bin, kq} = GM.extra(opts[:q], n, opts[:name_q], "q")
    if ky == 0 and kq > 0, do: raise(ArgumentError, "q needs a matching yend")
    if ky > kq, do: raise(ArgumentError, "q needs at least as many columns as yend")

    xcols = x_bin |> Regimes.bin_cols(n) |> tl()
    names = tl(names)

    %{
      kind: kind,
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
      set: Regimes.check_regimes!(regimes, n, length(names)),
      graph: graph,
      name_y: to_string(opts[:name_y]),
      warnings: warnings,
      opts: opts,
      at: nil
    }
  end

  # spreg's __init__ reconciliation of `regime_err_sep` and `regime_lag_sep` (combo only), then
  # the flags of every coefficient: the constant, `x`, `yend` and, for the combo, `W y`.
  defp reconcile(b) do
    {opts, notes} = if b.kind == :combo, do: lag_err_sep(b.opts), else: {b.opts, []}

    # spreg tests `col != True` over `cols2regi`, which also holds for the string 'all'.
    given = opts[:cols2regi]

    {given, notes} =
      if b.kind == :combo and opts[:regime_err_sep] and
           not (is_list(given) and Enum.all?(given)),
         do: {:all, notes ++ [@err_sep_all]},
         else: {given, notes}

    user = Regimes.cols2regi!(given, length(b.names) + b.ky)
    lag_flag = if b.kind == :combo, do: [opts[:regime_lag_sep]], else: []
    flags = [opts[:constant_regi] == :many] ++ user ++ lag_flag
    {Map.merge(b, %{opts: opts, cols2regi: user, flags: flags}), notes}
  end

  defp lag_err_sep(opts) do
    case {opts[:regime_lag_sep], opts[:regime_err_sep]} do
      {true, false} ->
        {Keyword.put(opts, :regime_err_sep, true),
         ["regime_err_sep set to True when regime_lag_sep=True."]}

      {false, true} ->
        {Keyword.put(opts, :regime_err_sep, false),
         ["regime_err_sep set to False when regime_lag_sep=False."]}

      _ ->
        {opts, []}
    end
  end

  # -- one fit on the regime design --------------------------------------------------------------

  defp pooled(b) do
    {ycols, qcols} = endogenous(b)
    names = %{y: yend_names(b), q: q_names(b)}
    spec = %{x: [List.duplicate(1.0, b.n) | b.xcols], y: ycols, q: qcols}
    first = design(b, spec, names)
    est = if b.opts[:estimator] == :kp98, do: kp98(b, spec, first, names), else: gmm(b, spec)
    nr = length(b.set)

    chow =
      Regimes.chow(
        est.coefficients,
        trim(est.vm, length(est.coefficients)),
        first.layout,
        nr,
        first.regi_names,
        b.opts[:spreg_compatible]
      )

    {krx, kfx, kry, kfy} = first.layout

    common =
      [
        n: b.n,
        kr: krx + kry,
        kf: kfx + kfy + 1,
        kryd: kry,
        nr: nr,
        regimes: b.set,
        betas: est.coefficients ++ [est.lam],
        vm: est.vm,
        std_err: est.std_err,
        z_stat: est.z_stat,
        lam: est.lam,
        sig2: est.sig2,
        pr2: est.pr2,
        u: est.u,
        predy: est.predy,
        e_filtered: est.e_filtered,
        chow: chow,
        mode: :pooled,
        constant_regi: b.opts[:constant_regi],
        cols2regi: b.cols2regi,
        regime_err_sep: false,
        names: first.xnames ++ first.ynames ++ ["lambda"],
        name_y: b.name_y,
        warnings: b.warnings ++ est.warnings
      ] ++ est.iteration

    pooled_result(b, common, first, ycols)
  end

  defp trim(vm, k), do: vm |> Enum.take(k) |> Enum.map(&Enum.take(&1, k))

  # KP98: first stage on the unfiltered regime design (OLS, or 2SLS with H = [X | q]), lambda from
  # its residuals, then the second stage on the filtered base columns put on the regime design.
  # `vm`, `std_err` and `z_stat` have no entry for lambda.
  defp kp98(b, spec, first, names) do
    ycols = spec.y
    h = first.x ++ first.q
    u0 = tsls_h(b.y, first.x, first.y, if(ycols == [], do: first.x, else: h)).u
    {lam, degenerate?} = lambda(u0, b.graph)

    xs = filter(b, spec.x, lam)
    [ys] = filter(b, [b.y], lam)
    second = design(b, %{spec | x: xs, y: filter(b, ycols, lam)}, names)

    fit2 =
      if ycols == [],
        do: tsls_h(ys, second.x, [], second.x),
        else: tsls_h(ys, second.x, second.y, h)

    z = first.x ++ first.y
    predy = z |> Enum.zip_with(& &1) |> Enum.map(&Regimes.dot(&1, fit2.betas))
    u = Enum.zip_with(b.y, predy, &(&1 - &2))

    %{
      coefficients: fit2.betas,
      lam: lam,
      vm: fit2.vm,
      std_err: fit2.std_err,
      z_stat: fit2.z_stat,
      sig2: fit2.sig2,
      pr2: pearson2(b.y, predy),
      u: u,
      predy: predy,
      e_filtered: e_filtered(b.graph, u, lam),
      iteration: [estimator: :kp98],
      warnings: if(degenerate?, do: [@degenerate], else: [])
    }
  end

  # Het and Hom: the native estimator on the base columns with the regime design (spreg's
  # `Regimes_Frame`) built inside, so its second stage filters the base columns and then
  # expands them. `vm`, `std_err` and `z_stat` include lambda.
  defp gmm(b, spec) do
    {x_flags, y_flags} = Enum.split(b.flags, length(b.names) + 1)
    pos = b.set |> Enum.with_index() |> Map.new()
    o = b.opts

    settings = %{
      estimator: if(o[:estimator] == :het, do: 1, else: 2),
      max_iter: o[:max_iter],
      epsilon: o[:epsilon] * 1.0,
      step1c: o[:step1c],
      a1: Map.fetch!(%{hom_sc: 0, hom: 1, het: 2}, o[:a1]),
      hard_bound: false,
      combo: false,
      w_lags: 1,
      lag_q: true,
      skip_const: true,
      at: b.at && Enum.map(b.at, &(&1 * 1.0)),
      regimes: %{
        index: Enum.map(b.regimes, &Map.fetch!(pos, &1)),
        nr: length(b.set),
        x_flags: x_flags,
        y_flags: Enum.take(y_flags, length(spec.y))
      }
    }

    e =
      b.y_bin
      |> Native.regression_gm_error(
        block(spec.x),
        block(spec.y),
        block(spec.q),
        b.graph.ref,
        settings
      )
      |> GM.entries()

    k = length(e.betas)

    %{
      coefficients: Enum.drop(e.betas, -1),
      lam: hd(e.lam),
      vm: GM.rows(e.vm, k),
      std_err: e.std_err,
      z_stat: GM.pairs(e.z_stat),
      sig2: e[:sig2] && hd(e.sig2),
      pr2: hd(e.pr2),
      u: e.u,
      predy: e.predy,
      e_filtered: e.e_filtered,
      iteration: [
        estimator: o[:estimator],
        iteration: round(hd(e.iteration)),
        iter_stop: GM.iter_stop(e),
        step1c: if(o[:estimator] == :het, do: o[:step1c]),
        max_iter: o[:max_iter],
        epsilon: o[:epsilon] * 1.0,
        a1: if(o[:estimator] == :hom, do: o[:a1]),
        lambdas: e.lambdas
      ],
      warnings: GM.warnings(e)
    }
  end

  defp pooled_result(%{kind: :error, ky: 0}, common, first, _ycols),
    do: struct!(GMErrorRegimes, [name_x: first.xnames ++ ["lambda"]] ++ common)

  defp pooled_result(%{kind: :error}, common, first, _ycols) do
    struct!(
      GMErrorRegimes,
      [name_x: first.xnames, name_yend: first.ynames, name_q: first.qnames] ++ common
    )
  end

  defp pooled_result(%{kind: :combo} = b, common, first, ycols) do
    rho = Enum.at(common[:betas], -2)

    {predy_e, e_pred, warn} =
      Regimes.prediction(b.graph, b.y, common[:predy], List.last(ycols), rho, false)

    struct!(
      GMComboRegimes,
      [
        name_x: first.xnames,
        name_yend: first.ynames,
        name_q: first.qnames,
        rho: rho,
        predy_e: predy_e,
        e_pred: e_pred,
        pr2_e: if(warn == [], do: pearson2(b.y, predy_e)),
        regime_lag_sep: false,
        w_lags: b.opts[:w_lags],
        lag_q: b.opts[:lag_q]
      ] ++ Keyword.update!(common, :warnings, &(&1 ++ warn))
    )
  end

  # The endogenous columns and instruments; the combo adds `W y` (last) and the spatial
  # instruments, built on the whole graph.
  defp endogenous(%{kind: :error} = b), do: {b.ycols, b.qcols}
  defp endogenous(%{kind: :combo} = b), do: Regimes.lag_endog(b, b.graph)

  defp yend_names(%{kind: :error} = b), do: b.yend_names
  defp yend_names(%{kind: :combo} = b), do: b.yend_names ++ ["W_" <> b.name_y]

  defp q_names(%{kind: :error} = b), do: b.q_names

  defp q_names(%{kind: :combo} = b) do
    lagged = if b.opts[:lag_q], do: b.q_names, else: []
    b.q_names ++ Regimes.lag_names(b.names ++ lagged, b.opts[:w_lags])
  end

  # The regime design (`Regimes_Frame`) of the columns `x`, `y` and `q` (instruments always by
  # regime): the regime-specific columns of every regime, then the global ones. Returns the
  # columns, their names and the layout `{krx, kfx, kry, kfy}`.
  defp design(b, spec, names) do
    n_x = length(b.names) + 1
    {x_flags, y_flags} = Enum.split(b.flags, n_x)
    y_flags = Enum.take(y_flags, length(spec.y))
    all = List.duplicate(true, length(spec.q))

    {xd, xnames, krx, kfx, x_regi} =
      Regimes.regimize(
        Enum.zip_with([["CONSTANT" | b.names], spec.x, x_flags], &List.to_tuple/1),
        b.regimes,
        b.set
      )

    {yd, ynames, kry, kfy, y_regi} =
      Regimes.regimize(
        Enum.zip_with([names.y, spec.y, y_flags], &List.to_tuple/1),
        b.regimes,
        b.set
      )

    {qd, qnames, _, _, _} =
      Regimes.regimize(Enum.zip_with([names.q, spec.q, all], &List.to_tuple/1), b.regimes, b.set)

    if krx + kry == 0, do: raise(ArgumentError, "at least one column must vary by regime")

    %{
      x: xd,
      y: yd,
      q: qd,
      xnames: xnames,
      ynames: ynames,
      qnames: qnames,
      layout: {krx, kfx, kry, kfy},
      regi_names: x_regi ++ y_regi
    }
  end

  # -- native pieces -----------------------------------------------------------------------------

  defp tsls_h(y, x, yend, h) do
    {:ok, list} =
      Native.regression_tsls_h(
        Regimes.bin(y),
        block(x),
        block(yend),
        block(h)
      )
      |> Regimes.ok!()

    e = Map.new(list, fn {k, v} -> {String.to_existing_atom(k), Input.from_f64_binary(v)} end)
    k = length(e.betas)

    %{
      e
      | sig2: hd(e.sig2),
        vm: Enum.chunk_every(e.vm, k),
        z_stat: e.z_stat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)
    }
  end

  defp block(cols), do: {IO.iodata_to_binary(Enum.map(cols, &Regimes.bin/1)), length(cols)}

  defp lambda(u, graph) do
    {:ok, list} = u |> Regimes.bin() |> Native.regression_error_lambda(graph.ref) |> Regimes.ok!()
    by_name = Map.new(list, fn {k, v} -> {k, Input.from_f64_binary(v)} end)
    {hd(by_name["lam"]), hd(by_name["degenerate"]) != 0.0}
  end

  defp filter(_b, [], _lam), do: []

  defp filter(b, cols, lam) do
    {:ok, out} =
      cols
      |> block()
      |> Native.regression_sp_filter(b.graph.ref, lam * 1.0)
      |> Regimes.ok!()

    Regimes.bin_cols(out, b.n)
  end

  defp e_filtered(graph, u, lam) do
    Enum.zip_with(u, Graph.lag(graph, u), &(&1 - lam * &2))
  end

  defp pearson2(a, b) do
    n = length(a)
    {ma, mb} = {Enum.sum(a) / n, Enum.sum(b) / n}
    {da, db} = {Enum.map(a, &(&1 - ma)), Enum.map(b, &(&1 - mb))}
    r = Regimes.dot(da, db) / :math.sqrt(Regimes.dot(da, da) * Regimes.dot(db, db))
    r * r
  end

  # -- one model per regime ----------------------------------------------------------------------

  defp separate(b) do
    subs = Regimes.regime_graphs(b.graph, b.regimes, b.set, 0)
    ids = Map.new(b.set, fn r -> {r, Regimes.indices(b.regimes, r)} end)

    # spreg's check_const_regi: a column constant within any regime is dropped everywhere.
    keep =
      for {name, col} <- Enum.zip(b.names, b.xcols),
          Enum.all?(b.set, &Regimes.varies?(Regimes.take(col, ids[&1]))),
          do: {name, col}

    dropped = b.names -- Enum.map(keep, &elem(&1, 0))
    ats = if b.at, do: Map.new(Enum.zip(b.set, b.at)), else: %{}
    fits = Map.new(b.set, fn r -> {r, regime_fit(b, ids[r], keep, subs[r], ats[r])} end)
    fit_list = Enum.map(b.set, &fits[&1])
    nr = length(b.set)
    combo? = b.kind == :combo

    x_names = ["CONSTANT" | Enum.map(keep, &elem(&1, 0))]
    y_names = yend_names(b)
    q_all = q_names(b)
    kr = length(x_names) + length(y_names)
    prefix = fn r, list -> Enum.map(list, &"#{r}_#{&1}") end
    z_names = Enum.flat_map(b.set, &prefix.(&1, x_names ++ y_names ++ ["lambda"]))

    betas = Enum.flat_map(fit_list, & &1.betas)
    vm = Regimes.block_diagonal(Enum.map(fit_list, & &1.vm))

    # KP98's covariance has no entry for lambda; Het and Hom count it as one more coefficient
    # of every block (spreg's `kr`) and test it across regimes with the others.
    {coefficients, kr, chow_names} =
      if b.opts[:estimator] == :kp98,
        do: {Enum.flat_map(fit_list, &Enum.drop(&1.betas, -1)), kr, x_names ++ y_names},
        else: {betas, kr + 1, x_names ++ y_names ++ ["lambda"]}

    common = [
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
      lams: Map.new(b.set, &{&1, fits[&1].lam}),
      u: Regimes.scatter(b.n, ids, b.set, &fits[&1].u),
      predy: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy),
      e_filtered: Regimes.scatter(b.n, ids, b.set, &fits[&1].e_filtered),
      chow: Regimes.chow(coefficients, vm, {kr, 0, 0, 0}, nr, chow_names),
      estimator: b.opts[:estimator],
      lambdas: b.opts[:estimator] != :kp98 && Map.new(b.set, &{&1, fits[&1].lambdas}),
      mode: :separate,
      multi: fits,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      regime_err_sep: true,
      names: z_names,
      name_y: b.name_y,
      warnings: b.warnings ++ dropped_warning(dropped) ++ Regimes.island_warning(b.set, subs)
    ]

    names = [
      name_x: Enum.flat_map(b.set, &prefix.(&1, x_names)),
      name_yend: Enum.flat_map(b.set, &prefix.(&1, y_names)),
      name_q: Enum.flat_map(b.set, &prefix.(&1, q_all))
    ]

    if combo? do
      struct!(
        GMComboRegimes,
        names ++
          [
            rhos: Map.new(b.set, &{&1, fits[&1].rho}),
            predy_e: Regimes.scatter(b.n, ids, b.set, &fits[&1].predy_e),
            e_pred: Regimes.scatter(b.n, ids, b.set, &fits[&1].e_pred),
            regime_lag_sep: true,
            w_lags: b.opts[:w_lags],
            lag_q: b.opts[:lag_q]
          ] ++ common
      )
    else
      error_names = if b.ky == 0, do: [name_x: z_names, name_yend: nil, name_q: nil], else: names
      struct!(GMErrorRegimes, error_names ++ common)
    end
  end

  defp dropped_warning([]), do: []

  defp dropped_warning(names),
    do: [
      "Variable(s) #{inspect(names)} removed for being constant for at least one of the regimes."
    ]

  defp regime_fit(b, rows, keep, sub, at) do
    pick = fn cols -> Enum.map(cols, &Regimes.take(&1, rows)) end
    extra = fn cols -> if cols == [], do: nil, else: pick.(cols) end

    opts =
      [
        yend: extra.(b.ycols),
        q: extra.(b.qcols),
        name_x: Enum.map(keep, &elem(&1, 0)),
        name_y: b.name_y,
        name_yend: if(b.ky == 0, do: nil, else: b.yend_names),
        name_q: if(b.kq == 0, do: nil, else: b.q_names)
      ]

    y = Regimes.take(b.y, rows)
    x = pick.(Enum.map(keep, &elem(&1, 1)))
    opts = opts ++ estimator_opts(b.opts)

    case b.kind do
      :error -> GM.error(y, x, sub, opts, at)
      :combo -> GM.combo(y, x, sub, opts ++ [w_lags: b.opts[:w_lags], lag_q: b.opts[:lag_q]], at)
    end
  end

  # The options the base models take for the Het and Hom estimators.
  defp estimator_opts(%{estimator: :kp98}), do: []

  defp estimator_opts(o),
    do: [
      estimator: o[:estimator],
      max_iter: o[:max_iter],
      epsilon: o[:epsilon],
      step1c: o[:step1c],
      a1: o[:a1]
    ]

  # -- options -----------------------------------------------------------------------------------

  @defaults [
    yend: nil,
    q: nil,
    constant_regi: :many,
    cols2regi: :all,
    regime_err_sep: false,
    estimator: :kp98,
    max_iter: 1,
    epsilon: 1.0e-5,
    step1c: false,
    # spreg's regimes classes default to A1 = 'het' (the base GM_Error_Hom to 'hom_sc')
    a1: :het,
    spreg_compatible: false,
    name_x: nil,
    name_y: "dep_var",
    name_yend: nil,
    name_q: nil,
    w_lags: 1,
    lag_q: true,
    regime_lag_sep: false
  ]

  defp validate!(kind, opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    allowed = if kind == :combo, do: @common ++ @combo, else: @common

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    merged = Keyword.merge(@defaults, opts)
    check_estimator!(merged[:estimator], Keyword.keys(opts))
    check_choices!(merged)
    opts = merged

    for key <- @booleans,
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    opts
  end

  defp check_estimator!(:kp98, given) do
    if Enum.any?(given, &(&1 in @iteration_opts)),
      do:
        raise(
          ArgumentError,
          "max_iter, epsilon, step1c and a1 only apply to the :het and :hom estimators"
        )
  end

  defp check_estimator!(:het, given) do
    if :a1 in given, do: raise(ArgumentError, "a1 is an option of the :hom estimator only")
  end

  defp check_estimator!(:hom, given) do
    if :step1c in given,
      do: raise(ArgumentError, "step1c is an option of the :het estimator only")
  end

  defp check_estimator!(other, _given),
    do: raise(ArgumentError, ":estimator must be :kp98, :het or :hom, got #{inspect(other)}")

  defp check_choices!(opts) do
    unless opts[:constant_regi] in [:many, :one],
      do:
        raise(
          ArgumentError,
          ":constant_regi must be :many or :one, got #{inspect(opts[:constant_regi])}"
        )

    unless is_integer(opts[:max_iter]) and opts[:max_iter] >= 1,
      do:
        raise(
          ArgumentError,
          "max_iter must be a positive integer, got #{inspect(opts[:max_iter])}"
        )

    unless is_number(opts[:epsilon]) and opts[:epsilon] >= 0,
      do: raise(ArgumentError, "epsilon must be a non-negative number")

    unless opts[:a1] in [:hom_sc, :hom, :het],
      do: raise(ArgumentError, "a1 must be :hom_sc, :hom or :het, got #{inspect(opts[:a1])}")

    unless is_integer(opts[:w_lags]) and opts[:w_lags] >= 1,
      do: raise(ArgumentError, "w_lags must be a positive integer, got #{inspect(opts[:w_lags])}")
  end
end
