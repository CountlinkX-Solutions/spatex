defmodule Spatex.Regression.SURerrorML do
  @moduledoc """
  Result of `Spatex.Regression.sur_error_ml/4` (`spreg.SURerrorML`): seemingly unrelated
  regressions with a spatial error process in every equation, estimated by maximum likelihood.

  Per-equation fields are lists with one entry per equation, in order.

    * `n`, `n_eq`, `k` - observations, equations and, per equation, the number of coefficients.
    * `lam` - the ML spatial error parameter of every equation (spreg's `lamsur`); `lam_ols` -
      the equation-by-equation ML error estimates (`lamols`) that start the search.
    * `b_sur0`, `llik` - the coefficients and log-likelihood (constant included) of the classic
      iterated SUR (`spreg.SUR` with `iter: true`) that provides the starting residuals.
    * `b_sur` - the ML SUR coefficients on the spatially filtered variables (the constant first,
      `"Constant_<eq>"`); `varb` - covariance of all coefficients stacked; `sur_inf` - per
      equation, one `{std_err, z, p}` per coefficient.
    * `sig`, `corr` - the cross-equation covariance of the filtered residuals and its correlation.
    * `clik_err`, `clik_sur_err` - concentrated log-likelihoods (no constant) of the equation by
      equation ML error models and of the ML SUR error model; `llik_err`, `llik_sur_err` - the same
      with the constant `-n/2 (n_eq (1 + ln 2 pi))` added.
    * `lr_test` - `{stat, df, p}` likelihood-ratio test of a diagonal `sig` (`nonspat_diag: true`,
      the default); `lr_lambda` - `{stat, df, p}` likelihood-ratio test of the spatial error
      parameters against the classic SUR (`spat_diag: true`); `nil` otherwise.
    * `vm` - asymptotic covariance of the lambdas and the upper-triangular elements of `sig`
      (`n_eq + n_eq (n_eq + 1) / 2` square, `vm: true`); `lam_setp` - per equation
      `{std_err, z, p}` of lambda; `lam_test` - `{stat, df, p}` Wald test that the lambdas are equal
      across equations; `join_lam` - Wald test that they are jointly zero (needs `vm: true` and
      `spat_diag: true`). `nil` when not requested.
    * `chow` - with the same number of coefficients in every equation, one `{stat, df, p}` Wald
      test per coefficient of its equality across equations; `nil` otherwise.
    * `residuals`, `predicted` - one list of `n` values per equation (`y - X b`, on the original,
      unfiltered variables).
    * `iterations` - the number of lambda searches of the estimation loop; `converged` - whether
      the loop stopped on its tolerance before the 2000 round cap.
    * `name_y`, `name_x` - names, per equation (`name_x` starts with `"Constant_<eq>"`).
    * `warnings`.
  """

  @enforce_keys [:n, :n_eq]
  defstruct [
    :n,
    :n_eq,
    :k,
    :lam,
    :lam_ols,
    :b_sur0,
    :llik,
    :b_sur,
    :varb,
    :sur_inf,
    :sig,
    :corr,
    :clik_err,
    :clik_sur_err,
    :llik_err,
    :llik_sur_err,
    :lr_test,
    :lr_lambda,
    :vm,
    :lam_setp,
    :lam_test,
    :join_lam,
    :chow,
    :residuals,
    :predicted,
    :iterations,
    :converged,
    :name_y,
    :name_x,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.SurErrorMl do
  @moduledoc false
  # `Spatex.Regression.sur_error_ml/4`: spreg's BaseSURerrorML. Starting lambdas from one ML
  # error fit per equation, starting residuals from the iterated SUR, then rounds of (SUR GLS on
  # the variables filtered with the current lambdas and sigma from the filtered previous
  # residuals; search of the lambdas that maximize the concentrated likelihood of the new
  # residuals) until the objective moves by at most `epsilon`.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{Regimes, Sur, SURerrorML}

  @allowed [
    :epsilon,
    :nonspat_diag,
    :spat_diag,
    :vm,
    :max_dense_n,
    :name_bigy,
    :name_bigx
  ]
  @max_rounds 2000

  # `path` replaces the searches by spreg's own lambdas (test hook): the equation-by-equation
  # start (`lamols`) followed by the lambdas each search returned, in order.
  @spec fit([term()], [term()], term(), keyword(), [[number()]] | nil) :: SURerrorML.t()
  def fit(ys, xs, graph, opts, path \\ nil) do
    opts = validate!(opts)

    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "the graph must be a Spatex.Graph")

    {eqs, n, names_y} = Sur.equations("sur_error_ml", ys, xs, opts[:name_bigy], opts[:name_bigx])

    unless graph.n == n,
      do: raise(ArgumentError, "the graph has #{graph.n} nodes but the equations have #{n} rows")

    estimate(%{eqs: eqs, ys: ys, xs: xs, n: n, graph: graph, opts: opts}, names_y, path)
  end

  defp estimate(%{eqs: eqs, n: n, graph: graph, opts: opts} = ctx, names_y, path) do
    n_eq = length(eqs)
    ks = Enum.map(eqs, &length(&1.x))
    {lam_ols, fun_start, path} = starts(ctx, path)
    reg0 = Sur.fit(ctx.ys, ctx.xs, iter: true, nonspat_diag: false)
    ctx = Map.merge(ctx, %{lags: lags(eqs, graph), ks: ks})
    loop = iterate(ctx, reg0.residuals, lam_ols, fun_start, path)
    {b_sur, varb, sig} = loop.est
    lam = loop.lam

    const = -(n / 2) * (n_eq * (1.0 + :math.log(2.0 * :math.pi())))
    llik_err = const - fun_start
    llik_sur = const - loop.fun
    vm = if opts[:vm], do: vm(graph, n, lam, sig, opts)

    %SURerrorML{
      n: n,
      n_eq: n_eq,
      k: ks,
      lam: lam,
      lam_ols: lam_ols,
      b_sur0: reg0.b_sur,
      llik: reg0.llik,
      b_sur: b_sur,
      varb: varb,
      sur_inf: Sur.inference(b_sur, varb),
      sig: sig,
      corr: Sur.correlation(sig),
      clik_err: -fun_start,
      clik_sur_err: -loop.fun,
      llik_err: llik_err,
      llik_sur_err: llik_sur,
      lr_test: if(opts[:nonspat_diag], do: lr_diag(n_eq, llik_sur, llik_err)),
      lr_lambda: if(opts[:spat_diag], do: lr_lambda(n_eq, llik_sur, reg0.llik)),
      vm: vm,
      lam_setp: vm && lam_setp(lam, vm),
      chow: Sur.chow(ks, b_sur, varb),
      residuals: loop.bige,
      predicted: for({e, b} <- Enum.zip(eqs, b_sur), do: Sur.mat_vec(Sur.transpose(e.x), b)),
      iterations: loop.count,
      converged: loop.count < @max_rounds,
      name_y: Enum.map(names_y, &to_string/1),
      name_x: Enum.map(eqs, & &1.names),
      warnings: Enum.flat_map(eqs, & &1.warnings) ++ round_warning(loop.count)
    }
    |> wald_tests(vm, n_eq)
  end

  defp round_warning(count) when count >= @max_rounds,
    do: ["the estimation loop reached #{@max_rounds} rounds"]

  defp round_warning(_count), do: []

  defp wald_tests(r, nil, _n_eq), do: r

  defp wald_tests(r, vm, n_eq) do
    joint = if r.lr_lambda, do: wald_lambda(r.lam, vm, n_eq, identity_rows(n_eq))
    %{r | lam_test: wald_lambda(r.lam, vm, n_eq, equality_rows(n_eq)), join_lam: joint}
  end

  # equation-by-equation ML error fits: the starting lambdas and their summed objective
  # (`path`, when given, supplies the lambdas and leaves the rest of the script)
  defp starts(ctx, path) do
    nf = ctx.n * 1.0
    opts = [epsilon: ctx.opts[:epsilon], max_dense_n: ctx.opts[:max_dense_n]]

    singles =
      for {y, x} <- Enum.zip(ctx.ys, ctx.xs) do
        fit = Regression.ml_error(y, x, ctx.graph, opts)
        {fit.lam, -(fit.logll + nf / 2 * :math.log(2 * :math.pi()) + nf / 2)}
      end

    fun = singles |> Enum.map(&elem(&1, 1)) |> Enum.sum()

    case path do
      [start | rest] -> {Enum.map(start, &(&1 * 1.0)), fun, rest}
      nil -> {Enum.map(singles, &elem(&1, 0)), fun, nil}
    end
  end

  # spreg's `bigylag` and `bigXlag` (the constant is lagged too)
  defp lags(eqs, graph),
    do: for(e <- eqs, do: %{y: Graph.lag(graph, e.y), x: Enum.map(e.x, &Graph.lag(graph, &1))})

  defp iterate(ctx, bige, lam, fun1, path) do
    round_(ctx, %{bige: bige, lam: lam, fun0: 0.0, fun: fun1, est: nil, count: 0, path: path})
  end

  # one round: SUR step with the current lambdas from the previous residuals, then the search
  defp round_(ctx, st) do
    if continue?(ctx, st) do
      est = sur_step(ctx, st.bige, st.lam)
      bige = Sur.residuals(ctx.eqs, elem(est, 0))
      {lam, fun, rest} = search(ctx, bige, st.lam, st.path)

      round_(ctx, %{
        st
        | bige: bige,
          lam: lam,
          fun0: st.fun,
          fun: fun,
          est: est,
          count: st.count + 1,
          path: rest
      })
    else
      st
    end
  end

  defp continue?(_ctx, %{count: count}) when count >= @max_rounds, do: false
  defp continue?(_ctx, %{path: path}) when is_list(path), do: path != []
  defp continue?(_ctx, %{est: nil}), do: true
  defp continue?(ctx, st), do: abs(st.fun0 - st.fun) > ctx.opts[:epsilon]

  defp search(ctx, bige, lam, nil) do
    r = native_search(ctx, bige, lam, true)
    {r.lam, r.fun, nil}
  end

  defp search(ctx, bige, _lam, [next | rest]) do
    lam2 = Enum.map(next, &(&1 * 1.0))
    r = native_search(ctx, bige, lam2, false)
    {lam2, r.fun, rest}
  end

  defp native_search(ctx, bige, lam, optimize) do
    {:ok, list} =
      bige
      |> List.flatten()
      |> Regimes.bin()
      |> Native.regression_sur_error_search(
        length(bige),
        ctx.graph.ref,
        Regimes.bin(lam),
        {optimize, ctx.opts[:max_dense_n]}
      )
      |> Regimes.ok!()

    by_name = Map.new(list, fn {k, v} -> {k, Input.from_f64_binary(v)} end)
    %{lam: by_name["lam"], fun: hd(by_name["fun"])}
  end

  # SUR GLS on the variables filtered with `lam`; sigma from the filtered previous residuals
  defp sur_step(ctx, bige, lam) do
    filtered =
      for {{e, l}, r} <- Enum.zip(Enum.zip(ctx.eqs, ctx.lags), lam) do
        %{y: filter(e.y, l.y, r), x: Enum.zip_with(e.x, l.x, &filter(&1, &2, r))}
      end

    {sxx, sxy} = Sur.cross_products(filtered)
    sp_e = for {u, r} <- Enum.zip(bige, lam), do: filter(u, Graph.lag(ctx.graph, u), r)
    Sur.sur_est(sxx, sxy, sp_e, ctx.ks, ctx.n)
  end

  defp filter(v, lag, lam), do: Enum.zip_with(v, lag, &(&1 - lam * &2))

  # -- tests -------------------------------------------------------------------------------

  defp lr_diag(n_eq, llik_sur, llik_err) do
    m = n_eq * (n_eq - 1) / 2
    stat = 2.0 * (llik_sur - llik_err)
    {stat, trunc(m), Sur.chi2_sf(stat, m)}
  end

  defp lr_lambda(n_eq, llik_sur, llik) do
    stat = 2.0 * (llik_sur - llik)
    {stat, n_eq, Sur.chi2_sf(stat, n_eq)}
  end

  # spreg's `surerrvm`: asymptotic covariance of the lambdas and the upper triangle of sigma
  defp vm(graph, n, lam, sig, opts) do
    n_eq = length(lam)
    sigi = Sur.inverse(sig)
    t = traces(graph, lam, opts[:max_dense_n])
    pairs = for i <- 0..(n_eq - 1), j <- i..(n_eq - 1), do: {i, j}

    sisj = Sur.hadamard(Sur.hadamard(sigi, sig), t.cross)
    vll = Enum.map(Enum.with_index(sisj), fn {row, i} -> add_diagonal(row, i, t.tr_dd) end)
    vlsig = for i <- 0..(n_eq - 1), do: Enum.map(pairs, &lam_sig(sigi, t.tr_d, i, &1))

    vsig =
      for {i, j} <- pairs, do: Enum.map(pairs, fn {h, k} -> n * sig_sig(sigi, i, j, h, k) end)

    top = Enum.zip_with(vll, vlsig, &(&1 ++ &2))
    bottom = Enum.zip_with(Sur.transpose(vlsig), vsig, &(&1 ++ &2))
    Sur.inverse(top ++ bottom)
  end

  defp add_diagonal(row, i, diag),
    do: List.update_at(row, i, &(&1 + Enum.at(diag, i)))

  # sigma^-1[i, other] * tr(D_i) in the column of the pair containing equation i
  defp lam_sig(sigi, tr_d, i, {a, b}) when i in [a, b] do
    other = if i == a, do: b, else: a
    Sur.at(sigi, i, other) * Enum.at(tr_d, i)
  end

  defp lam_sig(_sigi, _tr_d, _i, _pair), do: 0.0

  defp sig_sig(sigi, i, i, h, h), do: 0.5 * Sur.at(sigi, i, h) ** 2
  defp sig_sig(sigi, i, i, h, k), do: Sur.at(sigi, i, h) * Sur.at(sigi, i, k)
  defp sig_sig(sigi, i, j, h, h), do: Sur.at(sigi, i, h) * Sur.at(sigi, j, h)

  defp sig_sig(sigi, i, j, h, k),
    do: Sur.at(sigi, i, h) * Sur.at(sigi, j, k) + Sur.at(sigi, i, k) * Sur.at(sigi, j, h)

  defp traces(graph, lam, max_dense_n) do
    {:ok, list} =
      graph.ref
      |> Native.regression_sur_error_traces(Regimes.bin(lam), max_dense_n)
      |> Regimes.ok!()

    by_name = Map.new(list, fn {k, v} -> {k, Input.from_f64_binary(v)} end)

    %{
      tr_d: by_name["tr_d"],
      tr_dd: by_name["tr_dd"],
      cross: Enum.chunk_every(by_name["cross"], length(lam))
    }
  end

  defp lam_setp(lam, vm) do
    for {l, i} <- Enum.with_index(lam) do
      se = :math.sqrt(Sur.at(vm, i, i))
      {se, l / se, :math.erfc(abs(l / se) / :math.sqrt(2))}
    end
  end

  # R of spreg's `buildR(kr=1, kf=0, nr=n_eq)`: lambda 0 against each other lambda
  defp equality_rows(n_eq), do: Enum.map(1..(n_eq - 1), &unit_row(n_eq, [{0, 1.0}, {&1, -1.0}]))
  defp identity_rows(n_eq), do: Enum.map(0..(n_eq - 1), &unit_row(n_eq, [{&1, 1.0}]))

  defp unit_row(n, entries),
    do: Enum.map(0..(n - 1), fn c -> entries |> List.keyfind(c, 0, {c, 0.0}) |> elem(1) end)

  defp wald_lambda(lam, vm, n_eq, rows) do
    vlam = vm |> Enum.take(n_eq) |> Enum.map(&Enum.take(&1, n_eq))
    {stat, df} = Sur.wald(lam, vlam, rows)
    {stat, df, Sur.chi2_sf(stat, df)}
  end

  # -- options -----------------------------------------------------------------------------

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError,
              "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)} " <>
                "(regimes are not supported by sur_error_ml)"
    end

    opts =
      Keyword.merge(
        [epsilon: 1.0e-7, nonspat_diag: true, spat_diag: false, vm: false, max_dense_n: 5_000],
        opts
      )

    for key <- [:nonspat_diag, :spat_diag, :vm],
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    unless is_number(opts[:epsilon]) and opts[:epsilon] > 0,
      do: raise(ArgumentError, "epsilon must be a positive number")

    unless is_integer(opts[:max_dense_n]) and opts[:max_dense_n] > 0,
      do: raise(ArgumentError, "max_dense_n must be a positive integer")

    opts
  end
end
