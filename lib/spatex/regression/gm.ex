defmodule Spatex.Regression.GM do
  @moduledoc false
  # Option handling and result assembly for `Spatex.Regression.tsls/5`, `gm_lag/4`, `gm_error/4`
  # and `gm_combo/4`.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{GMCombo, GMError, GMLag, ML, TSLS}

  @robust %{nil => 0, white: 1, hac: 2}
  @impacts %{nil => 0, simple: 1, full: 2, power: 3, all: 4}
  @estimators %{kp98: 0, het: 1, hom: 2}
  @a1s %{hom_sc: 0, hom: 1, het: 2}

  @base [:constant, :name_x, :name_y, :name_yend, :name_q]
  @tsls_opts @base ++ [:robust, :gwk, :sig2n_k]
  @lag_opts @base ++
              [
                :yend,
                :q,
                :w_lags,
                :lag_q,
                :robust,
                :gwk,
                :sig2n_k,
                :spat_diag,
                :spat_impacts,
                :hard_bound,
                :slx_lags,
                :slx_vars
              ]
  @gmm_opts [:yend, :q, :estimator, :max_iter, :epsilon, :step1c, :a1, :hard_bound]
  @error_opts @base ++ @gmm_opts ++ [:slx_lags, :slx_vars]
  @combo_opts @base ++ @gmm_opts ++ [:w_lags, :lag_q]

  @keys ~w(betas lam rho vm varb std_err z_stat sig2 utu u predy predy_e e_pred e_filtered pr2
           pr2_e ak_test mult_simple mult_full mult_power power_terms iteration iter_stop_max
           lambdas warn_lam warn_rho warn_degenerate warn_rho_near)a

  @defaults [
    slx_lags: 0,
    slx_vars: :all,
    constant: true,
    name_x: nil,
    name_y: "dep_var",
    name_yend: nil,
    name_q: nil,
    yend: nil,
    q: nil,
    robust: nil,
    gwk: nil,
    sig2n_k: false,
    w_lags: 1,
    lag_q: true,
    spat_diag: true,
    spat_impacts: :simple,
    hard_bound: false,
    estimator: :kp98,
    max_iter: 1,
    epsilon: 1.0e-5,
    step1c: false,
    a1: :hom_sc
  ]

  # -- tsls ----------------------------------------------------------------------------

  @spec tsls(term(), term(), term(), term(), keyword()) :: TSLS.t()
  def tsls(y, x, yend, q, opts) do
    opts = validate!(opts, @tsls_opts)
    opts = Keyword.merge(opts, yend: yend, q: q)
    {p, opts} = prepare(y, x, nil, opts, false)

    if p.kyend == 0, do: raise(ArgumentError, "yend has no columns")
    if p.kq == 0, do: raise(ArgumentError, "q has no columns")
    if opts[:robust] == :hac and opts[:gwk] == nil, do: hac_error()

    e =
      Native.regression_tsls(
        p.y,
        {p.x, p.kx},
        {p.yend, p.kyend},
        {p.q, p.kq},
        Map.fetch!(@robust, opts[:robust]),
        opts[:sig2n_k],
        opts[:gwk] && opts[:gwk].ref
      )
      |> entries()

    k = p.kx + p.kyend

    %TSLS{
      n: p.n,
      k: k,
      betas: e.betas,
      vm: rows(e.vm, k),
      varb: rows(e.varb, k),
      std_err: e.std_err,
      z_stat: pairs(e.z_stat),
      sig2: hd(e.sig2),
      utu: hd(e.utu),
      u: e.u,
      predy: e.predy,
      pr2: hd(e.pr2),
      robust: opts[:robust],
      name_x: p.names ++ p.yend_names,
      name_y: to_string(opts[:name_y]),
      name_yend: p.yend_names,
      name_q: p.q_names,
      warnings: p.warnings
    }
  end

  # -- gm_lag --------------------------------------------------------------------------

  @spec lag(term(), term(), term(), keyword()) :: GMLag.t()
  def lag(y, x, graph, opts) do
    opts = validate!(opts, @lag_opts)
    {p, opts} = prepare(y, x, graph, opts)
    check_lag_opts!(opts)

    e =
      Native.regression_gm_lag(
        p.y,
        {p.x, p.kx},
        {p.yend, p.kyend},
        {p.q, p.kq},
        p.graph.ref,
        opts[:gwk] && opts[:gwk].ref,
        lag_settings(opts, p)
      )
      |> entries()

    k = p.kx + p.kyend + 1
    name_y = to_string(opts[:name_y])
    names = p.names ++ p.yend_names ++ ["W_" <> name_y]

    {slopes, slope_betas, thetas} = ML.durbin_terms(p.names, Enum.take(e.betas, p.kx), opts)
    {multipliers, effects} = ML.impacts_of(e, slopes, slope_betas, thetas)

    %GMLag{
      n: p.n,
      k: k,
      betas: e.betas,
      rho: hd(e.rho),
      vm: rows(e.vm, k),
      varb: rows(e.varb, k),
      std_err: e.std_err,
      z_stat: pairs(e.z_stat),
      sig2: hd(e.sig2),
      u: e.u,
      predy: e.predy,
      predy_e: e.predy_e,
      e_pred: e.e_pred,
      pr2: hd(e.pr2),
      pr2_e: e[:pr2_e] && hd(e.pr2_e),
      ak_test: e[:ak_test] && List.to_tuple(e.ak_test),
      sp_multipliers: multipliers,
      impacts: effects,
      power_terms: e[:power_terms] && round(hd(e.power_terms)),
      robust: opts[:robust],
      w_lags: opts[:w_lags],
      lag_q: opts[:lag_q],
      name_x: names,
      name_y: name_y,
      name_yend: p.yend_names,
      name_q: p.q_names,
      warnings: p.warnings ++ warnings(e)
    }
  end

  defp check_lag_opts!(opts) do
    impacts = opts[:spat_impacts]

    unless Map.has_key?(@impacts, impacts) do
      raise ArgumentError,
            "spat_impacts must be :simple, :full, :power, :all or nil, got #{inspect(impacts)}"
    end

    if opts[:robust] == :hac and opts[:gwk] == nil, do: hac_error()
    check_lags!(opts)
    for key <- [:spat_diag, :sig2n_k, :lag_q], do: boolean!(opts, key)
  end

  defp lag_settings(opts, p) do
    kx_nc = if opts[:constant], do: p.kx - 1, else: p.kx
    kx_base = if opts[:slx_lags] > 0, do: ML.base_count(kx_nc, opts), else: 0

    %{
      w_lags: opts[:w_lags],
      lag_q: opts[:lag_q],
      robust: Map.fetch!(@robust, opts[:robust]),
      sig2n_k: opts[:sig2n_k],
      spat_diag: opts[:spat_diag],
      impacts: Map.fetch!(@impacts, opts[:spat_impacts]),
      hard_bound: opts[:hard_bound],
      skip_const: opts[:constant],
      slx_lags: opts[:slx_lags],
      kx_base: kx_base
    }
  end

  # -- gm_error and gm_combo -----------------------------------------------------------

  @spec error(term(), term(), term(), keyword(), [float()] | nil) :: GMError.t()
  def error(y, x, graph, opts, at \\ nil) do
    opts = validate!(opts, @error_opts)
    {p, opts} = prepare(y, x, graph, opts)
    e = run_error(p, opts, false, at)
    kz = p.kx + p.kyend

    struct!(
      GMError,
      common_fields(p, opts, e, kz) ++
        [
          k: kz,
          lam: hd(e.lam),
          name_x: p.names ++ p.yend_names ++ ["lambda"]
        ]
    )
  end

  @spec combo(term(), term(), term(), keyword(), [float()] | nil) :: GMCombo.t()
  def combo(y, x, graph, opts, at \\ nil) do
    opts = validate!(opts, @combo_opts)
    {p, opts} = prepare(y, x, graph, opts)
    check_lags!(opts)
    boolean!(opts, :lag_q)
    e = run_error(p, opts, true, at)
    kz = p.kx + p.kyend + 1
    name_y = to_string(opts[:name_y])

    struct!(
      GMCombo,
      common_fields(p, opts, e, kz) ++
        [
          k: kz,
          rho: hd(e.rho),
          lam: hd(e.lam),
          predy_e: e.predy_e,
          e_pred: e.e_pred,
          pr2_e: e[:pr2_e] && hd(e.pr2_e),
          w_lags: opts[:w_lags],
          lag_q: opts[:lag_q],
          name_x: p.names ++ p.yend_names ++ ["W_" <> name_y, "lambda"]
        ]
    )
  end

  defp run_error(p, opts, combo?, at) do
    check_error_opts!(p, opts)

    settings = %{
      estimator: Map.fetch!(@estimators, opts[:estimator]),
      max_iter: opts[:max_iter],
      epsilon: opts[:epsilon] * 1.0,
      step1c: opts[:step1c],
      a1: Map.fetch!(@a1s, opts[:a1]),
      hard_bound: opts[:hard_bound],
      combo: combo?,
      w_lags: opts[:w_lags],
      lag_q: opts[:lag_q],
      skip_const: opts[:constant],
      at: at && Enum.map(at, &(&1 * 1.0)),
      regimes: nil
    }

    Native.regression_gm_error(
      p.y,
      {p.x, p.kx},
      {p.yend, p.kyend},
      {p.q, p.kq},
      p.graph.ref,
      settings
    )
    |> entries()
  end

  defp check_error_opts!(p, opts) do
    unless Map.has_key?(@estimators, opts[:estimator]) do
      raise ArgumentError,
            "estimator must be :kp98, :het or :hom, got #{inspect(opts[:estimator])}"
    end

    unless Map.has_key?(@a1s, opts[:a1]) do
      raise ArgumentError, "a1 must be :hom_sc, :hom or :het, got #{inspect(opts[:a1])}"
    end

    unless is_integer(opts[:max_iter]) and opts[:max_iter] >= 1 do
      raise ArgumentError, "max_iter must be a positive integer, got #{inspect(opts[:max_iter])}"
    end

    unless is_number(opts[:epsilon]) and opts[:epsilon] >= 0 do
      raise ArgumentError, "epsilon must be a non-negative number"
    end

    boolean!(opts, :step1c)

    if p.kyend > 0 and p.kq == 0 do
      raise ArgumentError, "yend needs instruments: pass q"
    end
  end

  defp common_fields(p, opts, e, _kz) do
    [
      n: p.n,
      estimator: opts[:estimator],
      betas: e.betas,
      vm: rows(e.vm, length(hd_row(e.vm))),
      std_err: e.std_err,
      z_stat: pairs(e.z_stat),
      sig2: e[:sig2] && hd(e.sig2),
      u: e.u,
      predy: e.predy,
      e_filtered: e.e_filtered,
      pr2: hd(e.pr2),
      lambdas: e.lambdas,
      name_y: to_string(opts[:name_y]),
      name_yend: p.yend_names,
      name_q: p.q_names,
      warnings: p.warnings ++ warnings(e)
    ] ++ iteration_fields(opts, e)
  end

  defp iteration_fields(opts, e) do
    iterated? = Map.has_key?(e, :iteration)
    estimator = opts[:estimator]

    [
      iteration: if(iterated?, do: round(hd(e.iteration))),
      iter_stop: if(iterated?, do: iter_stop(e)),
      step1c: if(estimator == :het, do: opts[:step1c]),
      max_iter: if(iterated?, do: opts[:max_iter]),
      epsilon: if(iterated?, do: opts[:epsilon] * 1.0),
      a1: if(estimator == :hom, do: opts[:a1])
    ]
  end

  # The covariance is square: its side is the square root of its length.
  defp hd_row(flat), do: List.duplicate(0, round(:math.sqrt(length(flat))))

  @doc false
  def iter_stop(e) do
    if hd(e.iter_stop_max) == 1.0,
      do: "Maximum number of iterations reached.",
      else: "Convergence threshold (epsilon) reached."
  end

  # -- input ---------------------------------------------------------------------------

  defp validate!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    opts = Keyword.merge(@defaults, opts)
    boolean!(opts, :constant)
    boolean!(opts, :hard_bound)

    unless opts[:robust] in [nil, :white, :hac] do
      raise ArgumentError, "robust must be nil, :white or :hac, got #{inspect(opts[:robust])}"
    end

    opts
  end

  defp boolean!(opts, key) do
    unless is_boolean(opts[key]), do: raise(ArgumentError, "#{key} must be a boolean")
  end

  defp check_lags!(opts) do
    unless is_integer(opts[:w_lags]) and opts[:w_lags] >= 1 do
      raise ArgumentError, "w_lags must be a positive integer, got #{inspect(opts[:w_lags])}"
    end
  end

  defp hac_error, do: raise(ArgumentError, "robust: :hac requires a kernel graph in :gwk")

  # `require_graph?` is false for `tsls/5`, which has no weights.
  defp prepare(y, x, graph, opts, require_graph? \\ true) do
    check_graphs!(graph, opts[:gwk], require_graph?)
    {y_bin, n, names, x_bin, kx, warnings} = Regression.prepare_design(y, x, opts, graph)
    check_sizes!(graph, opts[:gwk], n)

    {yend_names, yend_bin, kyend} = extra(opts[:yend], n, opts[:name_yend], "yend")
    {q_names, q_bin, kq} = extra(opts[:q], n, opts[:name_q], "q")

    p = %{
      graph: graph,
      n: n,
      y: y_bin,
      x: x_bin,
      kx: kx,
      names: names,
      warnings: warnings,
      yend: yend_bin,
      kyend: kyend,
      yend_names: yend_names,
      q: q_bin,
      kq: kq,
      q_names: q_names
    }

    {p, opts}
  end

  defp check_graphs!(graph, gwk, require_graph?) do
    if (graph != nil or require_graph?) and not match?(%Graph{}, graph) do
      raise ArgumentError, "graph must be a Spatex.Graph, got #{inspect(graph, limit: 3)}"
    end

    unless gwk == nil or match?(%Graph{}, gwk) do
      raise ArgumentError, "gwk must be a Spatex.Graph"
    end
  end

  defp check_sizes!(graph, gwk, n) do
    if graph && graph.n != n,
      do: raise(ArgumentError, "graph has #{graph.n} nodes but y has #{n} rows")

    if gwk && gwk.n != n,
      do: raise(ArgumentError, "gwk has #{gwk.n} nodes but y has #{n} rows")
  end

  # Extra endogenous variables or instruments: `{names, f64 binary, count}`.
  @doc false
  def extra(nil, _n, _names, _prefix), do: {[], <<>>, 0}
  def extra([], _n, _names, _prefix), do: {[], <<>>, 0}

  def extra(cols, n, names, prefix) do
    {given, bins} = Regression.extra_columns(cols, n, prefix)
    count = length(bins)

    names =
      cond do
        names != nil and length(names) == count ->
          Enum.map(names, &to_string/1)

        names != nil ->
          raise ArgumentError, "name_#{prefix} has #{length(names)} names for #{count} columns"

        given != nil ->
          given

        true ->
          Enum.map(1..count//1, &"#{prefix}_#{&1}")
      end

    {names, IO.iodata_to_binary(bins), count}
  end

  # -- results -------------------------------------------------------------------------

  @doc false
  def entries({:error, reason}), do: raise(ArgumentError, to_string(reason))

  def entries({:ok, list}) do
    Map.new(list, fn {name, bin} ->
      {Enum.find(@keys, &(Atom.to_string(&1) == name)) ||
         raise(ArgumentError, "unexpected native result #{inspect(name)}"),
       Input.from_f64_binary(bin)}
    end)
  end

  @doc false
  def rows(flat, k), do: Enum.chunk_every(flat, k)

  @doc false
  def pairs(flat), do: flat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)

  @doc false
  def warnings(e) do
    [
      {:warn_lam, "The spatial error parameter lambda is outside (-0.99, 0.99)."},
      {:warn_rho,
       "*** WARNING: Estimate for spatial lag coefficient is outside the boundary (-1, 1). ***"},
      {:warn_rho_near,
       "The spatial lag parameter is outside (-0.99, 0.99); it was clamped to +-0.9 in the iterations."},
      {:warn_degenerate,
       "The moment equations have a zero scale (spreg's optimizer fails here); lambda was set to 0."}
    ]
    |> Enum.filter(fn {key, _} -> Map.has_key?(e, key) end)
    |> Enum.map(&elem(&1, 1))
  end
end
