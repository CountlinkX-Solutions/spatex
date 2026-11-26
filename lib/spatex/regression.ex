defmodule Spatex.Regression do
  @moduledoc """
  Spatial regression, after `spreg` (PySAL).

  `ols/3` fits ordinary least squares with spreg's full diagnostic suite: coefficient inference,
  White or HAC robust covariances, Jarque-Bera, Breusch-Pagan, Koenker-Bassett and White
  heteroskedasticity tests, the condition number, variance inflation factors, and, given a spatial
  `graph`, the Lagrange multiplier tests for spatial lag and error dependence and Moran's I of the
  residuals. `ml_lag/4` and `ml_error/4` are the maximum-likelihood spatial lag and error models.
  `tsls/5`, `gm_lag/4`, `gm_error/4` and `gm_combo/4` are the two-stage least squares and GMM
  estimators (Kelejian-Prucha, heteroskedasticity-robust and homoskedastic). Results are checked
  against spreg 1.9.1 at 1e-9; GMM lambdas, which spreg finds with an early-stopping L-BFGS-B, to
  5e-6 (see `docs/parity.md`).

  Like spreg, no weights are ever transformed implicitly: the graph is used exactly as given.
  Row-standardize it first (`Spatex.Graph.transform(graph, :r)`) for the textbook diagnostics.

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0]
      iex> ols = Spatex.Regression.ols(y, [x], name_x: ["X"])
      iex> {ols.n, ols.k, ols.name_x}
      {8, 2, ["CONSTANT", "X"]}
      iex> Float.round(ols.r2, 4)
      0.0121
  """

  alias Spatex.{Graph, Input, Native}

  alias Spatex.Regression.{
    GM,
    GMCombo,
    GMComboRegimes,
    GMError,
    GMErrorRegimes,
    GMLag,
    GMLagRegimes,
    ML,
    MLError,
    MLErrorRegimes,
    MLLag,
    OLS,
    TSLS
  }

  alias Spatex.Regression.{
    MLLagRegimes,
    OLSRegimes,
    Panel,
    PanelFE,
    PanelGMKKP,
    PanelRE,
    PanelREError,
    Regimes,
    RegimesError,
    RegimesLag,
    RegimesML,
    RegimesMLError
  }

  alias Spatex.Regression.{
    SUR,
    Sur,
    SURerrorGM,
    SurErrorGm,
    SURerrorML,
    SurErrorMl,
    SURlagIV,
    SurLagIv,
    ThreeSLS,
    ThreeSls,
    TSLSRegimes
  }

  @ols_opts [
    :constant,
    :name_x,
    :name_y,
    :robust,
    :sig2n_k,
    :graph,
    :gwk,
    :nonspat_diag,
    :spat_diag,
    :moran,
    :white_test,
    :vif,
    :slx_lags,
    :slx_vars
  ]

  @pair_keys ~w(f_stat lm_error lm_lag rlm_error rlm_lag lm_sarma lm_wx lm_spdurbin rlm_wx
                rlm_durlag lm_slxerr)a
  @scalar_keys ~w(utu sig2 sig2n sig2n_k mean_y std_y r2 ar2 logll aic schwarz sig2ml
                  condition_number)a

  @doc """
  Ordinary least squares of `y` on `x` with diagnostics (`spreg.OLS`).

  `x` is a list of columns, a map of `name => column` (columns ordered by name), an `{n, k}`
  `Nx.Tensor`, an `Explorer.DataFrame` (all its columns), or a single column. Columns may be lists,
  `Nx` vectors or `Explorer.Series`. Like spreg, constant columns of `x` are dropped (with an entry
  in `warnings`) and, with `constant: true`, an intercept named `"CONSTANT"` is prepended.

  ## Options

    * `:constant` - add the intercept (default `true`). With `false` the columns are used as given
      (spreg's `BaseOLS`); spatial diagnostics then need the constant and are rejected.
    * `:name_x`, `:name_y` - variable names (defaults `var_1`, `var_2`, ... and `dep_var`; a map or
      DataFrame supplies its own names).
    * `:robust` - `nil` (default), `:white` or `:hac`.
    * `:sig2n_k` - estimate sigma-squared with `n - k` instead of `n` (default `true`).
    * `:gwk` - kernel `Spatex.Graph` for `robust: :hac`; its diagonal must be exactly 1 and its
      off-diagonal weights in `[0, 1]`. The HAC covariance is not symmetrized, like spreg's.
    * `:graph` - spatial weights for `:spat_diag` / `:moran`, used as given.
    * `:nonspat_diag` - F, log-likelihood, AIC/BIC, condition number, Jarque-Bera, Breusch-Pagan
      and Koenker-Bassett tests (default `true`).
    * `:white_test` - White's test (default `false`; unavailable with robust errors).
    * `:vif` - variance inflation factors (default `false`).
    * `:slx_lags` - spatial lags of `x` added as regressors (spreg's `slx_lags`, default `0`):
      `W x, ..., W^slx_lags x` of the non-constant columns, named `W_x`, `W2_x`, ...; needs the
      weights (`:graph`). With SLX terms the WX-based LM tests (`lm_wx`, `lm_spdurbin`, `rlm_wx`, `rlm_durlag`, `lm_slxerr`) are `nil`, as in spreg.
    * `:slx_vars` - `:all` (default) or one boolean per non-constant `x` column selecting the
      columns that get SLX terms.
    * `:spat_diag` - LM tests; `:moran` - Moran's I of the residuals. Both need `:graph`; spatial
      diagnostics are skipped for `robust: :hac`, as in spreg. Graphs with a link-free structure
      raise.

  Raises `ArgumentError` on invalid input, including an exactly singular `X'X`.

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> ols = Spatex.Regression.ols(y, %{"A" => [1, 2, 0.5, 4, 3, 2, 6, 5]})
      iex> Float.round(hd(ols.betas), 4)
      4.5072
  """
  @spec ols(term(), term(), keyword()) :: OLS.t()
  def ols(y, x, opts \\ []) do
    opts = validate!(opts)
    {y_bin, n} = Input.to_f64_binary(y, name: "y")
    {names, columns} = columns!(x, n)
    {names, columns, warnings} = design(names, columns, opts)
    flags = flags!(opts, n)
    {names, columns} = slx(names, columns, opts, opts[:graph])

    names = if opts[:constant], do: ["CONSTANT" | names], else: names
    x_bin = IO.iodata_to_binary(columns)
    k = length(names)

    entries =
      y_bin
      |> Native.regression_ols(
        x_bin,
        k,
        robust_code(opts[:robust]),
        opts[:sig2n_k],
        flags,
        opts[:graph] && opts[:graph].ref,
        opts[:gwk] && opts[:gwk].ref
      )
      |> unwrap()
      |> Map.new(fn {key, bin} -> {entry_key(key), Input.from_f64_binary(bin)} end)

    r = build(entries, n, k, names, warnings, opts)

    # spreg runs the WX-based LM tests only on models without SLX terms.
    if opts[:slx_lags] > 0,
      do: %{r | lm_wx: nil, lm_spdurbin: nil, rlm_wx: nil, rlm_durlag: nil, lm_slxerr: nil},
      else: r
  end

  @doc """
  OLS with coefficients that vary by regime (`spreg.OLS_Regimes`).

  `regimes` holds one label per observation (a list, `Nx` vector or `Explorer.Series`; see `Spatex.Input.to_labels/2`), sorted for the coefficient order. `x` and the input
  rules are those of `ols/3`; the intercept is added by the function. Returns a
  `Spatex.Regression.OLSRegimes` with Chow tests of equal coefficients across regimes.

  ## Options

    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column: `true` for a coefficient per
      regime, `false` for a global one.
    * `:regime_err_sep` - `true` (default): with every coefficient by regime and
      `constant_regi: :many`, one OLS per regime (each with its own error variance); otherwise
      the pooled fit is re-weighted by each regime's residual variance (unless `:robust` is
      set). `false` fits one pooled OLS with a common error variance.
    * `:robust` - `nil` (default) or `:white`. `:sig2n_k` (default `true`), `:name_x`, `:name_y`
      as in `ols/3`. Columns constant within a regime are dropped in the separate fits, with a
      warning, as in spreg.

  Spatial diagnostics, `:hac` and `slx_lags` are not available with regimes.

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> r = Spatex.Regression.ols_regimes(y, [[1, 2, 0.5, 4, 3, 2, 6, 5]], [0, 0, 0, 0, 1, 1, 1, 1])
      iex> {r.mode, r.name_x}
      {:separate, ["0_CONSTANT", "0_var_1", "1_CONSTANT", "1_var_1"]}
  """
  @spec ols_regimes(term(), term(), [term()], keyword()) :: OLSRegimes.t()
  def ols_regimes(y, x, regimes, opts \\ []), do: Regimes.ols(y, x, regimes, opts)

  @doc """
  Two-stage least squares with coefficients that vary by regime (`spreg.TSLS_Regimes`).

  `x`, `yend` and `q` follow the input rules of `tsls/5` (the intercept is added by the
  function); `regimes` holds one label per observation (see `Spatex.Input.to_labels/2`), sorted
  for the coefficient order. Returns a `Spatex.Regression.TSLSRegimes` with Chow tests of equal
  coefficients across regimes.

  ## Options

    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column followed by one per `yend`
      column: `true` for a coefficient per regime, `false` for a global one. The instruments
      always vary by regime.
    * `:regime_err_sep` - `false` (default, as in spreg): one pooled 2SLS on the regime design.
      `true`: with every coefficient by regime and `constant_regi: :many`, one 2SLS per regime
      (each with its own error variance; columns of `x` constant within a regime are dropped
      with a warning); otherwise ignored. It is disabled, with a warning, under `robust: :hac`.
    * `:robust` - `nil` (default), `:white` or `:hac` (with `:gwk`, a kernel `Spatex.Graph` over
      all observations). `:sig2n_k` (default `true`, as in spreg's regimes models),
      `:name_x`, `:name_y`, `:name_yend`, `:name_q` as in `tsls/5`.

    * `:spreg_compatible` - `false` (default): the Chow restrictions use the real coefficient
      layout (exogenous regime blocks, exogenous globals, endogenous regime blocks, endogenous
      globals). `true` reproduces spreg, whose `Chow` assumes the globals come last; with global
      exogenous columns and regime-specific endogenous ones (for example `constant_regi: :one`)
      its endogenous and joint tests compare other coefficients. Coefficients and covariance
      are never affected, nor are the other cases (see `docs/parity.md`).

  Spatial diagnostics (`spat_diag`), `robust: 'ogmm'`, `slx_lags` and `TSLS_Endog_Regimes` are
  not ported.

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> yend = [2.0, 1.0, 3.5, 0.5, 4.0, 6.5, 1.0, 5.0, 3.0, 2.5, 5.5, 3.0]
      iex> q = [[1.0, 0.0, 3.0, 1.0, 3.5, 6.0, 0.5, 4.0, 2.5, 3.0, 4.0, 2.0], [0.5, 2.5, 1.0, 3.0, 2.0, 1.0, 4.5, 0.0, 3.5, 1.5, 2.5, 0.5]]
      iex> r = Spatex.Regression.tsls_regimes(y, [x], [yend], q, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
      iex> {r.mode, r.name_x, r.name_yend}
      {:pooled, ["0_CONSTANT", "0_var_1", "1_CONSTANT", "1_var_1"], ["0_yend_1", "1_yend_1"]}
  """
  @spec tsls_regimes(term(), term(), term(), term(), [term()], keyword()) :: TSLSRegimes.t()
  def tsls_regimes(y, x, yend, q, regimes, opts \\ []),
    do: Regimes.tsls(y, x, yend, q, regimes, opts)

  @doc """
  Spatial lag model with coefficients that vary by regime (`spreg.GM_Lag_Regimes`).

  `y = rho W y + X beta + e` by spatial two-stage least squares, with `W y` instrumented as in
  `gm_lag/4` and the coefficients of every regime (or some of them) estimated separately.
  `x`, `:yend` and `:q` follow the input rules of `gm_lag/4` (the intercept is added by the
  function); `regimes` holds one label per observation (see `Spatex.Input.to_labels/2`), sorted
  for the coefficient order; `graph` is used as given (row-standardize it first). Returns a
  `Spatex.Regression.GMLagRegimes` with Chow tests of equal coefficients across regimes.

  ## Options

    * `:yend`, `:q` - extra endogenous variables and their instruments. `:w_lags` (default `1`)
      and `:lag_q` (default `true`) build the spatial instruments, as in `gm_lag/4`.
    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column followed by one per `:yend`
      column: `true` for a coefficient per regime, `false` for a global one. The instruments
      always vary by regime; `W y` varies by regime only with `:regime_lag_sep`.
    * `:regime_lag_sep` - `false` (default): one `rho` for all regimes, `W y` and its
      instruments built on the whole graph, one pooled 2SLS on the regime design.
      `true`: a coefficient of `W y` per regime, whose `W y` and instruments are built on the
      regime's own subgraph (edges between regimes are dropped; a row-standardized graph is
      row-standardized again, so regime islands get a zero lag). With every coefficient by
      regime and `constant_regi: :many` this runs one `gm_lag/4` per regime (`mode: :separate`);
      otherwise it fits the pooled design on the union of the regime subgraphs. A regime needs at
      least as many observations as there are variables plus two.
    * `:regime_err_sep` - spreg only honors it together with `:regime_lag_sep` (which forces
      it on); alone it is switched off with a warning. It is also switched off, with a warning,
      under `robust: :hac`.
    * `:robust` - `:white` (default, as in spreg's regimes models), `nil` or `:hac` (with
      `:gwk`, a kernel `Spatex.Graph` over all observations; the spatial diagnostics are then
      switched off). `:sig2n_k` (default `false`, as in `gm_lag/4`).
    * `:spat_diag` - the Anselin-Kelejian test (default `true`); `:spat_impacts` - `:simple`
      (default), `:full`, `:power`, `:all` or `nil`, for the non-constant exogenous
      coefficients, as in `gm_lag/4`. `:hard_bound` - raise when `|rho| >= 1`.
    * `:name_x`, `:name_y`, `:name_yend`, `:name_q` - as in `gm_lag/4`.
    * `:spreg_compatible` - `false` (default): the Chow restrictions use the real coefficient
      layout (exogenous regime blocks, exogenous globals, endogenous regime blocks, endogenous
      globals). `true` reproduces spreg, whose `Chow` assumes the globals come last; with global
      exogenous columns and regime-specific endogenous ones (`constant_regi: :one` or a global
      `x` column together with `:regime_lag_sep` or a regime-specific `:yend`) its endogenous and
      joint tests compare other coefficients. Coefficients and covariance are never affected,
      nor are the other cases (see `docs/parity.md`).

  `slx_lags`, `robust: 'ogmm'`, `cores`, the text summary and `GM_Lag_Endog_Regimes`
  (endogenous regimes) are not ported.

  ## Examples

      iex> ring = fn o -> for i <- 0..5, j <- [rem(i + 1, 6), rem(i + 5, 6)], do: {o + i, o + j} end
      iex> {f, t} = Enum.unzip(ring.(0) ++ ring.(6))
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> regimes = List.duplicate(0, 6) ++ List.duplicate(1, 6)
      iex> r = Spatex.Regression.gm_lag_regimes(y, [x], regimes, g, robust: nil, regime_lag_sep: true)
      iex> {r.mode, r.name_x, r.name_yend}
      {:separate, ["0_CONSTANT", "0_var_1", "1_CONSTANT", "1_var_1"], ["0_W_dep_var", "1_W_dep_var"]}
  """
  @spec gm_lag_regimes(term(), term(), [term()], Graph.t(), keyword()) :: GMLagRegimes.t()
  def gm_lag_regimes(y, x, regimes, graph, opts \\ []),
    do: RegimesLag.fit(y, x, regimes, graph, opts)

  @doc """
  Maximum-likelihood spatial lag model with coefficients that vary by regime
  (`spreg.ML_Lag_Regimes`).

  `y = rho W y + X beta + e` by maximum likelihood (the base `ml_lag/4` fit), with the
  coefficients of every regime (or some of them) estimated separately. `x` follows the input
  rules of `ml_lag/4` (the intercept is added by the function); `regimes` holds one label per
  observation (see `Spatex.Input.to_labels/2`), sorted for the coefficient order; `graph` is
  used as given (row-standardize it first). Returns a `Spatex.Regression.MLLagRegimes` with
  Chow tests of equal coefficients across regimes.

  ## Options

    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column: `true` for a coefficient per
      regime, `false` for a global one.
    * `:regime_lag_sep` - `false` (default): one `rho` for all regimes, one ML fit on the regime
      design. `true`: a `rho` per regime, one `ml_lag/4` fit per regime on the regime's own
      subgraph (edges between regimes are dropped; a row-standardized graph is row-standardized
      again, so regime islands get a zero lag). spreg requires every coefficient to vary by
      regime and `constant_regi: :many` here, and so does this function (it raises otherwise).
      A regime needs at least as many observations as there are variables plus two; columns
      constant within a regime are dropped with a warning (spreg fails with a shape error).
    * `:method` - `:full` (default), `:ord` or `:lu`; `:epsilon` (`1.0e-7`) and `:max_dense_n`
      as in `ml_lag/4`.
    * `:spat_impacts` - `:simple` (default), `:full`, `:power`, `:all` or `nil`: the multipliers
      and the effects of the non-constant coefficients.
    * `:name_x`, `:name_y` - variable names.

  spreg's class has no `regime_err_sep` (it is always `false`; passing it raises here, as spreg
  raises a `TypeError`). `slx_lags` / `slx_vars` (and with them the common-factor test
  `spat_diag`), `cores`, `vm` and the text summary are not ported. Unlike the 2SLS regimes
  models the Chow restrictions are always on the real coefficient layout (`rho` is the last
  global column, or one more regime-specific column), so there is no `spreg_compatible` switch.

  ## Examples

      iex> ring = fn o -> for i <- 0..5, j <- [rem(i + 1, 6), rem(i + 5, 6)], do: {o + i, o + j} end
      iex> {f, t} = Enum.unzip(ring.(0) ++ ring.(6))
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> regimes = List.duplicate(0, 6) ++ List.duplicate(1, 6)
      iex> r = Spatex.Regression.ml_lag_regimes(y, [x], regimes, g, regime_lag_sep: true)
      iex> {r.mode, r.name_x}
      {:separate, ["0_CONSTANT", "0_var_1", "0_W_dep_var", "1_CONSTANT", "1_var_1", "1_W_dep_var"]}
  """
  @spec ml_lag_regimes(term(), term(), [term()], Graph.t(), keyword()) :: MLLagRegimes.t()
  def ml_lag_regimes(y, x, regimes, graph, opts \\ []),
    do: RegimesML.fit(y, x, regimes, graph, opts)

  @doc false
  # Evaluates every quantity at given rho(s) (test hook): a number for the pooled fit, a list in
  # regime order for `regime_lag_sep`.
  def ml_lag_regimes_at(y, x, regimes, graph, at, opts \\ []),
    do: RegimesML.fit(y, x, regimes, graph, opts, at)

  @doc """
  Maximum-likelihood spatial error model with coefficients that vary by regime
  (`spreg.ML_Error_Regimes`).

  `y = X beta + u`, `u = lambda W u + e` by maximum likelihood (the base `ml_error/4` fit), with
  the coefficients of every regime (or some of them) estimated separately. `x` follows the input
  rules of `ml_error/4` (the intercept is added by the function); `regimes` holds one label per
  observation (see `Spatex.Input.to_labels/2`), sorted for the coefficient order; `graph` is
  used as given (row-standardize it first). Returns a `Spatex.Regression.MLErrorRegimes` with
  Chow tests of equal coefficients across regimes.

  ## Options

    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column: `true` for a coefficient per
      regime, `false` for a global one.
    * `:regime_err_sep` - `false` (default): one `lambda` for all regimes, one ML fit on the
      regime design. `true`: a `lambda` per regime, one `ml_error/4` fit per regime on the
      regime's own subgraph (edges between regimes are dropped; a row-standardized graph is
      row-standardized again, so regime islands get a zero lag). spreg requires every
      coefficient to vary by regime and `constant_regi: :many` here, and so does this function
      (it raises otherwise). Columns constant within a regime are dropped with a warning (spreg
      fails with a shape error).
    * `:method` - `:full` (default), `:ord` or `:lu`; `:epsilon` (`1.0e-7`) and `:max_dense_n`
      as in `ml_error/4`.
    * `:name_x`, `:name_y` - variable names.

  In the pooled fit spreg lags the original columns and then expands them by regime
  (`regimes(W [1 | X])`), which is not the lag of the regime design; this function reproduces
  that construction, so the pooled fit is not `ml_error/4` on the regime design wherever
  neighbours lie in another regime. `slx_lags` / `slx_vars`, `regime_lag_sep` (ignored by spreg
  here), `cores`, `vm` and the text summary are not ported. The Chow restrictions are always on
  the real coefficient layout (`lambda` is the last global column, or one more regime-specific
  column), so there is no `spreg_compatible` switch.

  ## Examples

      iex> ring = fn o -> for i <- 0..5, j <- [rem(i + 1, 6), rem(i + 5, 6)], do: {o + i, o + j} end
      iex> {f, t} = Enum.unzip(ring.(0) ++ ring.(6))
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> regimes = List.duplicate(0, 6) ++ List.duplicate(1, 6)
      iex> r = Spatex.Regression.ml_error_regimes(y, [x], regimes, g, regime_err_sep: true)
      iex> {r.mode, r.name_x}
      {:separate, ["0_CONSTANT", "0_var_1", "0_lambda", "1_CONSTANT", "1_var_1", "1_lambda"]}
  """
  @spec ml_error_regimes(term(), term(), [term()], Graph.t(), keyword()) :: MLErrorRegimes.t()
  def ml_error_regimes(y, x, regimes, graph, opts \\ []),
    do: RegimesMLError.fit(y, x, regimes, graph, opts)

  @doc false
  # Evaluates every quantity at given lambda(s) (test hook): a number for the pooled fit, a list
  # in regime order for `regime_err_sep`.
  def ml_error_regimes_at(y, x, regimes, graph, at, opts \\ []),
    do: RegimesMLError.fit(y, x, regimes, graph, opts, at)

  @doc """
  Spatial error model with coefficients that vary by regime (`spreg.GM_Error_Regimes`, with
  `:yend` / `:q` `spreg.GM_Endog_Error_Regimes`), with the Kelejian-Prucha, heteroskedastic or
  homoskedastic estimator.

  `y = X beta + u`, `u = lambda W u + e` by GMM, with the coefficients of every regime (or some of
  them) estimated separately. `x`, `:yend` and `:q` follow the input rules of `gm_error/4` (the
  intercept is added by the function); `regimes` holds one label per observation (see
  `Spatex.Input.to_labels/2`), sorted for the coefficient order; `graph` is used as given
  (row-standardize it first). Returns a `Spatex.Regression.GMErrorRegimes` with Chow tests of
  equal coefficients across regimes.

  ## Options

    * `:yend`, `:q` - extra endogenous variables and their instruments (the instruments always
      vary by regime).
    * `:constant_regi` - `:many` (default, an intercept per regime) or `:one` (a global one).
    * `:cols2regi` - `:all` (default) or one boolean per `x` column followed by one per `:yend`
      column: `true` for a coefficient per regime, `false` for a global one.
    * `:regime_err_sep` - `false` (default): one `lambda` and one pooled fit on the regime
      design. `true`: one `gm_error/4` per regime on the regime's own subgraph (edges between
      regimes are dropped; a row-standardized graph is row-standardized again), each with its own
      `lambda` (`mode: :separate`); spreg requires every coefficient to vary by regime, and
      this raises otherwise. Columns of `x` constant within a regime are dropped with a warning
      (spreg fails with a shape error there).
    * `:estimator` - `:kp98` (default, `GM_Error_Regimes`), `:het` (`GM_Error_Het_Regimes`) or
      `:hom` (`GM_Error_Hom_Regimes`), each with its Endog variant. `:max_iter` (default `1`),
      `:epsilon` (`1.0e-5`) and, for `:het`, `:step1c` (`false`) or, for `:hom`, `:a1`
      (`:hom_sc`, `:hom` or `:het`; spreg's regimes classes default to `:het`) control the
      iterations as in `gm_error/4`; passing them with `:kp98`, `:step1c` with `:hom` or `:a1`
      with `:het` raises. With `:het` / `:hom` the covariance, standard errors and z statistics
      include `lambda`, the separate Chow tests treat `lambda` as one more coefficient, and
      `:het` reports no `sig2`.
    * `:name_x`, `:name_y`, `:name_yend`, `:name_q` - as in `gm_error/4`.
    * `:spreg_compatible` - `false` (default): the Chow restrictions use the real coefficient
      layout (exogenous regime blocks, exogenous globals, endogenous regime blocks, endogenous
      globals). `true` reproduces spreg, whose `Chow` assumes the globals come last; with global
      exogenous columns and regime-specific endogenous ones (for example `constant_regi: :one`
      with `:yend`) its endogenous and joint tests compare other coefficients. Coefficients and
      covariance are never affected, nor are the other cases (see `docs/parity.md`).

  `lambda` is found like `gm_error/4`'s (to ~5e-6 of spreg's L-BFGS-B result). The iterations
  replicate spreg's regimes classes: the coefficients come from the last second stage, which uses
  the lambda of the previous step, and every class but `GM_Error_Het_Regimes` filters with the
  first-step lambda in every iteration (see `docs/parity.md`). `slx_lags`, `hard_bound`,
  `inv_method`, the text summary and `GMM_Error_Regimes` are not ported.

  ## Examples

      iex> ring = fn o -> for i <- 0..5, j <- [rem(i + 1, 6), rem(i + 5, 6)], do: {o + i, o + j} end
      iex> {f, t} = Enum.unzip(ring.(0) ++ ring.(6))
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> regimes = List.duplicate(0, 6) ++ List.duplicate(1, 6)
      iex> r = Spatex.Regression.gm_error_regimes(y, [x], regimes, g)
      iex> {r.mode, r.name_x}
      {:pooled, ["0_CONSTANT", "0_var_1", "1_CONSTANT", "1_var_1", "lambda"]}
  """
  @spec gm_error_regimes(term(), term(), [term()], Graph.t(), keyword()) :: GMErrorRegimes.t()
  def gm_error_regimes(y, x, regimes, graph, opts \\ []),
    do: RegimesError.fit(:error, y, x, regimes, graph, opts)

  @doc """
  Spatial lag model with spatially autocorrelated errors and coefficients that vary by regime
  (`spreg.GM_Combo_Regimes`, `GM_Combo_Het_Regimes`, `GM_Combo_Hom_Regimes`).

  `y = rho W y + X beta + u`, `u = lambda W u + e`: `W y` is instrumented as in `gm_combo/4`
  (`:w_lags`, `:lag_q`), then the error model of `gm_error_regimes/5` is fitted. Returns a
  `Spatex.Regression.GMComboRegimes`.

  ## Options

  As `gm_error_regimes/5` (`:yend`, `:q`, `:constant_regi`, `:cols2regi` over `x` then `:yend`,
  `:estimator` (`:kp98`, `:het` or `:hom`) with `:max_iter`, `:epsilon`, `:step1c` and `:a1`,
  names, `:spreg_compatible`; `W y` is a global coefficient), and:

    * `:regime_lag_sep` - `false` (default): one `rho` for all regimes, `W y` and its instruments
      built on the whole graph. `true`: a coefficient of `W y` per regime, one `gm_combo/4` per
      regime on the regime's subgraph; every coefficient must vary by regime (`constant_regi:
      :one` raises). spreg ties `:regime_err_sep` to it: `regime_lag_sep: true` switches
      `regime_err_sep` on, `regime_err_sep: true` alone is switched off, both with a warning;
      with `regime_lag_sep: true` a `:cols2regi` other than all-true is reset to `:all` with
      spreg's warning (which it also emits for the default `:all`).

  ## Examples

      iex> ring = fn o -> for i <- 0..5, j <- [rem(i + 1, 6), rem(i + 5, 6)], do: {o + i, o + j} end
      iex> {f, t} = Enum.unzip(ring.(0) ++ ring.(6))
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]
      iex> regimes = List.duplicate(0, 6) ++ List.duplicate(1, 6)
      iex> r = Spatex.Regression.gm_combo_regimes(y, [x], regimes, g, regime_lag_sep: true)
      iex> {r.mode, r.name_yend}
      {:separate, ["0_W_dep_var", "1_W_dep_var"]}
  """
  @spec gm_combo_regimes(term(), term(), [term()], Graph.t(), keyword()) :: GMComboRegimes.t()
  def gm_combo_regimes(y, x, regimes, graph, opts \\ []),
    do: RegimesError.fit(:combo, y, x, regimes, graph, opts)

  # Internal test hooks: the Het / Hom regimes models fed spreg's lambdas in call order (one list
  # for the pooled fit, one per regime for the separate fits) instead of the optimizer.
  @doc false
  def gm_error_regimes_at(y, x, regimes, graph, lambdas, opts \\ []),
    do: RegimesError.fit(:error, y, x, regimes, graph, opts, lambdas)

  @doc false
  def gm_combo_regimes_at(y, x, regimes, graph, lambdas, opts \\ []),
    do: RegimesError.fit(:combo, y, x, regimes, graph, opts, lambdas)

  @doc """
  Seemingly unrelated regressions (`spreg.SUR`): one OLS-type equation per element of `ys` /
  `xs`, estimated jointly by feasible GLS with a common error covariance across equations.

  `ys` is a list of `y` vectors, `xs` a list of the matching designs (each as `x` in `ols/3`; a
  constant `"Constant_<eq>"` is added to every equation). Returns a `Spatex.Regression.SUR`.

  ## Options

    * `:iter` - iterate the FGLS step until the log-determinant of the error covariance moves
      by at most `:epsilon` (default `1.0e-5`), for at most `:maxiter + 1` rounds (default `5`);
      default `false` (two-step).
    * `:nonspat_diag` - likelihood-ratio and Breusch-Pagan LM tests of a diagonal covariance
      (default `true`).
    * `:spat_diag` - spatial LM error and lag tests (default `false`); needs `:graph`, used as
      given.
    * `:name_bigy` - one name per equation; `:name_bigx` - one list of regressor names per
      equation.

  Regimes are not provided; see `three_sls/5` for the instrumented system and `sur_error_gm/4`
  for the spatial error SUR.
  """
  @spec sur([term()], [term()], keyword()) :: SUR.t()
  def sur(ys, xs, opts \\ []), do: Sur.fit(ys, xs, opts)

  @doc """
  Three-stage least squares for a system of equations with endogenous regressors
  (`spreg.ThreeSLS`).

  `ys`, `xs`, `yends` and `qs` are lists with one entry per equation (at least two): the `y`
  vector, the exogenous design, the endogenous variables and the additional instruments, each
  given as in `tsls/5` (a constant `"Constant_<eq>"` is added to every `xs` design). Every
  equation needs at least one endogenous variable and at least as many instruments as endogenous
  variables. Returns a `Spatex.Regression.ThreeSLS`.

  Estimation follows spreg: 2SLS per equation, the cross-equation covariance `sig = E'E / n` of
  those residuals, then one GLS step on the instrumented regressors (`[X, H (H'H)^-1 H'Yend]`),
  with no iteration. The Chow tests of coefficient equality across equations exist when every
  equation has the same number of coefficients. spreg reports no other non-spatial diagnostic for
  `ThreeSLS`, so there is no `:nonspat_diag` option.

  ## Options

    * `:name_bigy` - one name per equation; `:name_bigx`, `:name_bigyend`, `:name_bigq` - one list
      of names per equation.

  `regimes` is not supported (any `:regimes` option raises `ArgumentError`).

  ## Examples

      iex> y1 = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
      iex> y2 = [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
      iex> e = [0.3, 1.1, 0.9, 2.0, 1.4, 0.2, 2.5, 1.9, 0.6, 1.0, 0.1, 1.2]
      iex> q1 = [1.2, 0.4, 2.2, 0.8, 3.1, 1.7, 0.9, 2.8, 1.1, 0.5, 2.6, 1.9]
      iex> q2 = [0.2, 1.9, 0.7, 2.4, 1.0, 0.6, 2.9, 0.3, 1.8, 2.2, 0.8, 1.4]
      iex> r = Spatex.Regression.three_sls([y1, y2], [[x], [x]], [[e], [e]], [[q1, q2], [q1, q2]])
      iex> {r.n_eq, r.k, length(r.b_3sls), length(r.chow)}
      {2, [3, 3], 2, 3}
  """
  @spec three_sls([term()], [term()], [term()], [term()], keyword()) :: ThreeSLS.t()
  def three_sls(ys, xs, yends, qs, opts \\ []), do: ThreeSls.fit(ys, xs, yends, qs, opts)

  @doc """
  Spatial lag panel with individual fixed effects, `y = rho W y + X beta + mu_i + e`
  (`spreg.ML_LagFE`), by maximum likelihood on the data demeaned by area.

  `y` is an `n x t` matrix, one row per area of `graph` and one column per period: a list of
  rows, a `{n, t}` `Nx` tensor or an `Explorer.DataFrame` (see `Spatex.Input.to_matrix/2`); `x` is a list of such matrices, one per regressor, or a map `name => matrix`
  (ordered by name). No intercept is used: the fixed effects absorb it. `graph` is used as given
  (row-standardize it first). Returns a `Spatex.Regression.PanelFE`.

  ## Options

    * `:time_effects` - add period dummies (`time_2`, ..., `time_t`; default `false`).
    * `:spat_impacts` - `:simple` (default), `:full`, `:power`, `:all` or `nil`, as in
      `ml_lag/4`.
    * `:epsilon` - Brent tolerance (default `1.0e-7`); `rho` is searched in `[-0.99, 0.99]`.
    * `:name_x`, `:name_y` - variable names.

  A regressor without variation over time is absorbed by the fixed effects and raises. The
  variance follows spreg literally (its `x' W A^-1 X b` cross term uses the raw regressors).
  Dense `n x n` matrices are built, so `n` should stay in the low thousands.
  """
  @spec ml_lag_fe(term(), term(), Graph.t(), keyword()) :: PanelFE.t()
  def ml_lag_fe(y, x, graph, opts \\ []), do: Panel.fit(:lag, y, x, graph, opts)

  @doc """
  Spatial lag panel with random effects, `y = rho W y + X beta + mu_i + e` (`spreg.ML_LagRE`).

  Inputs and options are those of `ml_lag_fe/4`; an intercept `"CONSTANT"` is added. spreg
  alternates a bounded search for the variance ratio `phi` (quasi-demeaning
  `v - (1 - phi) mean_t(v)`) with one for `rho` until `phi` changes by less than `1e-7` (at
  most 100 rounds). Returns a `Spatex.Regression.PanelRE`.

  `ml_error_re/4` is the corresponding error model.
  """
  @spec ml_lag_re(term(), term(), Graph.t(), keyword()) :: PanelRE.t()
  def ml_lag_re(y, x, graph, opts \\ []), do: Panel.fit(:lag_re, y, x, graph, opts)

  @doc """
  Spatial error panel with random effects, `y = X beta + mu_i + u`, `u = lambda W u + e`
  (`spreg.ML_ErrorRE`, the maintained class of `spreg/panel.py`).

  Inputs follow `ml_lag_re/4` (an intercept `"CONSTANT"` is added); the options are
  `:time_effects`, `:name_x` and `:name_y`. Returns a `Spatex.Regression.PanelREError`: the
  coefficients, `lam`, `phi` (the variance ratio `sigma_u^2 / sigma^2`) and `sig2_u`.

  spreg concentrates the likelihood over `(lambda, phi)` with the eigen-decomposition of `W`
  (`numpy.linalg.eig`) and maximizes it with L-BFGS-B (`lambda` in `[-0.999, 0.999]`); the
  coefficients and variances are then computed with the exact GLS transform. Spatex evaluates the
  same formulas (to `1e-9` at a given `(lambda, phi)`) and finds the optimum with a nested bounded
  search accurate to about `1e-10`, so `lam` and `phi` agree with spreg's to about `1e-5` (spreg
  stops at `ftol = 1e-9`).

  The statistic only depends on `V diag(1/c) V'`, so the signs LAPACK gives the eigenvectors do
  not matter. Where eigenvalues are repeated the result of `numpy.linalg.eig` depends on its
  basis (non-orthogonal even for a symmetric `W`, e.g. a square rook lattice) and is not a
  likelihood: for a symmetric `W` Spatex uses the orthonormal eigenvectors (the exact likelihood,
  a documented deviation); for a non-symmetric `W` with repeated eigenvalues, or with complex
  eigenvalues, it raises `ArgumentError`. Dense `n x n` matrices are built.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
      iex> y = [[1.0, 2.0, 1.5], [3.0, 2.5, 4.0], [2.0, 1.0, 1.6], [5.0, 4.0, 4.4]]
      iex> x = [[0.2, 0.8, 0.5], [1.1, 0.9, 1.6], [0.4, 0.1, 0.3], [2.0, 1.7, 1.8]]
      iex> r = Spatex.Regression.ml_error_re(y, [x], g, name_x: ["x"])
      iex> r.name_x
      ["CONSTANT", "x", "lambda", "sigma2_u"]
  """
  @spec ml_error_re(term(), term(), Graph.t(), keyword()) :: PanelREError.t()
  def ml_error_re(y, x, graph, opts \\ []), do: Panel.fit(:error_re, y, x, graph, opts)

  @doc false
  # Evaluates every quantity at `{lam, phi}` (the reported variance ratio) instead of
  # optimizing (test hook).
  def ml_error_re_at(y, x, graph, {_lam, _phi} = at, opts \\ []),
    do: Panel.fit(:error_re, y, x, graph, opts, at)

  @doc """
  Kapoor-Kelejian-Prucha GM estimator of a spatial error panel with random effects,
  `y = X beta + mu_i + u`, `u = lambda W u + e` (`spreg.GM_ErrorRE`; `spreg.GM_KKP` in the
  deprecated `sp_panels.py` is the same estimator).

  Inputs and options follow `ml_error_re/4` (an intercept `"CONSTANT"` is added; `y` and every
  `x` variable are `n x t` matrices): `:time_effects`, `:name_x`, `:name_y`, and

    * `:full_weights` (default `false`) - weight the six moment conditions with the full matrix
      `Xi = diag(sigma_v^4 / (t - 1), sigma_1^4) (x) Tau` instead of `Tau = I`.

  `:regimes` is not supported (spreg raises `NotImplementedError`) and raises `ArgumentError`, as
  does `slx_lags` (an unknown option). Returns a `Spatex.Regression.PanelGMKKP`.

  Three stages: OLS and the within moments give a first `lambda` and `sigma_v^2`; the six moments
  (within and between) are then minimized, weighted, for `(lambda, sigma_v^2, sigma_1^2)`; a
  feasible GLS regression (spatial filter, then `I - theta Q1` with
  `theta = 1 - sigma_v / sigma_1`) gives the coefficients and `vm`. The graph is used as given.

  spreg minimizes each stage with L-BFGS-B, which stops early (and, with `:full_weights`, can
  stop far from the minimum). Spatex evaluates the same formulas (to `1e-9` at spreg's own
  optimizer results) and finds the exact minimizer: the variance parameters are profiled out as
  a non-negative least-squares problem and `lambda` is minimized on `[-0.99, 0.99]`, walking
  downhill from spreg's start. The estimates therefore differ from spreg's by its optimizer
  error; see the GM_KKP notes in `docs/parity.md`. Sparse; no dense `n x n` matrix is built.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
      iex> y = [[1.0, 2.0, 1.5], [3.0, 2.5, 4.0], [2.0, 1.0, 1.6], [5.0, 4.0, 4.4]]
      iex> x = [[0.2, 0.8, 0.5], [1.1, 0.9, 1.6], [0.4, 0.1, 0.3], [2.0, 1.7, 1.8]]
      iex> r = Spatex.Regression.gm_kkp(y, [x], g, name_x: ["x"])
      iex> r.name_x
      ["CONSTANT", "x", "lambda", "sigma2_v", "sigma2_1"]
  """
  @spec gm_kkp(term(), term(), Graph.t(), keyword()) :: PanelGMKKP.t()
  def gm_kkp(y, x, graph, opts \\ []), do: Panel.fit(:gm_kkp, y, x, graph, opts)

  @doc false
  # Evaluates every formula at spreg's optimizer results `{lambda_1, sigma_v^2, lambda,
  # sigma_v^2, sigma_1^2}` (first-step then weighted stage) instead of optimizing (test hook).
  def gm_kkp_at(y, x, graph, {_, _, _, _, _} = at, opts \\ []),
    do: Panel.fit(:gm_kkp, y, x, graph, opts, at)

  @doc """
  Spatial error panel with individual fixed effects, `y = X beta + mu_i + u`,
  `u = lambda W u + e` (`spreg.ML_ErrorFE`).

  Inputs and options are those of `ml_lag_fe/4` without `:spat_impacts`; `lam` is searched in
  `[-1, 1]`. As in spreg the information criteria do not count `lam`.
  """
  @spec ml_error_fe(term(), term(), Graph.t(), keyword()) :: PanelFE.t()
  def ml_error_fe(y, x, graph, opts \\ []), do: Panel.fit(:error, y, x, graph, opts)

  @doc """
  Maximum-likelihood spatial lag model `y = rho W y + X beta + e` (`spreg.ML_Lag`).

  `x` and the input rules are those of `ols/3` (constant columns dropped, intercept `"CONSTANT"`
  prepended with `constant: true`). `graph` is a `Spatex.Graph` used exactly as given; like spreg
  nothing is transformed implicitly, so row-standardize it first
  (`Spatex.Graph.transform(graph, :r)`) for the usual `rho` in (-1, 1) interpretation.

  `rho` is found with SciPy's bounded Brent search over (-1, 1) (`xatol: epsilon`), ported
  literally, so it agrees with spreg to rounding of the objective (~1e-10 or better with
  `method: :full`; see "Ord method").

  ## Options

    * `:method` - how the log-determinant `ln|I - rho W|` is evaluated: `:full` (default, a dense
      LU per evaluation), `:ord` (Ord's eigenvalue method: one eigendecomposition, then `O(n)`
      per evaluation) or `:lu` (spreg's sparse-LU option; evaluated like `:full`, see below).
    * `:epsilon` - Brent tolerance and power-expansion threshold (default `1.0e-7`).
    * `:spat_impacts` - `:simple` (default), `:full`, `:power`, `:all` or `nil`: average direct,
      indirect and total impact multipliers (`sp_multipliers`) and per-regressor `impacts`.
    * `:constant`, `:name_x`, `:name_y` - as in `ols/3`.
    * `:slx_lags` - spatial lags of `x` added as regressors (spreg's `slx_lags`, default `0`):
      `W x, ..., W^slx_lags x` of the non-constant columns, named `W_x`, `W2_x`, ...; needs the
      weights; with `ml_lag/4` the model is the spatial Durbin model and the impacts follow spreg (`total = ati (beta + theta)`, `direct = adi beta`, `indirect = total - direct`).
    * `:slx_vars` - `:all` (default) or one boolean per non-constant `x` column selecting the
      columns that get SLX terms.
    * `:max_dense_n` - largest `n` accepted (default `5_000`). All methods build dense `n x n`
      matrices for the asymptotic variance (about `n * n * 8` bytes each, several at once), so
      larger problems raise `ArgumentError`; raise the limit if you have the memory.

  ## Ord method

  `:ord` uses the eigenvalues of `W` itself. spreg replaces `W` by a symmetrized binary matrix
  whenever the *pattern* of `W` is symmetric, which is only valid for row-standardized binary
  weights: for other weights (inverse distance, say) spreg's `rho` is wrong, and its standard
  errors use the symmetrized matrix instead of `W` in every symmetric case. Spatex does neither;
  `:ord` results therefore equal `:full` up to the optimizer tolerance and differ from spreg's
  `ord` standard errors (see `docs/parity.md`).

  ## Differences from spreg

  `:lu` evaluates the determinant with a dense LU (spreg uses a sparse LU without the sign
  check); the optimum is identical. The log-determinant is a sum of logarithms, so it cannot
  overflow for large `n`. `regimes` and the common-factor `spat_diag` are not ported.

  Raises `ArgumentError` on invalid input or a singular `X'X`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3], List.duplicate(0.5, 8))
      iex> y = [3.0, 1.5, 4.0, 1.0]
      iex> x = [1.0, 2.0, 0.5, 4.0]
      iex> m = Spatex.Regression.ml_lag(y, [x], g, max_dense_n: 100)
      iex> {m.n, m.k, List.last(m.name_x)}
      {4, 3, "W_dep_var"}
  """
  @spec ml_lag(term(), term(), Graph.t(), keyword()) :: MLLag.t()
  def ml_lag(y, x, graph, opts \\ []), do: ML.lag(y, x, graph, opts)

  @doc """
  Maximum-likelihood spatial error model `y = X beta + u`, `u = lambda W u + e`
  (`spreg.ML_Error`).

  Options and input rules are those of `ml_lag/4` without `:spat_impacts`. `lam` is found with the
  same bounded Brent search. The coefficient covariance is block diagonal (`beta` and `lam` are
  uncorrelated), and, as in spreg, `k` and the information criteria do not count `lam`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3], List.duplicate(0.5, 8))
      iex> y = [3.0, 1.5, 4.0, 1.0]
      iex> x = [1.0, 2.0, 0.5, 4.0]
      iex> m = Spatex.Regression.ml_error(y, [x], g)
      iex> {m.n, m.k, List.last(m.name_x)}
      {4, 2, "lambda"}
  """
  @spec ml_error(term(), term(), Graph.t(), keyword()) :: MLError.t()
  def ml_error(y, x, graph, opts \\ []), do: ML.error(y, x, graph, opts)

  @doc false
  # Evaluates every quantity at a given rho (test hook: checks the formulas independently of the
  # optimizer).
  def ml_lag_at(y, x, graph, rho, opts \\ []), do: ML.lag(y, x, graph, opts, rho)

  @doc false
  def ml_error_at(y, x, graph, lam, opts \\ []), do: ML.error(y, x, graph, opts, lam)

  @doc """
  Two-stage least squares (`spreg.TSLS`) of `y` on `x` and the endogenous columns `yend`,
  instrumented by `x` and `q`.

  `x`, `yend` and `q` follow the input rules of `ols/3` (lists of columns, maps of
  `name => column`, `Nx` tensors, `Explorer` data frames). `yend` and `q` need at least one column
  each, and `q` at least as many columns as `yend`.

  ## Options

    * `:constant` - prepend an intercept named `"CONSTANT"` (default `true`), as in `ols/3`.
    * `:robust` - `nil` (default), `:white` or `:hac` (needs `:gwk`, a kernel `Spatex.Graph` with
      unit diagonal).
    * `:sig2n_k` - estimate sigma-squared (and rescale the White covariance) with `n - k`
      (default `false`, like spreg's 2SLS).
    * `:name_x`, `:name_y`, `:name_yend`, `:name_q` - variable names.

  Spatial diagnostics of `spreg.TSLS` (`spat_diag`) and regimes are not ported; use `gm_lag/4` for
  the Anselin-Kelejian test. Raises `ArgumentError` on invalid input or singular `H'H` /
  `Z'H (H'H)^-1 H'Z`.

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5]
      iex> yend = [2.0, 1.0, 3.5, 0.5, 4.0, 6.5, 1.0, 5.0, 3.0, 2.5]
      iex> q = [1.0, 0.0, 3.0, 1.0, 3.5, 6.0, 0.5, 4.0, 2.5, 3.0]
      iex> m = Spatex.Regression.tsls(y, [x], [yend], [q])
      iex> {m.n, m.k, m.name_x}
      {10, 3, ["CONSTANT", "var_1", "yend_1"]}
  """
  @spec tsls(term(), term(), term(), term(), keyword()) :: TSLS.t()
  def tsls(y, x, yend, q, opts \\ []), do: GM.tsls(y, x, yend, q, opts)

  @doc """
  Spatial two-stage least squares lag model `y = rho W y + X beta + e` (`spreg.GM_Lag`).

  `W y` is instrumented by `W X, ..., W^w_lags X` (and by the lags of the extra instruments with
  `lag_q: true`). Nothing is optimized, so every result agrees with spreg to rounding.

  `x` and the input rules are those of `ols/3`; `graph` is a `Spatex.Graph` used exactly as
  given (spreg never transforms weights implicitly).

  ## Options

    * `:yend`, `:q` - extra endogenous variables and their instruments.
    * `:w_lags` - spatial lag orders used as instruments (default `1`); `:lag_q` - also lag the
      extra instruments (default `true`).
    * `:robust` - `nil`, `:white` or `:hac` (needs `:gwk`; disables the Anselin-Kelejian test);
      `:sig2n_k` - `n - k` scaling (default `false`).
    * `:spat_diag` - the Anselin-Kelejian test `ak_test` (default `true`).
    * `:spat_impacts` - `:simple` (default), `:full`, `:power`, `:all` or `nil`, as in `ml_lag/4`.
    * `:hard_bound` - raise instead of warn when `|rho| >= 1` (default `false`).
    * `:slx_lags` - spatial lags of `x` added as regressors (spreg's `slx_lags`, default `0`):
      `W x, ..., W^slx_lags x` of the non-constant columns, named `W_x`, `W2_x`, ...; needs the
      weights; the instruments of `W y` become the lags `slx_lags + 1 ..= slx_lags + w_lags` of all original columns (with `:yend`, the lags of `W^(slx_lags + 1) x`), and the impacts follow the spatial Durbin rule of `ml_lag/4`.
    * `:slx_vars` - `:all` (default) or one boolean per non-constant `x` column selecting the
      columns that get SLX terms.
    * `:constant`, `:name_x`, `:name_y`, `:name_yend`, `:name_q` - as in `tsls/5`.

  Regimes are not ported. Raises `ArgumentError` on invalid input.

  ## Examples

      iex> ring = for i <- 0..7, j <- [rem(i + 1, 8), rem(i + 7, 8)], do: {i, j}
      iex> {f, t} = Enum.unzip(ring)
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 16))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0]
      iex> m = Spatex.Regression.gm_lag(y, [x], g)
      iex> {m.n, m.k, List.last(m.name_x), m.w_lags}
      {8, 3, "W_dep_var", 1}
  """
  @spec gm_lag(term(), term(), Graph.t(), keyword()) :: GMLag.t()
  def gm_lag(y, x, graph, opts \\ []), do: GM.lag(y, x, graph, opts)

  @doc """
  GMM spatial error model `y = X beta + u`, `u = lambda W u + e`.

  The `:estimator` option selects spreg's class: `:kp98` (default, `GM_Error`; Kelejian and
  Prucha 1998), `:het` (`GM_Error_Het`; Arraiz et al. 2010, robust to heteroskedasticity) or
  `:hom` (`GM_Error_Hom`; Drukker, Egger and Prucha 2013). With `:yend` / `:q` the corresponding
  `GM_Endog_Error*` model is fitted (2SLS in the second stage).

  `lambda` minimizes a quadratic form in the sample moments. spreg minimizes it with L-BFGS-B,
  which stops early (about 1e-6 away from the minimum); Spatex finds the exact minimizer on
  `[-0.99, 0.99]`, so `lam` agrees with spreg to about 5e-6 and quantities derived from it to
  about 1e-5 relative (see `docs/parity.md`). Every other formula is verified at 1e-9 by
  feeding spreg's own lambdas.

  ## Options

    * `:estimator` - `:kp98`, `:het` or `:hom`.
    * `:max_iter`, `:epsilon` - iteration limit (default `1`) and convergence threshold
      (`1.0e-5`) of `:het` / `:hom`.
    * `:step1c` - `:het` only: weighted second step before iterating (default `false`).
    * `:a1` - `:hom` only: `:hom_sc` (default), `:hom` or `:het`.
    * `:yend`, `:q` - endogenous variables and instruments.
    * `:hard_bound` - raise when `|lambda| >= 0.99` instead of warning (default `false`).
    * `:slx_lags` - spatial lags of `x` added as regressors (spreg's `slx_lags`, default `0`):
      `W x, ..., W^slx_lags x` of the non-constant columns, named `W_x`, `W2_x`, ...; needs the
      weights (the SLX-error model); not available in `gm_combo/4`.
    * `:slx_vars` - `:all` (default) or one boolean per non-constant `x` column selecting the
      columns that get SLX terms.
    * `:constant`, `:name_x`, `:name_y`, `:name_yend`, `:name_q` - as in `tsls/5`.

  As in spreg, `:het` and `:hom` evaluate the coefficients at the previous iteration's lambda
  (`betas` lag `lam` by one step) and `:het` reports no `sig2`. Raises `ArgumentError` on invalid
  input.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3], List.duplicate(0.5, 8))
      iex> m = Spatex.Regression.gm_error([3.0, 1.5, 4.0, 1.0], [[1.0, 2.0, 0.5, 4.0]], g)
      iex> {m.estimator, m.n, List.last(m.name_x)}
      {:kp98, 4, "lambda"}
  """
  @spec gm_error(term(), term(), Graph.t(), keyword()) :: GMError.t()
  def gm_error(y, x, graph, opts \\ []), do: GM.error(y, x, graph, opts)

  @doc """
  Spatial lag model with GMM spatial error, `y = rho W y + X beta + u`, `u = lambda W u + e`
  (`GM_Combo`, `GM_Combo_Het`, `GM_Combo_Hom`).

  `W y` is instrumented as in `gm_lag/4` (`:w_lags`, `:lag_q`) and the errors are modelled as in
  `gm_error/4`; the options of both apply (`:estimator` defaults to `:kp98`). With `:het`, the
  coefficient of `W y` is clamped to +-0.9 in the iterations, as spreg does.

  ## Examples

      iex> ring = for i <- 0..7, j <- [rem(i + 1, 8), rem(i + 7, 8)], do: {i, j}
      iex> {f, t} = Enum.unzip(ring)
      iex> g = Spatex.Graph.from_arrays(f, t, List.duplicate(0.5, 16))
      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0]
      iex> m = Spatex.Regression.gm_combo(y, [x], g)
      iex> {m.estimator, m.k, List.last(m.name_x)}
      {:kp98, 3, "lambda"}
  """
  @spec gm_combo(term(), term(), Graph.t(), keyword()) :: GMCombo.t()
  def gm_combo(y, x, graph, opts \\ []), do: GM.combo(y, x, graph, opts)

  @doc false
  # Test hooks: evaluate at spreg's own lambdas (first step, then each iteration, in call order),
  # bypassing the moment optimizer, to verify every formula independently of it.
  def gm_error_at(y, x, graph, lambdas, opts \\ []), do: GM.error(y, x, graph, opts, lambdas)

  @doc false
  def gm_combo_at(y, x, graph, lambdas, opts \\ []), do: GM.combo(y, x, graph, opts, lambdas)

  @doc false
  # Extra columns (endogenous variables, instruments): `{names | nil, [f64 binary]}`.
  def extra_columns(cols, n, _prefix), do: columns!(cols, n)

  @doc false
  # Shared input handling: `{y_binary, n, names, x_binary, k, warnings}` with the intercept
  # prepended and constant columns dropped according to `opts[:constant]` / `opts[:name_x]`.
  def prepare_design(y, x, opts, graph \\ nil) do
    {y_bin, n} = Input.to_f64_binary(y, name: "y")
    {names, columns} = columns!(x, n)
    {names, columns, warnings} = design(names, columns, opts)
    {names, columns} = slx(names, columns, opts, graph)
    names = if opts[:constant], do: ["CONSTANT" | names], else: names
    {y_bin, n, names, IO.iodata_to_binary(columns), length(names), warnings}
  end

  @doc """
  A printable text summary of an OLS fit, modelled on spreg's report (concise, not byte-identical).

  ## Examples

      iex> y = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
      iex> text = y |> Spatex.Regression.ols([[1, 2, 0.5, 4, 3, 2, 6, 5]]) |> Spatex.Regression.summary()
      iex> String.contains?(text, "ORDINARY LEAST SQUARES")
      true
  """
  @spec summary(OLS.t()) :: String.t()
  def summary(%OLS{} = r) do
    [
      "REGRESSION\n----------\nSUMMARY OF OUTPUT: ORDINARY LEAST SQUARES\n",
      "Dependent variable : #{r.name_y}",
      "Observations       : #{r.n}",
      "Variables          : #{r.k}",
      "R-squared          : #{fmt(r.r2)}",
      "Adjusted R-squared : #{fmt(r.ar2)}",
      "Sum squared resid. : #{fmt(r.utu)}",
      "Sigma-square       : #{fmt(r.sig2)}",
      "S.E. of regression : #{fmt(sqrt(r.sig2))}",
      r.robust && "Standard errors   : #{r.robust}",
      top_diagnostics(r),
      coefficient_table(r),
      regression_diagnostics(r),
      vif_table(r),
      spatial_diagnostics(r),
      warning_lines(r)
    ]
    |> Enum.reject(&(&1 in [nil, false, ""]))
    |> Enum.join("\n")
    |> Kernel.<>("\n")
  end

  # -- input ---------------------------------------------------------------------------

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @ols_opts do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(@ols_opts)}"
    end

    opts =
      Keyword.merge(
        [
          constant: true,
          robust: nil,
          sig2n_k: true,
          graph: nil,
          gwk: nil,
          nonspat_diag: true,
          spat_diag: false,
          moran: false,
          white_test: false,
          vif: false,
          name_x: nil,
          name_y: "dep_var",
          slx_lags: 0,
          slx_vars: :all
        ],
        opts
      )

    unless opts[:robust] in [nil, :white, :hac] do
      raise ArgumentError, "robust must be nil, :white or :hac, got #{inspect(opts[:robust])}"
    end

    for key <- [:constant, :sig2n_k, :nonspat_diag, :spat_diag, :moran, :white_test, :vif] do
      unless is_boolean(opts[key]), do: raise(ArgumentError, "#{key} must be a boolean")
    end

    graph!(opts[:graph], :graph)
    graph!(opts[:gwk], :gwk)
    opts
  end

  @doc false
  # spreg's `slx_lags` / `slx_vars` (`flex_wx`): appends `W x, ..., W^s x` of the selected
  # non-constant columns, block by block, named `W_x`, `W2_x`, ... `names` excludes the constant
  # (the first of `columns` when `constant: true`).
  def slx(names, columns, opts, graph) do
    lags = Keyword.get(opts, :slx_lags, 0)

    unless is_integer(lags) and lags >= 0,
      do: raise(ArgumentError, ":slx_lags must be a non-negative integer, got #{inspect(lags)}")

    if lags == 0 do
      {names, columns}
    else
      unless match?(%Graph{}, graph),
        do: raise(ArgumentError, "slx_lags requires the spatial weights (a Spatex.Graph)")

      cols = if opts[:constant], do: tl(columns), else: columns
      selected = slx_vars!(Keyword.get(opts, :slx_vars, :all), length(names))
      base = for {{name, col}, true} <- Enum.zip(Enum.zip(names, cols), selected), do: {name, col}

      blocks = slx_blocks(base, graph, lags)

      {names ++ Enum.map(blocks, &elem(&1, 0)), columns ++ Enum.map(blocks, &elem(&1, 1))}
    end
  end

  # `[{"W_x", Wx}, ..., {"W2_x", W^2 x}, ...]`: one block of lagged columns per order.
  defp slx_blocks(base, graph, lags) do
    lag = fn {name, col} -> {name, graph.ref |> Native.graph_lag(col) |> unwrap()} end

    base
    |> Stream.iterate(&Enum.map(&1, lag))
    |> Stream.drop(1)
    |> Enum.take(lags)
    |> Enum.with_index(1)
    |> Enum.flat_map(fn {level, order} ->
      prefix = if order == 1, do: "W_", else: "W#{order}_"
      Enum.map(level, fn {name, col} -> {prefix <> name, col} end)
    end)
  end

  defp slx_vars!(:all, k), do: List.duplicate(true, k)

  defp slx_vars!(flags, k) when is_list(flags) do
    unless length(flags) == k and Enum.all?(flags, &is_boolean/1) do
      raise ArgumentError,
            ":slx_vars must be :all or #{k} booleans (one per non-constant x column), " <>
              "got #{inspect(flags)}"
    end

    if Enum.any?(flags), do: flags, else: raise(ArgumentError, ":slx_vars selects no column")
  end

  defp slx_vars!(other, _k),
    do:
      raise(ArgumentError, ":slx_vars must be :all or a list of booleans, got #{inspect(other)}")

  defp graph!(nil, _), do: :ok
  defp graph!(%Graph{}, _), do: :ok

  defp graph!(other, key),
    do: raise(ArgumentError, "#{key} must be a Spatex.Graph, got #{inspect(other, limit: 3)}")

  defp flags!(opts, n) do
    spatial? = opts[:spat_diag] or opts[:moran]

    cond do
      opts[:robust] == :hac and opts[:gwk] == nil ->
        raise ArgumentError, "robust: :hac requires a kernel graph in the :gwk option"

      spatial? and opts[:graph] == nil ->
        raise ArgumentError, "spat_diag and moran require the :graph option"

      spatial? and not opts[:constant] ->
        raise ArgumentError, "spatial diagnostics require constant: true"

      true ->
        :ok
    end

    check_size!(opts[:graph], "graph", n)
    check_size!(opts[:gwk], "gwk", n)
    {opts[:nonspat_diag], opts[:white_test], opts[:vif], opts[:spat_diag], opts[:moran]}
  end

  defp check_size!(nil, _label, _n), do: :ok
  defp check_size!(%Graph{n: n}, _label, n), do: :ok

  defp check_size!(%Graph{n: gn}, label, n),
    do: raise(ArgumentError, "#{label} has #{gn} nodes but y has #{n} rows")

  defp robust_code(nil), do: 0
  defp robust_code(:white), do: 1
  defp robust_code(:hac), do: 2

  # {names, [column]} where each column is a validated f64 binary of length n.
  defp columns!(x, n) do
    {names, cols} = split_columns(x)
    names = names && Enum.map(names, &to_string/1)

    if cols == [], do: raise(ArgumentError, "x has no columns")

    bins =
      cols
      |> Enum.with_index(1)
      |> Enum.map(fn {col, i} ->
        {bin, _} = Input.to_f64_binary(col, name: "x column #{i}", length: n)
        bin
      end)

    {names, bins}
  end

  defp split_columns(%{__struct__: Explorer.DataFrame} = df) do
    names = call(Explorer.DataFrame, :names, [df])
    {names, Enum.map(names, &call(Explorer.DataFrame, :pull, [df, &1]))}
  end

  defp split_columns(%{__struct__: Nx.Tensor} = t) do
    case call(Nx, :rank, [t]) do
      1 ->
        {nil, [t]}

      2 ->
        cols = Nx |> call(:transpose, [t]) |> then(&call(Nx, :to_list, [&1]))
        {nil, cols}

      rank ->
        raise ArgumentError, "x tensor must have rank 1 or 2, got rank #{rank}"
    end
  end

  defp split_columns(%{__struct__: Explorer.Series} = s), do: {nil, [s]}

  defp split_columns(%{} = map) when not is_struct(map) do
    pairs = map |> Enum.sort_by(fn {k, _} -> to_string(k) end)
    {Enum.map(pairs, &elem(&1, 0)), Enum.map(pairs, &elem(&1, 1))}
  end

  defp split_columns([]), do: {nil, []}

  defp split_columns(list) when is_list(list) do
    if Enum.all?(list, &(is_number(&1) or &1 in [:nan, :infinity, :neg_infinity])),
      do: {nil, [list]},
      else: {nil, list}
  end

  defp split_columns(other),
    do: raise(ArgumentError, "unsupported x #{inspect(other, limit: 3)}; expected columns")

  # Names, constant-column removal and (when asked) the intercept column.
  defp design(given, bins, opts) do
    names = resolve_names(given || nil, opts[:name_x], length(bins))

    if opts[:constant] do
      {kept, dropped} =
        names
        |> Enum.zip(bins)
        |> Enum.split_with(fn {_, bin} -> varies?(bin) end)

      if kept == [], do: raise(ArgumentError, "every x column is constant; nothing to regress on")

      warnings =
        case dropped do
          [] ->
            []

          _ ->
            [
              "Variable(s) #{inspect(Enum.map(dropped, &elem(&1, 0)))} removed for being constant."
            ]
        end

      ones = :binary.copy(<<1.0::float-64-native>>, div(byte_size(hd(bins)), 8))
      {Enum.map(kept, &elem(&1, 0)), [ones | Enum.map(kept, &elem(&1, 1))], warnings}
    else
      {names, bins, []}
    end
  end

  defp resolve_names(given, nil, count) do
    given || Enum.map(1..count, &"var_#{&1}")
  end

  defp resolve_names(_given, name_x, count) do
    names = Enum.map(name_x, &to_string/1)

    if length(names) != count do
      raise ArgumentError, "name_x has #{length(names)} names for #{count} x columns"
    end

    names
  end

  defp varies?(bin) do
    [first | rest] = for <<v::float-64-native <- bin>>, do: v
    Enum.any?(rest, &(&1 != first))
  end

  # -- result --------------------------------------------------------------------------

  defp build(e, n, k, names, warnings, opts) do
    fields =
      [
        n: n,
        k: k,
        betas: e.betas,
        vm: rows(e.vm, k),
        xtx: rows(e.xtx, k),
        xtxi: rows(e.xtxi, k),
        std_err: e.std_err,
        t_stat: Enum.chunk_every(e.t_stat, 2) |> Enum.map(&List.to_tuple/1),
        predy: e.predy,
        u: e.u,
        robust: opts[:robust],
        name_x: names,
        name_y: to_string(opts[:name_y]),
        warnings: warnings ++ extra_warnings(opts)
      ] ++
        for(key <- @scalar_keys, Map.has_key?(e, key), do: {key, hd(e[key])}) ++
        for(key <- @pair_keys, Map.has_key?(e, key), do: {key, List.to_tuple(e[key])}) ++
        diagnostics(e)

    struct!(OLS, fields)
  end

  defp extra_warnings(opts) do
    base =
      if opts[:white_test] and opts[:robust] != nil,
        do: [
          "White test not available when standard errors are estimated by HAC or White correction."
        ],
        else: []

    if opts[:spat_diag] and opts[:robust] == :hac,
      do: base ++ ["Spatial diagnostics are not available for HAC estimation."],
      else: base
  end

  defp diagnostics(e) do
    [
      jarque_bera: map_of(e[:jarque_bera], [:df, :jb, :pvalue]),
      breusch_pagan: map_of(e[:breusch_pagan], [:df, :bp, :pvalue]),
      koenker_bassett: map_of(e[:koenker_bassett], [:df, :kb, :pvalue]),
      white: white(e),
      vif: vif(e[:vif]),
      moran_res: e[:moran_res] && List.to_tuple(e[:moran_res])
    ]
  end

  defp map_of(nil, _), do: nil

  defp map_of([df | rest], [_ | keys]) do
    %{df: round(df)} |> Map.merge(Map.new(Enum.zip(keys, rest)))
  end

  defp white(%{white: w}), do: map_of(w, [:df, :wh, :pvalue])
  defp white(%{white_not_computed: _}), do: {:not_computed, :multicollinearity}
  defp white(_), do: nil

  defp vif(nil), do: nil

  defp vif(flat) do
    flat
    |> Enum.chunk_every(2)
    |> Enum.map(fn
      [:nan, :nan] -> nil
      [v, t] -> {v, t}
    end)
  end

  defp rows(flat, k), do: Enum.chunk_every(flat, k)

  # Dispatch through a runtime module argument so the optional dependencies are never
  # referenced as compile-time remote calls.
  defp call(mod, fun, args), do: apply(mod, fun, args)

  defp unwrap({:ok, list}), do: list
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  @entry_keys ~w(betas vm xtx xtxi std_err t_stat predy u utu sig2 sig2n sig2n_k mean_y std_y r2 ar2
                 f_stat logll aic schwarz sig2ml condition_number jarque_bera breusch_pagan
                 koenker_bassett white white_not_computed vif lm_error lm_lag rlm_error rlm_lag
                 lm_sarma lm_wx lm_spdurbin rlm_wx rlm_durlag lm_slxerr moran_res)a

  defp entry_key(name) do
    Enum.find(@entry_keys, &(Atom.to_string(&1) == name)) ||
      raise ArgumentError, "unexpected native result #{inspect(name)}"
  end

  # -- summary -------------------------------------------------------------------------

  defp fmt(v) when is_float(v), do: :erlang.float_to_binary(v, decimals: 6)
  defp fmt(v) when is_integer(v), do: Integer.to_string(v)
  defp fmt(v), do: to_string(v)

  defp fmtp(v) when is_float(v) and v > 0.0 and v < 1.0e-5,
    do: :erlang.float_to_binary(v, scientific: 2)

  defp fmtp(v) when is_float(v), do: :erlang.float_to_binary(v, decimals: 5)
  defp fmtp(v), do: to_string(v)

  defp sqrt(v) when is_float(v) and v >= 0, do: :math.sqrt(v)
  defp sqrt(_), do: :nan

  defp top_diagnostics(%OLS{f_stat: nil}), do: nil

  defp top_diagnostics(r) do
    {f, fp} = r.f_stat

    [
      "F-statistic        : #{fmt(f)} (p #{fmtp(fp)})",
      "Log likelihood     : #{fmt(r.logll)}",
      "Akaike info crit.  : #{fmt(r.aic)}",
      "Schwarz criterion  : #{fmt(r.schwarz)}"
    ]
    |> Enum.join("\n")
  end

  defp coefficient_table(r) do
    header =
      "\n#{pad("Variable", 18)}#{pad("Coefficient", 14)}#{pad("Std.Error", 14)}#{pad("t-Statistic", 14)}Probability"

    lines =
      for {name, b, se, {t, p}} <- zip4(r.name_x, r.betas, r.std_err, r.t_stat) do
        pad(name, 18) <> pad(fmt(b), 14) <> pad(fmt(se), 14) <> pad(fmt(t), 14) <> fmtp(p)
      end

    Enum.join([header | lines], "\n")
  end

  defp zip4(a, b, c, d), do: Enum.zip([a, b, c, d]) |> Enum.map(& &1)

  defp pad(s, w), do: String.pad_trailing(to_string(s), w)

  defp regression_diagnostics(%OLS{f_stat: nil}), do: nil

  defp regression_diagnostics(r) do
    rows =
      [
        {"MULTICOLLINEARITY CONDITION NUMBER", "", fmt(r.condition_number), ""},
        {"Jarque-Bera", r.jarque_bera.df, fmt(r.jarque_bera.jb), fmtp(r.jarque_bera.pvalue)},
        {"Breusch-Pagan", r.breusch_pagan.df, fmt(r.breusch_pagan.bp),
         fmtp(r.breusch_pagan.pvalue)},
        {"Koenker-Bassett", r.koenker_bassett.df, fmt(r.koenker_bassett.kb),
         fmtp(r.koenker_bassett.pvalue)}
      ] ++ white_row(r.white)

    "\nREGRESSION DIAGNOSTICS\n" <>
      Enum.map_join(rows, "\n", fn {name, df, value, p} ->
        pad(name, 36) <> pad(df, 5) <> pad(value, 14) <> p
      end)
  end

  defp white_row(nil), do: []
  defp white_row({:not_computed, _}), do: [{"White", "", "not computed (multicollinearity)", ""}]
  defp white_row(w), do: [{"White", w.df, fmt(w.wh), fmtp(w.pvalue)}]

  defp vif_table(%OLS{vif: nil}), do: nil

  defp vif_table(r) do
    lines =
      r.name_x
      |> Enum.zip(r.vif)
      |> Enum.reject(fn {_, v} -> v == nil end)
      |> Enum.map(fn {name, {v, t}} -> pad(name, 18) <> pad(fmt(v), 14) <> fmt(t) end)

    Enum.join(
      [
        "\nVARIANCE INFLATION FACTOR",
        pad("Variable", 18) <> pad("VIF", 14) <> "Tolerance" | lines
      ],
      "\n"
    )
  end

  defp spatial_diagnostics(%OLS{lm_error: nil, moran_res: nil}), do: nil

  defp spatial_diagnostics(r) do
    rows =
      [
        moran_row(r.moran_res),
        {"Lagrange Multiplier (lag)", 1, r.lm_lag},
        {"Robust LM (lag)", 1, r.rlm_lag},
        {"Lagrange Multiplier (error)", 1, r.lm_error},
        {"Robust LM (error)", 1, r.rlm_error},
        {"Lagrange Multiplier (SARMA)", 2, r.lm_sarma}
      ]
      |> Enum.reject(fn row -> is_nil(row) or elem(row, 2) == nil end)

    "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n" <>
      Enum.map_join(rows, "\n", fn {name, df, {v, p}} ->
        pad(name, 36) <> pad(df, 5) <> pad(fmt(v), 14) <> fmtp(p)
      end)
  end

  defp moran_row(nil), do: nil
  defp moran_row({_i, z, p}), do: {"Moran's I (error) z-value", "", {z, p}}

  defp warning_lines(%OLS{warnings: []}), do: nil
  defp warning_lines(r), do: "\nWARNINGS\n" <> Enum.join(r.warnings, "\n")

  @doc """
  Seemingly unrelated regressions with spatial error processes, estimated by generalized moments
  (`spreg.SURerrorGM`).

  `ys` is a list of `y` vectors and `xs` the matching designs as in `sur/3` (a constant
  `"Constant_<eq>"` is added to every equation); `graph` is the `Spatex.Graph` used as given
  (spreg never row-standardizes implicitly). Returns a `Spatex.Regression.SURerrorGM`.

  Estimation follows spreg: OLS per equation; the spatial error parameter of every equation is
  the Kelejian-Prucha moment estimate of its OLS residuals (as in `gm_error/4`); the variables
  and the OLS residuals are filtered with those parameters (`v - lambda W v`, the constant
  included), and one SUR GLS step on the filtered variables gives the coefficients, with
  `sig = E'E / n` of the filtered OLS residuals. There is no iteration. The Chow tests of
  coefficient equality across equations exist when every equation has the same number of
  coefficients. spreg reports no further diagnostic for this class (its `spat_diag` and `vm`
  options are not implemented there, `nonspat_diag` only affects its text summary), so there are
  no such options; the lambda equality test belongs to the ML estimator.

  spreg minimizes the moment objective with L-BFGS-B, which stops up to ~5e-6 from the minimizer;
  Spatex returns the exact minimizer (see the spreg GM notes in `docs/parity.md`).

  ## Options

    * `:name_bigy` - one name per equation; `:name_bigx` - one list of regressor names per
      equation.

  `regimes` is not supported (any `:regimes` option raises `ArgumentError`).

  ## Examples

      iex> y1 = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
      iex> y2 = [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
      iex> focal = Enum.flat_map(0..11, &[&1, &1])
      iex> neighbor = Enum.flat_map(0..11, &[rem(&1 + 1, 12), rem(&1 + 11, 12)])
      iex> g = Spatex.Graph.from_arrays(focal, neighbor) |> Spatex.Graph.transform(:r)
      iex> r = Spatex.Regression.sur_error_gm([y1, y2], [[x], [x]], g)
      iex> {r.n_eq, r.k, length(r.lam), length(r.chow)}
      {2, [2, 2], 2, 2}
  """
  @spec sur_error_gm([term()], [term()], Graph.t(), keyword()) :: SURerrorGM.t()
  def sur_error_gm(ys, xs, graph, opts \\ []), do: SurErrorGm.fit(ys, xs, graph, opts)

  @doc false
  def sur_error_gm_at(ys, xs, graph, lambdas, opts \\ []),
    do: SurErrorGm.fit(ys, xs, graph, opts, lambdas)

  @doc """
  Seemingly unrelated regressions with a spatial error process in every equation, estimated by
  maximum likelihood (`spreg.SURerrorML`, spreg 1.9.1).

  `ys` is a list with the dependent variable of each equation (at least two), `xs` the list of
  regressor sets (a list of columns each; the constant is added per equation) and `graph` a
  `Spatex.Graph`. Returns a `Spatex.Regression.SURerrorML`.

  The estimation is spreg's: the lambda of each equation is first estimated by `ml_error/4` (with
  `epsilon` as tolerance), the residuals of the iterated classic SUR start the loop, and every round
  runs a SUR GLS step on the variables filtered with the current lambdas (sigma from the filtered
  residuals of the previous round) and then maximizes the concentrated likelihood
  `-n/2 ln|sigma| + sum_r ln|I - lambda_r W|` over all lambdas, until it moves by at most
  `epsilon`. The reported `b_sur`, `varb` and `sig` are those of the last SUR step, `lam` the
  output of the last search. The log-Jacobian is the exact `ln|det(I - lambda W)|` (as spreg's
  sparse LU) from a dense LU, so `n` is bounded by `:max_dense_n`.

  Reported tests: the likelihood-ratio test of a diagonal sigma (`lr_test`, `nonspat_diag`), the
  likelihood-ratio test of the spatial parameters against the classic SUR (`lr_lambda`,
  `spat_diag`), with `vm: true` the asymptotic covariance of the lambdas and sigma, the standard
  errors of the lambdas (`lam_setp`), the Wald test that the lambdas are equal (`lam_test`) and,
  with `spat_diag`, that they are jointly zero (`join_lam`), and the Chow tests of coefficient
  equality across equations (`chow`, equal-size equations).

  spreg searches the lambdas with L-BFGS-B (finite-difference gradient, loose tolerances); Spatex
  minimizes the same objective with analytic gradients, so its lambdas are the tighter optimum
  and agree with spreg's to ~1e-4 (see the spreg spatial SUR notes in `docs/parity.md`).

  ## Options

    * `:epsilon` - tolerance of the loop and of the per-equation ML fits (default `1.0e-7`).
    * `:nonspat_diag` (default `true`), `:spat_diag` (default `false`), `:vm` (default `false`) -
      the tests above.
    * `:max_dense_n` - cap on `n` (default 5000).
    * `:name_bigy` - one name per equation; `:name_bigx` - one list of regressor names per equation.

  `regimes` is not supported (any `:regimes` option raises `ArgumentError`).

  ## Examples

      iex> y1 = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
      iex> y2 = [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
      iex> focal = Enum.flat_map(0..11, &[&1, &1])
      iex> neighbor = Enum.flat_map(0..11, &[rem(&1 + 1, 12), rem(&1 + 11, 12)])
      iex> g = Spatex.Graph.from_arrays(focal, neighbor) |> Spatex.Graph.transform(:r)
      iex> r = Spatex.Regression.sur_error_ml([y1, y2], [[x], [x]], g, vm: true)
      iex> {r.n_eq, r.k, length(r.lam), length(r.vm), length(r.chow)}
      {2, [2, 2], 2, 5, 2}
  """
  @spec sur_error_ml([term()], [term()], Graph.t(), keyword()) :: SURerrorML.t()
  def sur_error_ml(ys, xs, graph, opts \\ []), do: SurErrorMl.fit(ys, xs, graph, opts)

  # Test hook: replays spreg's own lambda path (the lambdas each search returned) instead of
  # searching.
  @doc false
  def sur_error_ml_at(ys, xs, graph, path, opts \\ []),
    do: SurErrorMl.fit(ys, xs, graph, opts, path)

  @doc """
  Seemingly unrelated regressions with a spatial lag in every equation, estimated by three-stage
  least squares (`spreg.SURlagIV`).

  `ys` and `xs` are lists with one entry per equation (at least two): the `y` vector and the
  exogenous design, each given as in `ols/3` (a constant `"Constant_<eq>"` is added to every
  design). `graph` is a `Spatex.Graph` used as given (row-standardize it first). Returns a
  `Spatex.Regression.SURlagIV`.

  Every equation gets `W y` as its last endogenous variable (after the user's `:yends`) and, as
  instruments, `[Q, W X, ..., W^p X, W Q, ..., W^p Q]` with `p = :w_lags` and `Q` the user
  instruments `:qs` (the lags of `Q` only with `lag_q: true`), then `three_sls/5` estimates the
  system: 2SLS per equation, `sig = E'E / n` of those residuals, one GLS step. Reported: the
  coefficients (the spatial lag coefficient of each equation is `rho`), `varb`, standard errors
  and z statistics, the cross-equation `sig` and `corr`, the Chow tests of coefficient equality
  across equations (equal-size equations) and, with `spat_diag: true`, the Wald test that the
  spatial lag coefficients are jointly zero (`joinrho`). spreg computes no spatial impacts, no
  other diagnostic and no other `vm` for this class (its `nonspat_diag` and `vm` flags only change
  the text summary), so Spatex has no such options.

  ## Options

    * `:yends`, `:qs` - per equation, the additional endogenous variables and their instruments
      (both or neither; unnamed columns are `var_<i>`).
    * `:w_lags` - order of the spatial lags used as instruments (integer >= 1, default 1).
    * `:lag_q` - also lag the user instruments (default `true`).
    * `:spat_diag` - report `joinrho` (default `false`).
    * `:name_bigy` - one name per equation; `:name_bigx`, `:name_bigyend`, `:name_bigq` - one list
      of names per equation (of the user variables).

  `regimes` is not supported (any `:regimes` option raises `ArgumentError`).

  ## Examples

      iex> y1 = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
      iex> y2 = [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
      iex> x = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
      iex> focal = Enum.flat_map(0..11, &[&1, &1])
      iex> neighbor = Enum.flat_map(0..11, &[rem(&1 + 1, 12), rem(&1 + 11, 12)])
      iex> g = Spatex.Graph.from_arrays(focal, neighbor) |> Spatex.Graph.transform(:r)
      iex> r = Spatex.Regression.sur_lag_iv([y1, y2], [[x], [x]], g, spat_diag: true)
      iex> {r.n_eq, r.k, length(r.rho), length(r.chow), elem(r.joinrho, 1)}
      {2, [3, 3], 2, 3, 2}
  """
  @spec sur_lag_iv([term()], [term()], Graph.t(), keyword()) :: SURlagIV.t()
  def sur_lag_iv(ys, xs, graph, opts \\ []), do: SurLagIv.fit(ys, xs, graph, opts)
end
