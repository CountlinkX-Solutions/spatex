defmodule Spatex.GWR do
  @moduledoc """
  Geographically weighted regression (Gaussian, Poisson and binomial families), bandwidth
  selection and multiscale GWR (MGWR), after `mgwr` (PySAL).

  `fit/4` estimates one weighted regression per observation, with weights that decay with
  distance, giving spatially varying coefficients, standard errors, local R-squared and the
  diagnostics of `mgwr.gwr.GWRResults`. `select_bandwidth/4` picks the bandwidth by
  golden-section (or equal-interval) search over AICc, AIC, BIC or the cross-validation score.
  `mgwr/4` lets every covariate have its own bandwidth (backfitting) and computes standard
  errors from the per-covariate hat matrices. Results are checked against mgwr 2.2.1 on the
  Georgia example and on seeded synthetic data.

  ## Families

  `family: :gaussian` (the default) is weighted least squares. `family: :poisson` (log link,
  optional `offset:` exposure) and `family: :binomial` (logit link, 0/1 responses) are fitted
  at every location by iteratively re-weighted least squares exactly as `spglm.iwls` does,
  including its starting values and its stopping rule (the smallest coefficient change is
  compared with `iwls_tol`, default `1.0e-5`, at most `iwls_max_iter` = 20 iterations). Their
  information criteria are deviance based (`AIC = deviance + 2 tr(S)`), and results carry
  `d2`, `adj_d2` (percent deviance explained) and the local `p_dev`.

  ## Bandwidths and kernels

  With `fixed: false` (the default, as in mgwr) the bandwidth is a number of nearest neighbors,
  the focal point included, and the kernel adapts to the local point density. With `fixed: true`
  it is a distance in the units of the coordinates (kilometers with `spherical: true`, where
  points are `{longitude, latitude}` in degrees). The kernels are `:bisquare` (the default,
  zero beyond the bandwidth), `:gaussian` and `:exponential` (never zero).

  ## Memory and cost

  Neither the hat matrix nor the weight matrix is ever stored: each location's weights are
  recomputed on the fly and only per-location vectors are kept, so memory is `O(n * k)`. A fit
  costs `O(n^2 * k^2)` time (in parallel over locations; a GLM multiplies that by the IWLS
  iterations) and a bandwidth search repeats that a few dozen times. Both raise for more than
  `max_n` observations (default `10_000`) unless you raise the limit explicitly.

  MGWR inference needs one `n x n` hat matrix per covariate. They are never held in full: they
  are built in column blocks (`chunk_size`, default 256), so memory is about
  `8 * (k + 2) * n * chunk_size` bytes, but the time is `O(iterations * k * n^3)`. `mgwr/4`
  therefore refuses inference above `max_n` observations (default `5_000`); see its docs.

  ## Examples

      iex> pts = for r <- 0..3, c <- 0..3, do: {c * 1.0, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 1.0 + 2.0 * xv + rem(i, 3) * 0.1
      iex> r = Spatex.GWR.fit(pts, y, [x], bw: :infinity)
      iex> {r.n, r.k, Float.round(r.tr_s, 6)}
      {16, 2, 2.0}
  """

  alias Spatex.GWR.{
    Bandwidth,
    Collinearity,
    Design,
    Intervals,
    MGWR,
    MGWRResult,
    Prediction,
    Result,
    SpatialVariability,
    Variability
  }

  alias Spatex.{Input, Native}

  @criteria [:aicc, :aic, :bic, :cv]
  @families [:gaussian, :poisson, :binomial]
  @default_max_n 10_000
  @default_max_cells 4_000_000

  @glm_opts [:family, :offset, :iwls_tol, :iwls_max_iter]
  @fit_opts [:bw, :kernel, :fixed, :constant, :spherical, :sigma2_v1, :max_n] ++ @glm_opts
  @fit_only_opts [:hat_matrix, :max_cells]
  @predict_opts @fit_opts ++ [:exog_scale, :exog_resid]
  @collinearity_opts [:bw, :bws, :kernel, :fixed, :constant, :spherical, :max_n]
  @select_opts @glm_opts ++
                 [
                   :criterion,
                   :kernel,
                   :fixed,
                   :constant,
                   :spherical,
                   :bw_min,
                   :bw_max,
                   :include_global,
                   :search,
                   :interval,
                   :x_glob,
                   :tol,
                   :max_iter,
                   :max_n
                 ]

  @doc """
  Fits Gaussian GWR at every observation.

  `points` are `{x, y}` pairs (or anything `Spatex.Input.to_points_binary/1` accepts), `y` the
  response and `x` the regressors: a list of columns, a single column, a `name => column` map
  (ordered by name) or an `{n, k}` `Nx.Tensor`. Like mgwr, constant columns of `x` are dropped
  and an intercept is prepended unless `constant: false`.

  ## Options

    * `:bw` (required) - the bandwidth: a neighbor count (adaptive), a distance (`fixed: true`)
      or `:infinity` for the global model. A fractional adaptive count is truncated, with a
      warning, like mgwr. An adaptive count must be between 2 and `n`.
    * `:kernel` - `:bisquare` (default), `:gaussian` or `:exponential`.
    * `:fixed` - fixed distance bandwidth (default `false`, adaptive).
    * `:constant` - add an intercept (default `true`).
    * `:spherical` - great-circle distances in km for `{lon, lat}` degrees (default `false`).
    * `:sigma2_v1` - residual variance `resid_ss / (n - tr(S))` (default `true`, mgwr's default)
      instead of `resid_ss / (n - 2 tr(S) + tr(S'S))`; `enp` follows.
    * `:max_n` - refuse more observations than this (default #{@default_max_n}).
    * `:family` - `:gaussian` (default), `:poisson` or `:binomial`. Poisson needs a
      non-negative `y`; binomial a `y` of `0`s and `1`s.
    * `:offset` - Poisson exposure (one strictly positive number per observation; default
      ones). Not allowed for the other families.
    * `:iwls_tol`, `:iwls_max_iter` - IWLS stopping rule of the GLM families (defaults
      `1.0e-5` and `20`, mgwr's).
    * `:hat_matrix` - keep the full `n x n` hat matrix `S` in `result.hat_matrix` (mgwr's
      `hat_matrix=True`; `S y` are the fitted values and `diag(S)` is `influ`): `true` for a list
      of `n` rows, `:tensor` for an `{n, n}` `Nx.Tensor` of type `{:f, 64}` (needs the optional
      `:nx` dependency), `false` (default) to keep the `O(n k)` memory of an ordinary fit. For the
      GLM families the entries are the final IWLS weighted operator, as mgwr's.
    * `:max_cells` - with `hat_matrix`, refuse more than this many cells `n^2` (default
      #{@default_max_cells}, `n = 2000`).

  GLM results leave `local_r2`, `r2` and `adj_r2` as `nil` and fill `deviance`,
  `resid_deviance`, `p_dev` and `iterations` instead; for a zero Poisson count Spatex uses
  `0 * ln(0) = 0` in the local deviance where mgwr returns `NaN`.

  Raises `ArgumentError` on invalid input or a singular local design matrix.

  ## Examples

      iex> pts = for r <- 0..4, c <- 0..4, do: {c * 1.0, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 2.0 * xv + rem(i, 4) * 0.1
      iex> r = Spatex.GWR.fit(pts, y, x, bw: 12, kernel: :gaussian)
      iex> {length(r.params), length(hd(r.params))}
      {25, 2}

  With `hat_matrix: true` the full hat matrix is kept; `S y` are the fitted values:

      iex> pts = for r <- 0..4, c <- 0..4, do: {c * 1.0, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 2.0 * xv + rem(i, 4) * 0.1
      iex> r = Spatex.GWR.fit(pts, y, x, bw: 12, kernel: :gaussian, hat_matrix: true)
      iex> {length(r.hat_matrix), length(hd(r.hat_matrix))}
      {25, 25}
  """
  @spec fit(term(), term(), term(), keyword()) :: Result.t()
  def fit(points, y, x, opts \\ []) do
    opts = validate!(opts, @fit_opts ++ @fit_only_opts, :fit)
    model = model!(points, y, x, opts)
    check_hat_size!(opts, model)
    {bw, global, warnings} = bandwidth!(opts, model)
    {glm, offset} = glm_args(opts, model)

    raw =
      model.coords
      |> Native.gwr_fit(
        model.y,
        model.x,
        {Atom.to_string(opts[:kernel]), opts[:fixed], opts[:spherical], model.k},
        glm,
        offset,
        {bw, global},
        {opts[:sigma2_v1], opts[:hat_matrix] != false}
      )
      |> unwrap()

    {hat, raw} = List.keytake(raw, "hat", 0) |> then(fn {{_, bin}, rest} -> {bin, rest} end)

    entries =
      Map.new(raw, fn {key, bin} -> {String.to_existing_atom(key), Input.from_f64_binary(bin)} end)

    entries
    |> build(model, opts, global_bw(bw, global, opts), warnings ++ model.warnings)
    |> Map.put(:hat_matrix, hat_matrix(hat, opts[:hat_matrix], model.n))
  end

  @doc """
  Predicts at new locations (`mgwr.gwr.GWR.predict`).

  The model is calibrated on `points`, `y`, `x` exactly as in `fit/4`; at every point of
  `new_points` the weighted regression of the calibration sample (weights from the distances
  to the calibration locations) gives local coefficients, their standard errors and the
  prediction `x_new . params`. `x_new` has the same columns as `x` (a list of columns, a single
  column, a `name => column` map or an `{m, k}` tensor with one row per new point); constant
  columns of `x` are dropped from `x_new` too and the intercept is added unless `constant:
  false`.

  Nothing of size `m x n` is stored: weights are recomputed per prediction point, in parallel,
  so memory is `O(m k)` plus `O(n)` per worker. The calibration sample is limited by `max_n`.

  ## Options

  As in `fit/4` (`:bw` is required) plus

    * `:exog_scale` and `:exog_resid` - the residual variance and the response residuals of the
      calibration sample (`n` values) to use for the standard errors and the local R-squared.
      Both or neither: by default (mgwr's behavior) a model is fitted at the calibration
      locations first and its `scale` and `resid` are used. Giving them skips that fit.

  For `family: :poisson` and `:binomial` the predictions are the linear predictor (log mean or
  log-odds), as in mgwr, and there is no local R-squared. Unlike mgwr, more prediction points
  than calibration observations are supported (mgwr fails with an `IndexError`).

  Returns a `Spatex.GWR.Prediction`. Raises `ArgumentError` on invalid input or when a local
  design matrix is singular, for example a fixed bandwidth with no calibration point in range
  of a prediction point.

  ## Examples

      iex> pts = for r <- 0..4, c <- 0..4, do: {c * 1.0, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 1.0 + 2.0 * xv + rem(i, 4) * 0.1
      iex> p = Spatex.GWR.predict(pts, y, x, [{1.5, 1.5}], [[1.5]], bw: :infinity)
      iex> {p.m, length(hd(p.params)), Float.round(hd(p.predictions), 3)}
      {1, 2, 4.142}
  """
  @spec predict(term(), term(), term(), term(), term(), keyword()) :: Prediction.t()
  def predict(points, y, x, new_points, x_new, opts \\ []) do
    opts = validate!(opts, @predict_opts, :predict)
    model = model!(points, y, x, opts)
    {bw, global, warnings} = bandwidth!(opts, model)
    {glm, offset} = glm_args(opts, model)
    {new_coords, m} = Input.to_points_binary(new_points)
    p = Design.regressors!(x_new, model, opts[:constant], m)

    entries =
      model.coords
      |> Native.gwr_predict(
        model.y,
        model.x,
        {Atom.to_string(opts[:kernel]), opts[:fixed], opts[:spherical], model.k},
        glm,
        offset,
        {bw, global, opts[:sigma2_v1]},
        {new_coords, p, exog_args(opts, model)}
      )
      |> unwrap()
      |> Map.new(fn {key, bin} -> {String.to_existing_atom(key), Input.from_f64_binary(bin)} end)

    glm? = opts[:family] != :gaussian

    %Prediction{
      m: m,
      n: model.n,
      k: model.k,
      family: opts[:family],
      bw: global_bw(bw, global, opts),
      kernel: opts[:kernel],
      fixed: opts[:fixed],
      spherical: opts[:spherical],
      constant: opts[:constant],
      params: rows(entries.params, model.k),
      bse: rows(entries.bse, model.k),
      tvalues: rows(entries.tvalues, model.k),
      predictions: entries.predictions,
      local_r2: gaussian_only(entries.local_r2, glm?),
      scale: scalar(entries.scale),
      warnings: warnings ++ model.warnings
    }
  end

  @doc """
  Local collinearity diagnostics of the geographically weighted design matrix
  (`GWRResults.local_collinearity` and `MGWRResults.local_collinearity` of mgwr).

  `points` and `x` are as in `fit/4` (the response plays no part, so none is passed; constant
  columns of `x` are dropped and an intercept is added unless `constant: false`). For every
  location the kernel weights are normalized and give

    * the local correlation coefficients and variance inflation factors of the explanatory
      variables (the intercept left out) - with `:bw` only;
    * the local condition number and the variance-decomposition proportions of the smallest
      singular value of the column-scaled weighted design (Belsley, Kuh and Welsch).

  With `:bws`, one bandwidth per design column as an MGWR has (pass `mgwr/4`'s `bws`; the
  design must be the one the MGWR was fitted on, standardized if it was), only the condition
  number and the decomposition proportions exist, as in mgwr.

  At least two explanatory variables are needed (mgwr crashes inside numpy with fewer; Spatex
  raises). A singular local correlation matrix raises `ArgumentError`.

  ## Options

    * `:bw` - the bandwidth: a neighbor count (adaptive), a distance (`fixed: true`) or
      `:infinity`. Exactly one of `:bw` and `:bws` is required.
    * `:bws` - a list with one such bandwidth per design column (the intercept first).
    * `:kernel`, `:fixed`, `:constant`, `:spherical`, `:max_n` - as in `fit/4`.

  Returns a `Spatex.GWR.Collinearity`.

  ## Examples

      iex> pts = for r <- 0..4, c <- 0..4, do: {c * 1.0, r * 1.0}
      iex> x1 = for {cx, _} <- pts, do: cx
      iex> x2 = for {_, cy} <- pts, do: cy
      iex> x3 = for {{cx, cy}, i} <- Enum.with_index(pts), do: cx * cy + rem(i, 3)
      iex> r = Spatex.GWR.local_collinearity(pts, [x1, x2, x3], bw: 18)
      iex> {r.n, r.k, length(hd(r.vif)), length(hd(r.vdp))}
      {25, 4, 3, 4}
  """
  @spec local_collinearity(term(), term(), keyword()) :: Collinearity.t()
  def local_collinearity(points, x, opts \\ []) do
    Design.check_known!(opts, @collinearity_opts)

    opts =
      Keyword.merge(
        [
          kernel: :bisquare,
          fixed: false,
          constant: true,
          spherical: false,
          max_n: @default_max_n
        ],
        opts
      )

    Design.check_common!(opts)
    model = Design.model!(points, nil, x, opts)
    {bandwidths, warnings} = collinearity_bandwidths!(opts, model)
    shape = {Atom.to_string(opts[:kernel]), opts[:fixed], opts[:spherical], model.k}

    {entries, result} =
      case bandwidths do
        {:single, bw, global} ->
          native =
            Native.gwr_collinearity(
              model.coords,
              model.x,
              shape,
              if(global, do: nil, else: bw),
              opts[:constant]
            )

          {native, %{bw: global_bw(bw, global, opts), bws: nil}}

        {:multi, list} ->
          native = Native.mgwr_collinearity(model.coords, model.x, shape, native_bws(list))

          {native,
           %{bw: nil, bws: Enum.map(list, fn {bw, global} -> global_bw(bw, global, opts) end)}}
      end

    entries =
      entries
      |> unwrap()
      |> Map.new(fn {key, bin} -> {String.to_existing_atom(key), Input.from_f64_binary(bin)} end)

    [ncor, nvif] = entries.shape |> Enum.map(&trunc/1)
    single? = result.bws == nil
    first = if opts[:constant], do: 1, else: 0

    %Collinearity{
      n: model.n,
      k: model.k,
      constant: opts[:constant],
      kernel: opts[:kernel],
      fixed: opts[:fixed],
      spherical: opts[:spherical],
      bw: result.bw,
      bws: result.bws,
      corr: if(single?, do: Enum.chunk_every(entries.corr, ncor)),
      corr_pairs: if(single?, do: pairs(first, model.k)),
      vif: if(single?, do: Enum.chunk_every(entries.vif, nvif)),
      condition_number: entries.condition_number,
      vdp: rows(entries.vdp, model.k),
      warnings: warnings ++ model.warnings
    }
  end

  defp pairs(first, k), do: for(i <- first..(k - 2)//1, j <- (i + 1)..(k - 1), do: {i, j})

  defp native_bws(list), do: Enum.map(list, fn {bw, global} -> if(global, do: nil, else: bw) end)

  defp collinearity_bandwidths!(opts, model) do
    case {Keyword.has_key?(opts, :bw), Keyword.has_key?(opts, :bws)} do
      {false, false} ->
        raise ArgumentError, "bw (GWR) or bws (MGWR, one per design column) is required"

      {true, true} ->
        raise ArgumentError, "give either bw or bws, not both"

      {true, false} ->
        {bw, global, warnings} = bandwidth!(opts, model)
        {{:single, bw, global}, warnings}

      {false, true} ->
        bws = opts[:bws]

        unless is_list(bws) and length(bws) == model.k do
          raise ArgumentError,
                "bws needs one bandwidth per design column (#{model.k}, the intercept first)"
        end

        {list, warnings} =
          Enum.map_reduce(bws, [], fn bw, acc ->
            {value, global, warn} = bandwidth!(Keyword.put(opts, :bw, bw), model)
            {{value, global}, acc ++ warn}
          end)

        {{:multi, list}, warnings}
    end
  end

  defp exog_args(opts, model) do
    case {opts[:exog_scale], opts[:exog_resid]} do
      {nil, nil} ->
        nil

      {scale, resid} when scale != nil and resid != nil ->
        unless is_number(scale) and scale > 0,
          do: raise(ArgumentError, "exog_scale must be a positive number")

        {bin, _} = Input.to_f64_binary(resid, name: "exog_resid", length: model.n)
        {scale * 1.0, bin}

      _ ->
        raise ArgumentError, "exog_scale and exog_resid must both be given or both omitted"
    end
  end

  @doc """
  Selects the bandwidth that minimizes an information criterion (`mgwr.sel_bw.Sel_BW`).

  Arguments are as in `fit/4`. Returns a `Spatex.GWR.Bandwidth`.

  ## Options

    * `:criterion` - `:aicc` (default), `:aic`, `:bic` or `:cv`.
    * `:kernel`, `:fixed`, `:constant`, `:spherical`, `:max_n`, `:family`, `:offset`,
      `:iwls_tol`, `:iwls_max_iter` - as in `fit/4`. For the GLM families the criteria are the
      deviance based ones (`:cv` is computed from the response residuals).
    * `:search` - `:golden_section` (default), `:interval` or `:scipy`.
    * `:bw_min`, `:bw_max` - search bounds. By default adaptive searches run from
      `40 + 2 * (columns + 1)` neighbors to `n`, which therefore needs `n` of at least that size
      (mgwr crashes inside numpy for smaller samples; Spatex raises an `ArgumentError`), and
      fixed searches from half the smallest inter-point distance to twice the largest.
    * `:include_global` - also evaluate the global model (`bw: :infinity`) and keep it when it
      is at least as good (mgwr's `bw_max=np.inf`). Cannot be combined with a numeric `:bw_max`.
    * `:x_glob` - mgwr's `X_glob`: extra columns (any shape accepted for `x`) that only change
      the default search bounds. Like mgwr's `Sel_BW`, which has no semi-parametric fit, Spatex
      uses only their number: the default adaptive lower bound becomes
      `40 + 2 * (local columns + x_glob columns + 1)` neighbors. The values are never read (each
      column must still have `n` numbers) and explicit `:bw_min`/`:bw_max` take precedence.
    * `:interval` - step of the `:interval` search, which evaluates `bw_min, bw_min + interval,
      ...` up to `bw_max` (all three are required).
    * `:tol` - golden-section tolerance on the change of the criterion (default `1.0e-6`);
      `:max_iter` - iteration cap (default `200`).

  `search: :scipy` is mgwr's `search_method='scipy'`: SciPy's bounded Brent minimizer
  (`minimize_scalar(method='bounded')`, `xatol = 1.0e-5`, at most 500 evaluations, a literal port)
  over the same default bounds. As in mgwr the result `bw` is the optimizer's float (an adaptive
  fit truncates it to a neighbor count, so `fit/4` warns about it), the fixed bandwidth is not
  rounded, `history` is empty, and `nfev` and `converged` come from SciPy's result. The lower and
  upper bounds must differ, and `:include_global` is not available with it (mgwr ignores
  `bw_max=np.inf` here).

  The golden-section search follows mgwr literally: the probes of adaptive searches are rounded
  to integers, convergence is on the criterion difference between the two probes, and the
  reported fixed bandwidth is rounded to two decimals.

  ## Examples

      iex> pts = for r <- 0..5, c <- 0..5, do: {c * 1.0 + 0.01 * r, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 1.0 + xv + :math.sin(i)
      iex> b = Spatex.GWR.select_bandwidth(pts, y, [x], fixed: true, kernel: :gaussian)
      iex> b.bw > 0
      true

  The SciPy search returns the optimizer's float and its evaluation count:

      iex> pts = for r <- 0..5, c <- 0..5, do: {c * 1.0 + 0.01 * r, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 1.0 + xv + :math.sin(i)
      iex> b = Spatex.GWR.select_bandwidth(pts, y, [x], fixed: true, kernel: :gaussian, search: :scipy)
      iex> {b.history, b.converged, b.nfev > 3}
      {[], true, true}
  """
  @spec select_bandwidth(term(), term(), term(), keyword()) :: Bandwidth.t()
  def select_bandwidth(points, y, x, opts \\ []) do
    opts = validate!(opts, @select_opts, :select_bandwidth)
    model = model!(points, y, x, opts)
    {method, step, bounds} = search!(opts)
    n_glob = Design.glob_columns!(opts[:x_glob], model.n)
    {glm, offset} = glm_args(opts, model)

    {head, history} =
      model.coords
      |> Native.gwr_select(
        model.y,
        model.x,
        {Atom.to_string(opts[:kernel]), opts[:fixed], opts[:spherical], model.k},
        glm,
        offset,
        {Atom.to_string(opts[:criterion]), method, step, opts[:include_global], elem(bounds, 0),
         elem(bounds, 1)},
        {model.n_loc + n_glob, opts[:constant], opts[:tol], opts[:max_iter]}
      )
      |> unwrap()

    [bw, score, nfev, status] = Input.from_f64_binary(head)

    pairs =
      history
      |> Input.from_f64_binary()
      |> Enum.chunk_every(2)
      |> Enum.map(fn [b, s] -> {integer_bw(b, opts[:fixed]), s} end)

    %Bandwidth{
      bw: if(method == "scipy", do: bw, else: integer_bw(bw, opts[:fixed])),
      nfev: if(method == "scipy", do: trunc(nfev)),
      converged: if(method == "scipy", do: status == 0.0),
      score: score,
      history: pairs,
      criterion: opts[:criterion],
      family: opts[:family],
      kernel: opts[:kernel],
      fixed: opts[:fixed],
      spherical: opts[:spherical],
      constant: opts[:constant]
    }
  end

  @doc """
  Multiscale GWR (Gaussian): a separate bandwidth for every covariate.

  The bandwidths come from backfitting (`mgwr.search.multi_bw`): starting from a GWR with one
  bandwidth (`init_bw`, searched when not given), each covariate's partial residual is
  smoothed by a univariate GWR whose bandwidth is searched with the same golden-section (or
  interval) search as `select_bandwidth/4`, cycling through the covariates until the relative
  change of the fitted surfaces drops below `:tol_multi`. Then the effective number of
  parameters per covariate (`enp_j`) and the standard errors follow from the hat matrices of
  the model (Li and Fotheringham, 2020). Arguments are as in `fit/4`.

  mgwr expects `y` and the regressors to be standardized (mean 0, standard deviation 1) so the
  bandwidths are comparable; nothing enforces it. Pass `standardize: true` to have Spatex do it
  (coefficients are then on the standardized scale).

  ## Options

    * `:kernel`, `:fixed`, `:constant`, `:spherical` - as in `fit/4`. With `:constant` the
      intercept is the first of the `k` columns, so per-column lists start with it.
    * `:criterion`, `:search`, `:interval`, `:tol`, `:max_iter` - the bandwidth search settings
      of `select_bandwidth/4`, used for the initial and every univariate search.
    * `:multi_bw_min`, `:multi_bw_max` - search bounds per column: a number (all columns) or a
      list of `k` numbers or `nil`s. The `bw_min`/`bw_max` of `select_bandwidth/4` do not exist
      here (mgwr ignores them for MGWR as well).
    * `:init_bw` - starting bandwidth (default: searched with a one-bandwidth GWR).
    * `:tol_multi` (`1.0e-5`), `:max_iter_multi` (`200`) - backfitting convergence.
    * `:rss_score` - converge on the relative change of the residual sum of squares instead of
      mgwr's change-of-surface score (default `false`).
    * `:bws_same_times` - stop searching bandwidths after this many iterations without change
      (default `5`).
    * `:standardize` - standardize `y` and the non-intercept columns first (default `false`).
    * `:inference` - compute `enp_j`, standard errors and the criteria (default `true`). With
      `false` only the bandwidths, coefficients, fitted values and `resid_ss`/`r2` are returned.
    * `:chunk_size` - hat-matrix block width (default 256); it changes memory, not results.
    * `:max_n` - inference is refused for more observations than this (default `5_000`): it
      costs `O(iterations * k * n^3)` time and about `8 * (k + 2) * n * chunk_size` bytes. Raise
      it deliberately or pass `inference: false`.

  Returns a `Spatex.GWR.MGWRResult`. Like mgwr's, the standard errors divide by the regressor
  values, so a regressor with zero entries (a dummy) yields non-finite standard errors; a
  warning names such columns. The fields mgwr cannot compute for MGWR (`influ`, `std_res`,
  `cooks_d`, `cv`, local R-squared) are `nil`.

  ## Examples

      iex> pts = for r <- 0..7, c <- 0..7, do: {c + 0.05 * rem(r * 7 + c, 5), r + 0.04 * rem(c * 3 + r, 7)}
      iex> x1 = for {px, py} <- pts, do: :math.sin(px) + :math.cos(py)
      iex> x2 = for {px, py} <- pts, do: :math.cos(2 * px) - 0.3 * py
      iex> y = for {{px, _}, a, b} <- Enum.zip([pts, x1, x2]), do: 1.0 + 0.2 * px * a + 0.5 * b
      iex> r = Spatex.GWR.mgwr(pts, y, [x1, x2], fixed: true, standardize: true)
      iex> {r.n, r.k, length(r.bws), r.converged}
      {64, 3, 3, true}
  """
  @spec mgwr(term(), term(), term(), keyword()) :: MGWRResult.t()
  def mgwr(points, y, x, opts \\ []), do: MGWR.run(points, y, x, opts)

  @doc """
  Confidence intervals for the bandwidth (`GWRResults.get_bws_intervals`, `MGWRResults.
  get_bws_intervals`), after Li et al. (2020).

  The evaluations `{bw, score}` of the bandwidth search are sorted by score, turned into Akaike
  weights `exp(-0.5 * (score - best))` (normalized to sum to one) and accumulated; the interval
  spans the smallest and largest bandwidth among the `n` best evaluations, `n` being the number
  of cumulative weights below `level` plus one. The result therefore depends on the search that
  produced the history: a coarse or short search gives a coarse interval.

  `source` is a `Spatex.GWR.Bandwidth` (from `select_bandwidth/4`; returns one `{lower, upper}`)
  or a `Spatex.GWR.MGWRResult` (from `mgwr/4`; returns one `{lower, upper}` per column, from the
  searches of the last backfitting iteration that searched, as mgwr does). An adaptive
  bandwidth gives neighbor counts, a fixed one distances. `search: :scipy` keeps no history,
  as in mgwr, so it raises `ArgumentError`.

  ## Options

    * `:level` - cumulative weight, in `(0, 1]` (default `0.95`).

  ## Examples

      iex> b = %Spatex.GWR.Bandwidth{history: [{10, 100.0}, {20, 101.0}, {30, 102.0}, {40, 110.0}]}
      iex> Spatex.GWR.bandwidth_intervals(b, level: 0.9)
      {10, 30}
  """
  @spec bandwidth_intervals(Bandwidth.t() | MGWRResult.t(), keyword()) ::
          {number(), number()} | [{number(), number()}]
  def bandwidth_intervals(source, opts \\ []), do: Intervals.run(source, opts)

  @doc """
  Monte Carlo test of the spatial variability of each coefficient surface
  (`GWRResults.spatial_variability`).

  The bandwidth is selected and the model fitted as in `select_bandwidth/4` and `fit/4`; the
  standard deviation of every coefficient surface is the statistic. Each permutation re-orders
  the coordinates among the observations (breaking the spatial pattern, keeping `y` and `x`),
  searches the bandwidth again and refits; a surface's p-value is the share of permutations
  whose standard deviation exceeds the observed one. A small p-value means the surface varies
  in space more than chance would.

  **This test is expensive**: every permutation repeats a bandwidth search (dozens of fits).
  `:max_n` therefore defaults to `2_000` here. Memory stays `O(n k)` per fit.

  Arguments are as in `fit/4`. Returns a `Spatex.GWR.SpatialVariability`.

  ## Options

  All options of `select_bandwidth/4` (kernel, search, criterion, bounds, family, ...), whose
  search is repeated for every permutation, and

    * `:permutations` - number of permutations (mgwr's `n_iters`; default `1000`).
    * `:seed` - an integer in `0..2^64-1`, or `nil` (default) to draw one; the seed used is
      recorded in the result. Replicate `r` depends only on `(seed, r)`. Unlike mgwr, which
      uses numpy's generator, Spatex uses its ChaCha8 streams, so p-values agree with mgwr's up
      to Monte Carlo error; the observed statistic agrees exactly.
    * `:keep_simulations` - keep the simulated standard deviations: `true` (default) as lists,
      `:tensor` as a `{permutations, k}` `Nx.Tensor` (optional `:nx` dependency), `false` to
      drop them.

  ## Examples

      iex> pts = for r <- 0..5, c <- 0..5, do: {c + 0.07 * rem(r * 5 + c, 4), r + 0.05 * c}
      iex> x = for {px, _} <- pts, do: :math.sin(px)
      iex> y = for {{px, py}, a} <- Enum.zip(pts, x), do: 1.0 + 0.2 * px * a + 0.1 * py + :math.cos(px * py)
      iex> r = Spatex.GWR.spatial_variability(pts, y, [x], fixed: true, kernel: :gaussian, permutations: 3, seed: 1)
      iex> {r.k, length(r.p_values), length(r.simulated_sd)}
      {2, 2, 3}
  """
  @spec spatial_variability(term(), term(), term(), keyword()) :: SpatialVariability.t()
  def spatial_variability(points, y, x, opts \\ []), do: Variability.run(points, y, x, opts)

  @doc false
  def select_option_names, do: @select_opts

  @doc """
  The t critical value `t.ppf(1 - alpha / 2, n - 1)` for a fitted model.

  Without `alpha` it is the value for the corrected 0.05 level (`result.critical_tval`), like
  mgwr's `critical_tval()`.

  ## Examples

      iex> pts = for r <- 0..4, c <- 0..4, do: {c * 1.0, r * 1.0}
      iex> x = for {cx, _} <- pts, do: cx
      iex> y = for {xv, i} <- Enum.with_index(x), do: 2.0 * xv + rem(i, 4) * 0.1
      iex> r = Spatex.GWR.fit(pts, y, x, bw: :infinity)
      iex> Float.round(Spatex.GWR.critical_tval(r, 0.05), 4)
      2.0639
  """
  @spec critical_tval(Result.t() | MGWRResult.t(), number() | nil) :: float() | [float()]
  def critical_tval(result, alpha \\ nil)
  def critical_tval(%{critical_tval: c}, nil), do: c

  def critical_tval(%{n: n}, alpha) when is_number(alpha),
    do: Native.gwr_t_critical(abs(alpha) * 1.0, n - 1.0)

  @doc """
  The t-values with those that are not significant set to `0.0` (mgwr's `filter_tvals`).

  A t-value is kept when `abs(t)` reaches the critical value (the strict inequality of mgwr).
  For an MGWR result the default critical value is the per-covariate list `critical_tval`.

  ## Options

    * `:critical_t` - the critical value to use directly (a list per column for MGWR).
    * `:alpha` - a plain significance level (two-sided) to derive it from.
    * Without either, the multiple-testing corrected 0.05 level (`result.critical_tval`).
  """
  @spec filter_tvals(Result.t() | MGWRResult.t(), keyword()) :: [[float() | atom()]]
  def filter_tvals(result, opts \\ [])

  def filter_tvals(%MGWRResult{tvalues: nil}, _opts) do
    raise ArgumentError, "this MGWR fit was computed with inference: false; there are no t-values"
  end

  def filter_tvals(%MGWRResult{} = result, opts) do
    criticals =
      cond do
        is_list(opts[:critical_t]) -> opts[:critical_t]
        opts[:critical_t] -> List.duplicate(opts[:critical_t], result.k)
        opts[:alpha] -> List.duplicate(critical_tval(result, opts[:alpha]), result.k)
        true -> result.critical_tval
      end

    Enum.map(result.tvalues, fn row ->
      row |> Enum.zip(criticals) |> Enum.map(fn {t, c} -> zero_unless_significant(t, c) end)
    end)
  end

  def filter_tvals(%Result{} = result, opts) do
    critical =
      cond do
        opts[:critical_t] -> opts[:critical_t]
        opts[:alpha] -> critical_tval(result, opts[:alpha])
        true -> result.critical_tval
      end

    Enum.map(result.tvalues, fn row -> Enum.map(row, &zero_unless_significant(&1, critical)) end)
  end

  defp zero_unless_significant(t, c) when is_number(t) and t < c and t > -c, do: 0.0
  defp zero_unless_significant(t, _c), do: t

  # -- input ---------------------------------------------------------------------------

  defp validate!(opts, allowed, fun) do
    Design.check_known!(opts, allowed)

    defaults =
      [
        kernel: :bisquare,
        fixed: false,
        constant: true,
        spherical: false,
        max_n: @default_max_n,
        family: :gaussian,
        offset: nil,
        iwls_tol: 1.0e-5,
        iwls_max_iter: 20
      ] ++ fun_defaults(fun)

    opts = Keyword.merge(defaults, opts)
    Design.check_common!(opts)
    check_family!(opts)
    check_fun_opts!(opts, fun)
    opts
  end

  defp check_hat_size!(opts, model) do
    if opts[:hat_matrix] != false and model.n * model.n > opts[:max_cells] do
      raise ArgumentError,
            "hat_matrix keeps n^2 = #{model.n * model.n} cells, above max_cells = " <>
              "#{opts[:max_cells]}; pass a larger max_cells to proceed"
    end
  end

  defp hat_matrix(_bin, false, _n), do: nil
  defp hat_matrix(bin, keep, n), do: Input.simulations(bin, keep, {n, n})

  defp check_family!(opts) do
    unless opts[:family] in @families do
      raise ArgumentError,
            "family must be one of #{inspect(@families)}, got #{inspect(opts[:family])}"
    end

    if opts[:offset] != nil and opts[:family] != :poisson do
      raise ArgumentError, "offset is only used with family: :poisson"
    end

    unless is_number(opts[:iwls_tol]) and opts[:iwls_tol] > 0 do
      raise ArgumentError, "iwls_tol must be a positive number"
    end

    unless is_integer(opts[:iwls_max_iter]) and opts[:iwls_max_iter] >= 1 do
      raise ArgumentError, "iwls_max_iter must be a positive integer"
    end
  end

  defp fun_defaults(:fit),
    do: [sigma2_v1: true, hat_matrix: false, max_cells: @default_max_cells]

  defp fun_defaults(:predict), do: [sigma2_v1: true, exog_scale: nil, exog_resid: nil]

  defp fun_defaults(:select_bandwidth),
    do: [
      criterion: :aicc,
      search: :golden_section,
      include_global: false,
      tol: 1.0e-6,
      max_iter: 200,
      bw_min: nil,
      bw_max: nil,
      interval: nil,
      x_glob: nil
    ]

  defp check_fun_opts!(opts, fun) when fun in [:fit, :predict] do
    unless is_boolean(opts[:sigma2_v1]), do: raise(ArgumentError, "sigma2_v1 must be a boolean")
    if fun == :fit, do: check_hat_opts!(opts)
  end

  defp check_fun_opts!(opts, :select_bandwidth) do
    unless opts[:criterion] in @criteria do
      raise ArgumentError,
            "criterion must be one of #{inspect(@criteria)}, got #{inspect(opts[:criterion])}"
    end

    unless opts[:search] in [:golden_section, :interval, :scipy] do
      raise ArgumentError, "search must be :golden_section, :interval or :scipy"
    end

    unless is_boolean(opts[:include_global]),
      do: raise(ArgumentError, "include_global must be a boolean")

    check_search_numbers!(opts)

    if opts[:include_global] and opts[:bw_max] != nil do
      raise ArgumentError, "include_global cannot be combined with a numeric bw_max"
    end
  end

  defp check_hat_opts!(opts) do
    case opts[:hat_matrix] do
      keep when is_boolean(keep) ->
        :ok

      :tensor ->
        Input.require_nx!("hat_matrix: :tensor")

      other ->
        raise ArgumentError, "hat_matrix must be true, false or :tensor, got #{inspect(other)}"
    end

    unless is_integer(opts[:max_cells]) and opts[:max_cells] > 0,
      do: raise(ArgumentError, "max_cells must be a positive integer")
  end

  defp check_search_numbers!(opts) do
    unless is_number(opts[:tol]) and opts[:tol] >= 0, do: raise(ArgumentError, "tol must be >= 0")

    unless is_integer(opts[:max_iter]) and opts[:max_iter] >= 1,
      do: raise(ArgumentError, "max_iter must be a positive integer")

    for key <- [:bw_min, :bw_max, :interval] do
      v = opts[key]

      unless is_nil(v) or (is_number(v) and v > 0),
        do: raise(ArgumentError, "#{key} must be a positive number")
    end
  end

  defp model!(points, y, x, opts), do: Design.model!(points, y, x, opts)

  # Family settings for the NIFs: `{family, iwls_tol, iwls_max_iter}` and the offset binary
  # (empty means ones).
  defp glm_args(opts, model) do
    offset =
      case opts[:offset] do
        nil ->
          <<>>

        off ->
          {bin, _} = Input.to_f64_binary(off, name: "offset", length: model.n)
          bin
      end

    {{Atom.to_string(opts[:family]), opts[:iwls_tol] * 1.0, opts[:iwls_max_iter]}, offset}
  end

  defp bandwidth!(opts, model) do
    case Keyword.fetch(opts, :bw) do
      :error ->
        raise ArgumentError, "bw is required"

      {:ok, :infinity} ->
        {0.0, true, []}

      {:ok, bw} when is_number(bw) ->
        bandwidth_value!(bw, opts[:fixed], model.n)

      {:ok, other} ->
        raise ArgumentError, "bw must be a number or :infinity, got #{inspect(other)}"
    end
  end

  defp bandwidth_value!(bw, true, _n) do
    if bw <= 0, do: raise(ArgumentError, "bw must be positive for a fixed bandwidth, got #{bw}")
    {bw * 1.0, false, []}
  end

  defp bandwidth_value!(bw, false, n) do
    count = trunc(bw)

    if count < 2 or count > n do
      raise ArgumentError, "bw must be between 2 and n = #{n} neighbors, got #{bw}"
    end

    warnings =
      if count == bw,
        do: [],
        else: ["adaptive bw #{bw} truncated to #{count} neighbors (mgwr uses int(bw))"]

    {count * 1.0, false, warnings}
  end

  defp global_bw(_bw, true, _opts), do: :infinity
  defp global_bw(bw, false, opts), do: if(opts[:fixed], do: bw, else: trunc(bw))

  defp search!(opts) do
    bounds = {opts[:bw_min] && opts[:bw_min] * 1.0, opts[:bw_max] && opts[:bw_max] * 1.0}

    case opts[:search] do
      :golden_section ->
        {"golden", 0.0, bounds}

      :scipy ->
        {"scipy", 0.0, bounds}

      :interval ->
        Enum.each([:bw_min, :bw_max, :interval], &require_option!(opts, &1))
        {"interval", opts[:interval] * 1.0, bounds}
    end
  end

  defp require_option!(opts, key) do
    unless opts[key],
      do: raise(ArgumentError, "the interval search needs #{key} (bw_min, bw_max, interval)")
  end

  defp integer_bw(:infinity, _fixed), do: :infinity
  defp integer_bw(bw, true), do: bw
  defp integer_bw(bw, false) when is_float(bw), do: trunc(bw)
  defp integer_bw(bw, false), do: bw

  # -- output --------------------------------------------------------------------------

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp build(e, model, opts, bw, warnings) do
    k = model.k
    glm? = opts[:family] != :gaussian

    %Result{
      n: model.n,
      k: k,
      bw: bw,
      family: opts[:family],
      kernel: opts[:kernel],
      fixed: opts[:fixed],
      spherical: opts[:spherical],
      constant: opts[:constant],
      params: rows(e.params, k),
      bse: rows(e.bse, k),
      tvalues: rows(e.tvalues, k),
      local_r2: gaussian_only(e.local_r2, glm?),
      predy: e.predy,
      resid: e.resid,
      influ: e.influ,
      std_res: e.std_res,
      cooks_d: e.cooks_d,
      resid_ss: scalar(e.resid_ss),
      tr_s: scalar(e.tr_s),
      tr_sts: scalar(e.tr_sts),
      enp: scalar(e.enp),
      sigma2: scalar(e.sigma2),
      df_resid: scalar(e.df_resid),
      llf: scalar(e.llf),
      aic: scalar(e.aic),
      aicc: scalar(e.aicc),
      bic: scalar(e.bic),
      cv: scalar(e.cv),
      r2: gaussian_only(scalar(e.r2), glm?),
      adj_r2: gaussian_only(scalar(e.adj_r2), glm?),
      scale: scalar(e.scale),
      d2: scalar(e.d2),
      adj_d2: scalar(e.adj_d2),
      null_deviance: scalar(e.null_deviance),
      global_deviance: scalar(e.global_deviance),
      deviance: glm_only(e.deviance, glm?),
      resid_deviance: glm_only(e.resid_deviance, glm?),
      p_dev: glm_only(e.p_dev, glm?),
      iterations: glm_only(Enum.map(e.iterations, &trunc/1), glm?),
      adj_alpha: e.adj_alpha,
      critical_tval: scalar(e.critical_tval),
      warnings: warnings
    }
  end

  defp gaussian_only(_value, true), do: nil
  defp gaussian_only(value, false), do: value
  defp glm_only(value, true), do: value
  defp glm_only(_value, false), do: nil

  defp scalar([v]), do: v
  defp rows(values, k), do: Enum.chunk_every(values, k)
end
