defmodule Spatex.Regression.PanelFE do
  @moduledoc """
  Result of `Spatex.Regression.ml_lag_fe/4` and `ml_error_fe/4` (`spreg.ML_LagFE`,
  `spreg.ML_ErrorFE`): a spatial panel with individual fixed effects.

    * `model` - `:lag` (`y = rho W y + X b + mu_i + e`) or `:error`
      (`y = X b + mu_i + u`, `u = lambda W u + e`); `rho` / `lam` the spatial parameter.
    * `betas`, `vm`, `std_err`, `z_stat` - the slopes followed by `rho` / `lam` (`vm1` adds
      `sigma^2`); there is no intercept, the fixed effects absorb it.
    * `fixed_effects` - the `n` individual effects `mu_i` estimated from the area means;
      `mean_fixed_effects` their mean.
    * `logll`, `aic`, `schwarz`, `sig2` - as spreg (`:lag` counts `rho` in `k`, `:error` does not
      count `lam`).
    * `u`, `predy` (and `predy_e`, `e_pred` for `:lag`, `e_filtered` for `:error`) - stacked by
      period on the demeaned data: `n * t` values, period 1 first.
    * `sp_multipliers`, `impacts` - for `:lag`, as in `Spatex.Regression.ml_lag/4`.
  """

  @enforce_keys [:model, :n, :t]
  defstruct [
    :model,
    :n,
    :t,
    :betas,
    :rho,
    :lam,
    :vm,
    :vm1,
    :std_err,
    :z_stat,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :e_filtered,
    :fixed_effects,
    :mean_fixed_effects,
    :sp_multipliers,
    :impacts,
    :time_effects,
    :iterations,
    :converged,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.PanelRE do
  @moduledoc """
  Result of `Spatex.Regression.ml_lag_re/4` (`spreg.ML_LagRE`): a spatial lag panel with random
  effects, `y = rho W y + X b + mu_i + e`.

    * `betas`, `vm`, `std_err`, `z_stat` - `"CONSTANT"`, the slopes, `rho` and `phi` (the
      variance ratio `sigma_e^2 / (sigma_e^2 + t sigma_mu^2)` of the quasi-demeaning
      `v - (1 - phi) mean_t(v)`); `vm1` adds `sigma^2`.
    * `logll`, `aic`, `schwarz`, `sig2` - as spreg (`rho` and `phi` count in `k`).
    * `u`, `predy`, `predy_e`, `e_pred` - on the quasi-demeaned data, stacked by period.
    * `sp_multipliers`, `impacts` - as in `Spatex.Regression.ml_lag/4`; `iterations` - rounds
      of the alternating `phi` / `rho` search.
  """

  @enforce_keys [:n, :t]
  defstruct [
    :n,
    :t,
    :betas,
    :rho,
    :phi,
    :vm,
    :vm1,
    :std_err,
    :z_stat,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :sp_multipliers,
    :impacts,
    :time_effects,
    :iterations,
    :converged,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.PanelREError do
  @moduledoc """
  Result of `Spatex.Regression.ml_error_re/4` (`spreg.ML_ErrorRE`): a spatial error panel with
  random effects, `y = X b + mu_i + u`, `u = lambda W u + e`.

    * `betas`, `vm`, `std_err`, `z_stat` - `"CONSTANT"`, the slopes, `lambda` and `sigma2_u`
      (spreg's order, see `name_x`); `vm` has the slopes' block `sig2 (X*' X*)^-1` and the
      `lambda` / `phi` block of spreg's information matrix (`vm1[:2, :2]`; its second entry is
      the variance of `phi`, not of `sigma2_u`), with zero cross blocks.
    * `lam` - the spatial error parameter; `phi` - the variance ratio `sigma_u^2 / sigma^2`;
      `sig2` - the residual variance of the filtered model; `sig2_u = phi * sig2`.
    * `logll`, `aic`, `schwarz` - as spreg (`k` is the number of columns of `X`, constant
      included; `lambda` and `phi` are not counted).
    * `u`, `predy` - residuals `y - X b` and fitted values, stacked by period (`n * t` values,
      period 1 first), on the raw data; `e_filtered` - the residuals of the transformed model.
    * `iterations` - objective evaluations of the optimizer.
  """

  @enforce_keys [:n, :t]
  defstruct [
    :n,
    :t,
    :betas,
    :lam,
    :phi,
    :sig2_u,
    :vm,
    :std_err,
    :z_stat,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :u,
    :predy,
    :e_filtered,
    :time_effects,
    :iterations,
    :converged,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.PanelGMKKP do
  @moduledoc """
  Result of `Spatex.Regression.gm_kkp/4` (`spreg.GM_ErrorRE`, formerly `GM_KKP`): the
  Kapoor-Kelejian-Prucha GM estimator of a spatial error panel with random effects,
  `y = X b + mu_i + u`, `u = lambda W u + e`.

    * `betas` - `"CONSTANT"`, the slopes, then `lambda`, `sigma2_v` and `sigma2_1`, as spreg
      (`name_x` lists them in that order, without the stray space spreg puts in front of
      `sigma2_v`). `lam`, `sig2_v` (the variance of `e`) and `sig2_1` (`sigma_v^2 + t sigma_mu^2`)
      repeat the last three.
    * `vm`, `std_err`, `z_stat` - for the `k` regression coefficients only, as spreg
      (`vm = s2 (X*' X*)^-1` of the feasible GLS regression, `s2` with `n t - k`).
    * `theta` - `1 - sigma_v / sigma_1`, the weight of the area means in the GLS transform;
      `first_step` - `[lambda, sigma_v^2, sigma_1^2]` of the first (unweighted) stage.
    * `pr2` - the squared correlation of `y` and `predy`.
    * `u`, `predy` - residuals `y - X b` and fitted values on the raw data, stacked by period
      (`n * t` values, period 1 first); `e_filtered` - `u - lambda W u` per period.
  """

  @enforce_keys [:n, :t]
  defstruct [
    :n,
    :t,
    :betas,
    :lam,
    :sig2_v,
    :sig2_1,
    :theta,
    :first_step,
    :vm,
    :std_err,
    :z_stat,
    :pr2,
    :u,
    :predy,
    :e_filtered,
    :time_effects,
    :full_weights,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.Panel do
  @moduledoc false
  # `Spatex.Regression.ml_lag_fe/4` and `ml_error_fe/4`: panel layout, time effects and result
  # assembly around the `spatex_core::regression::panel` estimators.

  alias Spatex.{Graph, Input, Native}
  alias Spatex.Regression.{ML, PanelFE, PanelGMKKP, PanelRE, PanelREError}

  @allowed [:time_effects, :epsilon, :spat_impacts, :full_weights, :regimes, :name_x, :name_y]
  @impacts %{nil => 0, :simple => 1, :full => 2, :power => 3, :all => 4}
  @keys ~w(rho lam phi sig2_u sig2_v sig2_1 theta first_step pr2 vm vm1 std_err z_stat logll aic schwarz sig2 iterations converged u predy
           predy_e e_pred e_filtered mu_i betas mult_simple mult_full mult_power)a

  @spec fit(:lag | :error | :lag_re | :error_re | :gm_kkp, term(), term(), Graph.t(), keyword()) ::
          PanelFE.t() | PanelRE.t() | PanelREError.t() | PanelGMKKP.t()
  def fit(model, y, x, graph, opts, at \\ nil)

  def fit(model, y, x, %Graph{} = graph, opts, at) do
    opts = validate!(opts, model)
    n = graph.n
    {y_rows, t} = matrix!(y, n, "y")
    {names, x_mats} = variables!(x, n, t, opts[:name_x])

    {cols, names} =
      with_time_effects(Enum.map(x_mats, &stack/1), names, n, t, opts[:time_effects])

    # spreg's prepare_panel(add_constant: true) for the random-effects model.
    {cols, names} =
      if model in [:lag_re, :error_re, :gm_kkp],
        do: {[List.duplicate(1.0, n * t) | cols], ["CONSTANT" | names]},
        else: {cols, names}

    {y_bin, _} = Input.to_f64_binary(stack(y_rows), name: "y")

    x_bin =
      cols |> Enum.map(&elem(Input.to_f64_binary(&1, name: "x"), 0)) |> IO.iodata_to_binary()

    native(model, y_bin, x_bin, length(cols), graph, {t, opts, at})
    |> entries()
    |> build(model, {n, t, names}, opts)
  end

  def fit(_model, _y, _x, other, _opts, _at),
    do: raise(ArgumentError, "graph must be a Spatex.Graph, got #{inspect(other, limit: 3)}")

  defp native(:gm_kkp, y_bin, x_bin, k, graph, {t, opts, at}) do
    Native.regression_panel_gm_kkp(
      y_bin,
      x_bin,
      k,
      graph.ref,
      {t, opts[:full_weights], Tuple.to_list(at || {}) |> Enum.map(&(&1 * 1.0))}
    )
  end

  defp native(:error_re, y_bin, x_bin, k, graph, {t, _opts, at}) do
    {lam, phi} = at || {0.0, 0.0}

    Native.regression_panel_error_re(
      y_bin,
      x_bin,
      k,
      graph.ref,
      {t, at != nil, lam * 1.0, phi * 1.0}
    )
  end

  defp native(model, y_bin, x_bin, k, graph, {t, opts, _at}) do
    settings =
      {Atom.to_string(model), t, opts[:epsilon] * 1.0, Map.fetch!(@impacts, opts[:spat_impacts])}

    Native.regression_panel_fe(y_bin, x_bin, k, graph.ref, settings)
  end

  # Stacks an `n x t` matrix by period (spreg's long format).
  defp stack(rows), do: rows |> Enum.zip_with(& &1) |> List.flatten()

  defp with_time_effects(cols, names, _n, _t, false), do: {cols, names}

  # libpysal's `np.repeat(np.eye(T), n, axis=0)[:, 1:]`: one dummy per period after the first.
  defp with_time_effects(cols, names, n, t, true) do
    dummies = for s <- 2..t//1, do: Enum.flat_map(1..t, &List.duplicate(dummy(&1, s), n))
    {cols ++ dummies, names ++ Enum.map(2..t//1, &"time_#{&1}")}
  end

  defp dummy(p, p), do: 1.0
  defp dummy(_p, _s), do: 0.0

  defp build(e, :gm_kkp, {n, t, names}, opts) do
    k = length(names)

    %PanelGMKKP{
      n: n,
      t: t,
      betas: e.betas,
      lam: hd(e.lam),
      sig2_v: hd(e.sig2_v),
      sig2_1: hd(e.sig2_1),
      theta: hd(e.theta),
      first_step: e.first_step,
      vm: Enum.chunk_every(e.vm, k),
      std_err: e.std_err,
      z_stat: e.z_stat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1),
      pr2: hd(e.pr2),
      u: e.u,
      predy: e.predy,
      e_filtered: e.e_filtered,
      time_effects: opts[:time_effects],
      full_weights: opts[:full_weights],
      name_x: names ++ ["lambda", "sigma2_v", "sigma2_1"],
      name_y: to_string(opts[:name_y])
    }
  end

  defp build(e, :error_re, {n, t, names}, opts) do
    k = length(names)
    converged = hd(e.converged) == 1.0

    %PanelREError{
      n: n,
      t: t,
      betas: e.betas,
      lam: hd(e.lam),
      phi: hd(e.phi),
      sig2_u: hd(e.sig2_u),
      vm: Enum.chunk_every(e.vm, k + 2),
      std_err: e.std_err,
      z_stat: e.z_stat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1),
      logll: hd(e.logll),
      aic: hd(e.aic),
      schwarz: hd(e.schwarz),
      sig2: hd(e.sig2),
      u: e.u,
      predy: e.predy,
      e_filtered: e.e_filtered,
      time_effects: opts[:time_effects],
      iterations: round(hd(e.iterations)),
      converged: converged,
      name_x: names ++ ["lambda", "sigma2_u"],
      name_y: to_string(opts[:name_y]),
      warnings: if(converged, do: [], else: ["The optimizer did not converge."])
    }
  end

  defp build(e, :lag_re, {n, t, names}, opts) do
    k = length(names)
    name_y = to_string(opts[:name_y])
    converged = hd(e.converged) == 1.0
    slopes = tl(names)
    {multipliers, effects} = ML.impacts_of(e, slopes, e.betas |> Enum.take(k) |> tl())

    %PanelRE{
      n: n,
      t: t,
      betas: e.betas,
      rho: hd(e.rho),
      phi: hd(e.phi),
      vm: Enum.chunk_every(e.vm, k + 2),
      vm1: Enum.chunk_every(e.vm1, k + 3),
      std_err: e.std_err,
      z_stat: e.z_stat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1),
      logll: hd(e.logll),
      aic: hd(e.aic),
      schwarz: hd(e.schwarz),
      sig2: hd(e.sig2),
      u: e.u,
      predy: e.predy,
      predy_e: e.predy_e,
      e_pred: e.e_pred,
      sp_multipliers: multipliers,
      impacts: effects,
      time_effects: opts[:time_effects],
      iterations: round(hd(e.iterations)),
      converged: converged,
      name_x: names ++ ["W_" <> name_y, "phi"],
      name_y: name_y,
      warnings: if(converged, do: [], else: ["The phi / rho iterations did not converge."])
    }
  end

  defp build(e, model, {n, t, names}, opts) do
    k = length(names)
    kc = k + 1
    name_y = to_string(opts[:name_y])
    converged = hd(e.converged) == 1.0

    {multipliers, effects} =
      if model == :lag, do: ML.impacts_of(e, names, Enum.take(e.betas, k)), else: {nil, nil}

    %PanelFE{
      model: model,
      n: n,
      t: t,
      betas: e.betas,
      rho: first(e[:rho]),
      lam: first(e[:lam]),
      vm: Enum.chunk_every(e.vm, kc),
      vm1: Enum.chunk_every(e.vm1, if(model == :lag, do: kc + 1, else: 2)),
      std_err: e.std_err,
      z_stat: e.z_stat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1),
      logll: hd(e.logll),
      aic: hd(e.aic),
      schwarz: hd(e.schwarz),
      sig2: hd(e.sig2),
      u: e.u,
      predy: e.predy,
      predy_e: e[:predy_e],
      e_pred: e[:e_pred],
      e_filtered: e[:e_filtered],
      fixed_effects: e.mu_i,
      mean_fixed_effects: Enum.sum(e.mu_i) / n,
      sp_multipliers: multipliers,
      impacts: effects,
      time_effects: opts[:time_effects],
      iterations: round(hd(e.iterations)),
      converged: converged,
      name_x: names ++ [if(model == :lag, do: "W_" <> name_y, else: "lambda")],
      name_y: name_y,
      warnings: if(converged, do: [], else: ["The optimizer did not converge."])
    }
  end

  defp first(nil), do: nil
  defp first([v | _]), do: v

  defp validate!(opts, model) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    allowed =
      case model do
        :error -> @allowed -- [:spat_impacts, :full_weights, :regimes]
        :error_re -> @allowed -- [:spat_impacts, :epsilon, :full_weights, :regimes]
        :gm_kkp -> @allowed -- [:spat_impacts, :epsilon]
        _ -> @allowed -- [:full_weights, :regimes]
      end

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    opts =
      Keyword.merge(
        [
          time_effects: false,
          epsilon: 1.0e-7,
          spat_impacts: :simple,
          full_weights: false,
          name_x: nil,
          name_y: "y"
        ],
        opts
      )

    no_regimes!(opts)

    opts =
      if model in [:error, :error_re, :gm_kkp],
        do: Keyword.put(opts, :spat_impacts, nil),
        else: opts

    check_values!(opts)
  end

  defp no_regimes!(opts) do
    if opts[:regimes] != nil do
      raise ArgumentError,
            "regimes are not supported by gm_kkp/4 (spreg raises NotImplementedError for GM_ErrorRE)"
    end
  end

  defp check_values!(opts) do
    unless is_boolean(opts[:time_effects]),
      do: raise(ArgumentError, ":time_effects must be a boolean")

    unless is_boolean(opts[:full_weights]),
      do: raise(ArgumentError, ":full_weights must be a boolean")

    unless is_number(opts[:epsilon]) and opts[:epsilon] > 0,
      do: raise(ArgumentError, ":epsilon must be a positive number")

    unless Map.has_key?(@impacts, opts[:spat_impacts]),
      do: raise(ArgumentError, ":spat_impacts must be :simple, :full, :power, :all or nil")

    opts
  end

  # An `n x t` matrix (one row per area, one column per period) of numbers.
  defp matrix!(m, n, label) do
    rows = Input.to_matrix(m, name: label, rows: n)
    t = length(hd(rows))
    if t < 2, do: raise(ArgumentError, "#{label} must hold at least 2 periods")
    {rows, t}
  end

  defp variables!(x, n, t, given) when is_map(x) and not is_struct(x) do
    names = x |> Map.keys() |> Enum.sort()
    variables!(Enum.map(names, &Map.fetch!(x, &1)), n, t, given || Enum.map(names, &to_string/1))
  end

  defp variables!(x, n, t, given) when is_list(x) and x != [] do
    mats =
      Enum.map(Enum.with_index(x, 1), fn {m, j} ->
        {rows, tj} = matrix!(m, n, "x variable #{j}")
        if tj != t, do: raise(ArgumentError, "x variable #{j} has #{tj} periods, y has #{t}")
        rows
      end)

    names = given || Enum.map(1..length(mats), &"var_#{&1}")

    unless length(names) == length(mats),
      do: raise(ArgumentError, ":name_x has #{length(names)} names for #{length(mats)} variables")

    {Enum.map(names, &to_string/1), mats}
  end

  defp variables!(_x, _n, _t, _given),
    do: raise(ArgumentError, "x must be a non-empty list (or map) of n x t matrices")

  defp entries({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp entries({:ok, list}) do
    Map.new(list, fn {name, bin} ->
      {Enum.find(@keys, &(Atom.to_string(&1) == name)) ||
         raise(ArgumentError, "unexpected native result #{inspect(name)}"),
       Input.from_f64_binary(bin)}
    end)
  end
end
