defmodule Spatex.Regression.SURerrorGM do
  @moduledoc """
  Result of `Spatex.Regression.sur_error_gm/4` (`spreg.SURerrorGM`): seemingly unrelated
  regressions with a spatial error process in every equation, estimated by generalized moments
  and feasible GLS.

  Per-equation fields are lists with one entry per equation, in order.

    * `n`, `n_eq`, `k` - observations, equations and, per equation, the number of coefficients.
    * `lam` - the spatial error parameter of every equation: the Kelejian-Prucha moment estimate
      (`gm_error/4`) of the OLS residuals of that equation. spreg reports no standard error for
      it, nor a covariance of `sig`.
    * `b_ols` - the equation-by-equation OLS coefficients that start the estimation.
    * `b_sur` - the SUR coefficients on the spatially filtered variables (the constant first,
      `"Constant_<eq>"`); `varb` - covariance of all coefficients stacked; `sur_inf` - per
      equation, one `{std_err, z, p}` per coefficient.
    * `sig`, `corr` - the cross-equation covariance `E'E / n` of the filtered OLS residuals
      (`e - lambda W e`) and its correlation matrix.
    * `chow` - with the same number of coefficients in every equation, one `{stat, df, p}` Wald
      test per coefficient of its equality across equations; `nil` otherwise.
    * `residuals`, `predicted` - one list of `n` values per equation (`y - X b`, on the original,
      unfiltered variables).
    * `name_y`, `name_x` - names, per equation (`name_x` starts with `"Constant_<eq>"`).
    * `warnings` - for example a degenerate moment scale, for which lambda is set to 0.
  """

  @enforce_keys [:n, :n_eq]
  defstruct [
    :n,
    :n_eq,
    :k,
    :lam,
    :b_ols,
    :b_sur,
    :varb,
    :sur_inf,
    :sig,
    :corr,
    :chow,
    :residuals,
    :predicted,
    :name_y,
    :name_x,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.SurErrorGm do
  @moduledoc false
  # `Spatex.Regression.sur_error_gm/4`: spreg's BaseSURerrorGM. OLS per equation, lambda of each
  # equation from the KP98 moments of its OLS residuals (the optimizer of `gm_error/4`), then
  # the SUR GLS step on the variables filtered with those lambdas, `sigma` from the filtered OLS
  # residuals. The residuals reported are those of the unfiltered equations.

  alias Spatex.{Graph, Input, Native}
  alias Spatex.Regression.{Regimes, Sur, SURerrorGM}

  @allowed [:name_bigy, :name_bigx]
  @degenerate "The moment equations have a zero scale (spreg's optimizer fails here); lambda was set to 0."

  # `lambdas` bypasses the moment optimizer (test hook: spreg's own lambdas).
  @spec fit([term()], [term()], term(), keyword(), [number()] | nil) :: SURerrorGM.t()
  def fit(ys, xs, graph, opts, lambdas \\ nil) do
    opts = validate!(opts)

    unless match?(%Graph{}, graph),
      do: raise(ArgumentError, "the graph must be a Spatex.Graph")

    {eqs, n, names_y} = Sur.equations("sur_error_gm", ys, xs, opts[:name_bigy], opts[:name_bigx])

    unless graph.n == n,
      do: raise(ArgumentError, "the graph has #{graph.n} nodes but the equations have #{n} rows")

    if lambdas && length(lambdas) != length(eqs),
      do: raise(ArgumentError, "one lambda per equation is needed")

    estimate(eqs, n, names_y, graph, lambdas)
  end

  defp estimate(eqs, n, names_y, graph, lambdas) do
    n_eq = length(eqs)
    ks = Enum.map(eqs, &length(&1.x))
    {xx, xy} = Sur.cross_products(eqs)

    b_ols =
      for r <- 0..(n_eq - 1), do: Sur.mat_vec(Sur.inverse(Sur.at(xx, r, r)), Sur.at(xy, r, r))

    e_ols = Sur.residuals(eqs, b_ols)
    {lam, warnings} = lambdas(e_ols, graph, lambdas)

    filtered =
      for {e, l} <- Enum.zip(eqs, lam) do
        %{y: filter(graph, e.y, l), x: Enum.map(e.x, &filter(graph, &1, l))}
      end

    {sxx, sxy} = Sur.cross_products(filtered)
    sp_e = for {u, l} <- Enum.zip(e_ols, lam), do: filter(graph, u, l)
    {b_sur, varb, sig} = Sur.sur_est(sxx, sxy, sp_e, ks, n)

    %SURerrorGM{
      n: n,
      n_eq: n_eq,
      k: ks,
      lam: lam,
      b_ols: b_ols,
      b_sur: b_sur,
      varb: varb,
      sur_inf: Sur.inference(b_sur, varb),
      sig: sig,
      corr: Sur.correlation(sig),
      chow: Sur.chow(ks, b_sur, varb),
      residuals: Sur.residuals(eqs, b_sur),
      predicted: for({e, b} <- Enum.zip(eqs, b_sur), do: Sur.mat_vec(Sur.transpose(e.x), b)),
      name_y: Enum.map(names_y, &to_string/1),
      name_x: Enum.map(eqs, & &1.names),
      warnings: Enum.flat_map(eqs, & &1.warnings) ++ warnings
    }
  end

  defp lambdas(e_ols, graph, nil) do
    pairs = Enum.map(e_ols, &kp98_lambda(&1, graph))

    {Enum.map(pairs, &elem(&1, 0)),
     if(Enum.any?(pairs, &elem(&1, 1)), do: [@degenerate], else: [])}
  end

  defp lambdas(_e_ols, _graph, given), do: {Enum.map(given, &(&1 * 1.0)), []}

  defp kp98_lambda(u, graph) do
    {:ok, list} = u |> Regimes.bin() |> Native.regression_error_lambda(graph.ref) |> Regimes.ok!()
    by_name = Map.new(list, fn {k, v} -> {k, Input.from_f64_binary(v)} end)
    {hd(by_name["lam"]), hd(by_name["degenerate"]) != 0.0}
  end

  # v - lambda W v (spreg's filter_dict)
  defp filter(graph, v, lam), do: Enum.zip_with(v, Graph.lag(graph, v), &(&1 - lam * &2))

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        opts

      bad ->
        raise ArgumentError,
              "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)} " <>
                "(regimes are not supported by sur_error_gm)"
    end
  end
end
