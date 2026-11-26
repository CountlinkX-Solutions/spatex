defmodule Spatex.Regression.ThreeSLS do
  @moduledoc """
  Result of `Spatex.Regression.three_sls/5` (`spreg.ThreeSLS`): three-stage least squares for a
  system of equations with endogenous regressors.

  Per-equation fields are lists with one entry per equation, in order.

    * `n`, `n_eq`, `k` - observations, equations and, per equation, the number of coefficients
      (constant, exogenous and endogenous).
    * `b_2sls`, `residuals_2sls` - the equation-by-equation 2SLS coefficients and residuals that
      start the estimation.
    * `b_3sls` - the 3SLS coefficients (constant, exogenous, then endogenous); `varb` - covariance
      of all coefficients stacked; `tsls_inf` - per equation, one `{std_err, z, p}` per coefficient.
    * `sig`, `corr` - the cross-equation error covariance (`E'E / n` of the 2SLS residuals) and its
      correlation matrix.
    * `chow` - with the same number of coefficients in every equation, one `{stat, df, p}` Wald
      test per coefficient of its equality across equations; `nil` otherwise.
    * `residuals`, `predicted` - one list of `n` values per equation (from the 3SLS coefficients and
      the observed endogenous variables).
    * `name_y`, `name_x`, `name_yend`, `name_q` - names, per equation (`name_x` starts with
      `"Constant_<eq>"`).
  """

  @enforce_keys [:n, :n_eq]
  defstruct [
    :n,
    :n_eq,
    :k,
    :b_2sls,
    :residuals_2sls,
    :b_3sls,
    :varb,
    :tsls_inf,
    :sig,
    :corr,
    :chow,
    :residuals,
    :predicted,
    :name_y,
    :name_x,
    :name_yend,
    :name_q,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.ThreeSls do
  @moduledoc false
  # `Spatex.Regression.three_sls/5`: spreg's BaseThreeSLS. 2SLS per equation, the cross-equation
  # covariance of those residuals, then GLS on the instrumented system. The GLS machinery
  # (`Sur.sur_est/5`, inference, Chow test) is the SUR one: it only needs the cross products of
  # the regressors, which for 3SLS are those of Z-hat = [X | H (H'H)^-1 H'Yend].

  alias Spatex.{Input, Regression}
  alias Spatex.Regression.{Sur, ThreeSLS}

  @allowed [:name_bigy, :name_bigx, :name_bigyend, :name_bigq]

  @spec fit([term()], [term()], [term()], [term()], keyword()) :: ThreeSLS.t()
  def fit(ys, xs, yends, qs, opts) do
    opts = validate!(opts)

    unless Enum.all?([ys, xs, yends, qs], &is_list/1) and length(ys) >= 2,
      do: raise(ArgumentError, "three_sls needs lists of at least two equations")

    unless length(xs) == length(ys) and length(yends) == length(ys) and length(qs) == length(ys),
      do: raise(ArgumentError, "ys, xs, yends and qs need the same number of equations")

    n_eq = length(ys)
    names_y = opts[:name_bigy] || Enum.map(0..(n_eq - 1), &"dep_var_#{&1 + 1}")

    eqs =
      for r <- 0..(n_eq - 1) do
        equation(r, Enum.at(ys, r), Enum.at(xs, r), Enum.at(yends, r), Enum.at(qs, r), opts)
      end

    n = length(hd(eqs).y)

    unless Enum.all?(eqs, &(length(&1.y) == n)),
      do: raise(ArgumentError, "every equation needs the same number of observations")

    estimate(eqs, n, names_y)
  end

  defp equation(r, y, x, yend, q, opts) do
    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: nth(opts[:name_bigx], r))

    {_, _, yend_names, yend_bin, _, _} =
      Regression.prepare_design(y, yend, constant: false, name_x: nth(opts[:name_bigyend], r))

    {_, _, q_names, q_bin, _, _} =
      Regression.prepare_design(y, q, constant: false, name_x: nth(opts[:name_bigq], r))

    x_cols = columns(x_bin, n)
    yend_cols = columns(yend_bin, n)
    q_cols = columns(q_bin, n)

    if length(q_cols) < length(yend_cols),
      do:
        raise(
          ArgumentError,
          "equation #{r + 1} has fewer instruments (#{length(q_cols)}) than endogenous variables " <>
            "(#{length(yend_cols)})"
        )

    names = ["Constant_#{r + 1}" | tl(names)]
    h = x_cols ++ q_cols
    yhat = project(h, yend_cols)

    %{
      y: Input.from_f64_binary(y_bin),
      z: x_cols ++ yend_cols,
      zhat: x_cols ++ yhat,
      names: names,
      yend_names: yend_names,
      q_names: q_names,
      warnings: warnings
    }
  end

  defp nth(nil, _r), do: nil
  defp nth(list, r), do: Enum.at(list, r)

  defp columns(bin, n), do: bin |> Input.from_f64_binary() |> Enum.chunk_every(n)

  # H (H'H)^-1 H' v for every column v: the first-stage fitted values.
  defp project(h, cols) do
    hhi = h |> Sur.cross(h) |> Sur.inverse()

    for v <- cols do
      coef = Sur.mat_vec(hhi, Enum.map(h, &Sur.dot(&1, v)))

      h
      |> Enum.zip(coef)
      |> Enum.map(fn {col, c} -> Enum.map(col, &(&1 * c)) end)
      |> Enum.zip_with(&Enum.sum/1)
    end
  end

  defp estimate(eqs, n, names_y) do
    n_eq = length(eqs)
    ks = Enum.map(eqs, &length(&1.z))
    zhzh = for a <- eqs, do: for(b <- eqs, do: Sur.cross(a.zhat, b.zhat))
    zhy = for a <- eqs, do: for(b <- eqs, do: Enum.map(a.zhat, &Sur.dot(&1, b.y)))

    # residuals use the observed endogenous variables (Z), the estimation uses Z-hat
    zeqs = Enum.map(eqs, &%{x: &1.z, y: &1.y})

    b_2sls =
      for r <- 0..(n_eq - 1), do: Sur.mat_vec(Sur.inverse(Sur.at(zhzh, r, r)), Sur.at(zhy, r, r))

    e_2sls = Sur.residuals(zeqs, b_2sls)
    {b_3sls, varb, sig} = Sur.sur_est(zhzh, zhy, e_2sls, ks, n)

    %ThreeSLS{
      n: n,
      n_eq: n_eq,
      k: ks,
      b_2sls: b_2sls,
      residuals_2sls: e_2sls,
      b_3sls: b_3sls,
      varb: varb,
      tsls_inf: Sur.inference(b_3sls, varb),
      sig: sig,
      corr: Sur.correlation(sig),
      chow: Sur.chow(ks, b_3sls, varb),
      residuals: Sur.residuals(zeqs, b_3sls),
      predicted: for({e, b} <- Enum.zip(zeqs, b_3sls), do: Sur.mat_vec(Sur.transpose(e.x), b)),
      name_y: Enum.map(names_y, &to_string/1),
      name_x: Enum.map(eqs, & &1.names),
      name_yend: Enum.map(eqs, & &1.yend_names),
      name_q: Enum.map(eqs, & &1.q_names),
      warnings: Enum.flat_map(eqs, & &1.warnings)
    }
  end

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        opts

      bad ->
        raise ArgumentError,
              "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)} " <>
                "(regimes are not supported by three_sls)"
    end
  end
end
