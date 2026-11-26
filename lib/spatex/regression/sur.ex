defmodule Spatex.Regression.SUR do
  @moduledoc """
  Result of `Spatex.Regression.sur/3` (`spreg.SUR`): seemingly unrelated regressions by
  feasible GLS.

  Per-equation fields are lists with one entry per equation, in order.

    * `b_ols`, `b_sur` - OLS and SUR coefficients (the constant first, named
      `"Constant_<eq>"` as in spreg); `name_x` / `name_y` name them.
    * `varb` - covariance of all SUR coefficients stacked; `sur_inf` - per equation, one
      `{std_err, z, p}` per coefficient.
    * `sig`, `sig_ols`, `corr` - the error covariance used by the last estimation step, the one
      of the OLS residuals, and the correlation matrix of `sig`.
    * `llik`, `ldet_s0`, `ldet_s1`, `iterations` - log-likelihood, log-determinants of the
      diagonal OLS covariance and of the last `sig`, and the FGLS rounds.
    * `lr_test`, `lm_test` - `{stat, df, p}` likelihood-ratio and Breusch-Pagan LM tests of a
      diagonal error covariance (`nil` with `nonspat_diag: false`).
    * `chow` - with the same number of regressors in every equation, one `{stat, df, p}` Wald
      test per coefficient of its equality across equations; `nil` otherwise.
    * `lm_error`, `lm_lag` - `{stat, df, p}` spatial LM tests with `spat_diag: true`.
    * `residuals`, `predicted` - one list of `n` values per equation.
  """

  @enforce_keys [:n, :n_eq]
  defstruct [
    :n,
    :n_eq,
    :b_ols,
    :b_sur,
    :varb,
    :sur_inf,
    :sig,
    :sig_ols,
    :corr,
    :llik,
    :ldet_s0,
    :ldet_s1,
    :iterations,
    :lr_test,
    :lm_test,
    :chow,
    :lm_error,
    :lm_lag,
    :residuals,
    :predicted,
    :name_y,
    :name_x,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.Sur do
  @moduledoc false
  # `Spatex.Regression.sur/3`: spreg's SUR estimator and diagnostics, with small dense matrix
  # algebra in Elixir (the matrices are n_eq x n_eq and sum(K) x sum(K); only the cross
  # products touch the n observations).

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.SUR

  @allowed [
    :iter,
    :maxiter,
    :epsilon,
    :nonspat_diag,
    :spat_diag,
    :graph,
    :name_bigy,
    :name_bigx
  ]

  @spec fit([term()], [term()], keyword()) :: SUR.t()
  def fit(ys, xs, opts) do
    opts = validate!(opts)
    {eqs, n, names_y} = equations("sur", ys, xs, opts[:name_bigy], opts[:name_bigx])
    estimate(eqs, n, names_y, opts)
  end

  # The per-equation designs shared by the SUR family: `%{y, x, names, warnings}` with the
  # columns of `x` (the constant `Constant_<eq>` first), the number of observations and the
  # equation names.
  @doc false
  def equations(who, ys, xs, name_y, name_x) do
    unless is_list(ys) and is_list(xs) and length(ys) == length(xs) and length(ys) >= 2,
      do:
        raise(
          ArgumentError,
          "#{who} needs lists of at least two y vectors and as many x designs"
        )

    n_eq = length(ys)
    names_y = name_y || Enum.map(0..(n_eq - 1), &"dep_var_#{&1 + 1}")
    names_x = name_x || List.duplicate(nil, n_eq)

    eqs =
      for {{y, x, nx}, r} <- Enum.with_index(Enum.zip([ys, xs, names_x])) do
        {y_bin, n, names, x_bin, _k, warnings} =
          Regression.prepare_design(y, x, constant: true, name_x: nx)

        cols = x_bin |> Input.from_f64_binary() |> Enum.chunk_every(n)

        %{
          y: Input.from_f64_binary(y_bin),
          x: cols,
          names: ["Constant_#{r + 1}" | tl(names)],
          warnings: warnings
        }
      end

    n = length(hd(eqs).y)

    unless Enum.all?(eqs, &(length(&1.y) == n)),
      do: raise(ArgumentError, "every equation needs the same number of observations")

    {eqs, n, names_y}
  end

  # Cross products `X_r'X_t` and `X_r'y_t` of every pair of equations.
  @doc false
  def cross_products(eqs) do
    xx = for a <- eqs, do: for(b <- eqs, do: cross(a.x, b.x))
    xy = for a <- eqs, do: for(b <- eqs, do: Enum.map(a.x, &dot(&1, b.y)))
    {xx, xy}
  end

  defp estimate(eqs, n, names_y, opts) do
    ks = Enum.map(eqs, &length(&1.x))
    {xx, xy} = cross_products(eqs)

    b_ols = for r <- 0..(length(eqs) - 1), do: mat_vec(inverse(at(xx, r, r)), at(xy, r, r))

    e_ols = residuals(eqs, b_ols)
    {b_sur, varb, sig} = sur_est(xx, xy, e_ols, ks, n)
    resids = residuals(eqs, b_sur)
    sig_ols = sig
    ldet_s0 = sig_ols |> diagonal() |> Enum.map(&:math.log/1) |> Enum.sum()
    det1 = log_det(sig)

    {b_sur, varb, sig, resids, det1, iterations} =
      if opts[:iter],
        do: iterate({b_sur, varb, sig, resids, ldet_s0, det1, 0}, eqs, xx, xy, ks, n, opts),
        else: {b_sur, varb, sig, resids, det1, 1}

    n_eq = length(eqs)
    predicted = for {e, b} <- Enum.zip(eqs, b_sur), do: mat_vec(transpose(e.x), b)

    %SUR{
      n: n,
      n_eq: n_eq,
      b_ols: b_ols,
      b_sur: b_sur,
      varb: varb,
      sur_inf: inference(b_sur, varb),
      sig: sig,
      sig_ols: sig_ols,
      corr: correlation(sig),
      llik: -(n / 2) * (n_eq * (1 + :math.log(2 * :math.pi())) + det1),
      ldet_s0: ldet_s0,
      ldet_s1: det1,
      iterations: iterations,
      lr_test: when?(opts[:nonspat_diag], fn -> lr_test(n, n_eq, ldet_s0, det1) end),
      lm_test: when?(opts[:nonspat_diag], fn -> lm_test(n, n_eq, sig_ols) end),
      chow: chow(ks, b_sur, varb),
      lm_error: when?(opts[:spat_diag], fn -> lm_error(opts[:graph], resids, sig) end),
      lm_lag:
        when?(opts[:spat_diag], fn -> lm_lag(opts[:graph], eqs, resids, predicted, sig, varb) end),
      residuals: resids,
      predicted: predicted,
      name_y: Enum.map(names_y, &to_string/1),
      name_x: Enum.map(eqs, & &1.names),
      warnings: Enum.flat_map(eqs, & &1.warnings)
    }
  end

  defp when?(true, fun), do: fun.()
  defp when?(false, _fun), do: nil

  # spreg's iterated FGLS: re-estimate from the latest residuals while the log-determinant of
  # sigma moves by more than `epsilon` (at most `maxiter + 1` rounds).
  defp iterate({b, varb, sig, resids, det0, det1, k}, eqs, xx, xy, ks, n, opts) do
    if abs(det1 - det0) > opts[:epsilon] and k <= opts[:maxiter] do
      {b2, varb2, sig2} = sur_est(xx, xy, resids, ks, n)
      resids2 = residuals(eqs, b2)
      iterate({b2, varb2, sig2, resids2, det1, log_det(sig2), k + 1}, eqs, xx, xy, ks, n, opts)
    else
      {b, varb, sig, resids, det1, k}
    end
  end

  # spreg's sur_est: sigma from the residuals, then GLS with sigma^-1 (x) I.
  @doc false
  def sur_est(xx, xy, resids, ks, n) do
    sig = for a <- resids, do: for(b <- resids, do: dot(a, b) / n)
    sigi = inverse(sig)
    n_eq = length(ks)

    xsigx =
      for r <- 0..(n_eq - 1), row <- 0..(Enum.at(ks, r) - 1) do
        Enum.flat_map(0..(n_eq - 1), fn t ->
          Enum.map(Enum.at(at(xx, r, t), row), &(&1 * at(sigi, r, t)))
        end)
      end

    xsigy =
      for r <- 0..(n_eq - 1), row <- 0..(Enum.at(ks, r) - 1) do
        Enum.sum(for t <- 0..(n_eq - 1), do: at(sigi, r, t) * Enum.at(at(xy, r, t), row))
      end

    varb = inverse(xsigx)
    {split(mat_vec(varb, xsigy), ks), varb, sig}
  end

  @doc false
  def residuals(eqs, betas) do
    for {e, b} <- Enum.zip(eqs, betas) do
      fit = mat_vec(transpose(e.x), b)
      Enum.zip_with(e.y, fit, &(&1 - &2))
    end
  end

  # -- diagnostics ------------------------------------------------------------------------
  @doc false
  def inference(b_sur, varb) do
    se = varb |> diagonal() |> Enum.map(&:math.sqrt/1)

    b_sur
    |> List.flatten()
    |> Enum.zip_with(se, fn b, s -> {s, b / s, :math.erfc(abs(b / s) / :math.sqrt(2))} end)
    |> split(Enum.map(b_sur, &length/1))
  end

  defp lr_test(n, n_eq, ldet0, ldet1) do
    m = div(n_eq * (n_eq - 1), 2)
    stat = n * (ldet0 - ldet1)
    {stat, m, chi2_sf(stat, m)}
  end

  defp lm_test(n, n_eq, sig) do
    r = correlation(sig)
    tr = r |> List.flatten() |> Enum.map(&(&1 * &1)) |> Enum.sum()
    m = div(n_eq * (n_eq - 1), 2)
    stat = n / 2 * (tr - n_eq)
    {stat, m, chi2_sf(stat, m)}
  end

  # Cross-equation Wald tests of each coefficient (spreg's sur_chow); every equation needs the
  # same number of regressors.
  @doc false
  def chow(ks, b_sur, varb) do
    if Enum.uniq(ks) == [hd(ks)] do
      kr = hd(ks)
      b = List.flatten(b_sur)

      for i <- 0..(kr - 1) do
        {stat, df} = wald(b, varb, chow_rows(i, kr, length(ks)))
        {stat, df, chi2_sf(stat, df)}
      end
    end
  end

  # spreg's buildR1var: row j compares coefficient `i` of equation 0 with that of equation j + 1.
  defp chow_rows(i, kr, nr) do
    for j <- 0..(nr - 2), do: Enum.map(0..(kr * nr - 1), &restriction(&1, i, kr + i + j * kr))
  end

  defp restriction(c, plus, _minus) when c == plus, do: 1.0
  defp restriction(c, _plus, minus) when c == minus, do: -1.0
  defp restriction(_c, _plus, _minus), do: 0.0

  @doc false
  def wald(b, vm, r) do
    rb = Enum.map(r, &dot(&1, b))
    rvr = mat_mul(mat_mul(r, vm), transpose(r))
    {dot(rb, mat_vec(inverse(rvr), rb)), length(r)}
  end

  # spreg's surLMe: score from sigma^-1 * (E' W E) (element-wise), information from tr(WW) and
  # tr(W'W).
  defp lm_error(graph, resids, sig) do
    {tr_ww, tr_wtw} = traces(graph)
    we = Enum.map(resids, &lag(graph, &1))
    ewe = for a <- resids, do: for(b <- we, do: dot(a, b))
    sigi = inverse(sig)
    score = column_sums(hadamard(sigi, ewe))
    denom = add(scale(identity(length(sig)), tr_ww), scale(hadamard(sigi, sig), tr_wtw))
    stat = dot(score, mat_vec(inverse(denom), score))
    {stat, length(sig), chi2_sf(stat, length(sig))}
  end

  # spreg's surLMlag (its score uses the matrix product sigma^-1 (E' W Y)).
  defp lm_lag(graph, eqs, resids, predicted, sig, varb) do
    {tr_ww, tr_wtw} = traces(graph)
    n_eq = length(sig)
    wy = Enum.map(eqs, &lag(graph, &1.y))
    ewy = for a <- resids, do: for(b <- wy, do: dot(a, b))
    sigi = inverse(sig)
    score = column_sums(mat_mul(sigi, ewy))
    first = add(scale(identity(n_eq), tr_ww), scale(hadamard(sigi, sig), tr_wtw))
    wyp = Enum.map(predicted, &lag(graph, &1))
    inner = for a <- wyp, do: for(b <- wyp, do: dot(a, b))
    ipp = add(first, hadamard(sigi, inner))

    bp =
      for {e, r} <- Enum.with_index(eqs), col <- e.x do
        Enum.zip_with(Enum.map(wyp, &dot(col, &1)), Enum.at(sigi, r), &(&1 * &2))
      end

    i_inner = add(ipp, scale(mat_mul(mat_mul(transpose(bp), varb), bp), -1.0))
    stat = dot(score, mat_vec(inverse(i_inner), score))
    {stat, n_eq, chi2_sf(stat, n_eq)}
  end

  defp traces(graph) do
    adj = Graph.to_adjacency(graph)
    index = graph.ids |> Enum.with_index() |> Map.new()
    w = Map.new(adj, fn {a, b, v} -> {{index[a], index[b]}, v} end)
    tr_ww = Enum.reduce(w, 0.0, fn {{i, j}, v}, acc -> acc + v * Map.get(w, {j, i}, 0.0) end)
    tr_wtw = Enum.reduce(w, 0.0, fn {_k, v}, acc -> acc + v * v end)
    {tr_ww, tr_wtw}
  end

  defp lag(graph, v) do
    {bin, _} = Input.to_f64_binary(v, name: "values")
    graph.ref |> Native.graph_lag(bin) |> unwrap() |> Input.from_f64_binary()
  end

  defp unwrap({:ok, v}), do: v
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  @doc false
  def chi2_sf(x, df) do
    [p] = (x * 1.0) |> Native.dynamics_chi2_sf(df * 1.0) |> Input.from_f64_binary()
    p
  end

  @doc false
  def correlation(sig) do
    s = sig |> diagonal() |> Enum.map(&:math.sqrt/1)
    for {row, si} <- Enum.zip(sig, s), do: Enum.zip_with(row, s, &(&1 / (si * &2)))
  end

  # -- options ------------------------------------------------------------------------------

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(@allowed)}"
    end

    opts =
      Keyword.merge(
        [
          iter: false,
          maxiter: 5,
          epsilon: 1.0e-5,
          nonspat_diag: true,
          spat_diag: false,
          graph: nil
        ],
        opts
      )

    for key <- [:iter, :nonspat_diag, :spat_diag],
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    if opts[:spat_diag] and not match?(%Graph{}, opts[:graph]),
      do: raise(ArgumentError, "spat_diag: true needs the :graph option (a Spatex.Graph)")

    opts
  end

  # -- small dense matrices (lists of rows) -------------------------------------------------
  @doc false
  def cross(a, b), do: for(ca <- a, do: for(cb <- b, do: dot(ca, cb)))
  @doc false
  def dot(a, b), do: a |> Enum.zip_with(b, &(&1 * &2)) |> Enum.sum()
  @doc false
  def at(m, i, j), do: m |> Enum.at(i) |> Enum.at(j)
  @doc false
  def transpose(m), do: Enum.zip_with(m, & &1)
  @doc false
  def mat_vec(m, v), do: Enum.map(m, &dot(&1, v))
  @doc false
  def mat_mul(a, b), do: Enum.map(a, fn row -> Enum.map(transpose(b), &dot(row, &1)) end)
  @doc false
  def hadamard(a, b), do: Enum.zip_with(a, b, fn ra, rb -> Enum.zip_with(ra, rb, &(&1 * &2)) end)
  defp add(a, b), do: Enum.zip_with(a, b, fn ra, rb -> Enum.zip_with(ra, rb, &(&1 + &2)) end)
  defp scale(m, f), do: Enum.map(m, fn row -> Enum.map(row, &(&1 * f)) end)
  defp column_sums(m), do: m |> transpose() |> Enum.map(&Enum.sum/1)
  @doc false
  def diagonal(m), do: m |> Enum.with_index() |> Enum.map(fn {row, i} -> Enum.at(row, i) end)

  defp identity(k), do: for(i <- 0..(k - 1), do: Enum.map(0..(k - 1), &delta(i, &1)))

  defp delta(i, i), do: 1.0
  defp delta(_i, _j), do: 0.0
  @doc false
  def split(flat, ks) do
    {parts, []} = Enum.map_reduce(ks, flat, fn k, rest -> Enum.split(rest, k) end)
    parts
  end

  # Gauss-Jordan inverse with partial pivoting.
  @doc false
  def inverse(a) do
    n = length(a)
    aug = for {row, i} <- Enum.with_index(a), do: row ++ Enum.at(identity(n), i)
    aug |> eliminate(0, n) |> Enum.map(&Enum.drop(&1, n))
  end

  defp eliminate(m, col, n) when col == n, do: m

  defp eliminate(m, col, n) do
    {pivot, idx} =
      m
      |> Enum.with_index()
      |> Enum.drop(col)
      |> Enum.max_by(fn {row, _} -> abs(Enum.at(row, col)) end)

    m = m |> List.replace_at(idx, Enum.at(m, col)) |> List.replace_at(col, pivot)
    p = Enum.at(pivot, col)
    normalized = Enum.map(pivot, &(&1 / p))

    m
    |> Enum.with_index()
    |> Enum.map(fn
      {_row, ^col} -> normalized
      {row, _} -> Enum.zip_with(row, normalized, &(&1 - Enum.at(row, col) * &2))
    end)
    |> eliminate(col + 1, n)
  end

  # log |det| by Gaussian elimination with partial pivoting.
  defp log_det(a), do: a |> pivots(0, length(a), 0.0)

  defp pivots(_m, col, n, acc) when col == n, do: acc

  defp pivots(m, col, n, acc) do
    {pivot, idx} =
      m
      |> Enum.with_index()
      |> Enum.drop(col)
      |> Enum.max_by(fn {row, _} -> abs(Enum.at(row, col)) end)

    m = m |> List.replace_at(idx, Enum.at(m, col)) |> List.replace_at(col, pivot)
    p = Enum.at(pivot, col)

    m =
      m
      |> Enum.with_index()
      |> Enum.map(fn
        {row, i} when i > col ->
          f = Enum.at(row, col) / p
          Enum.zip_with(row, pivot, &(&1 - f * &2))

        {row, _} ->
          row
      end)

    pivots(m, col + 1, n, acc + :math.log(abs(p)))
  end
end
