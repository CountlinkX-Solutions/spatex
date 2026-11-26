defmodule Spatex.Regression.OLSRegimes do
  @moduledoc """
  Result of `Spatex.Regression.ols_regimes/4` (`spreg.OLS_Regimes`).

    * `betas`, `vm`, `std_err` - coefficients in spreg's layout: the regime-specific columns of
      every regime in turn (`"<regime>_<name>"`), then the global ones (`"_Global_<name>"`);
      `name_x` names them.
    * `kr`, `kf`, `nr` - regime-specific columns per regime, global columns and regimes;
      `regimes` - the regime labels in order.
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      are equal across regimes, jointly and per regime-specific column.
    * `u`, `predy` - residuals and fitted values of every observation.
    * `mode` - `:separate` (one OLS per regime, `multi` holds them by regime label),
      `:groupwise` (a pooled fit re-weighted by each regime's residual variance; `u` and `predy`
      apply its betas to the original data) or `:pooled`.
    * `ols` - the pooled `Spatex.Regression.OLS` fit of the `:groupwise` / `:pooled` modes.
  """

  @enforce_keys [:n, :mode]
  defstruct [
    :n,
    :kr,
    :kf,
    :nr,
    :regimes,
    :betas,
    :vm,
    :std_err,
    :chow,
    :u,
    :predy,
    :mode,
    :multi,
    :ols,
    :constant_regi,
    :cols2regi,
    :robust,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.TSLSRegimes do
  @moduledoc """
  Result of `Spatex.Regression.tsls_regimes/6` (`spreg.TSLS_Regimes`).

    * `betas`, `vm`, `std_err`, `z_stat` (`[{z, p}]`) - coefficients in `names` order. In
      `:pooled` mode this is spreg's layout: the regime-specific exogenous columns of every
      regime in turn (`"<regime>_<name>"`), the global exogenous ones (`"_Global_<name>"`), then
      the same for the endogenous columns, so `names == name_x ++ name_yend`. In `:separate`
      mode every regime's `[constant, x..., yend...]` block follows the previous one.
    * `name_x`, `name_yend`, `name_q` - names as spreg lists them (the instruments always vary
      by regime). `kr`, `kf`, `kryd`, `nr` - regime-specific columns per regime (exogenous plus
      endogenous), global columns, regime-specific endogenous columns, and regimes; `regimes` -
      the labels in order.
    * `chow` - `%{joint: {wald, p}, regi: [{name, wald, p}]}`: Wald tests that the coefficients
      are equal across regimes, jointly and per regime-specific column.
    * `u`, `predy` - residuals and fitted values of every observation.
    * `mode` - `:separate` (one 2SLS per regime, `multi` holds the `Spatex.Regression.TSLS`
      fits by regime label) or `:pooled` (one 2SLS on the regime design, `tsls` holds it).
    * `cols2regi` - the flags over the `x` then `yend` columns; `constant_regi`, `robust`.
  """

  @enforce_keys [:n, :mode]
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
    :chow,
    :u,
    :predy,
    :mode,
    :multi,
    :tsls,
    :constant_regi,
    :cols2regi,
    :robust,
    :names,
    :name_x,
    :name_yend,
    :name_q,
    :name_y,
    warnings: []
  ]

  @type t :: %__MODULE__{}
end

defmodule Spatex.Regression.Regimes do
  @moduledoc false
  # `Spatex.Regression.ols_regimes/4` and `tsls_regimes/6`: spreg's regime designs, fits and Chow
  # tests, built on the fits of `Spatex.Regression.ols/3` and `tsls/5`.

  alias Spatex.{Graph, Input, Native, Regression}
  alias Spatex.Regression.{GM, OLSRegimes, TSLSRegimes}

  @allowed [:constant_regi, :cols2regi, :regime_err_sep, :robust, :sig2n_k, :name_x, :name_y]

  @spec ols(term(), term(), [term()], keyword()) :: OLSRegimes.t()
  def ols(y, x, regimes, opts) do
    opts = validate!(opts)

    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: opts[:name_x])

    regimes = Input.to_labels(regimes, name: "regimes", length: n)

    cols = x_bin |> Input.from_f64_binary() |> Enum.chunk_every(n) |> tl()
    names = tl(names)
    y = Input.from_f64_binary(y_bin)
    set = regimes |> Enum.uniq() |> Enum.sort()
    if length(set) < 2, do: raise(ArgumentError, "regimes needs at least two distinct labels")
    cols2regi = cols2regi!(opts[:cols2regi], length(names))

    base = %{
      n: n,
      y: y,
      cols: cols,
      names: names,
      regimes: regimes,
      set: set,
      cols2regi: cols2regi,
      warnings: warnings,
      opts: opts
    }

    if opts[:regime_err_sep] and Enum.all?(cols2regi) and opts[:constant_regi] == :many,
      do: separate(base),
      else: pooled(base)
  end

  # -- separate regressions (regime_err_sep, every column by regime, a constant per regime) --

  defp separate(b) do
    ids = Map.new(b.set, fn r -> {r, indices(b.regimes, r)} end)

    # spreg's check_const_regi: a column constant within any regime is dropped everywhere.
    keep =
      for {{name, col}, j} <- Enum.with_index(Enum.zip(b.names, b.cols)),
          Enum.all?(b.set, &varies?(take(col, ids[&1]))),
          do: {name, col, j}

    dropped = b.names -- Enum.map(keep, &elem(&1, 0))

    warnings =
      if dropped == [],
        do: b.warnings,
        else:
          b.warnings ++
            [
              "Variable(s) #{inspect(dropped)} removed for being constant for at least one of the regimes."
            ]

    fits =
      Map.new(b.set, fn r ->
        rows = ids[r]

        fit =
          Regression.ols(take(b.y, rows), Enum.map(keep, &take(elem(&1, 1), rows)),
            name_x: Enum.map(keep, &"#{r}_#{elem(&1, 0)}"),
            name_y: "#{r}_#{b.opts[:name_y]}",
            robust: b.opts[:robust],
            sig2n_k: b.opts[:sig2n_k]
          )

        {r, %{fit | name_x: ["#{r}_CONSTANT" | tl(fit.name_x)]}}
      end)

    kr = length(keep) + 1
    betas = Enum.flat_map(b.set, &fits[&1].betas)
    vm = block_diagonal(Enum.map(b.set, &fits[&1].vm))
    u = scatter(b.n, ids, b.set, &fits[&1].u)
    predy = scatter(b.n, ids, b.set, &fits[&1].predy)

    result(b, :separate, betas, vm, kr, 0, %{
      u: u,
      predy: predy,
      multi: Map.new(fits, fn {r, f} -> {r, f} end),
      name_x: Enum.flat_map(b.set, &fits[&1].name_x),
      warnings: warnings,
      chow_names: ["CONSTANT" | Enum.map(keep, &elem(&1, 0))]
    })
  end

  # -- one regression on the regime design ------------------------------------------------

  defp pooled(b) do
    {design, names, kr, kf, regi_names} = design(b)
    opts = b.opts
    fit = fit(b.y, design, names, opts, opts[:robust])

    if opts[:regime_err_sep] and opts[:robust] == nil do
      groupwise(b, fit, design, {names, kr, kf, regi_names})
    else
      result(b, :pooled, fit.betas, fit.vm, kr, kf, %{
        u: fit.u,
        predy: fit.predy,
        ols: fit,
        name_x: names,
        warnings: b.warnings,
        chow_names: regi_names
      })
    end
  end

  # spreg's _get_weighted_var: every observation divided by its regime's residual standard
  # deviation, then OLS again; RegressionProps_basic applies the new betas to the original data.
  defp groupwise(b, fit, design, {names, kr, kf, regi_names}) do
    scales = regime_scales(b, fit.u, length(design))
    weights = Enum.map(b.regimes, &Map.fetch!(scales, &1))
    weigh = fn col -> Enum.zip_with(col, weights, &(&1 * &2)) end
    fit2 = fit(weigh.(b.y), Enum.map(design, weigh), names, b.opts, nil)
    predy = design |> Enum.zip_with(& &1) |> Enum.map(&dot(&1, fit2.betas))

    result(b, :groupwise, fit2.betas, fit2.vm, kr, kf, %{
      u: Enum.zip_with(b.y, predy, &(&1 - &2)),
      predy: predy,
      ols: fit,
      name_x: names,
      warnings:
        b.warnings ++ ["Residuals treated as homoskedastic for the purpose of diagnostics."],
      chow_names: regi_names
    })
  end

  defp regime_scales(b, u, k) do
    Map.new(b.set, fn r ->
      ur = take(u, indices(b.regimes, r))
      denom = if b.opts[:sig2n_k], do: length(ur) - k, else: length(ur)
      {r, 1 / :math.sqrt(Enum.sum(Enum.map(ur, &(&1 * &1))) / denom)}
    end)
  end

  defp fit(y, design, names, opts, robust) do
    Regression.ols(y, design,
      constant: false,
      name_x: names,
      name_y: opts[:name_y],
      robust: robust,
      sig2n_k: opts[:sig2n_k],
      nonspat_diag: false
    )
  end

  # spreg's Regimes_Frame for the OLS design; the constant is regime-specific with
  # `constant_regi: :many` and global with `:one`.
  defp design(b) do
    ones = List.duplicate(1.0, b.n)
    const_regi = b.opts[:constant_regi] == :many

    regimize(
      [
        {"CONSTANT", ones, const_regi}
        | Enum.zip_with([b.names, b.cols, b.cols2regi], &List.to_tuple/1)
      ],
      b.regimes,
      b.set
    )
  end

  # spreg's regimeX_setup: from `{name, column, by_regime?}` triples, the regime-specific columns
  # of every regime (the column times the regime indicator), then the global columns. Returns
  # `{columns, names, kr, kf, regime_specific_names}`.
  def regimize(vars, regimes, set) do
    regi = for {name, col, true} <- vars, do: {name, col}
    glob = for {name, col, false} <- vars, do: {name, col}

    blocks =
      for r <- set, {name, col} <- regi do
        {"#{r}_#{name}", Enum.zip_with(col, regimes, fn v, l -> if l == r, do: v, else: 0.0 end)}
      end

    globals = for {name, col} <- glob, do: {"_Global_#{name}", col}
    cols = blocks ++ globals

    {Enum.map(cols, &elem(&1, 1)), Enum.map(cols, &elem(&1, 0)), length(regi), length(glob),
     Enum.map(regi, &elem(&1, 0))}
  end

  defp result(b, mode, betas, vm, kr, kf, extra) do
    %OLSRegimes{
      n: b.n,
      kr: kr,
      kf: kf,
      nr: length(b.set),
      regimes: b.set,
      betas: betas,
      vm: vm,
      std_err:
        vm |> Enum.with_index() |> Enum.map(fn {row, i} -> :math.sqrt(Enum.at(row, i)) end),
      chow: chow(betas, vm, {kr, kf, 0, 0}, length(b.set), extra.chow_names),
      u: extra.u,
      predy: extra.predy,
      mode: mode,
      multi: extra[:multi],
      ols: extra[:ols],
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      robust: b.opts[:robust],
      name_x: extra.name_x,
      name_y: to_string(b.opts[:name_y]),
      warnings: extra.warnings
    }
  end

  # -- TSLS_Regimes ---------------------------------------------------------------------------

  @tsls_allowed [
    :constant_regi,
    :cols2regi,
    :regime_err_sep,
    :robust,
    :gwk,
    :sig2n_k,
    :spreg_compatible,
    :name_x,
    :name_y,
    :name_yend,
    :name_q
  ]

  @spec tsls(term(), term(), term(), term(), term(), keyword()) :: TSLSRegimes.t()
  def tsls(y, x, yend, q, regimes, opts) do
    opts = validate_tsls!(opts)

    {y_bin, n, names, x_bin, _k, warnings} =
      Regression.prepare_design(y, x, constant: true, name_x: opts[:name_x])

    regimes = Input.to_labels(regimes, name: "regimes", length: n)
    {yend_names, yend_bin, ky} = GM.extra(yend, n, opts[:name_yend], "yend")
    {q_names, q_bin, kq} = GM.extra(q, n, opts[:name_q], "q")

    if ky == 0, do: raise(ArgumentError, "yend has no columns")
    if kq < ky, do: raise(ArgumentError, "q needs at least as many columns as yend")

    xcols = x_bin |> bin_cols(n) |> tl()
    names = tl(names)
    set = regimes |> Enum.uniq() |> Enum.sort()
    if length(set) < 2, do: raise(ArgumentError, "regimes needs at least two distinct labels")

    if n / length(set) < length(names) + 1,
      do: raise(ArgumentError, "not enough observations for the number of regimes and variables")

    {opts, warnings} = hac_without_err_sep(opts, warnings)

    b = %{
      n: n,
      y: Input.from_f64_binary(y_bin),
      xcols: xcols,
      names: names,
      ycols: bin_cols(yend_bin, n),
      yend_names: yend_names,
      qcols: bin_cols(q_bin, n),
      q_names: q_names,
      regimes: regimes,
      set: set,
      cols2regi: cols2regi!(opts[:cols2regi], length(names) + ky),
      warnings: warnings,
      opts: opts
    }

    if opts[:regime_err_sep] and Enum.all?(b.cols2regi) and opts[:constant_regi] == :many,
      do: tsls_separate(b),
      else: tsls_pooled(b)
  end

  def bin_cols(bin, n), do: bin |> Input.from_f64_binary() |> Enum.chunk_every(n)

  # spreg disables the separate regressions under HAC (the kernel couples the regimes).
  defp hac_without_err_sep(opts, warnings) do
    if opts[:robust] == :hac and opts[:regime_err_sep] do
      {Keyword.put(opts, :regime_err_sep, false),
       warnings ++
         [
           "Error by regimes is not available for HAC estimation. The error by regimes has been disabled for this model."
         ]}
    else
      {opts, warnings}
    end
  end

  defp tsls_opts(opts, extra) do
    [
      robust: opts[:robust],
      gwk: opts[:gwk],
      sig2n_k: opts[:sig2n_k],
      name_y: opts[:name_y]
    ] ++ extra
  end

  # One 2SLS on the regime design: x and yend by regime or global as `cols2regi` says, the
  # instruments always by regime (spreg: Regimes_Frame with cols2regi 'all').
  defp tsls_pooled(b), do: b |> tsls_pooled_design() |> elem(0)

  # The pooled fit and the columns of its regressor matrix `Z = [X | yend]`.
  @doc false
  def tsls_pooled_design(b) do
    kx = length(b.names)
    {x_flags, y_flags} = Enum.split(b.cols2regi, kx)
    const_regi = b.opts[:constant_regi] == :many
    ones = List.duplicate(1.0, b.n)

    {xd, xnames, krx, kfx, x_regi} =
      regimize(
        [
          {"CONSTANT", ones, const_regi}
          | Enum.zip_with([b.names, b.xcols, x_flags], &List.to_tuple/1)
        ],
        b.regimes,
        b.set
      )

    {yd, ynames, kry, kfy, y_regi} =
      regimize(
        Enum.zip_with([b.yend_names, b.ycols, y_flags], &List.to_tuple/1),
        b.regimes,
        b.set
      )

    {qd, qnames, _, _, _} =
      regimize(
        Enum.zip_with(
          [b.q_names, b.qcols, List.duplicate(true, length(b.qcols))],
          &List.to_tuple/1
        ),
        b.regimes,
        b.set
      )

    if krx + kry == 0, do: raise(ArgumentError, "at least one column must vary by regime")

    fit =
      Regression.tsls(
        b.y,
        xd,
        yd,
        qd,
        tsls_opts(b.opts, constant: false, name_x: xnames, name_yend: ynames, name_q: qnames)
      )

    kr = krx + kry
    kf = kfx + kfy

    result = %TSLSRegimes{
      n: b.n,
      kr: kr,
      kf: kf,
      kryd: kry,
      nr: length(b.set),
      regimes: b.set,
      betas: fit.betas,
      vm: fit.vm,
      std_err: fit.std_err,
      z_stat: fit.z_stat,
      chow:
        chow(
          fit.betas,
          fit.vm,
          {krx, kfx, kry, kfy},
          length(b.set),
          x_regi ++ y_regi,
          b.opts[:spreg_compatible]
        ),
      u: fit.u,
      predy: fit.predy,
      mode: :pooled,
      tsls: fit,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      robust: b.opts[:robust],
      names: xnames ++ ynames,
      name_x: xnames,
      name_yend: ynames,
      name_q: qnames,
      name_y: to_string(b.opts[:name_y]),
      warnings: b.warnings ++ fit.warnings
    }

    {result, xd ++ yd}
  end

  # One 2SLS per regime (regime_err_sep with everything by regime). As in spreg, an exogenous
  # column constant within any regime is dropped everywhere; `yend` and `q` are kept.
  defp tsls_separate(b) do
    ids = Map.new(b.set, fn r -> {r, indices(b.regimes, r)} end)

    keep =
      for {{name, col}, _} <- Enum.with_index(Enum.zip(b.names, b.xcols)),
          Enum.all?(b.set, &varies?(take(col, ids[&1]))),
          do: {name, col}

    dropped = b.names -- Enum.map(keep, &elem(&1, 0))

    warnings =
      if dropped == [],
        do: b.warnings,
        else:
          b.warnings ++
            [
              "Variable(s) #{inspect(dropped)} removed for being constant for at least one of the regimes."
            ]

    fits = Map.new(b.set, fn r -> {r, tsls_regime_fit(b, ids[r], keep, r)} end)
    kr = length(keep) + 1 + length(b.ycols)
    fit_list = Enum.map(b.set, &fits[&1])
    betas = Enum.flat_map(fit_list, & &1.betas)
    vm = block_diagonal(Enum.map(fit_list, & &1.vm))
    x_names = Enum.flat_map(b.set, &Enum.take(fits[&1].name_x, length(keep) + 1))
    ynames = Enum.flat_map(fit_list, & &1.name_yend)

    %TSLSRegimes{
      n: b.n,
      kr: kr,
      kf: 0,
      kryd: 0,
      nr: length(b.set),
      regimes: b.set,
      betas: betas,
      vm: vm,
      std_err: Enum.flat_map(fit_list, & &1.std_err),
      z_stat: Enum.flat_map(fit_list, & &1.z_stat),
      chow:
        chow(
          betas,
          vm,
          {kr, 0, 0, 0},
          length(b.set),
          ["CONSTANT" | Enum.map(keep, &elem(&1, 0))] ++ b.yend_names
        ),
      u: scatter(b.n, ids, b.set, &fits[&1].u),
      predy: scatter(b.n, ids, b.set, &fits[&1].predy),
      mode: :separate,
      multi: fits,
      constant_regi: b.opts[:constant_regi],
      cols2regi: b.cols2regi,
      robust: b.opts[:robust],
      names: Enum.flat_map(fit_list, & &1.name_x),
      name_x: x_names,
      name_yend: ynames,
      name_q: Enum.flat_map(fit_list, & &1.name_q),
      name_y: to_string(b.opts[:name_y]),
      warnings: warnings
    }
  end

  defp tsls_regime_fit(b, rows, keep, r) do
    pick = fn cols -> Enum.map(cols, &take(&1, rows)) end

    fit =
      Regression.tsls(
        take(b.y, rows),
        pick.(Enum.map(keep, &elem(&1, 1))),
        pick.(b.ycols),
        pick.(b.qcols),
        tsls_opts(Keyword.put(b.opts, :gwk, nil),
          name_x: Enum.map(keep, &elem(&1, 0)),
          name_yend: b.yend_names,
          name_q: b.q_names
        )
      )

    prefix = fn names -> Enum.map(names, &"#{r}_#{&1}") end
    x_names = prefix.(["CONSTANT" | Enum.map(keep, &elem(&1, 0))])
    yend_names = prefix.(b.yend_names)

    %{
      fit
      | name_x: x_names ++ yend_names,
        name_yend: yend_names,
        name_q: prefix.(b.q_names),
        name_y: "#{r}_#{b.opts[:name_y]}"
    }
  end

  # -- Chow tests ---------------------------------------------------------------------------

  # Chow restrictions: row j compares column `vari` of regime 0 with that of regime j + 1. The
  # coefficients are laid out as `kxr` exogenous regime blocks, `kxf` exogenous globals, `kyr`
  # endogenous regime blocks and `kyf` endogenous globals; `vari` runs over the regime-specific
  # exogenous columns, then the endogenous ones.
  #
  # spreg's `buildR1var` assumes the globals come last instead. That is the same layout unless
  # exogenous globals precede regime-specific endogenous columns (`kxf > 0 and kyr > 0`), where
  # its endogenous restrictions point at other coefficients. `spreg_compatible: true` reproduces
  # it; models without endogenous columns (`ols_regimes/4`) cannot hit the difference.
  defp restriction(vari, {kxr, kxf, kyr, kyf}, nr, true) do
    kf = kxf + kyf
    krexog = kxr
    kryd = kyr
    kr = kxr + kyr

    {kr_j, cbeg} =
      if vari < krexog, do: {krexog, vari}, else: {kryd, krexog * (nr - 1) + vari}

    for j <- 0..(nr - 2),
        do: Enum.map(0..(kr * nr + kf - 1), &coefficient(&1, cbeg, kr_j + cbeg + j * kr_j))
  end

  defp restriction(vari, {kxr, kxf, kyr, kyf}, nr, false) do
    total = nr * (kxr + kyr) + kxf + kyf

    {base, stride, v} =
      if vari < kxr, do: {0, kxr, vari}, else: {nr * kxr + kxf, kyr, vari - kxr}

    for j <- 0..(nr - 2),
        do: Enum.map(0..(total - 1), &coefficient(&1, base + v, base + v + (j + 1) * stride))
  end

  defp coefficient(c, plus, _minus) when c == plus, do: 1.0
  defp coefficient(c, _plus, minus) when c == minus, do: -1.0
  defp coefficient(_c, _plus, _minus), do: 0.0

  def chow(betas, vm, {kxr, _, kyr, _} = layout, nr, names, spreg_compatible \\ false) do
    per_var = for v <- 0..(kxr + kyr - 1), do: restriction(v, layout, nr, spreg_compatible)

    regi =
      for {r, name} <- Enum.zip(per_var, names), do: Tuple.insert_at(wald(betas, vm, r), 0, name)

    %{joint: wald(betas, vm, Enum.concat(per_var)), regi: regi}
  end

  defp wald(betas, vm, r) do
    rb = Enum.map(r, &dot(&1, betas))
    rv = Enum.map(r, fn row -> Enum.map(transpose(vm), &dot(row, &1)) end)
    rvr = Enum.map(rv, fn row -> Enum.map(r, &dot(row, &1)) end)
    w = dot(rb, solve(rvr, rb))
    df = length(r)

    [p] = (w * 1.0) |> Native.dynamics_chi2_sf(df * 1.0) |> Input.from_f64_binary()

    {w, p}
  end

  # Gauss-Jordan solve of the small symmetric positive definite system `a x = b`.
  defp solve(a, b) do
    n = length(b)
    rows = Enum.zip_with(a, b, fn row, v -> row ++ [v] end)

    reduced =
      Enum.reduce(0..(n - 1), rows, fn col, m ->
        {pivot_row, pivot_idx} =
          m
          |> Enum.with_index()
          |> Enum.drop(col)
          |> Enum.max_by(fn {row, _} -> abs(Enum.at(row, col)) end)

        m = m |> List.replace_at(pivot_idx, Enum.at(m, col)) |> List.replace_at(col, pivot_row)
        p = Enum.at(pivot_row, col)
        normalized = Enum.map(pivot_row, &(&1 / p))

        m
        |> Enum.with_index()
        |> Enum.map(fn
          {_row, ^col} ->
            normalized

          {row, _} ->
            f = Enum.at(row, col)
            Enum.zip_with(row, normalized, &(&1 - f * &2))
        end)
      end)

    Enum.map(reduced, &List.last/1)
  end

  # -- helpers ------------------------------------------------------------------------------

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
          constant_regi: :many,
          cols2regi: :all,
          regime_err_sep: true,
          robust: nil,
          sig2n_k: true,
          name_x: nil,
          name_y: "dep_var"
        ],
        opts
      )

    unless opts[:constant_regi] in [:many, :one],
      do:
        raise(
          ArgumentError,
          ":constant_regi must be :many or :one, got #{inspect(opts[:constant_regi])}"
        )

    unless opts[:robust] in [nil, :white],
      do: raise(ArgumentError, ":robust must be nil or :white, got #{inspect(opts[:robust])}")

    for key <- [:regime_err_sep, :sig2n_k],
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    opts
  end

  defp validate_tsls!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- @tsls_allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError,
              "unknown option(s) #{inspect(bad)}; expected #{inspect(@tsls_allowed)}"
    end

    opts =
      Keyword.merge(
        [
          constant_regi: :many,
          cols2regi: :all,
          regime_err_sep: false,
          robust: nil,
          gwk: nil,
          sig2n_k: true,
          spreg_compatible: false,
          name_x: nil,
          name_y: "dep_var",
          name_yend: nil,
          name_q: nil
        ],
        opts
      )

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

    for key <- [:regime_err_sep, :sig2n_k, :spreg_compatible],
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean")

    opts
  end

  def cols2regi!(:all, k), do: List.duplicate(true, k)

  def cols2regi!(flags, k) when is_list(flags) do
    unless length(flags) == k and Enum.all?(flags, &is_boolean/1),
      do: raise(ArgumentError, ":cols2regi must be :all or #{k} booleans, got #{inspect(flags)}")

    flags
  end

  def cols2regi!(other, _k),
    do:
      raise(ArgumentError, ":cols2regi must be :all or a list of booleans, got #{inspect(other)}")

  def indices(labels, r), do: for({l, i} <- Enum.with_index(labels), l == r, do: i)

  def take(list, ids) do
    t = List.to_tuple(list)
    Enum.map(ids, &elem(t, &1))
  end

  def varies?([first | rest]), do: Enum.any?(rest, &(&1 != first))
  def varies?([]), do: false

  def scatter(n, ids, set, values_of) do
    pairs = Enum.flat_map(set, fn r -> Enum.zip(ids[r], values_of.(r)) end)
    m = Map.new(pairs)
    Enum.map(0..(n - 1), &Map.fetch!(m, &1))
  end

  def block_diagonal(blocks) do
    sizes = Enum.map(blocks, &length/1)
    total = Enum.sum(sizes)

    {rows, _} =
      Enum.flat_map_reduce(blocks, 0, fn block, offset ->
        k = length(block)

        rows =
          Enum.map(
            block,
            &(List.duplicate(0.0, offset) ++ &1 ++ List.duplicate(0.0, total - offset - k))
          )

        {rows, offset + k}
      end)

    rows
  end

  # -- shared by the spatial regimes models ---------------------------------------------------

  @doc false
  def check_regimes!(regimes, n, kx) do
    set = regimes |> Enum.uniq() |> Enum.sort()
    if length(set) < 2, do: raise(ArgumentError, "regimes needs at least two distinct labels")

    if n / length(set) < kx + 1,
      do: raise(ArgumentError, "not enough observations for the number of regimes and variables")

    set
  end

  # spreg's w_regimes / w_regimes_union. The graph restricted to each regime's observations,
  # re-transformed as the graph was (a row-standardized graph stays row-standardized); a regime
  # must hold at least `min_n` observations.
  @doc false
  def regime_graphs(graph, regimes, set, min_n) do
    ids = Graph.ids(graph)
    label_of = ids |> Enum.zip(regimes) |> Map.new()
    adjacency = real_edges(graph)

    Map.new(set, fn r ->
      members = for {id, l} <- Enum.zip(ids, regimes), l == r, do: id

      if length(members) < min_n,
        do: raise(ArgumentError, "There are less observations than variables in regime #{r}.")

      edges = for {f, t, _} = e <- adjacency, label_of[f] == r, label_of[t] == r, do: e
      {r, build_graph(edges, members, graph.transform)}
    end)
  end

  @doc false
  def union_graph(graph, set, subs) do
    edges = Enum.flat_map(set, &real_edges(subs[&1]))
    build_graph(edges, Graph.ids(graph), graph.transform)
  end

  # `to_adjacency/1` lists an island as a zero self-loop.
  defp real_edges(graph),
    do: graph |> Graph.to_adjacency() |> Enum.reject(fn {f, t, w} -> f == t and w == 0.0 end)

  defp build_graph(edges, ids, transform) do
    {f, t, w} = unzip3(edges)
    g = Graph.from_arrays(f, t, w, ids: ids)
    if transform == :o, do: g, else: Graph.transform(g, transform)
  end

  defp unzip3(edges),
    do:
      {Enum.map(edges, &elem(&1, 0)), Enum.map(edges, &elem(&1, 1)),
       Enum.map(edges, &elem(&1, 2))}

  @doc false
  def island_warning(set, subs) do
    islands = for r <- set, Graph.isolates(subs[r]) != [], do: r

    case List.last(islands) do
      nil -> []
      r -> ["The regimes operation resulted in islands for regime #{r}."]
    end
  end

  # spreg's set_name_q_sp: the names of the spatially lagged instruments.
  @doc false
  def lag_names(names, w_lags) do
    counts =
      Enum.reduce(names, %{}, fn
        "W_" <> rest, acc -> Map.put_new(acc, rest, 2)
        name, acc -> Map.put(acc, name, 2)
      end)

    {out, _} =
      Enum.reduce(1..w_lags//1, {[], {MapSet.new(names), counts}}, fn _, acc ->
        Enum.reduce(names, acc, &lag_name/2)
      end)

    Enum.reverse(out)
  end

  defp lag_name("W_" <> rest, {out, {existing, counts}}) do
    lag = "W#{counts[rest]}_#{rest}"
    {[lag | out], {MapSet.put(existing, lag), Map.update!(counts, rest, &(&1 + 1))}}
  end

  defp lag_name(name, {out, {existing, counts}}) do
    if MapSet.member?(existing, "W_" <> name) do
      lag = "W#{counts[name]}_#{name}"
      {[lag | out], {MapSet.put(existing, lag), Map.update!(counts, name, &(&1 + 1))}}
    else
      lag = "W_" <> name
      {[lag | out], {MapSet.put(existing, lag), counts}}
    end
  end

  # `W y` (the last endogenous column) and the instruments, from the native `set_endog`.
  @doc false
  def lag_endog(b, w) do
    {:ok, list} =
      Native.regression_lag_endog(
        b.y_bin,
        {b.x_bin, length(b.names)},
        {b.yend_bin, b.ky},
        {b.q_bin, b.kq},
        w.ref,
        b.opts[:w_lags],
        b.opts[:lag_q]
      )
      |> ok!()

    by_name = Map.new(list)
    {bin_cols(by_name["yend"], b.n), bin_cols(by_name["q"], b.n)}
  end

  @doc false
  def prediction(w, y, predy, wy, rho, hard_bound) do
    {:ok, list} =
      Native.regression_sp_att(
        bin(y),
        bin(predy),
        bin(wy),
        w.ref,
        rho * 1.0,
        hard_bound
      )
      |> ok!()

    by_name = Map.new(list, fn {k, v} -> {k, Input.from_f64_binary(v)} end)

    warn =
      if Map.has_key?(by_name, "warn_rho"),
        do: [
          "*** WARNING: Estimate for spatial lag coefficient is outside the boundary (-1, 1). ***"
        ],
        else: []

    {by_name["predy_e"], by_name["e_pred"], warn}
  end

  @doc false
  def ok!({:error, reason}), do: raise(ArgumentError, to_string(reason))
  def ok!({:ok, _} = ok), do: ok

  @doc false
  def bin(list) when is_binary(list), do: list
  def bin(list), do: for(v <- list, into: <<>>, do: <<v * 1.0::float-native-64>>)

  def dot(a, b), do: a |> Enum.zip_with(b, &(&1 * &2)) |> Enum.sum()
  defp transpose(m), do: Enum.zip_with(m, & &1)
end
