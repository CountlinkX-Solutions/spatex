defmodule Spatex.RegressionGMOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference
  alias Spatex.Regression
  alias Spatex.Regression.{GMCombo, GMError, GMLag, TSLS}

  # spreg minimizes the GMM moment objective with L-BFGS-B, which stops early (lambda ~1e-6 from
  # the exact minimizer; up to 5e-6 here). Spatex finds the exact minimizer, so fitted models are
  # compared with `@lam_tol` for lambda and `@fit_tol` for everything derived from it (which moves
  # up to ~100x the lambda gap: betas reach 60).
  # The "at-lambda" tests feed spreg's own lambdas to the same code and demand 1e-9 everywhere;
  # TSLS and GM_Lag have no optimizer and are exact at 1e-9 in both modes.
  @lam_tol 5.0e-6
  @fit_tol 5.0e-4
  @exact_tol 1.0e-9

  defp graph(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  @opt_keys %{
    "sig2n_k" => :sig2n_k,
    "w_lags" => :w_lags,
    "lag_q" => :lag_q,
    "step1c" => :step1c,
    "max_iter" => :max_iter,
    "robust" => :robust
  }

  defp options(c) do
    base =
      for {k, v} <- c["opts"], Map.has_key?(@opt_keys, k) do
        {@opt_keys[k], if(k == "robust", do: String.to_existing_atom(v), else: v)}
      end

    base =
      case c["opts"]["A1"] do
        nil -> base
        a1 -> [{:a1, String.to_existing_atom(a1)} | base]
      end

    base =
      case c["opts"]["spat_impacts"] do
        nil -> base
        list when length(list) == 3 -> [{:spat_impacts, :all} | base]
        _ -> base
      end

    base
    |> maybe(:gwk, c["gwk"] && graph(c["gwk"]))
    |> Keyword.put(:name_x, c["name_x"])
    |> Keyword.put(:name_y, "y")
    |> maybe(:yend, c["yend"])
    |> maybe(:q, c["q"])
    |> maybe(:name_yend, c["yend_names"])
    |> maybe(:name_q, c["q_names"])
  end

  defp maybe(opts, _key, nil), do: opts
  defp maybe(opts, key, value), do: Keyword.put(opts, key, value)

  defp estimator("gm_error_het"), do: :het
  defp estimator("gm_endog_error_het"), do: :het
  defp estimator("gm_combo_het"), do: :het
  defp estimator("gm_error_hom"), do: :hom
  defp estimator("gm_endog_error_hom"), do: :hom
  defp estimator("gm_combo_hom"), do: :hom
  defp estimator(_), do: :kp98

  defp fit(c, mode) do
    g = graph(c["w"])
    opts = options(c)
    est = [estimator: estimator(c["model"])]

    case {c["model"], mode} do
      {"tsls", _} ->
        Regression.tsls(c["y"], c["x"], c["yend"], c["q"], Keyword.drop(opts, [:yend, :q]))

      {"gm_lag", _} ->
        Regression.gm_lag(c["y"], c["x"], g, opts)

      {"gm_combo" <> _, :fit} ->
        Regression.gm_combo(c["y"], c["x"], g, opts ++ est)

      {"gm_combo" <> _, :at} ->
        Regression.gm_combo_at(c["y"], c["x"], g, c["lambdas"], opts ++ est)

      {_, :fit} ->
        Regression.gm_error(c["y"], c["x"], g, opts ++ est)

      {_, :at} ->
        Regression.gm_error_at(c["y"], c["x"], g, c["lambdas"], opts ++ est)
    end
  end

  defp lam_of(%GMError{lam: lam}), do: lam
  defp lam_of(%GMCombo{lam: lam}), do: lam

  defp assert_z(z_stat, expected, tol, label) do
    assert length(z_stat) == length(expected), "#{label}: z_stat length"

    z_stat
    |> Enum.zip(expected)
    |> Enum.with_index()
    |> Enum.each(fn {{{z, p}, [ez, ep]}, i} ->
      Reference.assert_close(z, ez, tol)
      assert abs(p - ep) <= max(tol * ep, 1.0e-300) + tol * 1.0e-3, "#{label}: p[#{i}]"
    end)
  end

  # Attributes every model reports.
  defp check_common(r, e, tol, label) do
    Reference.assert_close(r.betas, e["betas"], tol)
    Reference.assert_close(r.vm, e["vm"], tol)
    Reference.assert_close(r.std_err, e["std_err"], tol)
    assert_z(r.z_stat, e["z_stat"], tol, label)
    Reference.assert_close(r.u, e["u"], tol)
    Reference.assert_close(r.predy, e["predy"], tol)
    Reference.assert_close(r.pr2, e["pr2"], tol)
    assert r.n == e["n"]
    assert r.name_x == (e["name_z"] || e["name_x"]), label

    if e["sig2"], do: Reference.assert_close(r.sig2, e["sig2"], tol)
  end

  defp check(%TSLS{} = r, e, _mode, tol, label) do
    check_common(r, e, tol, label)
    Reference.assert_close(r.varb, e["varb"], tol)
    assert r.k == e["k"]
  end

  defp check(%GMLag{} = r, e, _mode, tol, label) do
    check_common(r, e, tol, label)
    Reference.assert_close(r.rho, e["rho"], tol)
    Reference.assert_close(r.varb, e["varb"], tol)
    Reference.assert_close(r.predy_e, e["predy_e"], tol)
    Reference.assert_close(r.e_pred, e["e_pred"], tol)
    Reference.assert_close(r.pr2_e, e["pr2_e"], tol)
    assert r.k == e["k"]

    case e["ak_test"] do
      nil ->
        assert r.ak_test == nil

      [ak, p] ->
        {rak, rp} = r.ak_test
        Reference.assert_close(rak, ak, tol)
        Reference.assert_close(rp, p, tol)
    end

    for {name, [adi, aii, ati]} <- e["sp_multipliers"] do
      assert {a, i, t} = r.sp_multipliers[String.to_existing_atom(name)]
      Reference.assert_close([a, i, t], [adi, aii, ati], tol)
    end

    assert map_size(r.sp_multipliers) == map_size(e["sp_multipliers"])
  end

  defp check(r, e, mode, tol, label) when is_struct(r, GMError) or is_struct(r, GMCombo) do
    check_common(r, e, tol, label)
    assert_lam(lam_of(r), List.last(e["betas"]), mode)
    Reference.assert_close(r.e_filtered, e["e_filtered"], tol)

    if e["iteration"] do
      assert r.iteration == e["iteration"], label
      assert r.iter_stop == e["iter_stop"], label
    end

    if e["sig2"] == nil, do: assert(r.sig2 == nil)

    if is_struct(r, GMCombo) do
      Reference.assert_close(r.rho, e["rho"], tol)
      Reference.assert_close(r.predy_e, e["predy_e"], tol)
      Reference.assert_close(r.e_pred, e["e_pred"], tol)
      Reference.assert_close(r.pr2_e, e["pr2_e"], tol)
    end
  end

  defp assert_lam(lam, expected, :fit), do: Reference.assert_close(lam, expected, @lam_tol)
  defp assert_lam(lam, expected, :loose), do: Reference.assert_close(lam, expected, 1.0e-4)
  defp assert_lam(lam, expected, :at), do: Reference.assert_close(lam, expected, @exact_tol)

  defp all_cases(file), do: Reference.load!("regression/#{file}.json")["cases"]

  # GM_Endog_Error_Het clamps its last endogenous coefficient to +-0.9 even without W y (a spreg
  # bug, see "clamp deviation"); those cases are not part of the parity runs.
  defp deviation?(c), do: String.starts_with?(c["label"], "gm_endog_error_het/step1c3")
  defp cases(file), do: Enum.reject(all_cases(file), &deviation?/1)

  for file <- ~w(gm_columbus gm_synthetic gm_clamp gm_optimizer) do
    describe "#{file}" do
      test "TSLS and GM_Lag match spreg at 1e-9 (no optimizer)" do
        for c <- cases(unquote(file)), c["model"] in ["tsls", "gm_lag"] do
          check(fit(c, :fit), c["expected"], :fit, @exact_tol, c["label"])
        end
      end

      test "every formula matches spreg at 1e-9 when fed spreg's own lambdas" do
        for c <- cases(unquote(file)), c["model"] not in ["tsls", "gm_lag"] do
          check(fit(c, :at), c["expected"], :at, @exact_tol, c["label"])
        end
      end
    end
  end

  # gm_optimizer holds the cases where spreg's own lambda is not the minimizer (see below).
  for file <- ~w(gm_columbus gm_synthetic gm_clamp) do
    describe "#{file} fits" do
      test "fitted GMM models follow spreg within the optimizer tolerance" do
        for c <- cases(unquote(file)), c["model"] not in ["tsls", "gm_lag"] do
          check(fit(c, :fit), c["expected"], :fit, @fit_tol, c["label"])
        end
      end
    end
  end

  describe "optimizer" do
    test "lambda is within 5e-6 of spreg's L-BFGS-B result wherever that is the minimizer" do
      gaps =
        for file <- ~w(gm_columbus gm_synthetic gm_clamp),
            c <- cases(file),
            c["model"] not in ["tsls", "gm_lag"] do
          r = fit(c, :fit)
          {abs(lam_of(r) - List.last(c["expected"]["betas"])), c["label"]}
        end

      assert {worst, label} = Enum.max(gaps)
      assert worst < @lam_tol, "#{label}: #{worst}"
    end

    # known_deviations.exs: spreg's L-BFGS-B (forward-difference gradient, factr 1e7) misses the
    # minimizer of its own objective in these cases. `first_step` holds the first-step objective
    # at spreg's lambda and at the exact minimizer (SciPy bounded Brent, xatol 1e-12).
    for label <- [
          "gm_error/crime~inc+hoval/rook_binary",
          "gm_endog_error_het/crime~inc|hoval|discbd/queen_r"
        ] do
      test "#{label}: Spatex finds the minimizer of the moment objective spreg only approximates" do
        c = Enum.find(cases("gm_optimizer"), &(&1["label"] == unquote(label)))
        fs = c["first_step"]
        [pysal_first | _] = c["lambdas"]
        [spatex_first | _] = fit(c, :fit).lambdas

        assert abs(spatex_first - fs["lam_exact"]) < 1.0e-8
        assert abs(spatex_first - pysal_first) > 1.0e-4
        assert fs["f_pysal"] > fs["f_exact"]
      end
    end

    test "gm_error_hom with binary weights: flat objective, spreg's weighted step stops 1.6e-5 early" do
      c =
        Enum.find(
          cases("gm_optimizer"),
          &(&1["label"] == "gm_error_hom/crime~inc+hoval/rook_binary")
        )

      r = fit(c, :fit)
      gap = abs(r.lam - List.last(c["expected"]["betas"]))
      assert gap > @lam_tol and gap < 1.0e-4
      check(r, c["expected"], :loose, 2.0e-3, c["label"])
    end
  end

  describe "clamp deviation (GM_Endog_Error_Het without W y)" do
    # known_deviations.exs: spreg clamps `tsls.betas[-1]` to +-0.9 in both the first stage and
    # every iteration. That is meant for rho (the last column of Z in the combo models), but
    # GM_Endog_Error_Het applies it to whichever endogenous regressor comes last (HOVAL here, with
    # coefficient -1.35 unclamped). Spatex clamps only in the combo models.
    setup do
      cs = Map.new(all_cases("gm_clamp"), &{&1["label"], &1})

      %{
        pysal: cs["gm_endog_error_het/step1c3"],
        plain: cs["gm_endog_error_het/step1c3/unclamped"]
      }
    end

    test "spreg's clamped run pins the coefficient at -0.9", %{pysal: c, plain: plain} do
      assert Enum.at(c["expected"]["betas"], 2) == -0.9
      assert Enum.at(plain["expected"]["betas"], 2) < -1.0
    end

    test "Spatex equals spreg with the clamp disabled (1e-9, spreg's lambdas)", %{plain: plain} do
      check(fit(plain, :at), plain["expected"], :at, @exact_tol, plain["label"])
    end

    test "Spatex does not replicate the clamp", %{pysal: c} do
      r = fit(c, :at)
      refute abs(Enum.at(r.betas, 2) - Enum.at(c["expected"]["betas"], 2)) < 1.0e-3
    end

    test "the fitted model is the unclamped one within the optimizer tolerance", %{plain: plain} do
      check(fit(plain, :fit), plain["expected"], :fit, @fit_tol, plain["label"])
    end
  end

  describe "rho clamp (GM_Combo_Het)" do
    test "the replicated clamp keeps the iterated rho at 0.9" do
      for c <- cases("gm_clamp"), String.contains?(c["label"], "rho_clamp") do
        assert Enum.at(c["expected"]["betas"], 2) == 0.9
        r = fit(c, :at)
        assert Enum.at(r.betas, 2) == 0.9
      end
    end
  end
end
