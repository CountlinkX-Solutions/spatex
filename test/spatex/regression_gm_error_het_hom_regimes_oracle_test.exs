defmodule Spatex.RegressionGMErrorHetHomRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  # Fed spreg's own lambdas (the internal `*_at` hook) every other formula agrees with spreg at
  # `@exact`. With the optimizer, spreg's L-BFGS-B stops early (docs/parity.md, "spreg GM notes"):
  # lambda is compared at `@lam_tol`, everything derived from it at `@tol`.
  @exact 1.0e-8
  @lam_tol 5.0e-6
  @tol 1.0e-3

  defp sparse(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp graph(fx, "queen"), do: fx["graphs"]["queen"] |> sparse() |> Graph.transform(:r)
  defp graph(fx, "rook_binary"), do: sparse(fx["graphs"]["rook_binary"])

  defp options(c, compat) do
    base =
      [
        estimator: String.to_atom(c["estimator"]),
        name_x: c["x"],
        name_y: "CRIME",
        name_yend: c["yend"],
        name_q: c["q"]
      ] ++ if(compat, do: [spreg_compatible: true], else: [])

    Enum.reduce(c["opts"], base, fn
      {"constant_regi", v}, acc -> [{:constant_regi, String.to_atom(v)} | acc]
      {"A1", v}, acc -> [{:a1, String.to_atom(v)} | acc]
      {key, v}, acc -> [{String.to_atom(key), v} | acc]
    end)
  end

  # `at`: spreg's lambdas to consume instead of the optimizer (`nil`: optimize)
  defp fit(fx, c, compat, at) do
    cols = fn names -> Enum.map(names, &fx["columns"][&1]) end

    opts =
      options(c, compat)
      |> then(
        &if c["yend"] == [], do: &1, else: [{:yend, cols.(c["yend"])}, {:q, cols.(c["q"])} | &1]
      )

    args = [fx["y"], cols.(c["x"]), fx["regimes"][c["regimes"]], graph(fx, c["w"])]
    fun = if c["class"] == "combo", do: :gm_combo_regimes, else: :gm_error_regimes

    if at do
      apply(Regression, :"#{fun}_at", args ++ [lambdas_for(c, at), opts])
    else
      apply(Regression, fun, args ++ [opts])
    end
  end

  defp lambdas_for(%{"multi" => nil} = c, :spreg), do: c["lambdas"]

  defp lambdas_for(c, :spreg),
    do:
      c["lambdas_by_regime"]
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(&elem(&1, 1))

  defp assert_z(z_stat, expected, tol, label) do
    assert length(z_stat) == length(expected), "#{label}: z_stat length"

    for {{z, p}, [ez, ep]} <- Enum.zip(z_stat, expected) do
      Reference.assert_close(z, ez, tol)
      assert abs(p - ep) <= tol * max(ep, 1.0e-2), "#{label}: p"
    end
  end

  defp assert_pr2_e(nil, nil, _tol), do: :ok
  defp assert_pr2_e(got, expected, tol), do: assert_in_delta(got, expected, tol)

  # GM_Endog_Error_Het_Regimes (and so GM_Combo_Het_Regimes) names the lambda of each separate
  # regime "<r>lambda", without the underscore every other name has (known deviation).
  defp spreg_names(c),
    do: Enum.map(c["name_z"] || c["name_x"], &Regex.replace(~r/^(\d+)lambda$/, &1, "\\1_lambda"))

  defp load, do: Reference.load!("regression/gm_error_het_hom_regimes.json")

  defp limited?(c), do: Enum.any?(c["optim"], &(abs(&1["lam"] - &1["lam_exact"]) > @lam_tol))

  test "fed spreg's lambdas, every case matches spreg with the Chow tests on the real layout" do
    check_cases(:spreg, "chow_joint", "chow_regi", false, @exact)
  end

  test "spreg_compatible: true reproduces spreg's Chow restrictions exactly" do
    check_cases(:spreg, "spreg_chow_joint", "spreg_chow_regi", true, @exact)
  end

  test "with the optimizer the fits follow spreg within the optimizer tolerance" do
    check_cases(nil, "chow_joint", "chow_regi", false, @tol)
  end

  test "the Chow layout defect hits exactly the fits with global x and regime-wise endogenous columns" do
    flagged = for c <- load()["cases"], c["chow_layout_bug"], do: c["label"]

    expected =
      for est <- ~w(het hom),
          label <-
            ~w(endog_nsa_one_constant endog_nsa_mixed endog_three_one_constant combo_nsa_endog_one_constant combo_nsa_mixed),
          do: "#{est}_#{label}"

    assert Enum.sort(flagged) == Enum.sort(expected)

    for c <- load()["cases"], not c["chow_layout_bug"] do
      Reference.assert_close(c["chow_joint"], c["spreg_chow_joint"], 1.0e-8)
    end
  end

  test "the optimizer-limited cases are the flat-objective ones (binary weights, early stops)" do
    limited = for c <- load()["cases"], not c["clamped"], limited?(c), do: c["label"]

    assert Enum.sort(limited) ==
             ~w(het_combo_four_lagsep het_combo_nsa_lagsep_rook_binary het_error_nsa_rook_binary het_error_nsa_sep_rook_binary hom_combo_nsa_lagsep_rook_binary hom_error_nsa_rook_binary hom_error_nsa_sep_rook_binary)
  end

  # spreg's separate GM_Endog_Error_Het fits pin the last endogenous coefficient (HOVAL here) at
  # -0.9 (a spreg defect, see docs/parity.md); Spatex does not, and is compared with the twin
  # run of the same case with the clamp disabled.
  test "spreg's +-0.9 clamp of the separate Het endog fits is compared through unclamped twins" do
    cases = load()["cases"]
    clamped = for c <- cases, c["clamped"], do: c["label"]

    assert Enum.sort(clamped) ==
             ~w(het_endog_nsa_sep het_endog_nsa_sep_extra_iter3 het_endog_nsa_sep_rook_binary het_endog_three_sep)

    for label <- clamped do
      twin = Enum.find(cases, &(&1["label"] == label <> "/unclamped"))
      original = Enum.find(cases, &(&1["label"] == label))
      assert twin["no_clamp"] and not twin["clamped"]

      assert Enum.any?(
               Enum.zip(twin["betas"], original["betas"]),
               fn {a, b} -> abs(a - b) > 1.0e-6 end
             )
    end
  end

  defp check_cases(at, joint_key, regi_key, compat, tol) do
    fx = load()

    for c <- fx["cases"], not c["clamped"], at == :spreg or not limited?(c) do
      label = c["label"]
      r = fit(fx, c, compat, at)

      assert r.name_x == c["name_x"], label
      assert r.name_yend == c["name_yend"], label
      assert r.names == spreg_names(c), label
      assert {r.kr, r.kf, r.kryd, r.nr} == {c["kr"], c["kf"], c["kryd"], c["nr"]}, label
      assert r.regime_err_sep == c["regime_err_sep"], label
      assert c["multi"] != nil == (r.mode == :separate), label
      assert r.estimator == String.to_atom(c["estimator"]), label

      for phrase <- warning_phrases(c["warning"]) do
        assert Enum.any?(r.warnings, &String.contains?(&1, phrase)), "#{label}: #{phrase}"
      end

      check_lambdas(r, c, at, label)
      check_fit(r, c, joint_key, regi_key, tol, label)
    end
  end

  defp lambdas(%{mode: :pooled} = r), do: [r.lam]
  defp lambdas(r), do: Enum.map(r.regimes, &r.lams[&1])

  defp call_count(%{mode: :pooled} = r), do: length(r.lambdas)
  defp call_count(r), do: r.lambdas |> Map.values() |> Enum.map(&length/1) |> Enum.sum()

  # lambda against spreg's and against the exact minimizer of the objective spreg minimizes
  defp check_lambdas(r, c, at, label) do
    assert call_count(r) == length(c["optim"]), label
    lams = lambdas(r)

    if at == :spreg do
      for {lam, spreg} <- Enum.zip(lams, last_lambdas(c)), do: assert(lam == spreg, label)
    else
      for {lam, spreg} <- Enum.zip(lams, last_lambdas(c)),
          do: assert_in_delta(lam, spreg, @lam_tol, label)
    end
  end

  # the lambda each fit returns: the last optimizer call of its (regime's) run
  defp last_lambdas(%{"multi" => nil} = c), do: [List.last(c["lambdas"])]

  defp last_lambdas(c),
    do:
      c["lambdas_by_regime"]
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(&List.last(elem(&1, 1)))

  defp check_fit(r, c, joint_key, regi_key, tol, label) do
    Reference.assert_close(r.betas, c["betas"], tol)
    Reference.assert_close(List.flatten(r.vm), c["vm"], tol)
    Reference.assert_close(r.u, c["u"], tol)
    Reference.assert_close(r.predy, c["predy"], tol)
    Reference.assert_close(r.e_filtered, c["e_filtered"], tol)
    Reference.assert_close(Tuple.to_list(r.chow.joint), c[joint_key], tol)

    Reference.assert_close(
      Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
      List.flatten(c[regi_key]),
      tol
    )

    if c["multi"], do: check_multi(r, c, tol, label), else: check_pooled(r, c, tol, label)
  end

  defp warning_phrases(""), do: []

  defp warning_phrases(text),
    do:
      text
      |> String.split("\n", trim: true)
      |> Enum.map(&String.replace_prefix(&1, "Warning: ", ""))
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

  defp assert_sig2(nil, nil, _tol), do: :ok

  defp assert_sig2(got, expected, tol),
    do: assert_in_delta(got, expected, tol * max(expected, 1.0))

  defp check_pooled(r, c, tol, label) do
    Reference.assert_close(r.std_err, c["std_err"], tol)
    assert_z(r.z_stat, c["z_stat"], tol, label)
    assert_sig2(r.sig2, c["sig2"], tol)
    assert c["estimator"] == "het" == (r.sig2 == nil), label
    assert_in_delta r.pr2, c["pr2"], tol
    assert r.lam == List.last(r.betas)
    assert r.multi == nil and r.lams == nil
    assert r.iteration == c["iteration"], label

    if c["class"] == "combo" do
      assert_in_delta r.rho, c["rho"], tol
      assert r.rho == Enum.at(r.betas, -2)
      Reference.assert_close(r.predy_e, c["predy_e"], tol)
      Reference.assert_close(r.e_pred, c["e_pred"], tol)
      assert_pr2_e(r.pr2_e, c["pr2_e"], tol)
    end
  end

  defp check_multi(r, c, tol, label) do
    keys = Enum.sort(Map.keys(c["multi"]))
    assert Enum.map(r.multi, &to_string(elem(&1, 0))) |> Enum.sort() == keys

    for {key, m} <- c["multi"] do
      fit = Enum.find_value(r.multi, fn {k, v} -> to_string(k) == key && v end)
      Reference.assert_close(fit.betas, m["betas"], tol)
      Reference.assert_close(List.flatten(fit.vm), m["vm"], tol)
      Reference.assert_close(fit.e_filtered, m["e_filtered"], tol)
      assert_sig2(fit.sig2, m["sig2"], tol)
      assert fit.iteration == m["iteration"], label

      if m["std_err"] do
        Reference.assert_close(fit.std_err, m["std_err"], tol)
        assert_z(fit.z_stat, m["z_stat"], tol, label)
        assert_in_delta fit.pr2, m["pr2"], tol
      end

      if c["class"] == "combo" do
        assert_in_delta fit.rho, m["rho"], tol
        assert_pr2_e(fit.pr2_e, m["pr2_e"], tol)
        Reference.assert_close(fit.predy_e, m["predy_e"], tol)
        Reference.assert_close(fit.e_pred, m["e_pred"], tol)
      end
    end

    assert r.pr2 == nil and r.sig2 == nil
    assert r.lam == nil and Enum.sort(Enum.map(Map.keys(r.lams), &to_string/1)) == keys

    if c["class"] == "combo" do
      Reference.assert_close(r.predy_e, c["predy_e"], tol)
      assert r.rho == nil
      assert map_size(r.rhos) == length(keys)
    end
  end
end
