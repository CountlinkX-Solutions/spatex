defmodule Spatex.RegressionGMErrorRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  # spreg's L-BFGS-B stops early (docs/parity.md, "spreg GM notes"): lambda is compared at
  # `@lam_tol`, everything derived from it (betas move up to ~100x the lambda gap) at `@tol`.
  @lam_tol 5.0e-6
  @tol 1.0e-3
  @chow_tol 1.0e-3

  defp sparse(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp graph(fx, "queen"), do: fx["graphs"]["queen"] |> sparse() |> Graph.transform(:r)
  defp graph(fx, "rook_binary"), do: sparse(fx["graphs"]["rook_binary"])

  defp options(c, compat) do
    base =
      [name_x: c["x"], name_y: "CRIME", name_yend: c["yend"], name_q: c["q"]] ++
        if(compat, do: [spreg_compatible: true], else: [])

    Enum.reduce(c["opts"], base, fn
      {"constant_regi", v}, acc -> [{:constant_regi, String.to_existing_atom(v)} | acc]
      {key, v}, acc -> [{String.to_existing_atom(key), v} | acc]
    end)
  end

  defp fit(fx, c, compat) do
    cols = fn names -> Enum.map(names, &fx["columns"][&1]) end

    opts =
      options(c, compat)
      |> then(
        &if c["yend"] == [], do: &1, else: [{:yend, cols.(c["yend"])}, {:q, cols.(c["q"])} | &1]
      )

    args = [fx["y"], cols.(c["x"]), fx["regimes"][c["regimes"]], graph(fx, c["w"]), opts]

    case c["class"] do
      "combo" -> apply(Regression, :gm_combo_regimes, args)
      _ -> apply(Regression, :gm_error_regimes, args)
    end
  end

  defp assert_z(z_stat, expected, label) do
    assert length(z_stat) == length(expected), "#{label}: z_stat length"

    for {{z, p}, [ez, ep]} <- Enum.zip(z_stat, expected) do
      Reference.assert_close(z, ez, @tol)
      assert abs(p - ep) <= @tol * max(ep, 1.0e-2), "#{label}: p"
    end
  end

  # spreg omits the spatial pseudo R-squared when |rho| >= 1.
  defp assert_pr2_e(nil, nil), do: :ok
  defp assert_pr2_e(got, expected), do: assert_in_delta(got, expected, @tol)

  test "every case matches spreg with the Chow tests on the real coefficient layout" do
    check_cases("chow_joint", "chow_regi", false)
  end

  test "spreg_compatible: true reproduces spreg's Chow restrictions exactly" do
    check_cases("spreg_chow_joint", "spreg_chow_regi", true)
  end

  test "the Chow layout defect hits exactly the fits with global x and regime-wise endogenous columns" do
    fx = Reference.load!("regression/gm_error_regimes.json")
    flagged = for c <- fx["cases"], c["chow_layout_bug"], do: c["label"]

    assert Enum.sort(flagged) ==
             ~w(combo_nsa_endog_one_constant combo_nsa_mixed endog_nsa_mixed endog_nsa_one_constant endog_three_one_constant)

    for c <- fx["cases"], not c["chow_layout_bug"] do
      Reference.assert_close(c["chow_joint"], c["spreg_chow_joint"], 1.0e-8)
    end
  end

  # Models whose spreg lambda is further than `@lam_tol` from the exact minimizer of spreg's own
  # moment objective (a flat objective with binary weights, or an early L-BFGS-B stop): spreg's
  # derived quantities then follow its lambda, not the minimizer's. Those cases are checked for
  # layout, names and lambda only (see docs/parity.md, "spreg GM notes").
  defp limited?(c), do: Enum.any?(c["optim"], &(abs(&1["lam"] - &1["lam_exact"]) > @lam_tol))

  test "the optimizer-limited cases are exactly the flat-objective ones" do
    fx = Reference.load!("regression/gm_error_regimes.json")
    limited = for c <- fx["cases"], limited?(c), do: c["label"]

    assert Enum.sort(limited) ==
             ~w(combo_four_lagsep combo_nsa_lagsep_rook_binary combo_nsa_rook_binary combo_three_endog_pooled endog_nsa_sep_rook_binary error_nsa_rook_binary error_nsa_sep_rook_binary)
  end

  defp check_cases(joint_key, regi_key, compat) do
    fx = Reference.load!("regression/gm_error_regimes.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = fit(fx, c, compat)

      assert r.name_x == c["name_x"], label
      assert r.name_yend == c["name_yend"], label
      assert r.names == (c["name_z"] || c["name_x"]), label
      assert {r.kr, r.kf, r.kryd, r.nr} == {c["kr"], c["kf"], c["kryd"], c["nr"]}, label
      assert r.regime_err_sep == c["regime_err_sep"], label
      assert c["multi"] != nil == (r.mode == :separate), label

      for phrase <- warning_phrases(c["warning"]) do
        assert Enum.any?(r.warnings, &String.contains?(&1, phrase)), "#{label}: #{phrase}"
      end

      check_lambdas(r, c, label)

      unless limited?(c) do
        check_fit(r, c, joint_key, regi_key, label)
      end
    end
  end

  defp lambdas(%{mode: :pooled} = r), do: [r.lam]
  defp lambdas(r), do: Enum.map(r.regimes, &r.lams[&1])

  # lambda against spreg's, and against the exact minimizer of the objective spreg minimizes
  defp check_lambdas(r, c, label) do
    assert length(c["optim"]) == length(lambdas(r)), label

    for {lam, o} <- Enum.zip(lambdas(r), c["optim"]) do
      assert_in_delta lam, o["lam_exact"], @lam_tol
      assert o["f_exact"] <= o["f"] * (1 + 1.0e-6) + 1.0e-12, label
    end

    unless limited?(c) do
      for {lam, o} <- Enum.zip(lambdas(r), c["optim"]),
          do: assert_in_delta(lam, o["lam"], @lam_tol)
    end
  end

  defp check_fit(r, c, joint_key, regi_key, label) do
    Reference.assert_close(r.betas, c["betas"], @tol)
    Reference.assert_close(List.flatten(r.vm), c["vm"], @tol)
    Reference.assert_close(r.u, c["u"], @tol)
    Reference.assert_close(r.predy, c["predy"], @tol)
    Reference.assert_close(r.e_filtered, c["e_filtered"], @tol)
    Reference.assert_close(Tuple.to_list(r.chow.joint), c[joint_key], @chow_tol)

    Reference.assert_close(
      Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
      List.flatten(c[regi_key]),
      @chow_tol
    )

    if c["multi"], do: check_multi(r, c, label), else: check_pooled(r, c, label)
  end

  defp warning_phrases(""), do: []

  defp warning_phrases(text),
    do:
      text
      |> String.split("\n", trim: true)
      |> Enum.map(&String.replace_prefix(&1, "Warning: ", ""))
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

  defp check_pooled(r, c, label) do
    Reference.assert_close(r.std_err, c["std_err"], @tol)
    assert_z(r.z_stat, c["z_stat"], label)
    assert_in_delta r.sig2, c["sig2"], @tol * max(c["sig2"], 1.0)
    assert_in_delta r.pr2, c["pr2"], @tol
    assert r.lam == List.last(r.betas)
    assert r.multi == nil and r.lams == nil

    if c["class"] == "combo" do
      assert_in_delta r.rho, c["rho"], @tol
      assert r.rho == Enum.at(r.betas, -2)
      Reference.assert_close(r.predy_e, c["predy_e"], @tol)
      Reference.assert_close(r.e_pred, c["e_pred"], @tol)
      assert_pr2_e(r.pr2_e, c["pr2_e"])
    end
  end

  defp check_multi(r, c, label) do
    keys = Enum.sort(Map.keys(c["multi"]))
    assert Enum.map(r.multi, &to_string(elem(&1, 0))) |> Enum.sort() == keys

    for {key, m} <- c["multi"] do
      fit = Enum.find_value(r.multi, fn {k, v} -> to_string(k) == key && v end)
      Reference.assert_close(fit.betas, m["betas"], @tol)
      Reference.assert_close(List.flatten(fit.vm), m["vm"], @tol)
      Reference.assert_close(fit.std_err, m["std_err"], @tol)
      assert_z(fit.z_stat, m["z_stat"], label)
      assert_in_delta fit.sig2, m["sig2"], @tol * max(m["sig2"], 1.0)
      assert_in_delta fit.pr2, m["pr2"], @tol
      Reference.assert_close(fit.e_filtered, m["e_filtered"], @tol)

      if c["class"] == "combo" do
        assert_in_delta fit.rho, m["rho"], @tol
        assert_pr2_e(fit.pr2_e, m["pr2_e"])
        Reference.assert_close(fit.predy_e, m["predy_e"], @tol)
        Reference.assert_close(fit.e_pred, m["e_pred"], @tol)
      end
    end

    assert r.pr2 == nil and r.sig2 == nil
    assert r.lam == nil and Enum.sort(Enum.map(Map.keys(r.lams), &to_string/1)) == keys
    pieces = fn field -> Enum.flat_map(keys, &c["multi"][&1][field]) end
    Reference.assert_close(r.std_err, pieces.("std_err"), @tol)
    assert_z(r.z_stat, Enum.flat_map(keys, &c["multi"][&1]["z_stat"]), label)

    if c["class"] == "combo" do
      Reference.assert_close(r.predy_e, c["predy_e"], @tol)
      assert r.rho == nil
      assert map_size(r.rhos) == length(keys)
    end
  end
end
