defmodule Spatex.RegressionGMLagRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  @tol 1.0e-9
  @chow_tol 1.0e-8

  defp sparse(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp graph(fx, "queen"), do: fx["graphs"]["queen"] |> sparse() |> Graph.transform(:r)
  defp graph(fx, "rook_binary"), do: sparse(fx["graphs"]["rook_binary"])

  defp options(c, gwk, compat) do
    base =
      [name_x: c["x"], name_y: "CRIME", name_yend: c["yend"], name_q: c["q"]] ++
        if(compat, do: [spreg_compatible: true], else: [])

    Enum.reduce(c["opts"], base, fn
      {"robust", v}, acc -> [{:robust, v && String.to_existing_atom(v)} | acc]
      {"spat_impacts", v}, acc -> [{:spat_impacts, v && String.to_existing_atom(v)} | acc]
      {"constant_regi", v}, acc -> [{:constant_regi, String.to_existing_atom(v)} | acc]
      {key, v}, acc -> [{String.to_existing_atom(key), v} | acc]
    end)
    |> then(&if c["gwk"], do: [{:gwk, gwk} | &1], else: &1)
  end

  defp fit(fx, c, compat) do
    cols = fn names -> Enum.map(names, &fx["columns"][&1]) end

    opts =
      options(c, sparse(fx["gwk"]), compat)
      |> then(
        &if c["yend"] == [], do: &1, else: [{:yend, cols.(c["yend"])}, {:q, cols.(c["q"])} | &1]
      )

    Regression.gm_lag_regimes(
      fx["y"],
      cols.(c["x"]),
      fx["regimes"][c["regimes"]],
      graph(fx, c["w"]),
      opts
    )
  end

  # spreg omits the spatial pseudo R-squared when |rho| >= 1.
  defp assert_pr2_e(nil, nil), do: :ok
  defp assert_pr2_e(got, expected), do: assert_in_delta(got, expected, 1.0e-10)

  defp assert_z(z_stat, expected, label) do
    assert length(z_stat) == length(expected), "#{label}: z_stat length"

    for {{z, p}, [ez, ep]} <- Enum.zip(z_stat, expected) do
      Reference.assert_close(z, ez, @tol)
      assert abs(p - ep) <= max(@tol * ep, 1.0e-300) + 1.0e-12, "#{label}: p"
    end
  end

  test "every case matches spreg with the Chow tests on the real coefficient layout" do
    check_cases("chow_joint", "chow_regi", false)
  end

  test "spreg_compatible: true reproduces spreg's Chow restrictions exactly" do
    check_cases("spreg_chow_joint", "spreg_chow_regi", true)
  end

  test "the Chow layout defect hits exactly the fits with global exogenous and regime-wise endogenous columns" do
    fx = Reference.load!("regression/gm_lag_regimes.json")
    flagged = for c <- fx["cases"], c["chow_layout_bug"], do: c["label"]

    assert Enum.sort(flagged) ==
             ~w(four_lagsep_mixed nsa_endog_mixed nsa_endog_one_constant nsa_lagsep_endog_one_constant nsa_lagsep_mixed nsa_lagsep_mixed_rook_binary nsa_lagsep_one_constant)

    for c <- fx["cases"], not c["chow_layout_bug"] do
      Reference.assert_close(c["chow_joint"], c["spreg_chow_joint"], @chow_tol)
    end
  end

  test "the default robust estimator is spreg's: White" do
    fx = Reference.load!("regression/gm_lag_regimes.json")
    c = Enum.find(fx["cases"], &(&1["label"] == "nsa_pooled_default_white"))
    assert fit(fx, c, false).robust == :white
  end

  defp check_cases(joint_key, regi_key, compat) do
    fx = Reference.load!("regression/gm_lag_regimes.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = fit(fx, c, compat)

      assert r.name_x == c["name_x"], label
      assert r.name_yend == c["name_yend"], label
      assert r.name_q == c["name_q"], label
      assert {r.kr, r.kf, r.kryd, r.nr} == {c["kr"], c["kf"], c["kryd"], c["nr"]}, label
      assert {r.regime_lag_sep, r.regime_err_sep} == {c["regime_lag_sep"], c["regime_err_sep"]}
      assert c["multi"] != nil == (r.mode == :separate), label

      Reference.assert_close(r.betas, c["betas"], @tol)
      Reference.assert_close(List.flatten(r.vm), c["vm"], @tol)
      Reference.assert_close(r.u, c["u"], @tol)
      Reference.assert_close(r.predy, c["predy"], @tol)
      Reference.assert_close(r.predy_e, c["predy_e"], @tol)
      Reference.assert_close(r.e_pred, c["e_pred"], @tol)
      Reference.assert_close(Tuple.to_list(r.chow.joint), c[joint_key], @chow_tol)

      Reference.assert_close(
        Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
        List.flatten(c[regi_key]),
        @chow_tol
      )

      for phrase <- warning_phrases(c["warning"]) do
        assert Enum.any?(r.warnings, &String.contains?(&1, phrase)), "#{label}: #{phrase}"
      end

      if c["multi"] do
        check_multi(r, c, label)
      else
        check_pooled(r, c, label)
      end
    end
  end

  defp warning_phrases(""), do: []

  defp warning_phrases(text),
    do:
      text
      |> String.split("\n", trim: true)
      |> Enum.map(&String.replace_prefix(&1, "Warning: ", ""))
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == "" or String.starts_with?(&1, ["so regime_err_sep", "Hence"])))

  defp check_pooled(r, c, label) do
    assert r.tsls != nil, label
    Reference.assert_close(r.std_err, c["std_err"], @tol)
    assert_z(r.z_stat, c["z_stat"], label)
    Reference.assert_close(List.flatten(r.tsls.varb), c["varb"], @tol)
    assert_in_delta r.tsls.sig2, c["sig2"], @tol
    assert_in_delta r.tsls.pr2, c["pr2"], 1.0e-10
    assert_in_delta r.rho, c["rho"], @tol
    assert_pr2_e(r.pr2_e, c["pr2_e"])
    check_diagnostics(r, c, label)
  end

  defp check_diagnostics(r, c, label) do
    case c["ak_test"] do
      nil ->
        assert r.ak_test == nil, label

      [ak, p] ->
        {rak, rp} = r.ak_test
        Reference.assert_close(rak, ak, 1.0e-8)
        assert_in_delta rp, p, 1.0e-8
    end

    case c["sp_multipliers"] do
      nil ->
        assert r.sp_multipliers == nil, label

      mult ->
        assert Enum.sort(Map.keys(mult)) ==
                 Enum.sort(Enum.map(Map.keys(r.sp_multipliers), &Atom.to_string/1)),
               label

        for {method, values} <- mult do
          {adi, aii, ati} = r.sp_multipliers[String.to_existing_atom(method)]
          Reference.assert_close([adi, aii, ati], values, @tol)
        end
    end

    for {method, effects} <- c["impacts"] || %{} do
      got = r.impacts[String.to_existing_atom(method)]
      assert Enum.map(got, & &1.name) == Enum.map(effects, & &1["name"]), label

      for {g, e} <- Enum.zip(got, effects) do
        Reference.assert_close(
          [g.direct, g.indirect, g.total],
          [e["direct"], e["indirect"], e["total"]],
          @tol
        )
      end
    end
  end

  defp check_multi(r, c, label) do
    assert r.tsls == nil

    assert Enum.map(r.multi, &to_string(elem(&1, 0))) |> Enum.sort() ==
             Enum.sort(Map.keys(c["multi"]))

    for {key, m} <- c["multi"] do
      fit = Enum.find_value(r.multi, fn {k, v} -> to_string(k) == key && v end)
      assert fit.name_x == m["name_x"] ++ m["name_yend"], label
      assert fit.name_yend == m["name_yend"] -- [key <> "_W_CRIME"], label
      Reference.assert_close(fit.betas, m["betas"], @tol)
      Reference.assert_close(List.flatten(fit.vm), m["vm"], @tol)
      Reference.assert_close(fit.std_err, m["std_err"], @tol)
      assert_z(fit.z_stat, m["z_stat"], label)
      assert_in_delta fit.sig2, m["sig2"], @tol
      assert_in_delta fit.pr2, m["pr2"], 1.0e-10
      assert_in_delta fit.rho, m["rho"], @tol
      assert_pr2_e(fit.pr2_e, m["pr2_e"])
      Reference.assert_close(fit.predy_e, m["predy_e"], @tol)
      check_diagnostics(fit, m, label)
    end

    keys = Enum.sort(Map.keys(c["multi"]))
    pieces = fn field -> Enum.flat_map(keys, &c["multi"][&1][field]) end
    Reference.assert_close(r.std_err, pieces.("std_err"), @tol)
    assert_z(r.z_stat, Enum.flat_map(keys, &c["multi"][&1]["z_stat"]), label)
    last = c["multi"][List.last(keys)]
    assert_in_delta r.rho, last["rho"], @tol
  end
end
