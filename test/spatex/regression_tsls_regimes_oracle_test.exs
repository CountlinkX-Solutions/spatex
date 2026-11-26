defmodule Spatex.RegressionTSLSRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  @tol 1.0e-9
  @chow_tol 1.0e-8

  defp graph(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp options(c, gwk, compat) do
    base = if compat, do: [spreg_compatible: true], else: []

    base =
      base ++ [name_x: c["x"], name_y: "CRIME", name_yend: c["yend"], name_q: c["q"]]

    Enum.reduce(c["opts"], base, fn
      {"regime_err_sep", v}, acc -> [{:regime_err_sep, v} | acc]
      {"sig2n_k", v}, acc -> [{:sig2n_k, v} | acc]
      {"constant_regi", v}, acc -> [{:constant_regi, String.to_existing_atom(v)} | acc]
      {"cols2regi", v}, acc -> [{:cols2regi, v} | acc]
      {"robust", v}, acc -> [{:robust, String.to_existing_atom(v)} | acc]
    end)
    |> then(&if c["gwk"], do: [{:gwk, gwk} | &1], else: &1)
  end

  defp fit(fx, c, compat) do
    cols = fn names -> Enum.map(names, &fx["columns"][&1]) end

    Regression.tsls_regimes(
      fx["y"],
      cols.(c["x"]),
      cols.(c["yend"]),
      cols.(c["q"]),
      fx["regimes"][c["regimes"]],
      options(c, graph(fx["gwk"]), compat)
    )
  end

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

  test "the corrected and spreg Chow tests differ only when exogenous columns are global and endogenous ones vary" do
    fx = Reference.load!("regression/tsls_regimes.json")
    flagged = for c <- fx["cases"], c["chow_layout_bug"], do: c["label"]

    assert Enum.sort(flagged) ==
             ~w(nsa_mixed_err_sep nsa_mixed_global_open nsa_one_constant nsa_one_constant_err_sep three_one_constant)

    for c <- fx["cases"], not c["chow_layout_bug"] do
      assert c["chow_joint"] == c["spreg_chow_joint"] or
               Enum.zip(c["chow_joint"], c["spreg_chow_joint"])
               |> Enum.all?(fn {a, b} -> abs(a - b) <= 1.0e-8 * max(1.0, abs(b)) end),
             c["label"]
    end
  end

  defp check_cases(joint_key, regi_key, compat) do
    fx = Reference.load!("regression/tsls_regimes.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = fit(fx, c, compat)

      assert r.name_x == c["name_x"], label
      assert r.name_yend == c["name_yend"], label
      assert r.name_q == c["name_q"], label
      assert {r.kr, r.kf, r.kryd, r.nr} == {c["kr"], c["kf"], c["kryd"], c["nr"]}, label
      assert c["multi"] != nil == (r.mode == :separate), label

      Reference.assert_close(r.betas, c["betas"], @tol)
      Reference.assert_close(List.flatten(r.vm), c["vm"], @tol)
      Reference.assert_close(r.u, c["u"], @tol)
      Reference.assert_close(r.predy, c["predy"], @tol)
      Reference.assert_close(Tuple.to_list(r.chow.joint), c[joint_key], @chow_tol)

      Reference.assert_close(
        Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
        List.flatten(c[regi_key]),
        @chow_tol
      )

      if c["warning"] != "" do
        [phrase] =
          Regex.run(~r/(Error by regimes.*|removed for being constant.*)$/, c["warning"],
            capture: :first
          )

        assert Enum.any?(r.warnings, &String.contains?(&1, phrase)), label
      end

      if c["multi"] do
        check_multi(r, c, label)
      else
        Reference.assert_close(r.std_err, c["std_err"], @tol)
        assert_z(r.z_stat, c["z_stat"], label)
        Reference.assert_close(List.flatten(r.tsls.varb), c["varb"], @tol)
        assert_in_delta r.tsls.sig2, c["sig2"], @tol
        assert_in_delta r.tsls.utu, c["utu"], 1.0e-8
        assert_in_delta r.tsls.pr2, c["pr2"], 1.0e-10
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
      assert fit.name_yend == m["name_yend"], label
      assert fit.name_q == m["name_q"], label
      Reference.assert_close(fit.betas, m["betas"], @tol)
      Reference.assert_close(List.flatten(fit.vm), m["vm"], @tol)
      Reference.assert_close(fit.std_err, m["std_err"], @tol)
      assert_z(fit.z_stat, m["z_stat"], label)
      assert_in_delta fit.sig2, m["sig2"], @tol
      assert_in_delta fit.pr2, m["pr2"], 1.0e-10
    end

    keys = Enum.sort(Map.keys(c["multi"]))
    pieces = fn field -> Enum.flat_map(keys, &c["multi"][&1][field]) end
    Reference.assert_close(r.std_err, pieces.("std_err"), @tol)
    assert_z(r.z_stat, Enum.flat_map(keys, &c["multi"][&1]["z_stat"]), label)
  end
end
