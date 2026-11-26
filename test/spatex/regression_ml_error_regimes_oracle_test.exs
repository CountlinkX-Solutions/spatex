defmodule Spatex.RegressionMLErrorRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  # As the base ML tests: lam comes from a bounded Brent search (xatol 1e-7) on a flat objective,
  # so a fit may end within ~1e-7 of spreg's lam. Every formula is therefore verified at spreg's
  # own lam (`ml_error_regimes_at/6`, 1e-9) and the optimizer-driven fit is compared at a tolerance
  # that follows the lam gap.
  # two independent Brent runs (spreg's, ours) each stop within xatol 1e-7 of the optimum of a flat
  # objective: the smaller regimes of the separate fits can end ~1.5e-7 apart
  @lam_tol 3.0e-7
  @tol 1.0e-9
  @chow_tol 1.0e-8
  @fit_tol 2.0e-6
  # spreg's `ord` variance uses the symmetrized matrix; Spatex's ord uses the real W, so those
  # quantities are compared with spreg's `full` run (own lam, same optimum within xatol).
  @ord_var_tol 1.0e-5

  defp sparse(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp graph(fx, "queen"), do: fx["graphs"]["queen"] |> sparse() |> Graph.transform(:r)
  defp graph(fx, "rook_binary"), do: sparse(fx["graphs"]["rook_binary"])

  defp options(c) do
    base = [name_x: c["x"], name_y: "CRIME"]

    Enum.reduce(c["opts"], base, fn
      {"constant_regi", v}, acc -> [{:constant_regi, String.to_existing_atom(v)} | acc]
      {"method", v}, acc -> [{:method, String.to_existing_atom(String.downcase(v))} | acc]
      {key, v}, acc -> [{String.to_existing_atom(key), v} | acc]
    end)
  end

  defp args(fx, c) do
    cols = Enum.map(c["x"], &fx["columns"][&1])
    {fx["y"], cols, fx["regimes"][c["regimes"]], graph(fx, c["w"]), options(c)}
  end

  defp fit(fx, c) do
    {y, x, regimes, g, opts} = args(fx, c)
    Regression.ml_error_regimes(y, x, regimes, g, opts)
  end

  # Fits at spreg's lam (pooled) or lams (one per regime, in label order).
  defp fit_at(fx, c) do
    {y, x, regimes, g, opts} = args(fx, c)

    at =
      if c["multi"],
        do: c["multi"] |> Enum.sort() |> Enum.map(fn {_, m} -> m["lam"] end),
        else: c["lam"]

    Regression.ml_error_regimes_at(y, x, regimes, g, at, opts)
  end

  defp assert_z(z_stat, expected, tol, label) do
    assert length(z_stat) == length(expected), "#{label}: z_stat length"

    for {{z, p}, [ez, ep]} <- Enum.zip(z_stat, expected) do
      Reference.assert_close(z, ez, tol)
      assert abs(p - ep) <= max(tol * ep, 1.0e-300) + 1.0e-7 * ep + 1.0e-12, "#{label}: p"
    end
  end

  defp assert_chow(r, _c, chow_c, tol) do
    Reference.assert_close(Tuple.to_list(r.chow.joint), chow_c["chow_joint"], tol)

    Reference.assert_close(
      Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
      List.flatten(chow_c["chow_regi"]),
      tol
    )

    assert length(r.chow.regi) == length(chow_c["chow_regi"])
  end

  # -- at spreg's lam: everything at 1e-9 ---------------------------------------------------

  test "every case matches spreg at spreg's own lam" do
    fx = Reference.load!("regression/ml_error_regimes.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = fit_at(fx, c)
      ord? = c["opts"]["method"] == "ord"
      var = if ord?, do: c["full_ref"], else: c
      var_tol = if ord?, do: @ord_var_tol, else: @tol

      assert r.name_x == c["name_x"], label
      assert {r.kr, r.kf, r.nr} == {c["kr"], c["kf"], c["nr"]}, label
      assert r.regime_err_sep == c["regime_err_sep"], label
      assert c["multi"] != nil == (r.mode == :separate), label

      Reference.assert_close(r.betas, c["betas"], @tol)
      Reference.assert_close(List.flatten(r.vm), var["vm"], var_tol)
      Reference.assert_close(r.u, c["u"], @tol)
      Reference.assert_close(r.predy, c["predy"], @tol)
      Reference.assert_close(r.e_filtered, c["e_filtered"], @tol)
      assert_chow(r, c, var, if(ord?, do: @ord_var_tol, else: @chow_tol))

      for phrase <- warning_phrases(c["warning"]) do
        assert Enum.any?(r.warnings, &String.contains?(&1, phrase)), "#{label}: #{phrase}"
      end

      if c["multi"],
        do: check_multi(r, c, var, var_tol, label),
        else: check_pooled(r, c, var, var_tol, label)
    end
  end

  defp warning_phrases(""), do: []

  defp warning_phrases(text),
    do:
      text
      |> String.split("\n", trim: true)
      |> Enum.map(&String.replace_prefix(&1, "Warning: ", ""))
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

  defp check_pooled(r, c, var, var_tol, label) do
    assert r.ml != nil and r.multi == nil, label
    assert_in_delta r.lam, c["lam"], 1.0e-12
    assert r.lams == nil
    Reference.assert_close(List.flatten(r.vm1), var["vm1"], var_tol)
    Reference.assert_close(r.std_err, var["std_err"], var_tol)
    assert_z(r.z_stat, var["z_stat"], var_tol, label)

    for key <- ~w(logll aic schwarz sig2 utu pr2 mean_y std_y)a do
      Reference.assert_close(Map.fetch!(r, key), c[Atom.to_string(key)], @tol)
    end

    assert r.k == c["k"] and r.n == c["n"]
  end

  defp check_multi(r, c, var, var_tol, label) do
    assert r.ml == nil and r.lam == nil
    keys = Enum.sort(Map.keys(c["multi"]))
    assert Enum.map(r.multi, &to_string(elem(&1, 0))) |> Enum.sort() == keys
    Reference.assert_close(r.utu, c["utu"], @tol)
    Reference.assert_close([r.mean_y, r.std_y], [c["mean_y"], c["std_y"]], @tol)

    for key <- keys do
      m = c["multi"][key]
      v = var["multi"][key]
      fit = Enum.find_value(r.multi, fn {k, f} -> to_string(k) == key && f end)
      assert fit.name_x == m["name_x"], label
      assert_in_delta fit.lam, m["lam"], 1.0e-12
      assert Enum.find_value(r.lams, fn {k, f} -> to_string(k) == key && f end) == fit.lam
      Reference.assert_close(fit.betas, m["betas"], @tol)
      Reference.assert_close(List.flatten(fit.vm), v["vm"], var_tol)
      Reference.assert_close(List.flatten(fit.vm1), v["vm1"], var_tol)
      Reference.assert_close(fit.std_err, v["std_err"], var_tol)
      assert_z(fit.z_stat, v["z_stat"], var_tol, label)

      for k <- ~w(logll aic schwarz sig2 pr2 mean_y std_y)a do
        Reference.assert_close(Map.fetch!(fit, k), m[Atom.to_string(k)], @tol)
      end

      assert fit.k == m["k"] and fit.n == m["n"]
      Reference.assert_close(fit.e_filtered, m["e_filtered"], @tol)
    end

    pieces = fn field -> Enum.flat_map(keys, &var["multi"][&1][field]) end
    Reference.assert_close(r.std_err, pieces.("std_err"), var_tol)
    assert_z(r.z_stat, Enum.flat_map(keys, &var["multi"][&1]["z_stat"]), var_tol, label)
  end

  # -- the optimizer-driven fit --------------------------------------------------------------

  test "the optimizer reproduces spreg's lam to xatol and the derived quantities to the lam gap" do
    fx = Reference.load!("regression/ml_error_regimes.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = fit(fx, c)

      lams =
        if c["multi"],
          do:
            for(
              {key, m} <- Enum.sort(c["multi"]),
              do: {Map.fetch!(r.lams, key_of(r, key)), m["lam"]}
            ),
          else: [{r.lam, c["lam"]}]

      for {got, want} <- lams, do: assert_in_delta(got, want, @lam_tol, label)
      gap = lams |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
      tol = min(@fit_tol, max(1.0e-8, 1.0e3 * gap))

      Reference.assert_close(r.betas, c["betas"], tol)
      assert r.name_x == c["name_x"], label

      if c["multi"] do
        Enum.each(c["multi"], fn {key, m} ->
          fit = r.multi[key_of(r, key)]
          Reference.assert_close(fit.logll, m["logll"], 1.0e-9)
        end)
      else
        Reference.assert_close(r.logll, c["logll"], 1.0e-9)
        assert r.converged
      end
    end
  end

  defp key_of(r, key), do: Enum.find(r.regimes, &(to_string(&1) == key))

  # -- structure -------------------------------------------------------------------------------

  test "the fixture's Chow tests equal the ones on the real coefficient layout (no spreg defect)" do
    fx = Reference.load!("regression/ml_error_regimes.json")
    assert length(fx["cases"]) == 27
    # the generator asserts spreg's Chow equals the one built on the real layout in every case;
    # Spatex has no spreg_compatible switch here
    assert_raise ArgumentError, ~r/unknown option/, fn ->
      c = hd(fx["cases"])
      {y, x, regimes, g, opts} = args(fx, c)
      Regression.ml_error_regimes(y, x, regimes, g, [spreg_compatible: true] ++ opts)
    end
  end
end
