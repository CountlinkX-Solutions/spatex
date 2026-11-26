defmodule Spatex.ESDAGearyGJoinsOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.ESDA
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9
  @perm_seed 20_240_601

  defp graph(g) do
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  defp tag(label, fun) do
    fun.()
  rescue
    e in ExUnit.AssertionError ->
      reraise %{e | message: "#{label}: #{e.message}"}, __STACKTRACE__
  end

  # Two independent binomial estimates differ by at most 4 sqrt(2 p (1 - p) / P) (+ 1/P).
  defp assert_p_close(actual, expected, perms, label) do
    band = 4 * :math.sqrt(2 * expected * (1 - expected) / perms) + 1 / perms
    assert abs(actual - expected) <= band, "#{label}: p #{actual} vs PySAL #{expected} (#{band})"
  end

  defp assert_sim_close(m, perm, perms, keys, label) do
    {mean_k, se_k, z_k} = keys
    assert_p_close(m.p_sim, perm["p_sim"], perms, label)
    band = 4 * :math.sqrt(2) * perm[se_k] / :math.sqrt(perms)
    assert abs(Map.fetch!(m, mean_k) - perm[to_string(mean_k)]) <= band, "#{label}: mean"
    assert abs(Map.fetch!(m, se_k_atom(se_k)) - perm[se_k]) <= 0.05 * perm[se_k], "#{label}: se"
    assert abs(Map.fetch!(m, z_k) - perm["z_sim"]) <= 0.06 * abs(perm["z_sim"]) + 0.1
    assert length(m.sim) == perms
  end

  defp se_k_atom(k),
    do:
      k |> String.replace("sec_", "se_c_") |> String.replace("seg_", "se_g_") |> String.to_atom()

  for file <- ~w(geary_columbus_queen geary_lattice geary_islands) do
    describe file do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx["graph"])}
      end

      test "analytic quantities match PySAL to 1e-9", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          m =
            ESDA.geary(c["y"], g,
              permutations: 0,
              transform: Spatex.Reference.fixture_atom(c["transform"])
            )

          actual = %{
            "c" => m.c,
            "ec" => m.ec,
            "vc_norm" => m.vc_norm,
            "sec_norm" => m.se_c_norm,
            "z_norm" => m.z_norm,
            "p_norm" => m.p_norm,
            "vc_rand" => m.vc_rand,
            "sec_rand" => m.se_c_rand,
            "z_rand" => m.z_rand,
            "p_rand" => m.p_rand
          }

          tag(c["label"], fn -> Reference.assert_close(actual, c["analytic"], @tol) end)
        end
      end

      test "permutation inference agrees with PySAL statistically", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          perms = c["permutations"]
          t = Spatex.Reference.fixture_atom(c["transform"])
          m = ESDA.geary(c["y"], g, permutations: perms, seed: @perm_seed, transform: t)

          tag(c["label"], fn ->
            assert_sim_close(m, c["perm"], perms, {:ec_sim, "sec_sim", :z_sim}, c["label"])
          end)
        end
      end
    end
  end

  test "esda crashes on Geary with island graphs; Spatex handles them" do
    fx = Reference.load!("esda/geary_islands.json")
    assert fx["graph_crash"] =~ "broadcast"
    assert %ESDA.Geary{} = ESDA.geary(hd(fx["cases"])["y"], graph(fx["graph"]), permutations: 0)
  end

  for file <- ~w(g_columbus_band g_columbus_sparse_band g_columbus_queen g_lattice g_islands) do
    describe file do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx["graph"])}
      end

      test "analytic quantities match PySAL to 1e-9", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          m = ESDA.getis_ord_g(c["y"], g, permutations: 0)

          actual = %{
            "g" => m.g,
            "eg" => m.eg,
            "vg" => m.vg,
            "z_norm" => m.z_norm,
            "p_norm" => m.p_norm,
            "den_sum" => m.den_sum
          }

          tag(c["label"], fn -> Reference.assert_close(actual, c["analytic"], @tol) end)
        end
      end

      test "permutation inference agrees with PySAL statistically", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          perms = c["permutations"]
          m = ESDA.getis_ord_g(c["y"], g, permutations: perms, seed: @perm_seed)

          tag(c["label"], fn ->
            assert_sim_close(m, c["perm"], perms, {:eg_sim, "seg_sim", :z_sim}, c["label"])
          end)
        end
      end
    end
  end

  test "complete graph: VG is rounding noise, z_norm and p_norm are NaN like PySAL" do
    fx = Reference.load!("esda/g_complete.json")
    assert fx["pysal"]["z_norm"] == :nan and fx["pysal"]["p_norm"] == :nan
    m = ESDA.getis_ord_g(fx["y"], graph(fx["graph"]), permutations: 0)
    assert_in_delta m.g, fx["pysal"]["g"], @tol
    assert m.z_norm == :nan and m.p_norm == :nan
  end

  describe "join counts" do
    @analytic ~w(j bb ww bw chi2 chi2_p chi2_dof autocorr_pos autocorr_neg)

    defp jc_cases(file) do
      fx = Reference.load!("esda/#{file}.json")

      base = for c <- fx["cases"], do: {graph(fx["graph"]), c}

      extra =
        for key <- ["sparse", "path"], Map.has_key?(fx, key), c <- fx[key]["cases"] do
          {graph(fx[key]["graph"]), c}
        end

      base ++ extra
    end

    for file <- ~w(join_counts_columbus join_counts_lattice join_counts_islands) do
      test "#{file}: counts, table and chi-square match PySAL to 1e-9" do
        for {g, c} <- jc_cases(unquote(file)) do
          m = ESDA.join_counts(c["y"], g, permutations: 0, drop_islands: c["drop_islands"])
          actual = Map.new(@analytic, fn k -> {k, Map.fetch!(m, String.to_atom(k))} end)

          tag(c["label"], fn ->
            Reference.assert_close(actual, c["analytic"], @tol)
            Reference.assert_close(List.flatten(m.crosstab), List.flatten(c["crosstab"]), @tol)
            Reference.assert_close(List.flatten(m.expected), List.flatten(c["expected"]), @tol)
          end)
        end
      end

      test "#{file}: permutation inference agrees with PySAL statistically" do
        for {g, c} <- jc_cases(unquote(file)) do
          perms = c["permutations"]
          p = c["perm"]

          m =
            ESDA.join_counts(c["y"], g,
              permutations: perms,
              seed: @perm_seed,
              drop_islands: c["drop_islands"]
            )

          tag(c["label"], fn ->
            for k <- ~w(p_sim_bb p_sim_bw p_sim_chi2 p_sim_autocorr_pos p_sim_autocorr_neg) do
              assert_p_close(Map.fetch!(m, String.to_atom(k)), p[k], perms, k)
            end

            sd_bb = :math.sqrt(max(p["max_bb"] - p["min_bb"], 1.0) * 2)
            assert abs(m.mean_bb - p["mean_bb"]) <= 4 * sd_bb / :math.sqrt(perms) * :math.sqrt(2)
            sd_bw = :math.sqrt(max(p["max_bw"] - p["min_bw"], 1.0) * 2)
            assert abs(m.mean_bw - p["mean_bw"]) <= 4 * sd_bw / :math.sqrt(perms) * :math.sqrt(2)
            assert length(m.sim_bb) == perms
          end)
        end
      end
    end

    test "known deviation: PySAL's sim_autocorr_chi2 divides both terms by mean_pos squared" do
      for {g, c} <- jc_cases("join_counts_columbus") do
        p = c["perm"]
        pos = c["analytic"]["autocorr_pos"]
        neg = c["analytic"]["autocorr_neg"]
        {mp, mn} = {p["mean_autocorr_pos"], p["mean_autocorr_neg"]}

        # What esda computes: both squared deviations over mean_pos^2, then 1 - chi2.cdf(., 1).
        stat = (pos - mp) ** 2 / mp ** 2 + (neg - mn) ** 2 / mp ** 2
        pysal = :math.erfc(:math.sqrt(stat / 2))
        assert_in_delta pysal, p["sim_autocorr_chi2"], 1.0e-9

        # Spatex reports the Pearson statistic over each cell's own mean instead.
        m = ESDA.join_counts(c["y"], g, permutations: c["permutations"], seed: @perm_seed)
        pearson = (pos - mp) ** 2 / mp + (neg - mn) ** 2 / mn
        expected = :math.erfc(:math.sqrt(pearson / 2))
        assert abs(m.sim_autocorr_chi2 - expected) <= 0.05 * expected + 1.0e-3
        refute abs(expected - pysal) < 1.0e-6
      end
    end
  end
end
