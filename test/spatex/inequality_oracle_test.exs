defmodule Spatex.InequalityOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Inequality
  alias Spatex.Reference

  @tol 1.0e-9

  defp graph(fx, name) do
    g = fx["graphs"][name]

    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
      ids: Enum.to_list(0..(length(g["ids"]) - 1))
    )
  end

  defp assert_p_close(actual, expected, perms, label) do
    sd = :math.sqrt(2 * max(expected * (1 - expected), 1.0e-4) / perms) + 1 / perms

    assert abs(actual - expected) <= 5 * sd,
           "#{label}: #{actual} vs #{expected} (5 sd = #{5 * sd})"
  end

  setup_all do
    {:ok, fx: Reference.load!("dynamics/inequality.json")}
  end

  test "gini matches inequality.Gini", %{fx: fx} do
    for c <- fx["gini"], do: Reference.assert_close(Inequality.gini(c["x"]), c["g"], @tol)
  end

  test "theil matches inequality.Theil", %{fx: fx} do
    for c <- fx["theil"] do
      actual = Inequality.theil(c["y"])
      Reference.assert_close(List.wrap(actual), c["t"], @tol)
    end
  end

  test "theil_d matches inequality.TheilD", %{fx: fx} do
    for c <- fx["theil_d"] do
      d = Inequality.theil_d(c["y"], c["partition"])
      Reference.assert_close(List.wrap(d.t), List.wrap(c["t"]), @tol)
      Reference.assert_close(List.wrap(d.bg), List.wrap(c["bg"]), @tol)
      Reference.assert_close(List.wrap(d.wg), List.wrap(c["wg"]), @tol)
    end
  end

  test "theil_d permutation p-values match inequality.TheilDSim", %{fx: fx} do
    c = fx["theil_d_sim"]
    perms = c["permutations"]
    d = Inequality.theil_d(c["y"], c["partition"], permutations: perms, seed: 8)
    Reference.assert_close(d.bg, c["bg"], @tol)

    for {a, e} <- Enum.zip(d.bg_pvalue, c["bg_pvalue"]),
        do: assert_p_close(a, e, perms, "theil_d_sim")
  end

  test "spatial_gini matches inequality.Gini_Spatial", %{fx: fx} do
    for c <- fx["spatial_gini"] do
      l = c["label"]
      perms = c["permutations"]
      s = Inequality.spatial_gini(c["x"], graph(fx, c["graph"]), permutations: perms, seed: 12)

      Reference.assert_close(
        [s.g, s.wg, s.wcg, s.wcg_share, s.dtotal, s.polarization],
        [c["g"], c["wg"], c["wcg"], c["wcg_share"], c["dtotal"], c["polarization"]],
        @tol
      )

      assert_p_close(s.p_sim, c["p_sim"], perms, l)
      assert_p_close(s.polarization_p_sim, c["polarization_p_sim"], perms, l)
      # moments of the null distribution: Monte Carlo error of the mean is s / sqrt(P)
      se = c["s_wcg"] / :math.sqrt(perms)
      assert abs(s.e_wcg - c["e_wcg"]) <= 6 * se * :math.sqrt(2), l
      assert abs(s.s_wcg - c["s_wcg"]) <= 0.05 * c["s_wcg"], l
      assert abs(s.z_wcg - c["z_wcg"]) <= 0.25, l
    end
  end
end
