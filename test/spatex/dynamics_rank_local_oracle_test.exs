defmodule Spatex.DynamicsRankLocalOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Dynamics
  alias Spatex.Reference

  defp assert_chain(m, expected, label) do
    assert m.classes == expected["classes"], label
    assert m.transitions == expected["transitions"], label
    Reference.assert_close(List.flatten(m.p), List.flatten(expected["p"]), 1.0e-12)
  end

  describe "full_rank_markov/2 and geo_rank_markov/2" do
    setup do
      {:ok, fx: Reference.load!("dynamics/rank_markov.json")}
    end

    test "mexico ranks, transitions and probabilities match giddy", %{fx: fx} do
      fr = Dynamics.full_rank_markov(fx["y"])
      assert fr.ranks == fx["full_rank"]["ranks"]
      assert_chain(fr, fx["full_rank"], "full rank")

      gr = Dynamics.geo_rank_markov(fx["y"])
      assert gr.ranks == fx["geo_rank"]["ranks"]
      assert_chain(gr, fx["geo_rank"], "geo rank")
    end

    test "ties are ranked by position as in rankdata(method: ordinal)", %{fx: fx} do
      fr = Dynamics.full_rank_markov(fx["tied"])
      assert fr.ranks == fx["full_rank_tied"]["ranks"]
      assert_chain(fr, fx["full_rank_tied"], "full rank tied")
      assert_chain(Dynamics.geo_rank_markov(fx["tied"]), fx["geo_rank_tied"], "geo rank tied")
    end
  end

  defp graph(fx) do
    g = fx["graph"]
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Spatex.Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  # Two independent P-permutation estimates of p differ with sd sqrt(2 p (1 - p) / P); allow a
  # handful of sites beyond 4 sd and none beyond 6 sd (as the ESDA oracle tests do).
  defp assert_p_close(actual, expected, perms, label) do
    pairs = Enum.reject(Enum.zip(actual, expected), fn {a, e} -> a == :nan or e == nil end)

    d =
      for {a, e} <- pairs,
          do: abs(a - e) / (:math.sqrt(2 * max(e * (1 - e), 1.0e-4) / perms) + 1 / perms)

    assert Enum.count(d, &(&1 > 4)) <= 1, "#{label}: #{inspect(d)}"
    assert Enum.all?(d, &(&1 <= 6)), "#{label}: #{inspect(d)}"
  end

  describe "local tau family" do
    setup do
      {:ok, fx: Reference.load!("dynamics/tau_local.json")}
    end

    test "tau_local matches giddy exactly", %{fx: fx} do
      t = Dynamics.tau_local(fx["x"], fx["y"])
      assert_in_delta t.tau, fx["tau"], 1.0e-12
      Reference.assert_close(t.tau_local, fx["tau_local"], 1.0e-12)
    end

    test "tau_local_neighbor statistics exact, p-values statistically", %{fx: fx} do
      c = fx["neighbor"]
      perms = fx["permutations"]
      t = Dynamics.tau_local_neighbor(fx["x"], fx["y"], graph(fx), permutations: perms, seed: 1)
      Reference.assert_close(t.tau_ln, c["tau_ln"], 1.0e-12)
      Reference.assert_close(t.weights, c["weights"], 1.0e-12)
      assert t.sign == c["sign"]
      assert_p_close(t.p_sim, c["p"], perms, "neighbor")
    end

    test "tau_local_neighborhood statistics exact, p-values statistically", %{fx: fx} do
      c = fx["neighborhood"]
      perms = fx["permutations"]

      t =
        Dynamics.tau_local_neighborhood(fx["x"], fx["y"], graph(fx), permutations: perms, seed: 2)

      Reference.assert_close(t.tau_lnhood, c["tau_lnhood"], 1.0e-12)
      assert t.sign == c["sign"]
      assert_p_close(t.p_sim, c["p"], perms, "neighborhood")
    end

    test "tau_regional matrix exact, p-values statistically", %{fx: fx} do
      c = fx["regional"]
      perms = fx["permutations"]
      t = Dynamics.tau_regional(fx["x"], fx["y"], fx["regime"], permutations: perms, seed: 3)
      Reference.assert_close(List.flatten(t.tau_reg), List.flatten(c["tau_reg"]), 1.0e-12)
      assert_p_close(List.flatten(t.p_sim), List.flatten(c["p"]), perms, "regional")
      assert length(t.sim) == perms
    end
  end

  describe "lisa_spillover/3" do
    test "components and spill-overs match giddy on its quadrants and p-values" do
      fx = Reference.load!("dynamics/spillover.json")

      lm = %Spatex.Dynamics.LisaMarkov{
        q: fx["q"],
        classes: [1, 2, 3, 4],
        p_values: fx["p_values"],
        significance_level: fx["significance_level"],
        permutations: 99
      }

      for c <- fx["cases"] do
        r =
          Dynamics.lisa_spillover(lm, graph(fx),
            quadrant: c["quadrant"],
            neighbors_on: c["neighbors_on"]
          )

        label = "quadrant #{c["quadrant"]} neighbors_on #{c["neighbors_on"]}"
        assert r.components == c["components"], label
        assert r.spill_over == c["spill_over"], label
      end
    end
  end
end
