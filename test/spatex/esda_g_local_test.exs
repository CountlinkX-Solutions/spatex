defmodule Spatex.ESDAGLocalTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.ESDA
  alias Spatex.ESDA.GLocal
  alias Spatex.Graph

  doctest Spatex.ESDA.GLocal

  defp path4, do: Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])

  # Path 0-1-2-3-4 plus island 5.
  defp island_graph do
    Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], nil,
      ids: Enum.to_list(0..5)
    )
  end

  @y4 [1.0, 2.0, 3.0, 4.0]
  @y6 [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]

  describe "g_local/3 analytic part" do
    test "matches the closed form of the path example (row-standardized, no star)" do
      m = ESDA.g_local(@y4, path4(), permutations: 0)
      assert %GLocal{n: 4, star: false, w_transform: :r, permutations: 0, y_sum: 10.0} = m
      assert_in_delta Enum.at(m.gs, 0), 2 / 9, 1.0e-12
      assert_in_delta Enum.at(m.gs, 1), 0.25, 1.0e-12
      assert_in_delta hd(m.egs), 1 / 3, 1.0e-12
      assert_in_delta hd(m.vgs), 2 / 243, 1.0e-12
      assert_in_delta hd(m.zs), (2 / 9 - 1 / 3) / :math.sqrt(2 / 243), 1.0e-12
      assert length(m.p_norm) == 4 and Enum.all?(m.p_norm, &(&1 > 0 and &1 <= 1))
    end

    test "permutations: 0 leaves every simulation field nil" do
      m = ESDA.g_local(@y4, path4(), permutations: 0)
      assert %GLocal{sim: nil, p_sim: nil, eg_sim: nil, se_g_sim: nil, seed: nil} = m
      assert %GLocal{vg_sim: nil, z_sim: nil, p_z_sim: nil} = m
    end

    test "star: true assigns the row maximum as self weight" do
      m = ESDA.g_local(@y4, path4(), permutations: 0, star: true)
      assert m.star
      assert_in_delta hd(m.gs), 1.5 / 10, 1.0e-12
      assert_in_delta hd(m.egs), 0.25, 1.0e-12
    end

    test "a numeric star is the self weight, and a number is not the same as true" do
      m = ESDA.g_local(@y4, path4(), permutations: 0, star: 0.5)
      assert_in_delta hd(m.gs), (1 / 3 + 2 * 2 / 3) / 10, 1.0e-12
      assert m.star_arg == 0.5

      # `star: 1` assigns self weight 1; `star: true` uses the row maximum of the input weights
      # (here 1/2 and 1 for the row-standardized graph), so the two differ.
      r = Graph.transform(path4(), :r)
      y = [1.0, 2.0, 5.0, 4.0]
      yes = ESDA.g_local(y, r, permutations: 0, star: true)
      one = ESDA.g_local(y, r, permutations: 0, star: 1)
      assert_in_delta Enum.at(yes.gs, 1), (1 + 2 + 5) / 3 / 12, 1.0e-12
      assert_in_delta Enum.at(one.gs, 1), 2.5 / 12, 1.0e-12
    end

    test "star: nil infers G* from a nonzero diagonal" do
      g = Graph.from_arrays([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
      assert ESDA.g_local(@y4, g, permutations: 0, star: nil).star
      refute ESDA.g_local(@y4, g, permutations: 0, star: false).star
      refute ESDA.g_local(@y4, path4(), permutations: 0, star: nil).star
    end

    test "transform: :b and a binary-labelled graph" do
      b = ESDA.g_local(@y4, path4(), permutations: 0, transform: :b, star: true)
      assert_in_delta Enum.at(b.gs, 1), 6 / 10, 1.0e-12
      assert b.w_transform == :b
      assert ESDA.g_local(@y4, path4(), permutations: 0, transform: "B", star: true).gs == b.gs

      # A graph labelled :b gets self weight 1 even when the requested transform is :r.
      bin = Graph.transform(path4(), :b)
      m = ESDA.g_local(@y4, bin, permutations: 0, star: true)
      assert_in_delta hd(m.gs), 1.5 / 10, 1.0e-12
    end

    test "islands: zero statistic, NaN z" do
      m = ESDA.g_local(@y6, island_graph(), permutations: 0)
      assert Enum.at(m.gs, 5) == 0.0
      assert Enum.at(m.zs, 5) == :nan
      assert Enum.at(m.p_norm, 5) == :nan
    end
  end

  describe "g_local/3 permutations" do
    test "island gets the smallest p-value and the shapes are right" do
      m = ESDA.g_local(@y6, island_graph(), permutations: 99, seed: 7, keep_simulations: true)
      assert Enum.at(m.p_sim, 5) == 0.01
      assert length(m.sim) == 6 and Enum.all?(m.sim, &(length(&1) == 99))
      assert Enum.at(m.z_sim, 5) == :nan
      assert m.seed == 7 and m.alternative == :directed
      assert m.ids == Enum.to_list(0..5)
    end

    test "keep_simulations: false keeps p_sim only" do
      kept = ESDA.g_local(@y6, island_graph(), permutations: 99, seed: 7, keep_simulations: true)

      dropped =
        ESDA.g_local(@y6, island_graph(), permutations: 99, seed: 7, keep_simulations: false)

      assert dropped.sim == nil and dropped.eg_sim == nil and dropped.z_sim == nil
      assert dropped.p_sim == kept.p_sim
    end

    test "same seed identical, different seeds differ, seed: nil is recorded" do
      a = ESDA.g_local(@y6, island_graph(), permutations: 199, seed: 3, keep_simulations: true)

      assert a ==
               ESDA.g_local(@y6, island_graph(),
                 permutations: 199,
                 seed: 3,
                 keep_simulations: true
               )

      refute a.sim ==
               ESDA.g_local(@y6, island_graph(),
                 permutations: 199,
                 seed: 4,
                 keep_simulations: true
               ).sim

      b = ESDA.g_local(@y6, island_graph(), permutations: 99, keep_simulations: true)
      assert is_integer(b.seed)

      assert b ==
               ESDA.g_local(@y6, island_graph(),
                 permutations: 99,
                 seed: b.seed,
                 keep_simulations: true
               )
    end

    test "all alternatives run and directed never exceeds the one-sided p-values" do
      opts = [permutations: 199, seed: 5, star: true]
      directed = ESDA.g_local(@y6, island_graph(), [keep_simulations: true] ++ opts)

      for alt <- ~w(greater lesser two_sided folded)a do
        m = ESDA.g_local(@y6, island_graph(), [alternative: alt, keep_simulations: true] ++ opts)
        assert length(m.p_sim) == 6 and m.alternative == alt
      end

      greater =
        ESDA.g_local(@y6, island_graph(), [alternative: :greater, keep_simulations: true] ++ opts)

      lesser =
        ESDA.g_local(@y6, island_graph(), [alternative: :lesser, keep_simulations: true] ++ opts)

      for i <- 0..4 do
        d = Enum.at(directed.p_sim, i)
        assert d <= min(Enum.at(greater.p_sim, i), Enum.at(lesser.p_sim, i)) + 1.0e-12
      end
    end

    test "island_weight only changes the island's simulations" do
      a = ESDA.g_local(@y6, island_graph(), permutations: 49, seed: 2, keep_simulations: true)

      b =
        ESDA.g_local(@y6, island_graph(),
          permutations: 49,
          seed: 2,
          island_weight: 1.0,
          keep_simulations: true
        )

      assert Enum.take(a.sim, 5) == Enum.take(b.sim, 5)
      refute Enum.at(a.sim, 5) == Enum.at(b.sim, 5)
    end
  end

  describe "g_local/3 validation" do
    test "raises ArgumentError on invalid input" do
      g = path4()

      assert_raise ArgumentError, ~r/non-negative/, fn ->
        ESDA.g_local([1, 2, -3, 4], g)
      end

      assert_raise ArgumentError, ~r/sum/, fn ->
        ESDA.g_local([0, 0, 0, 0], g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.g_local([1, 2, 3], g)
      end

      assert_raise ArgumentError, fn ->
        ESDA.g_local([1, 2, 3, :nan], g)
      end

      assert_raise ArgumentError, ~r/transform/, fn ->
        ESDA.g_local(@y4, g, transform: :d)
      end

      assert_raise ArgumentError, ~r/star/, fn ->
        ESDA.g_local(@y4, g, star: "yes")
      end

      assert_raise ArgumentError, ~r/alternative/, fn ->
        ESDA.g_local(@y4, g, alternative: :up)
      end

      assert_raise ArgumentError, ~r/island_weight/, fn ->
        ESDA.g_local(@y4, g, island_weight: :x)
      end

      assert_raise ArgumentError, ~r/permutations/, fn ->
        ESDA.g_local(@y4, g, permutations: -1, keep_simulations: true)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
        ESDA.g_local(@y4, :nope)
      end

      assert_raise ArgumentError, fn -> ESDA.g_local(@y4, g, bogus: 1) end
    end
  end

  describe "fdr/2" do
    test "matches the esda examples" do
      assert ESDA.fdr([0.001, 0.01, 0.02, 0.04, 0.3, 0.5]) == 0.025000000000000005
      assert ESDA.fdr([0.5, 0.6]) == 0.025
      assert ESDA.fdr([0.5, 0.7, 0.9, 0.99], 0.1) == 0.1 / 4
    end

    test "raises ArgumentError on invalid input" do
      assert_raise ArgumentError, fn -> ESDA.fdr([]) end
      assert_raise ArgumentError, fn -> ESDA.fdr([0.1, :nan]) end
      assert_raise ArgumentError, ~r/alpha/, fn -> ESDA.fdr([0.1], 0.0) end
      assert_raise ArgumentError, ~r/alpha/, fn -> ESDA.fdr([0.1], "x") end
    end

    property "the cut-off lies between Bonferroni and alpha and ignores order" do
      check all(
              p <- list_of(float(min: 0.0, max: 1.0), min_length: 1, max_length: 40),
              alpha <- float(min: 0.001, max: 0.5)
            ) do
        cut = ESDA.fdr(p, alpha)
        assert cut >= alpha / length(p) - 1.0e-15 and cut <= alpha + 1.0e-15
        assert cut == ESDA.fdr(Enum.reverse(p), alpha)
        assert cut == ESDA.fdr(Enum.sort(p), alpha)
      end
    end
  end

  property "p_sim lies in [1/(P+1), 1] and the statistic is finite for positive y" do
    check all(
            y <- list_of(float(min: 0.1, max: 50.0), length: 6),
            star <- member_of([false, true, 0.5]),
            transform <- member_of([:r, :b]),
            seed <- integer(0..1000),
            max_runs: 25
          ) do
      m =
        ESDA.g_local(y, island_graph(),
          permutations: 19,
          seed: seed,
          star: star,
          transform: transform,
          keep_simulations: true
        )

      assert Enum.all?(m.p_sim, &(&1 >= 0.05 and &1 <= 1.0))
      assert Enum.all?(m.gs, &is_float/1)
    end
  end
end
