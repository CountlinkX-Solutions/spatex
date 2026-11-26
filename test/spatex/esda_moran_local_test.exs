defmodule Spatex.ESDAMoranLocalTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.ESDA
  alias Spatex.ESDA.MoranLocal
  alias Spatex.Graph

  doctest Spatex.ESDA.MoranLocal

  # Path 0-1-2-3-4 plus island 5 (an esda probe with an island).
  defp island_graph do
    Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], nil,
      ids: Enum.to_list(0..5)
    )
  end

  @y [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]
  @is [-0.53603604, -0.44144144, -0.79279279, -0.9009009, -0.56306306, 0.0]

  describe "moran_local/3" do
    test "matches the esda island probe" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 0)
      assert %MoranLocal{n: 6, permutations: 0, q: [2, 4, 2, 4, 2, 4]} = m

      for {a, e} <- Enum.zip(m.is, @is), do: assert_in_delta(a, e, 1.0e-8)
      assert Enum.at(m.lag, 5) == 0.0
      assert length(m.z) == 6
      assert Enum.all?(m.ei, &is_float/1)
      assert_in_delta hd(m.ei), -0.2, 1.0e-12
      assert_in_delta hd(m.se_i), :math.sqrt(hd(m.vi)), 1.0e-12
      assert length(m.ei_c) == 6 and length(m.vi_c) == 6
    end

    test "permutations: 0 leaves every simulation field nil" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 0)
      assert %MoranLocal{sim: nil, p_sim: nil, ei_sim: nil, se_i_sim: nil, seed: nil} = m
      assert %MoranLocal{vi_sim: nil, z_sim: nil, p_z_sim: nil} = m
    end

    test "geoda_quads swaps the low-high and low-low codes" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 0, geoda_quads: true)
      assert m.q == [3, 4, 3, 4, 3, 4]
      assert m.geoda_quads
    end

    test "the island has the smallest possible p-value and a NaN z_sim" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 99, seed: 7, keep_simulations: true)
      assert Enum.at(m.p_sim, 5) == 0.01
      assert Enum.at(m.z_sim, 5) == :nan
      assert Enum.at(m.p_z_sim, 5) == :nan
      assert length(m.sim) == 6 and Enum.all?(m.sim, &(length(&1) == 99))
      assert m.seed == 7
    end

    test "keep_simulations: false drops sim and the simulated moments but keeps p_sim" do
      kept =
        ESDA.moran_local(@y, island_graph(), permutations: 99, seed: 7, keep_simulations: true)

      dropped =
        ESDA.moran_local(@y, island_graph(), permutations: 99, seed: 7, keep_simulations: false)

      assert dropped.sim == nil
      assert dropped.ei_sim == nil and dropped.z_sim == nil
      assert dropped.p_sim == kept.p_sim
    end

    test "the same seed gives identical arrays and different seeds differ" do
      a = ESDA.moran_local(@y, island_graph(), permutations: 199, seed: 3, keep_simulations: true)

      assert a ==
               ESDA.moran_local(@y, island_graph(),
                 permutations: 199,
                 seed: 3,
                 keep_simulations: true
               )

      refute a.sim ==
               ESDA.moran_local(@y, island_graph(),
                 permutations: 199,
                 seed: 4,
                 keep_simulations: true
               ).sim
    end

    test "seed: nil draws and records a seed that reproduces the run" do
      a = ESDA.moran_local(@y, island_graph(), permutations: 99, keep_simulations: true)
      assert is_integer(a.seed)

      assert a ==
               ESDA.moran_local(@y, island_graph(),
                 permutations: 99,
                 seed: a.seed,
                 keep_simulations: true
               )
    end

    test "alternatives change p_sim consistently" do
      opts = [permutations: 199, seed: 5]

      greater =
        ESDA.moran_local(
          @y,
          island_graph(),
          [alternative: :greater, keep_simulations: true] ++ opts
        )

      lesser =
        ESDA.moran_local(
          @y,
          island_graph(),
          [alternative: :lesser, keep_simulations: true] ++ opts
        )

      directed = ESDA.moran_local(@y, island_graph(), [keep_simulations: true] ++ opts)

      two_sided =
        ESDA.moran_local(
          @y,
          island_graph(),
          [alternative: :two_sided, keep_simulations: true] ++ opts
        )

      folded =
        ESDA.moran_local(
          @y,
          island_graph(),
          [alternative: :folded, keep_simulations: true] ++ opts
        )

      assert directed.alternative == :directed

      for i <- 0..4 do
        g = Enum.at(greater.p_sim, i)
        l = Enum.at(lesser.p_sim, i)
        d = Enum.at(directed.p_sim, i)
        # directed folds the larger tail onto the smaller one: never above either one-sided
        # p-value (ties can only make those larger) and never above one half.
        assert d <= 0.5 + 1 / 200
        assert d <= min(g, l) + 1.0e-12
      end

      assert length(two_sided.p_sim) == 6 and length(folded.p_sim) == 6
    end

    test "default transform is :r and :b differs" do
      r = ESDA.moran_local(@y, island_graph(), permutations: 0)
      assert r.is == ESDA.moran_local(@y, Graph.transform(island_graph(), :r), permutations: 0).is
      refute r.is == ESDA.moran_local(@y, island_graph(), permutations: 0, transform: :b).is
    end

    test "raises ArgumentError on invalid input" do
      g = island_graph()

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.moran_local(List.duplicate(2, 6), g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.moran_local([1, 2, 3], g)
      end

      assert_raise ArgumentError, fn ->
        ESDA.moran_local([1, 2, 3, 4, 5, :nan], g)
      end

      assert_raise ArgumentError, ~r/alternative/, fn ->
        ESDA.moran_local(@y, g, alternative: :up)
      end

      assert_raise ArgumentError, ~r/geoda_quads/, fn ->
        ESDA.moran_local(@y, g, geoda_quads: 1)
      end

      assert_raise ArgumentError, ~r/permutations/, fn ->
        ESDA.moran_local(@y, g, permutations: -1, keep_simulations: true)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
        ESDA.moran_local(@y, :nope)
      end

      assert_raise ArgumentError, fn ->
        ESDA.moran_local(@y, g, bogus: 1)
      end
    end
  end

  describe "moran_local_bv/4" do
    test "with x == y it equals the univariate statistic" do
      uni = ESDA.moran_local(@y, island_graph(), permutations: 0)
      bv = ESDA.moran_local_bv(@y, @y, island_graph(), permutations: 0)
      assert bv.is == uni.is and bv.q == uni.q
      assert bv.ei == nil and bv.vi == nil
    end

    test "permutations run and the island is minimal" do
      x = [2.0, 7.0, 1.0, 3.0, 9.0, 5.0]

      m =
        ESDA.moran_local_bv(x, @y, island_graph(),
          permutations: 99,
          seed: 1,
          keep_simulations: true
        )

      assert Enum.at(m.p_sim, 5) == 0.01

      assert m ==
               ESDA.moran_local_bv(x, @y, island_graph(),
                 permutations: 99,
                 seed: 1,
                 keep_simulations: true
               )
    end

    test "rejects constant or mismatched inputs" do
      g = island_graph()

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.moran_local_bv(List.duplicate(1, 6), @y, g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.moran_local_bv([1, 2], @y, g)
      end
    end
  end

  describe "cluster_labels/2" do
    test "labels significant sites by quadrant and uses a strict threshold" do
      m = %MoranLocal{
        is: List.duplicate(0.0, 5),
        q: [1, 2, 3, 4, 1],
        p_sim: [0.01, 0.049, 0.05, 0.001, 0.2],
        n: 5,
        permutations: 99
      }

      assert MoranLocal.cluster_labels(m) ==
               ["High-High", "Low-High", "Insignificant", "High-Low", "Insignificant"]

      assert MoranLocal.cluster_labels(m, 0.1) ==
               ["High-High", "Low-High", "Low-Low", "High-Low", "Insignificant"]
    end

    test "requires permutation results" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 0)
      assert_raise ArgumentError, ~r/permutations/, fn -> MoranLocal.cluster_labels(m) end
    end

    test "works end to end" do
      m = ESDA.moran_local(@y, island_graph(), permutations: 99, seed: 9, keep_simulations: true)
      labels = MoranLocal.cluster_labels(m, 0.05)
      assert length(labels) == 6
      assert Enum.at(labels, 5) == "High-Low"
    end
  end

  property "p_sim stays within [1/(P+1), 1] and quadrants are 1..4" do
    check all(
            y <- list_of(float(min: -50.0, max: 50.0), length: 8),
            Enum.uniq(y) != [hd(y)],
            seed <- integer(0..1_000_000),
            max_runs: 25
          ) do
      next = Enum.map(0..7, &rem(&1 + 1, 8))
      g = Graph.from_arrays(Enum.to_list(0..7) ++ next, next ++ Enum.to_list(0..7), nil)
      m = ESDA.moran_local(y, g, permutations: 49, seed: seed, keep_simulations: true)
      assert Enum.all?(m.p_sim, &(&1 >= 1 / 50 and &1 <= 1.0))
      assert Enum.all?(m.q, &(&1 in 1..4))
    end
  end
end
