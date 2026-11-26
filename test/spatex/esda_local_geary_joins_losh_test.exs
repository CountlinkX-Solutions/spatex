defmodule Spatex.ESDALocalGearyJoinsLoshTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias Spatex.ESDA
  alias Spatex.ESDA.{GearyLocal, JoinCountsLocal, Losh}
  alias Spatex.Graph

  # Path 0-1-2-3-4 plus island 5.
  defp island_graph do
    Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], nil,
      ids: Enum.to_list(0..5)
    )
  end

  @x [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]
  @b [1, 1, 0, 1, 1, 1]

  describe "geary_local/3" do
    test "local_g is aligned with the graph and zero at the island" do
      m = ESDA.geary_local(@x, island_graph(), permutations: 0)
      assert %GearyLocal{n: 6, permutations: 0, p_sim: nil, sim: nil, labs: nil} = m
      assert length(m.local_g) == 6
      assert List.last(m.local_g) == 0.0
      assert Enum.all?(Enum.take(m.local_g, 5), &(&1 > 0.0))
    end

    test "the graph is row-standardized whatever its transform" do
      a = ESDA.geary_local(@x, island_graph(), permutations: 0)
      b = ESDA.geary_local(@x, Graph.transform(island_graph(), :b), permutations: 0)
      assert a.local_g == b.local_g
    end

    test "permutations give p_sim, sim and optional labels; the island is minimal" do
      m =
        ESDA.geary_local(@x, island_graph(),
          permutations: 99,
          seed: 5,
          labels: true,
          keep_simulations: true
        )

      assert length(m.p_sim) == 6 and List.last(m.p_sim) == 0.01
      assert length(m.sim) == 6 and Enum.all?(m.sim, &(length(&1) == 99))
      assert m.seed == 5 and m.sig == 0.05
      assert length(m.labs) == 6
      assert Enum.all?(m.labs, &(&1 in [1, 2, 3, 4, nil]))
      # esda quirk: the island's minimal p_sim is significant, and localG 0 < mean => outlier.
      assert List.last(m.labs) == 1
    end

    test "labels follow sig and keep_simulations: false keeps p_sim" do
      kept =
        ESDA.geary_local(@x, island_graph(), permutations: 99, seed: 5, keep_simulations: true)

      dropped =
        ESDA.geary_local(@x, island_graph(), permutations: 99, seed: 5, keep_simulations: false)

      assert dropped.sim == nil and dropped.p_sim == kept.p_sim and kept.labs == nil

      all =
        ESDA.geary_local(@x, island_graph(),
          permutations: 99,
          seed: 5,
          labels: true,
          sig: 1.0,
          keep_simulations: true
        )

      assert Enum.all?(all.labs, &(&1 in [1, 2, 3, nil]))
    end

    test "is deterministic for a seed and seed: nil records one" do
      a = ESDA.geary_local(@x, island_graph(), permutations: 49, seed: 1, keep_simulations: true)

      assert a ==
               ESDA.geary_local(@x, island_graph(),
                 permutations: 49,
                 seed: 1,
                 keep_simulations: true
               )

      b = ESDA.geary_local(@x, island_graph(), permutations: 49, keep_simulations: true)
      assert is_integer(b.seed)

      assert b ==
               ESDA.geary_local(@x, island_graph(),
                 permutations: 49,
                 seed: b.seed,
                 keep_simulations: true
               )
    end

    test "raises ArgumentError on invalid input" do
      g = island_graph()

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.geary_local(List.duplicate(2, 6), g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.geary_local([1, 2, 3], g)
      end

      assert_raise ArgumentError, fn ->
        ESDA.geary_local([1, 2, 3, 4, 5, :nan], g)
      end

      assert_raise ArgumentError, ~r/labels/, fn ->
        ESDA.geary_local(@x, g, labels: true, permutations: 0)
      end

      assert_raise ArgumentError, ~r/sig/, fn ->
        ESDA.geary_local(@x, g, sig: 0)
      end

      assert_raise ArgumentError, ~r/labels/, fn ->
        ESDA.geary_local(@x, g, labels: 1)
      end

      assert_raise ArgumentError, ~r/alternative/, fn ->
        ESDA.geary_local(@x, g, alternative: :up)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
        ESDA.geary_local(@x, :nope)
      end

      assert_raise ArgumentError, fn ->
        ESDA.geary_local(@x, g, bogus: 1)
      end
    end
  end

  describe "join_counts_local/3" do
    test "counts joined neighbors; islands count 0 and have a NaN p_sim" do
      m =
        ESDA.join_counts_local(@b, island_graph(),
          permutations: 99,
          seed: 3,
          keep_simulations: true
        )

      assert %JoinCountsLocal{n: 6, permutations: 99} = m
      assert m.ljc == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
      assert Enum.map(m.p_sim, &(&1 == :nan)) == [false, false, true, false, false, true]
      assert length(m.sim) == 6
    end

    test "permutations: 0 and keep_simulations: false" do
      m = ESDA.join_counts_local(@b, island_graph(), permutations: 0)
      assert m.p_sim == nil and m.sim == nil

      kept =
        ESDA.join_counts_local(@b, island_graph(),
          permutations: 49,
          seed: 3,
          keep_simulations: true
        )

      dropped =
        ESDA.join_counts_local(@b, island_graph(),
          permutations: 49,
          seed: 3,
          keep_simulations: false
        )

      assert dropped.sim == nil and dropped.p_sim == kept.p_sim
    end

    test "raises ArgumentError for non-binary or mismatched input" do
      g = island_graph()

      assert_raise ArgumentError, ~r/binary/, fn ->
        ESDA.join_counts_local([0, 1, 2, 0, 1, 1], g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.join_counts_local([0, 1], g)
      end

      assert_raise ArgumentError, fn ->
        ESDA.join_counts_local(@b, g, labels: true)
      end
    end
  end

  describe "losh/3" do
    test "island sites are NaN and the others equal the island-free graph" do
      m = ESDA.losh(@x, island_graph())
      sub = Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], nil)
      s = ESDA.losh(Enum.take(@x, 5), sub)
      assert %Losh{n: 6, a: 2.0, inference: nil, pval: nil} = m
      assert Enum.take(m.hi, 5) == s.hi
      assert Enum.take(m.var_hi, 5) == s.var_hi
      assert List.last(m.hi) == :nan and List.last(m.ylag) == :nan
    end

    test "chi-square inference adds p-values (both spellings)" do
      m = ESDA.losh(@x, island_graph(), inference: :chi2)
      assert m.inference == :chi2
      assert length(m.pval) == 6 and List.last(m.pval) == :nan
      assert Enum.all?(Enum.take(m.pval, 5), &(&1 >= 0.0 and &1 <= 1.0))
      assert ESDA.losh(@x, island_graph(), inference: :chi_square).pval == m.pval
    end

    test "chi-square with a != 2 warns and yields no p-values" do
      log =
        capture_log(fn ->
          m = ESDA.losh(@x, island_graph(), a: 1, inference: :chi2)
          assert m.pval == nil and m.a == 1.0
        end)

      assert log =~ "a = 2"
    end

    test "raises ArgumentError on invalid input" do
      g = island_graph()

      assert_raise ArgumentError, ~r/inference/, fn ->
        ESDA.losh(@x, g, inference: :permutation)
      end

      assert_raise ArgumentError, ~r/inference/, fn ->
        ESDA.losh(@x, g, inference: "chi-square")
      end

      assert_raise ArgumentError, ~r/a must/, fn -> ESDA.losh(@x, g, a: 0) end
      assert_raise ArgumentError, ~r/length/, fn -> ESDA.losh([1, 2], g) end
      assert_raise ArgumentError, fn -> ESDA.losh(@x, g, bogus: 1) end
    end
  end
end
