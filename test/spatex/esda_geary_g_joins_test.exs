defmodule Spatex.ESDAGearyGJoinsTest do
  use ExUnit.Case, async: true

  alias Spatex.ESDA
  alias Spatex.ESDA.{G, Geary, JoinCounts}
  alias Spatex.Graph

  defp path(n, islands \\ 0) do
    focal = Enum.flat_map(0..(n - 2)//1, &[&1, &1 + 1])
    neighbor = Enum.flat_map(0..(n - 2)//1, &[&1 + 1, &1])
    Graph.from_arrays(focal, neighbor, nil, ids: Enum.to_list(0..(n + islands - 1)))
  end

  @y8 [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]

  describe "geary/3" do
    test "matches the esda probe on a path with islands (default transform r)" do
      m = ESDA.geary(@y8, path(4, 4), permutations: 0)
      assert %Geary{ec: 1.0, n: 8, permutations: 0} = m
      assert_in_delta m.c, 0.42470441817050403, 1.0e-12
      assert_in_delta m.vc_norm, 0.4583333333333333, 1.0e-12
      assert_in_delta m.vc_rand, 0.494568343847255, 1.0e-12
      assert_in_delta m.se_c_norm, :math.sqrt(m.vc_norm), 1.0e-12
      assert_in_delta m.p_norm, 0.19772707226622627, 1.0e-12
      assert %Geary{sim: nil, p_sim: nil, ec_sim: nil, se_c_sim: nil, vc_sim: nil} = m
      assert %Geary{z_sim: nil, p_z_sim: nil, seed: nil} = m
    end

    test "transform: :b is honoured" do
      m = ESDA.geary(@y8, path(4, 4), permutations: 0, transform: :b)
      assert_in_delta m.c, 0.4472101223812487, 1.0e-12
    end

    test "non-finite results surface as atoms" do
      m = ESDA.geary([1, 5, 2], path(3), permutations: 0)
      assert m.vc_rand == :nan
      assert m.p_rand == :nan
    end

    test "permutations are seeded, summarised and optionally dropped" do
      a = ESDA.geary(@y8, path(4, 4), permutations: 99, seed: 5)
      b = ESDA.geary(@y8, path(4, 4), permutations: 99, seed: 5)
      assert a.sim == b.sim
      assert length(a.sim) == 99
      assert a.seed == 5
      assert a.p_sim >= 0.01 and a.p_sim <= 1.0
      assert_in_delta a.vc_sim, a.se_c_sim * a.se_c_sim, 1.0e-12

      d = ESDA.geary(@y8, path(4, 4), permutations: 99, seed: 5, keep_simulations: false)
      assert d.sim == nil
      assert d.p_sim == a.p_sim
    end

    test "validates its input" do
      assert_raise ArgumentError, ~r/constant/, fn -> ESDA.geary([2, 2, 2], path(3)) end
      assert_raise ArgumentError, ~r/length/, fn -> ESDA.geary([1, 2], path(3)) end
      assert_raise ArgumentError, fn -> ESDA.geary([1, 2, 3], :nope) end
      assert_raise ArgumentError, fn -> ESDA.geary([1, 2, 3], path(3), bogus: 1) end
      assert_raise ArgumentError, ~r/seed/, fn -> ESDA.geary([1, 2, 4], path(3), seed: -1) end
    end
  end

  describe "getis_ord_g/3" do
    test "matches the esda probe on a path with islands" do
      m = ESDA.getis_ord_g(@y8, path(4, 4), permutations: 0)
      assert %G{n: 8, permutations: 0, sim: nil, p_sim: nil} = m
      assert_in_delta m.g, 0.03178484107579462, 1.0e-12
      assert_in_delta m.eg, 0.10714285714285714, 1.0e-12
      assert_in_delta m.vg, 0.003553183069247571, 1.0e-10
      assert_in_delta m.z_norm, -1.2642142006042831, 1.0e-10
      assert_in_delta m.p_norm, 0.10307657760853384, 1.0e-10
      assert_in_delta m.den_sum, 818.0, 1.0e-9
    end

    test "weights are always binary, whatever the graph carries" do
      g = path(4, 4)
      a = ESDA.getis_ord_g(@y8, g, permutations: 0)
      assert a.g == ESDA.getis_ord_g(@y8, Graph.transform(g, :r), permutations: 0).g
    end

    test "has no transform option" do
      assert_raise ArgumentError, fn -> ESDA.getis_ord_g(@y8, path(8), transform: :r) end
    end

    test "negative or degenerate y is rejected" do
      assert_raise ArgumentError, ~r/non-negative/, fn ->
        ESDA.getis_ord_g([1, -2, 3, 4, 5], path(5))
      end

      assert_raise ArgumentError, fn -> ESDA.getis_ord_g([0, 0, 0, 0, 0], path(5)) end
    end

    test "permutations are seeded" do
      a = ESDA.getis_ord_g(@y8, path(4, 4), permutations: 99, seed: 5)
      assert a.sim == ESDA.getis_ord_g(@y8, path(4, 4), permutations: 99, seed: 5).sim
      assert is_float(a.eg_sim) and is_float(a.se_g_sim) and is_float(a.vg_sim)
      assert a.p_sim >= 0.01 and a.p_sim <= 1.0
    end
  end

  describe "join_counts/3" do
    @lattice_y [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

    defp lattice do
      edges =
        for r <- 0..3, c <- 0..3, {dr, dc} <- [{0, 1}, {1, 0}], r + dr < 4, c + dc < 4 do
          {r * 4 + c, (r + dr) * 4 + c + dc}
        end

      focal = Enum.flat_map(edges, fn {a, b} -> [a, b] end)
      neighbor = Enum.flat_map(edges, fn {a, b} -> [b, a] end)
      Graph.from_arrays(focal, neighbor, nil, ids: Enum.to_list(0..15))
    end

    test "checkerboard matches esda" do
      m = ESDA.join_counts(@lattice_y, lattice(), permutations: 0)
      assert %JoinCounts{chi2_dof: 1.0, n: 16} = m
      assert {m.j, m.bb, m.ww, m.bw} == {24.0, 0.0, 0.0, 24.0}
      assert m.crosstab == [[0.0, 12.0], [12.0, 0.0]]
      assert m.expected == [[6.0, 6.0], [6.0, 6.0]]
      assert_in_delta m.chi2, 20.166666666666668, 1.0e-9
      assert_in_delta m.chi2_p, 7.09790833090828e-06, 1.0e-12
      assert m.autocorr_pos == 0.0 and m.autocorr_neg == 24.0
      assert %JoinCounts{sim_bb: nil, p_sim_bb: nil, sim_autocorr_chi2: nil, seed: nil} = m
    end

    test "drop_islands toggles island self-loop rows" do
      y = [1, 1, 0, 1, 1, 0, 1, 0]
      drop = ESDA.join_counts(y, path(4, 4), permutations: 0)
      keep = ESDA.join_counts(y, path(4, 4), permutations: 0, drop_islands: false)
      assert {drop.bb, drop.ww, drop.bw} == {1.0, 0.0, 2.0}
      assert {keep.bb, keep.ww, keep.bw} == {2.0, 1.0, 2.0}
      assert drop.j == keep.j
    end

    test "permutations return simulations, summaries and upper-tail p-values" do
      m = ESDA.join_counts(@lattice_y, lattice(), permutations: 199, seed: 3)
      same = ESDA.join_counts(@lattice_y, lattice(), permutations: 199, seed: 3)
      assert m.sim_bb == same.sim_bb
      assert length(m.sim_bb) == 199 and length(m.sim_bw) == 199 and length(m.sim_chi2) == 199
      assert length(m.sim_autocurr_pos) == 199 and length(m.sim_autocurr_neg) == 199
      assert m.min_bb <= m.mean_bb and m.mean_bb <= m.max_bb
      assert m.min_bw <= m.mean_bw and m.mean_bw <= m.max_bw
      larger = Enum.count(m.sim_bw, &(&1 >= m.bw))
      assert_in_delta m.p_sim_bw, (larger + 1) / 200, 1.0e-12
      assert m.sim_autocorr_chi2 >= 0.0 and m.sim_autocorr_chi2 <= 1.0
      assert m.seed == 3

      d =
        ESDA.join_counts(@lattice_y, lattice(),
          permutations: 199,
          seed: 3,
          keep_simulations: false
        )

      assert d.sim_bb == nil
      assert d.p_sim_bb == m.p_sim_bb
    end

    test "validates its input" do
      assert_raise ArgumentError, ~r/binary/, fn -> ESDA.join_counts([0, 1, 2, 1], path(4)) end
      assert_raise ArgumentError, fn -> ESDA.join_counts([1, 1, 1, 1], path(4)) end
      assert_raise ArgumentError, ~r/length/, fn -> ESDA.join_counts([0, 1, 0], path(4)) end

      assert_raise ArgumentError, fn ->
        ESDA.join_counts([0, 1, 0, 1], path(4), drop_islands: :yes)
      end

      assert_raise ArgumentError, fn -> ESDA.join_counts([0, 1, 0, 1], path(4), transform: :r) end
    end
  end
end
