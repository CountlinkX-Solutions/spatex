defmodule Spatex.ESDALocalMultivariateTest do
  use ExUnit.Case, async: true

  alias Spatex.ESDA
  alias Spatex.Graph

  # Path 0-1-2-3-4 plus island 5.
  defp path_with_island do
    Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], ids: Enum.to_list(0..5))
  end

  describe "island_weight" do
    test "gives islands a pseudo-neighbor in the geary_local simulations" do
      g = path_with_island()
      x = [1.0, 4.0, 2.0, 8.0, 5.0, 3.0]

      island = fn opts ->
        ESDA.geary_local(x, g, [permutations: 19, seed: 1, keep_simulations: true] ++ opts).sim
        |> List.last()
      end

      assert Enum.all?(island.([]), &(&1 == 0.0))
      assert Enum.any?(island.(island_weight: 1.0), &(&1 > 0.0))
    end

    test "gives islands a pseudo-neighbor in the join_counts_local simulations" do
      g = path_with_island()
      y = [1, 1, 0, 1, 0, 1]

      island = fn opts ->
        ESDA.join_counts_local(y, g, [permutations: 19, seed: 1, keep_simulations: true] ++ opts).sim
        |> List.last()
      end

      assert Enum.all?(island.([]), &(&1 == 0.0))
      assert Enum.any?(island.(island_weight: 1.0), &(&1 > 0.0))
    end

    test "must be a number" do
      assert_raise ArgumentError, ~r/:island_weight/, fn ->
        ESDA.geary_local([1, 2, 3, 4, 5, 6], path_with_island(),
          island_weight: "1",
          keep_simulations: true
        )
      end
    end
  end

  describe "geary_local_mv" do
    test "averages the per-variable local Geary of z-scores" do
      g = path_with_island()
      a = [1.0, 4.0, 2.0, 8.0, 5.0, 3.0]
      b = [2.0, 2.5, 9.0, 1.0, 4.0, 7.0]
      m = ESDA.geary_local_mv([a, b], g, permutations: 0)
      ga = ESDA.geary_local(a, g, permutations: 0).local_g
      gb = ESDA.geary_local(b, g, permutations: 0).local_g

      for {mv, {x, y}} <- Enum.zip(m.local_g, Enum.zip(ga, gb)),
          do: assert_in_delta(mv, (x + y) / 2, 1.0e-12)

      assert m.p_sim == nil and m.sim == nil
      assert %{n: 6, k: 2, permutations: 0} = m
    end

    test "is seeded, keeps simulations on request and validates its input" do
      g = path_with_island()
      vars = [[1, 4, 2, 8, 5, 3], [2, 3, 9, 1, 4, 7]]
      a = ESDA.geary_local_mv(vars, g, permutations: 49, seed: 7, keep_simulations: true)

      assert a.p_sim ==
               ESDA.geary_local_mv(vars, g, permutations: 49, seed: 7, keep_simulations: true).p_sim

      assert length(a.sim) == 6 and length(hd(a.sim)) == 49
      assert ESDA.geary_local_mv(vars, g, permutations: 49, keep_simulations: false).sim == nil

      assert_raise ArgumentError, ~r/at least two variables/, fn ->
        ESDA.geary_local_mv([hd(vars)], g)
      end

      assert_raise ArgumentError, ~r/length/, fn ->
        ESDA.geary_local_mv([[1, 2, 3], [1, 2, 3]], g)
      end
    end
  end

  describe "join_counts_local_bv and join_counts_local_mv" do
    @x [1, 1, 0, 1, 1, 1]
    @z [1, 0, 1, 1, 1, 1]

    test "CLC and MV are the univariate counts of the product" do
      g = path_with_island()
      prod = Enum.zip_with(@x, @z, &(&1 * &2))
      uni = ESDA.join_counts_local(prod, g, permutations: 0).ljc
      assert ESDA.join_counts_local_bv(@x, @z, g, permutations: 0).ljc == uni
      assert ESDA.join_counts_local_mv([@x, @z], g, permutations: 0).ljc == uni

      a =
        ESDA.join_counts_local_mv([@x, @z], g, permutations: 49, seed: 3, keep_simulations: true)

      b = ESDA.join_counts_local(prod, g, permutations: 49, seed: 3, keep_simulations: true)
      assert a.p_sim == b.p_sim
    end

    test "BJC counts x=1, z=0 sites next to x=0, z=1 sites" do
      g = path_with_island()
      # Site 1 (x=1, z=0) has neighbor 2 (x=0, z=1): one join; every other site has none.
      m =
        ESDA.join_counts_local_bv(@x, @z, g,
          case: :bjc,
          permutations: 19,
          seed: 1,
          keep_simulations: true
        )

      assert m.ljc == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
      assert Enum.at(m.p_sim, 1) |> is_float()
      assert Enum.at(m.p_sim, 0) == :nan
    end

    test "inputs are validated" do
      g = path_with_island()

      assert_raise ArgumentError, ~r/:case/, fn ->
        ESDA.join_counts_local_bv(@x, @z, g, case: :abc)
      end

      assert_raise ArgumentError, ~r/binary/, fn ->
        ESDA.join_counts_local_bv([2, 1, 0, 1, 1, 1], @z, g)
      end

      assert_raise ArgumentError, ~r/at least two variables/, fn ->
        ESDA.join_counts_local_mv([@x], g)
      end
    end
  end
end
