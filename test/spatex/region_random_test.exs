defmodule Spatex.RegionRandomTest do
  use ExUnit.Case, async: true
  doctest Spatex.Region.RandomRegion
  doctest Spatex.Region.RandomRegions

  import Spatex.RegionHelper

  alias Spatex.Region
  alias Spatex.Region.RandomRegion
  alias Spatex.Region.RandomRegions

  defp sizes(%RandomRegion{regions: regions}), do: Enum.map(regions, &length/1)

  defp assert_partition(%RandomRegion{} = r, n) do
    assert r.feasible
    assert r.regions |> List.flatten() |> Enum.sort() == Enum.to_list(0..(n - 1))
    assert length(r.labels) == n
    assert r.n_regions == length(r.regions)
    assert r.labels |> Enum.uniq() |> Enum.sort() == Enum.to_list(0..(r.n_regions - 1))

    for {members, region} <- Enum.with_index(r.regions), a <- members do
      assert Enum.at(r.labels, a) == region
    end
  end

  describe "without a graph" do
    test "unconstrained partitions have between 2 and n - 1 regions" do
      for seed <- 1..30 do
        r = Region.random_region(20, seed: seed)
        assert_partition(r, 20)
        assert r.n_regions in 2..19
        assert r.seed == seed
      end
    end

    test "num_regions is exact" do
      for seed <- 1..30 do
        r = Region.random_region(15, num_regions: 4, seed: seed)
        assert_partition(r, 15)
        assert r.n_regions == 4
      end
    end

    test "cardinality sets the region sizes in order" do
      r = Region.random_region(12, cardinality: [6, 1, 5], seed: 3)
      assert_partition(r, 12)
      assert sizes(r) == [6, 1, 5]
      assert Region.random_region(12, cardinality: [6, 1, 5], num_regions: 3, seed: 3) == r
    end
  end

  describe "with a graph" do
    test "regions are contiguous and cardinalities exact, compact or not" do
      g = lattice(6, 6)

      for compact <- [false, true], seed <- 1..10 do
        r = Region.random_region(g, cardinality: [18, 9, 6, 3], compact: compact, seed: seed)
        assert_partition(r, 36)
        assert r |> sizes() |> Enum.sort() == [3, 6, 9, 18]
        assert Region.contiguous?(r.labels, g)
      end
    end

    test "num_regions and the graph alone give contiguous regions" do
      g = lattice(5, 5)

      for opts <- [[num_regions: 4], []], seed <- 1..10 do
        r = Region.random_region(g, [maxiter: 500, seed: seed] ++ opts)
        assert_partition(r, 25)
        assert Region.contiguous?(r.labels, g)
        if opts[:num_regions], do: assert(r.n_regions == 4)
      end
    end

    test "an impossible request is reported infeasible, like spopt" do
      edges = [{0, 1}, {1, 2}, {3, 4}, {4, 5}]
      {f, nb} = Enum.unzip(edges ++ Enum.map(edges, fn {a, b} -> {b, a} end))
      g = Spatex.Graph.from_arrays(f, nb, ids: Enum.to_list(0..5))

      r = Region.random_region(g, cardinality: [4, 2], maxiter: 5, max_swaps: 50, seed: 1)
      refute r.feasible
      assert r.labels == nil
      assert r.regions == []
      assert r.n_regions == 0
      assert r.attempts == 5
    end
  end

  describe "determinism and seeds" do
    test "a seed reproduces the partition and nil draws a recorded seed" do
      g = lattice(6, 6)
      a = Region.random_region(g, cardinality: [12, 12, 12], seed: 5)
      assert a == Region.random_region(g, cardinality: [12, 12, 12], seed: 5)
      refute a.labels == Region.random_region(g, cardinality: [12, 12, 12], seed: 6).labels

      drawn = Region.random_region(g, cardinality: [12, 12, 12], seed: nil)
      assert is_integer(drawn.seed)
      assert drawn == Region.random_region(g, cardinality: [12, 12, 12], seed: drawn.seed)
    end
  end

  describe "random_regions/2" do
    test "returns permutations solutions and the feasible subset, reproducibly" do
      g = lattice(5, 5)
      r = Region.random_regions(g, cardinality: [10, 8, 7], permutations: 12, seed: 8)
      assert %RandomRegions{seed: 8} = r
      assert length(r.solutions) == 12
      assert r.feasible_solutions == Enum.filter(r.solutions, & &1.feasible)
      assert Enum.all?(r.solutions, &Region.contiguous?(&1.labels, g))
      assert r == Region.random_regions(g, cardinality: [10, 8, 7], permutations: 12, seed: 8)
      # permutation 0 is the single-run result for the same seed
      assert hd(r.solutions).labels ==
               Region.random_region(g, cardinality: [10, 8, 7], seed: 8).labels
    end

    test "defaults to 99 permutations" do
      assert length(Region.random_regions(10, seed: 1).solutions) == 99
    end
  end

  describe "validation" do
    test "invalid requests raise ArgumentError" do
      assert_raise ArgumentError, ~r/cardinality/, fn ->
        Region.random_region(10, cardinality: [5, 4])
      end

      assert_raise ArgumentError, ~r/num_regions/, fn ->
        Region.random_region(10, num_regions: 3, cardinality: [5, 5])
      end

      assert_raise ArgumentError, ~r/too large/, fn ->
        Region.random_region(10, num_regions: 10)
      end

      assert_raise ArgumentError, ~r/at least 3/, fn -> Region.random_region(2) end
      assert_raise ArgumentError, ~r/unknown option/, fn -> Region.random_region(10, bogus: 1) end
      assert_raise ArgumentError, ~r/maxiter/, fn -> Region.random_region(10, maxiter: 0) end

      assert_raise ArgumentError, ~r/permutations/, fn ->
        Region.random_regions(10, permutations: 0)
      end

      assert_raise ArgumentError, ~r/compact/, fn -> Region.random_region(10, compact: 1) end
      assert_raise ArgumentError, ~r/areas/, fn -> Region.random_region(:x) end

      assert_raise ArgumentError, ~r/cardinality/, fn ->
        Region.random_region(10, cardinality: [10, 0])
      end
    end
  end
end
