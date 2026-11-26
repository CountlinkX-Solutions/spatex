defmodule Spatex.RegionTest do
  use ExUnit.Case, async: true
  doctest Spatex.Region

  import Spatex.RegionHelper

  alias Spatex.Graph
  alias Spatex.Region

  defp two_by_two_islands do
    # two disconnected 2x2 lattices (nodes 0-3 and 4-7)
    edges = [{0, 1}, {0, 2}, {1, 3}, {2, 3}, {4, 5}, {4, 6}, {5, 7}, {6, 7}]
    both = edges ++ Enum.map(edges, fn {a, b} -> {b, a} end)
    {f, n} = Enum.unzip(both)
    Graph.from_arrays(f, n, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..7))
  end

  describe "input handling" do
    test "attributes accept columns, a single column, a map and an Nx tensor" do
      g = lattice(2, 3)
      a = [1.0, 1.1, 9.0, 1.2, 1.3, 9.1]
      b = [0.0, 0.1, 4.0, 0.1, 0.0, 4.2]
      base = Region.ward_spatial([a, b], g, n_clusters: 2).labels
      assert Region.ward_spatial(%{"a" => a, "b" => b}, g, n_clusters: 2).labels == base

      assert Region.ward_spatial(
               Nx.tensor(Enum.zip_with([a, b], &Tuple.to_list(List.to_tuple(&1)))),
               g,
               n_clusters: 2
             ).labels == base

      assert Region.ward_spatial(a, g, n_clusters: 2).labels |> length() == 6
    end

    test "rows must match the graph and values must be finite" do
      g = lattice(2, 2)

      assert_raise ArgumentError, ~r/4 nodes/, fn ->
        Region.skater([[1.0, 2.0, 3.0]], g, n_clusters: 2)
      end

      assert_raise ArgumentError, ~r/non-finite|expected a number/, fn ->
        Region.skater([[1.0, :nan, 3.0, 4.0]], g, n_clusters: 2)
      end

      assert_raise ArgumentError, ~r/n_clusters/, fn ->
        Region.skater([[1.0, 2.0, 3.0, 4.0]], g, n_clusters: 5)
      end

      assert_raise ArgumentError, ~r/n_clusters/, fn ->
        Region.ward_spatial([[1.0, 2.0, 3.0, 4.0]], g, n_clusters: 0)
      end
    end

    test "an asymmetric graph raises unless symmetrize: true" do
      g = Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1], [1.0, 1.0, 1.0, 1.0], ids: [0, 1, 2])
      asym = Graph.from_arrays([0, 1], [1, 2], [1.0, 1.0], ids: [0, 1, 2])
      x = [[1.0, 2.0, 9.0]]
      assert %Region.Skater{} = Region.skater(x, g, n_clusters: 2)

      assert_raise ArgumentError, ~r/symmetric/, fn -> Region.skater(x, asym, n_clusters: 2) end

      assert %Region.Skater{} = Region.skater(x, asym, n_clusters: 2, symmetrize: true)
    end
  end

  describe "skater/3" do
    test "cuts the most dissimilar link" do
      g = lattice(1, 6)
      x = [[1.0, 1.1, 1.2, 9.0, 9.1, 9.2]]
      r = Region.skater(x, g, n_clusters: 2)
      assert r.labels == [0, 0, 0, 1, 1, 1]
      assert r.n_regions == 2
      assert_in_delta r.score, 0.2 + 0.2, 1.0e-9
    end

    test "n_clusters = 1 returns one region; n_clusters = n returns singletons" do
      g = lattice(2, 2)
      x = [[1.0, 5.0, 2.0, 8.0]]
      assert Region.skater(x, g, n_clusters: 1).labels == [0, 0, 0, 0]
      assert Region.skater(x, g, n_clusters: 4).n_regions == 4
    end

    test "euclidean dissimilarity is supported and floors are area counts" do
      g = lattice(1, 6)
      x = [[1.0, 1.0, 1.1, 5.0, 9.0, 9.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
      r = Region.skater(x, g, n_clusters: 3, dissimilarity: :euclidean, floor: 2)
      assert Enum.all?(regions(r.labels), &(length(&1) >= 2))
    end

    test "is deterministic" do
      g = lattice(5, 5)
      x = [Enum.map(0..24, &:math.sin(&1 * 1.7))]
      assert Region.skater(x, g, n_clusters: 4) == Region.skater(x, g, n_clusters: 4)
    end
  end

  describe "ward_spatial/3" do
    test "merges similar neighbors first" do
      g = lattice(1, 6)
      x = [[1.0, 1.1, 1.2, 9.0, 9.1, 9.2]]
      r = Region.ward_spatial(x, g, n_clusters: 2)
      assert regions(r.labels) == [[0, 1, 2], [3, 4, 5]]
      assert length(r.children) == 5
    end

    test "a single area" do
      g = Graph.from_arrays([0], [0], [0.0], ids: [0])
      assert Region.ward_spatial([[3.0]], g, n_clusters: 1).labels == [0]
    end
  end

  describe "region_kmeans/3" do
    test "regions grow from the seeds and stay contiguous" do
      g = lattice(4, 4)
      x = [Enum.map(0..15, &(div(&1, 4) * 1.0)), Enum.map(0..15, &(rem(&1, 4) * 1.0))]
      r = Region.region_kmeans(x, g, n_clusters: 3, seeds: [0, 5, 15])
      assert r.seeds == [0, 5, 15]
      assert Region.contiguous?(r.labels, g)
      assert length(Enum.uniq(r.labels)) == 3
      assert length(r.centroids) == 3
    end

    test "seed: draws reproducible seeds, one per component at least" do
      g = two_by_two_islands()
      x = [[1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0]]
      a = Region.region_kmeans(x, g, n_clusters: 3, seed: 7)
      assert a == Region.region_kmeans(x, g, n_clusters: 3, seed: 7)
      comps = Enum.map(a.seeds, &div(&1, 4))
      assert Enum.sort(Enum.uniq(comps)) == [0, 1]
    end

    test "raises when a component has no seed" do
      g = two_by_two_islands()
      x = [[1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0]]

      assert_raise ArgumentError, ~r/no seed/, fn ->
        Region.region_kmeans(x, g, n_clusters: 2, seeds: [0, 1])
      end
    end

    test "validates seeds" do
      g = lattice(2, 2)
      x = [[1.0, 2.0, 3.0, 4.0]]

      assert_raise ArgumentError, ~r/distinct/, fn ->
        Region.region_kmeans(x, g, n_clusters: 2, seeds: [1, 1])
      end

      assert_raise ArgumentError, ~r/seeds/, fn ->
        Region.region_kmeans(x, g, n_clusters: 2, seeds: [0])
      end

      assert_raise ArgumentError, ~r/seeds/, fn ->
        Region.region_kmeans(x, g, n_clusters: 2, seeds: [0, 9])
      end
    end
  end

  describe "azp/3" do
    test "an initial solution is respected as a starting point" do
      g = lattice(3, 4)
      x = [Enum.map(0..11, &(:math.sin(&1) * 5))]
      initial = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
      r = Region.azp(x, g, n_clusters: 3, initial: initial, seed: 1)
      assert r.initial_labels == initial
      assert r.objective <= Region.objective(x, initial, metric: :euclidean) + 1.0e-9
    end

    test "rejects a non-contiguous or mis-sized initial solution" do
      g = lattice(2, 3)
      x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

      assert_raise ArgumentError, ~r/contiguous/, fn ->
        Region.azp(x, g, n_clusters: 2, initial: [0, 1, 0, 1, 0, 1])
      end

      assert_raise ArgumentError, ~r/exactly 2/, fn ->
        Region.azp(x, g, n_clusters: 2, initial: [0, 0, 0, 0, 0, 0])
      end
    end

    test "tabu and annealing are deterministic per seed and validate their options" do
      g = lattice(4, 4)
      x = [Enum.map(0..15, &(rem(&1 * 5, 7) * 1.0))]

      for opts <- [[method: :tabu], [method: :annealing, init_temperature: 2.0]] do
        a = Region.azp(x, g, [n_clusters: 4, seed: 3] ++ opts)
        assert a == Region.azp(x, g, [n_clusters: 4, seed: 3] ++ opts)
      end

      assert_raise ArgumentError, ~r/:init_temperature/, fn ->
        Region.azp(x, g, n_clusters: 4, method: :annealing)
      end

      assert_raise ArgumentError, ~r/:cooling_factor/, fn ->
        Region.azp(x, g,
          n_clusters: 4,
          method: :annealing,
          init_temperature: 1,
          cooling_factor: 1
        )
      end

      assert_raise ArgumentError, ~r/:tabu_length/, fn ->
        Region.azp(x, g, n_clusters: 4, method: :tabu, tabu_length: 0)
      end

      assert_raise ArgumentError, ~r/:method/, fn ->
        Region.azp(x, g, n_clusters: 4, method: :reactive_tabu)
      end
    end

    test "n_clusters = 1 is one region (spopt crashes here)" do
      g = lattice(2, 2)
      r = Region.azp([[1.0, 2.0, 3.0, 4.0]], g, n_clusters: 1)
      assert r.labels == [0, 0, 0, 0]
    end

    test "more regions than components allows, fewer raises" do
      g = two_by_two_islands()
      x = [[1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0]]
      r = Region.azp(x, g, n_clusters: 3, seed: 2)
      assert Region.contiguous?(r.labels, g)

      assert_raise ArgumentError, ~r/components/, fn -> Region.azp(x, g, n_clusters: 1) end
    end

    test "is deterministic per seed and varies across seeds" do
      g = lattice(5, 5)
      x = [Enum.map(0..24, &:math.cos(&1 * 2.3))]
      a = Region.azp(x, g, n_clusters: 4, seed: 3)
      assert a == Region.azp(x, g, n_clusters: 4, seed: 3)
      assert a.initial_labels != Region.azp(x, g, n_clusters: 4, seed: 4).initial_labels
    end
  end

  describe "max_p/3" do
    test "regions meet the threshold" do
      g = lattice(4, 4)
      x = [Enum.map(0..15, &:math.sin(&1))]
      t = List.duplicate(1.0, 16)
      r = Region.max_p(x, g, threshold_variable: t, threshold: 4, seed: 1)
      assert r.p >= 2
      assert Region.contiguous?(r.labels, g)
      assert Enum.all?(regions(r.labels), &(length(&1) >= 4))
      assert r == Region.max_p(x, g, threshold_variable: t, threshold: 4, seed: 1)
    end

    test "threshold above the total raises" do
      g = lattice(2, 2)

      assert_raise ArgumentError, ~r/No feasible components/, fn ->
        Region.max_p([[1.0, 2.0, 3.0, 4.0]], g,
          threshold_variable: [1.0, 1.0, 1.0, 1.0],
          threshold: 10
        )
      end
    end

    test "infeasible components: :single needs coords, :drop removes them, :keep raises" do
      # component 0 (nodes 0-3) is feasible (4 areas), component 1 (nodes 4-5) is not
      f = [0, 1, 1, 2, 2, 3, 4, 5]
      n = [1, 0, 2, 1, 3, 2, 5, 4]
      g = Graph.from_arrays(f, n, List.duplicate(1.0, 8), ids: Enum.to_list(0..5))
      x = [[1.0, 2.0, 3.0, 4.0, 10.0, 11.0]]
      t = List.duplicate(1.0, 6)
      coords = [{0, 0}, {1, 0}, {2, 0}, {3, 0}, {10, 0}, {11, 0}]

      assert_raise ArgumentError, ~r/coords/, fn ->
        Region.max_p(x, g, threshold_variable: t, threshold: 3)
      end

      r = Region.max_p(x, g, threshold_variable: t, threshold: 3, coords: coords, seed: 1)
      assert length(r.labels) == 6
      assert Enum.all?(r.labels, &is_integer/1)

      d = Region.max_p(x, g, threshold_variable: t, threshold: 3, policy: :drop, seed: 1)
      assert Enum.drop(d.labels, 4) == [nil, nil]
      assert Enum.all?(Enum.take(d.labels, 4), &is_integer/1)

      assert_raise ArgumentError, ~r/infeasible/, fn ->
        Region.max_p(x, g, threshold_variable: t, threshold: 3, policy: :keep)
      end
    end

    test "top_n must be positive" do
      g = lattice(2, 2)

      assert_raise ArgumentError, ~r/top_n/, fn ->
        Region.max_p([[1.0, 2.0, 3.0, 4.0]], g,
          threshold_variable: [1.0, 1.0, 1.0, 1.0],
          threshold: 2,
          top_n: 0
        )
      end
    end
  end

  describe "helpers" do
    test "relabel/1 renumbers by first appearance" do
      assert Region.relabel([5, 5, 2, 9, 2]) == [0, 0, 1, 2, 1]
    end

    test "contiguous?/2" do
      g = lattice(2, 3)
      assert Region.contiguous?([0, 0, 0, 1, 1, 1], g)
      refute Region.contiguous?([0, 1, 0, 1, 0, 1], g)
    end

    test "objective/3 on a toy example" do
      x = [[0.0, 1.0, 5.0, 7.0]]
      labels = [0, 0, 1, 1]
      assert Region.objective(x, labels) == 3.0
      assert Region.objective(x, labels, kind: :center) == 3.0
      assert Region.objective(x, labels, kind: :center, metric: :sqeuclidean) == 0.5 + 2.0
    end

    test "standardize/1 returns z-scores with ddof = 0" do
      [z] = Region.standardize([[1.0, 2.0, 3.0]])
      assert_in_delta Enum.sum(z), 0.0, 1.0e-12
      assert_in_delta Enum.sum(Enum.map(z, &(&1 * &1))) / 3, 1.0, 1.0e-12
    end
  end
end
