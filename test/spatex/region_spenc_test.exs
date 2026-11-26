defmodule Spatex.RegionSpencTest do
  use ExUnit.Case, async: true

  import Spatex.RegionHelper

  alias Spatex.Graph
  alias Spatex.Region
  alias Spatex.Region.Spenc

  defp path(n) do
    {f, nb} = Enum.unzip(for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair)
    Graph.from_arrays(f, nb, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(n - 1)))
  end

  @x [[0.0, 0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3]]
  @opts [n_clusters: 2, gamma: 0.05]

  test "two groups are recovered by every assignment strategy" do
    for assign <- [:discretize, :kmeans, :hierarchical] do
      r = Region.spenc(@x, path(8), [assign_labels: assign, seed: 4] ++ @opts)
      assert %Spenc{assign_labels: ^assign, n_clusters: 2} = r
      assert regions(r.labels) == [[0, 1, 2, 3], [4, 5, 6, 7]], "#{assign}"
    end
  end

  test "a seed reproduces the result and nil draws and records one" do
    a = Region.spenc(@x, path(8), [seed: 9] ++ @opts)
    b = Region.spenc(@x, path(8), [seed: 9] ++ @opts)
    assert a == b
    assert a.seed == 9 and is_integer(a.start_row)

    c = Region.spenc(@x, path(8), @opts)
    assert is_integer(c.seed) and c.seed >= 0
    assert Region.spenc(@x, path(8), [seed: c.seed] ++ @opts).labels == c.labels
  end

  test "hierarchical is deterministic and records no seed or start row" do
    r = Region.spenc(@x, path(8), [assign_labels: :hierarchical] ++ @opts)
    assert r.seed == nil and r.start_row == nil and r.embedding == nil and r.eigenvalues == []
    assert r == Region.spenc(@x, path(8), [assign_labels: :hierarchical, seed: 3] ++ @opts)
  end

  test ":infinity selects the hierarchical bipartition with a floor" do
    r = Region.spenc(@x, path(8), n_clusters: :infinity, gamma: 0.05, floor: 3)
    assert r.assign_labels == :hierarchical
    assert regions(r.labels) == [[0, 1, 2, 3], [4, 5, 6, 7]]
  end

  test "the embedding, eigenvalues and affinity are exposed on request" do
    r = Region.spenc(@x, path(8), [keep_affinity: true] ++ @opts)
    assert length(r.embedding) == 8 and Enum.all?(r.embedding, &(length(&1) == 2))
    assert length(r.eigenvalues) == 2 and abs(hd(r.eigenvalues)) < 1.0e-10
    assert length(r.affinity_matrix) == 8
    assert Enum.at(Enum.at(r.affinity_matrix, 0), 1) == :math.exp(-0.05 * 0.01)

    t = Region.spenc(@x, path(8), [keep_embedding: :tensor, keep_affinity: :tensor] ++ @opts)
    assert Nx.shape(t.embedding) == {8, 2} and Nx.shape(t.affinity_matrix) == {8, 8}
    assert Nx.type(t.embedding) == {:f, 64}

    n = Region.spenc(@x, path(8), [keep_embedding: false] ++ @opts)
    assert n.embedding == nil and n.affinity_matrix == nil
  end

  test "graph weights multiply the affinity" do
    f = [0, 1, 1, 2]
    nb = [1, 0, 2, 1]
    g = Graph.from_arrays(f, nb, [2.0, 2.0, 1.0, 1.0])
    r = Region.spenc([[0.0, 1.0, 2.0]], g, n_clusters: 2, gamma: 1.0, keep_affinity: true)
    assert_in_delta Enum.at(Enum.at(r.affinity_matrix, 0), 1), 2.0 * :math.exp(-1.0), 1.0e-15
  end

  test "a disconnected graph warns" do
    g = Graph.from_arrays([0, 1, 2, 3], [1, 0, 3, 2])
    r = Region.spenc([[0.0, 0.1, 5.0, 5.1]], g, n_clusters: 2, seed: 1)
    assert Enum.any?(r.warnings, &(&1 =~ "disconnected"))
  end

  test "precomputed and nearest_neighbors affinities" do
    aff = for i <- 0..7, do: for(j <- 0..7, do: if(abs(i - j) == 1, do: 1.0, else: 0.0))
    r = Region.spenc(aff, path(8), n_clusters: 2, affinity: :precomputed, seed: 1)
    assert r.n_clusters == 2

    r = Region.spenc(@x, path(8), n_clusters: 2, affinity: :nearest_neighbors, n_neighbors: 3)
    assert length(r.labels) == 8
  end

  describe "validation" do
    test "options are checked" do
      bad = [
        [n_clusters: 8],
        [n_clusters: 0],
        [n_clusters: 2.0],
        [n_clusters: 2, affinity: :chi2],
        [n_clusters: 2, assign_labels: :other],
        [n_clusters: 2, eigen_solver: :amg],
        [n_clusters: 2, gamma: "1"],
        [n_clusters: 2, n_neighbors: 0],
        [n_clusters: 2, affinity: :nearest_neighbors, n_neighbors: 9],
        [n_clusters: 2, floor_weights: [1.0]],
        [n_clusters: 2, seed: -1],
        [n_clusters: 2, keep_embedding: :yes],
        [n_clusters: 2, bogus: 1]
      ]

      for opts <- bad do
        assert_raise ArgumentError, fn -> Region.spenc(@x, path(8), opts) end
      end
    end

    test "eigen_solver, eigen_tol and n_init are accepted and ignored" do
      a =
        Region.spenc(
          @x,
          path(8),
          [eigen_solver: :arpack, eigen_tol: 1.0e-7, n_init: 3, seed: 2] ++ @opts
        )

      assert a.labels == Region.spenc(@x, path(8), [seed: 2] ++ @opts).labels
    end

    test "an asymmetric affinity raises" do
      g = Graph.from_arrays([0, 1, 2], [1, 2, 0])

      assert_raise ArgumentError, ~r/symmetric/, fn ->
        Region.spenc([[0.0, 1.0, 2.0]], g, n_clusters: 2)
      end
    end

    test "the attribute table must match the graph and precomputed must be square" do
      assert_raise ArgumentError, fn -> Region.spenc([[1.0, 2.0]], path(8), n_clusters: 2) end

      assert_raise ArgumentError, fn ->
        Region.spenc(@x, path(8), n_clusters: 2, affinity: :precomputed)
      end
    end

    test "hierarchical raises when no cut can be accepted" do
      assert_raise ArgumentError, fn ->
        Region.spenc(@x, path(8), n_clusters: :infinity, gamma: 0.05, floor: 8)
      end
    end
  end
end
