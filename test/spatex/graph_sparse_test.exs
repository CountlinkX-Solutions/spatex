defmodule Spatex.GraphSparseTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph
  alias Spatex.Input

  defp hand do
    Graph.from_adjacency([
      {:a, :b, 1.0},
      {:a, :c, 2.0},
      {:b, :a, 3.0},
      {:c, :d, 0.5},
      {:d, :c, 1.5},
      {:d, :a, 2.5},
      {:e, :e, 0.0}
    ])
  end

  describe "to_sparse/1" do
    test "returns the CSR as Nx tensors with documented types" do
      sparse = Graph.to_sparse(hand())

      assert sparse.shape == {5, 5}
      assert Nx.type(sparse.indptr) == {:s, 64}
      assert Nx.type(sparse.indices) == {:u, 32}
      assert Nx.type(sparse.data) == {:f, 64}
      assert Nx.to_flat_list(sparse.indptr) == [0, 2, 3, 4, 6, 7]
      assert Nx.to_flat_list(sparse.indices) == [1, 2, 0, 3, 0, 2, 4]
      assert Nx.to_flat_list(sparse.data) == [1.0, 2.0, 3.0, 0.5, 2.5, 1.5, 0.0]
    end

    test "the island keeps its explicit zero self-loop" do
      sparse = Graph.to_sparse(Graph.from_arrays([:a], [:b], [1.0]))

      assert Nx.to_flat_list(sparse.indptr) == [0, 1, 2]
      assert Nx.to_flat_list(sparse.indices) == [1, 1]
      assert Nx.to_flat_list(sparse.data) == [1.0, 0.0]
    end

    test "reflects the current transform" do
      sparse = hand() |> Graph.transform(:b) |> Graph.to_sparse()
      assert Nx.to_flat_list(sparse.data) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    end
  end

  describe "Input.require_nx!/2" do
    test "passes when Nx is loaded" do
      assert Input.require_nx!("Spatex.Graph.to_sparse/1", true) == :ok
    end

    test "raises an ArgumentError naming the dependency when Nx is missing" do
      assert_raise ArgumentError, ~r/Spatex.Graph.to_sparse\/1 needs.*\{:nx, "~> 0.10"\}/, fn ->
        Input.require_nx!("Spatex.Graph.to_sparse/1", false)
      end
    end
  end
end
