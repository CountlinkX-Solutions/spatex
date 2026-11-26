defmodule Spatex.GraphSparseOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12

  for name <- ~w(sparse_queen_island sparse_kernel sparse_distance_band sparse_explicit_zero) do
    describe "fixture #{name}" do
      setup do
        fx = Reference.load!("graph/#{unquote(name)}.json")
        ids = fx["ids"]
        adj = fx["adjacency"]
        at = fn positions -> Enum.map(positions, &Enum.at(ids, &1)) end

        graph =
          Graph.from_arrays(at.(adj["focal"]), at.(adj["neighbor"]), adj["weight"], ids: ids)

        {:ok, fx: fx, graph: graph}
      end

      for t <- ~w(O R) do
        test "transform #{t}: indptr, indices, data and shape", %{fx: fx, graph: graph} do
          expected = fx["sparse"][unquote(t)]
          graph = if unquote(t) == "R", do: Graph.transform(graph, :r), else: graph

          sparse = Graph.to_sparse(graph)

          assert sparse.shape == List.to_tuple(expected["shape"])
          assert Nx.to_flat_list(sparse.indptr) == expected["indptr"]
          assert Nx.to_flat_list(sparse.indices) == expected["indices"]
          Reference.assert_close(Nx.to_flat_list(sparse.data), expected["data"], @tol)
        end
      end
    end
  end
end
