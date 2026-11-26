defmodule Spatex.GraphOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12

  for name <- ~w(columbus_queen lattice4x4_rook asymmetric_island) do
    describe "fixture #{name}" do
      setup do
        fx = Reference.load!("graph/#{unquote(name)}.json")
        ids = fx["ids"]
        original = fx["transforms"]["O"]["adjacency"]
        by_pos = fn positions -> Enum.map(positions, &Enum.at(ids, &1)) end

        graph =
          Graph.from_arrays(
            by_pos.(original["focal"]),
            by_pos.(original["neighbor"]),
            original["weight"],
            ids: ids
          )

        {:ok, fx: fx, graph: graph, by_pos: by_pos}
      end

      test "structure matches", %{fx: fx, graph: g, by_pos: by_pos} do
        assert Graph.n(g) == fx["n"]
        assert Graph.ids(g) == fx["ids"]
        assert Graph.cardinalities(g) == fx["cardinalities"]
        assert Graph.isolates(g) == by_pos.(fx["isolates"])
        assert Graph.symmetric?(g) == fx["symmetric"]
      end

      for t <- ~w(O R B D V) do
        test "transform #{t}: adjacency, s0/s1/s2 and lag", %{fx: fx, graph: g, by_pos: by_pos} do
          expected = fx["transforms"][unquote(t)]
          t_graph = Graph.transform(g, unquote(t))
          adjacency = expected["adjacency"]

          actual = Graph.to_adjacency(t_graph)
          assert Enum.map(actual, &elem(&1, 0)) == by_pos.(adjacency["focal"])
          assert Enum.map(actual, &elem(&1, 1)) == by_pos.(adjacency["neighbor"])
          Reference.assert_close(Enum.map(actual, &elem(&1, 2)), adjacency["weight"], @tol)

          Reference.assert_close(Graph.s0(t_graph), expected["s0"], @tol)
          Reference.assert_close(Graph.s1(t_graph), expected["s1"], @tol)
          Reference.assert_close(Graph.s2(t_graph), expected["s2"], @tol)

          Reference.assert_close(
            Graph.lag(t_graph, fx["variable"]["values"]),
            expected["lag"],
            @tol
          )
        end
      end
    end
  end
end
