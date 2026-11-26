defmodule Spatex.GraphKnnTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  @line [{0, 0}, {1, 0}, {2, 0}, {3.5, 0}, {10, 0}]

  describe "build_knn/3" do
    test "returns a graph of k neighbors per node with unit weights" do
      g = Graph.build_knn(@line, 1)
      assert %Graph{n: 5, ids: [0, 1, 2, 3, 4], transform: :o} = g

      assert Graph.to_adjacency(g) ==
               [{0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}, {3, 2, 1.0}, {4, 3, 1.0}]

      assert Graph.cardinalities(Graph.build_knn(@line, 3)) == [3, 3, 3, 3, 3]
    end

    test "accepts [x, y] lists, {xs, ys} pairs and Nx tensors" do
      expected = Graph.to_adjacency(Graph.build_knn(@line, 2))
      assert Graph.to_adjacency(Graph.build_knn(Enum.map(@line, &Tuple.to_list/1), 2)) == expected

      pair = {Enum.map(@line, &elem(&1, 0)), Enum.map(@line, &elem(&1, 1))}
      assert Graph.to_adjacency(Graph.build_knn(pair, 2)) == expected

      tensor = Nx.tensor(Enum.map(@line, &Tuple.to_list/1))
      assert Graph.to_adjacency(Graph.build_knn(tensor, 2)) == expected
    end

    test "labels nodes with :ids" do
      g = Graph.build_knn(@line, 1, ids: [:a, :b, :c, :d, :e])
      assert Graph.ids(g) == [:a, :b, :c, :d, :e]
      assert {:a, :b, 1.0} in Graph.to_adjacency(g)
      assert {:e, :d, 1.0} in Graph.to_adjacency(g)
    end

    test "coincident points raise by default and connect as a clique on request" do
      pts = [{0, 0}, {0, 0}, {1, 0}, {3, 0}, {3, 1}]
      assert_raise ArgumentError, ~r/coincident/, fn -> Graph.build_knn(pts, 2) end

      g = Graph.build_knn(pts, 2, coincident: :clique)

      assert Graph.to_adjacency(g) |> Enum.filter(&(elem(&1, 0) == 0)) ==
               [{0, 1, 1.0}, {0, 2, 1.0}, {0, 3, 1.0}]
    end

    test "validates its arguments" do
      assert_raise ArgumentError, ~r/k must be/, fn -> Graph.build_knn(@line, 0) end
      assert_raise ArgumentError, ~r/k must be/, fn -> Graph.build_knn(@line, 1.5) end
      assert_raise ArgumentError, ~r/at most 4/, fn -> Graph.build_knn(@line, 5) end

      assert_raise ArgumentError, ~r/:coincident/, fn ->
        Graph.build_knn(@line, 1, coincident: :shake)
      end

      assert_raise ArgumentError, ~r/ids/, fn -> Graph.build_knn(@line, 1, ids: [1, 2]) end

      assert_raise ArgumentError, ~r/unique/, fn ->
        Graph.build_knn(@line, 1, ids: [1, 1, 2, 3, 4])
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_knn(@line, 1, bogus: 1)
      end

      assert_raise ArgumentError, ~r/points/, fn -> Graph.build_knn([{0, :x}], 1) end

      assert_raise ArgumentError, ~r/non-finite/, fn ->
        Graph.build_knn({[0, :nan], [0, 1]}, 1)
      end
    end
  end

  describe "build_distance_band/3" do
    test "threshold is inclusive and distant nodes are islands" do
      g = Graph.build_distance_band(@line, 1.0)

      assert Graph.to_adjacency(g) ==
               [{0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}, {3, 3, 0.0}, {4, 4, 0.0}]

      assert Graph.isolates(g) == [3, 4]
    end

    test "binary: false weights by inverse distance power alpha" do
      inv = Graph.build_distance_band(@line, 1.5, binary: false)

      assert {2, 3, w} =
               Enum.find(Graph.to_adjacency(inv), &(elem(&1, 0) == 2 and elem(&1, 1) == 3))

      assert_in_delta w, 1 / 1.5, 1.0e-15

      sq = Graph.build_distance_band(@line, 1.5, binary: false, alpha: -2)

      assert {2, 3, w2} =
               Enum.find(Graph.to_adjacency(sq), &(elem(&1, 0) == 2 and elem(&1, 1) == 3))

      assert_in_delta w2, 1 / 2.25, 1.0e-15
    end

    test "labels nodes with :ids" do
      g = Graph.build_distance_band(@line, 1.0, ids: ~w(a b c d e))
      assert {"a", "b", 1.0} in Graph.to_adjacency(g)
      assert Graph.isolates(g) == ["d", "e"]
    end

    test "coincident points are linked when binary and rejected for inverse distance" do
      pts = [{0, 0}, {0, 0}, {5, 5}]
      assert {0, 1, 1.0} in Graph.to_adjacency(Graph.build_distance_band(pts, 1))

      assert_raise ArgumentError, ~r/coincident/, fn ->
        Graph.build_distance_band(pts, 1, binary: false)
      end
    end

    test "validates its arguments" do
      assert_raise ArgumentError, ~r/threshold/, fn -> Graph.build_distance_band(@line, 0) end
      assert_raise ArgumentError, ~r/threshold/, fn -> Graph.build_distance_band(@line, -1.0) end
      assert_raise ArgumentError, ~r/threshold/, fn -> Graph.build_distance_band(@line, :far) end

      assert_raise ArgumentError, ~r/alpha/, fn ->
        Graph.build_distance_band(@line, 1, binary: false, alpha: :x)
      end

      assert_raise ArgumentError, ~r/binary/, fn ->
        Graph.build_distance_band(@line, 1, binary: :yes)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_distance_band(@line, 1, bogus: :gaussian)
      end
    end
  end
end
