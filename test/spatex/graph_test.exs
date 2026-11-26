defmodule Spatex.GraphTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  doctest Spatex.Graph

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

  describe "from_arrays/4" do
    test "ids default to first appearance in focal then neighbor" do
      g = Graph.from_arrays([:b, :a], [:c, :b], [1, 2])
      assert Graph.ids(g) == [:b, :a, :c]
      assert Graph.n(g) == 3
      # :c has no outgoing edge, so it is an island
      assert Graph.isolates(g) == [:c]
      assert Graph.cardinalities(g) == [1, 1, 0]
    end

    test "weights default to 1.0 and ids: fixes the node order" do
      g = Graph.from_arrays([:a, :b], [:b, :a], ids: [:b, :a, :z])
      assert Graph.ids(g) == [:b, :a, :z]
      assert Graph.to_adjacency(g) == [{:b, :a, 1.0}, {:a, :b, 1.0}, {:z, :z, 0.0}]
    end

    test "integer and float weights are accepted" do
      g = Graph.from_arrays([0, 1], [1, 0], [1, 2.5])
      assert Graph.s0(g) == 3.5
    end

    test "raises on duplicate edges" do
      assert_raise ArgumentError, ~r/duplicate edge/, fn ->
        Graph.from_arrays([:a, :a], [:b, :b], [1.0, 2.0])
      end
    end

    test "raises on length mismatch" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        Graph.from_arrays([:a, :b], [:b], [1.0])
      end

      assert_raise ArgumentError, ~r/same length/, fn ->
        Graph.from_arrays([:a], [:b], [1.0, 2.0])
      end
    end

    test "raises on non-numeric weights" do
      assert_raise ArgumentError, ~r/weight: expected a number/, fn ->
        Graph.from_arrays([:a], [:b], [:x])
      end

      assert_raise ArgumentError, ~r/weight: expected a number/, fn ->
        Graph.from_arrays([:a], [:b], [nil])
      end
    end

    test "raises when ids: is not unique or misses nodes" do
      assert_raise ArgumentError, ~r/unique/, fn ->
        Graph.from_arrays([:a], [:b], [1.0], ids: [:a, :b, :a])
      end

      assert_raise ArgumentError, ~r/missing from :ids/, fn ->
        Graph.from_arrays([:a], [:b], [1.0], ids: [:a])
      end
    end

    test "empty input gives an empty graph" do
      g = Graph.from_arrays([], [])
      assert Graph.n(g) == 0
      assert Graph.to_adjacency(g) == []
      assert Graph.s0(g) == 0.0
    end
  end

  describe "from_adjacency/2" do
    test "accepts a list of triples" do
      g = hand()
      assert Graph.ids(g) == [:a, :b, :c, :d, :e]
      assert Graph.isolates(g) == [:e]
    end

    test "accepts a map of focal => [{neighbor, weight}], sorted by focal key" do
      g = Graph.from_adjacency(%{b: [{:a, 2.0}], a: [{:b, 1.0}, {:c, 3.0}], d: []})
      assert Graph.ids(g) == [:a, :b, :d, :c]

      assert Graph.to_adjacency(g) == [
               {:a, :b, 1.0},
               {:a, :c, 3.0},
               {:b, :a, 2.0},
               {:d, :d, 0.0},
               {:c, :c, 0.0}
             ]
    end

    test "raises on malformed entries" do
      assert_raise ArgumentError, ~r/adjacency/, fn -> Graph.from_adjacency([{:a, :b}]) end
      assert_raise ArgumentError, ~r/adjacency/, fn -> Graph.from_adjacency(:nope) end
    end
  end

  describe "accessors" do
    test "to_adjacency is in canonical order and includes island self-loops" do
      assert Graph.to_adjacency(hand()) == [
               {:a, :b, 1.0},
               {:a, :c, 2.0},
               {:b, :a, 3.0},
               {:c, :d, 0.5},
               {:d, :a, 2.5},
               {:d, :c, 1.5},
               {:e, :e, 0.0}
             ]
    end

    test "cardinalities align with ids; symmetric?; stats" do
      g = hand()
      assert Graph.cardinalities(g) == [2, 1, 1, 2, 0]
      refute Graph.symmetric?(g)
      assert Graph.s0(g) == 10.5
      assert Graph.s1(g) == 30.25
      assert Graph.s2(g) == 124.5
    end

    test "symmetric? honours tolerance" do
      g = Graph.from_arrays([0, 1], [1, 0], [1.0, 1.0 + 1.0e-14])
      assert Graph.symmetric?(g)
      refute Graph.symmetric?(g, 0.0)
    end
  end

  describe "transform/2" do
    test "accepts atoms and case-insensitive strings" do
      g = hand()
      assert Graph.transform(g, "r").transform == :r
      assert Graph.transform(g, "R").transform == :r
      assert Graph.transform(g, :b).transform == :b
      assert Graph.transform(g, :o) == g
    end

    test "is idempotent on the same label and chains from current weights" do
      r = Graph.transform(hand(), :r)
      assert Graph.transform(r, :r) == r

      assert Graph.to_adjacency(Graph.transform(r, :b)) ==
               Graph.to_adjacency(Graph.transform(hand(), :b))
    end

    test "does not mutate the source graph" do
      g = hand()
      _ = Graph.transform(g, :d)
      assert Graph.s0(g) == 10.5
      assert g.transform == :o
    end

    test "raises on unknown transforms and on restoring :o" do
      assert_raise ArgumentError, ~r/transform/, fn -> Graph.transform(hand(), :x) end
      assert_raise ArgumentError, ~r/transform/, fn -> Graph.transform(hand(), "zz") end

      assert_raise ArgumentError, ~r/original/, fn ->
        hand() |> Graph.transform(:r) |> Graph.transform(:o)
      end
    end
  end

  describe "lag/2" do
    test "computes Wy for lists and f64 binaries" do
      g = hand()
      assert Graph.lag(g, [1, 2, 3, 4, 5]) == [8.0, 3.0, 2.0, 7.0, 0.0]
      bin = for v <- [1.0, 2.0, 3.0, 4.0, 5.0], into: <<>>, do: <<v::float-64-native>>
      assert Graph.lag(g, bin) == [8.0, 3.0, 2.0, 7.0, 0.0]
    end

    test "raises on wrong length or non-numeric y" do
      assert_raise ArgumentError, ~r/length/, fn -> Graph.lag(hand(), [1.0, 2.0]) end

      assert_raise ArgumentError, ~r/expected a number/, fn ->
        Graph.lag(hand(), [1, 2, 3, 4, :x])
      end

      assert_raise ArgumentError, ~r/unsupported input/, fn -> Graph.lag(hand(), :nope) end
    end
  end

  test "inspect shows size and transform" do
    g = Graph.transform(hand(), :r)
    assert inspect(g) == "#Spatex.Graph<n: 5, nnz: 6, transform: :r>"
  end
end
