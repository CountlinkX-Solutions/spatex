defmodule Spatex.GraphFuzzyContiguityTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph
  alias Spatex.TestWKB, as: W

  # Unit squares at x = 0, 1.5, 4 and 20 (an island): gaps 0.5, 1.5 and 14.5.
  defp row do
    [W.cell(0, 0), W.cell(1.5, 0), W.cell(4, 0), W.cell(20, 0)]
  end

  defp neighbor_map(graph) do
    graph
    |> Graph.to_adjacency()
    |> Enum.reject(fn {f, n, _w} -> f == n end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
  end

  describe "build_fuzzy_contiguity/2" do
    test "without a buffer neighbors are the polygons that intersect" do
      assert neighbor_map(
               Graph.build_fuzzy_contiguity([W.cell(0, 0), W.cell(1, 0), W.cell(5, 5)])
             ) ==
               %{0 => [1], 1 => [0]}
    end

    test "a buffer closes gaps of up to twice its size" do
      graph = Graph.build_fuzzy_contiguity(row(), buffer: 0.25)
      assert neighbor_map(graph) == %{0 => [1], 1 => [0]}
      assert Graph.isolates(graph) == [2, 3]

      graph = Graph.build_fuzzy_contiguity(row(), buffer: 0.75)
      assert neighbor_map(graph) == %{0 => [1], 1 => [0, 2], 2 => [1]}
      assert Graph.isolates(graph) == [3]
    end

    test "tolerance is a share of half the shorter side of the total bounds" do
      # bounds 21 x 1: buffer = tolerance * 0.5 * 1
      assert neighbor_map(Graph.build_fuzzy_contiguity(row(), tolerance: 0.5)) == %{
               0 => [1],
               1 => [0]
             }

      assert neighbor_map(Graph.build_fuzzy_contiguity(row(), tolerance: 1.5)) == %{
               0 => [1],
               1 => [0, 2],
               2 => [1]
             }
    end

    test "predicates are directed: contains and within are converses" do
      nested = [W.box(0, 0, 4, 4), W.box(1, 1, 2, 2)]

      assert neighbor_map(Graph.build_fuzzy_contiguity(nested, predicate: :contains)) == %{
               0 => [1]
             }

      assert neighbor_map(Graph.build_fuzzy_contiguity(nested, predicate: :within)) == %{1 => [0]}

      assert Graph.build_fuzzy_contiguity(nested, predicate: :crosses) |> Graph.isolates() == [
               0,
               1
             ]
    end

    test "predicate: nil compares bounding boxes, which a buffer grows" do
      diagonal = [W.cell(0, 0), W.cell(2, 2)]
      assert Graph.build_fuzzy_contiguity(diagonal, predicate: nil) |> Graph.isolates() == [0, 1]

      assert neighbor_map(Graph.build_fuzzy_contiguity(diagonal, predicate: nil, buffer: 0.5)) ==
               %{0 => [1], 1 => [0]}
    end

    test "weights are 1.0, ids default to 0..n-1 and can be given" do
      graph = Graph.build_fuzzy_contiguity(row(), buffer: 0.25, ids: [:a, :b, :c, :d])
      assert Graph.ids(graph) == [:a, :b, :c, :d]
      assert {:a, :b, 1.0} in Graph.to_adjacency(graph)
      assert Graph.build_fuzzy_contiguity(row()) |> Graph.ids() == [0, 1, 2, 3]
    end

    test "multipolygons and holes" do
      ring =
        W.polygon([
          [{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 0}],
          [{3, 3}, {7, 3}, {7, 7}, {3, 7}, {3, 3}]
        ])

      inside = W.box(4, 4, 6, 6)
      assert Graph.build_fuzzy_contiguity([ring, inside]) |> Graph.isolates() == [0, 1]

      assert neighbor_map(Graph.build_fuzzy_contiguity([ring, inside], buffer: 0.5)) ==
               %{0 => [1], 1 => [0]}

      multi =
        W.multipolygon([
          [[{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}]],
          [[{5, 0}, {6, 0}, {6, 1}, {5, 1}, {5, 0}]]
        ])

      assert neighbor_map(Graph.build_fuzzy_contiguity([multi, W.cell(6, 0)])) == %{
               0 => [1],
               1 => [0]
             }
    end

    test "tolerance and buffer are exclusive, finite and not negative" do
      assert_raise ArgumentError, ~r/only one of :tolerance and :buffer/, fn ->
        Graph.build_fuzzy_contiguity(row(), tolerance: 0.1, buffer: 1.0)
      end

      for {key, value} <- [tolerance: -0.1, buffer: -1.0, buffer: :inf, tolerance: "0.1"] do
        assert_raise ArgumentError, ~r/:#{key}/, fn ->
          Graph.build_fuzzy_contiguity(row(), [{key, value}])
        end
      end
    end

    test "unsupported and unknown predicates raise" do
      assert_raise ArgumentError, ~r/dwithin.*not ported/, fn ->
        Graph.build_fuzzy_contiguity(row(), predicate: :dwithin)
      end

      assert_raise ArgumentError, ~r/:predicate/, fn ->
        Graph.build_fuzzy_contiguity(row(), predicate: :disjoint)
      end

      assert_raise ArgumentError, ~r/:predicate/, fn ->
        Graph.build_fuzzy_contiguity(row(), predicate: "intersects")
      end
    end

    test "a buffer only works with :intersects and nil" do
      assert_raise ArgumentError, ~r/buffer.*intersects/, fn ->
        Graph.build_fuzzy_contiguity(row(), buffer: 1.0, predicate: :touches)
      end

      assert %Graph{} = Graph.build_fuzzy_contiguity(row(), buffer: 0.0, predicate: :touches)
    end

    test "validates geometries, ids and options" do
      assert_raise ArgumentError, ~r/at least one/, fn -> Graph.build_fuzzy_contiguity([]) end
      assert_raise ArgumentError, ~r/list of WKB/, fn -> Graph.build_fuzzy_contiguity(:x) end

      assert_raise ArgumentError, ~r/geometry 1.*WKB/, fn ->
        Graph.build_fuzzy_contiguity([W.cell(0, 0), nil])
      end

      assert_raise ArgumentError, ~r/geometry 1.*Polygon/, fn ->
        Graph.build_fuzzy_contiguity([W.cell(0, 0), W.point(1, 1)])
      end

      assert_raise ArgumentError, ~r/ids/, fn -> Graph.build_fuzzy_contiguity(row(), ids: [1]) end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_fuzzy_contiguity(row(), bogus: 1)
      end
    end
  end
end
