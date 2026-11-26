defmodule Spatex.GraphContiguityTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph
  alias Spatex.TestWKB, as: W

  # 2x2 grid, row-major: 0 1 / 2 3
  defp grid, do: [W.cell(0, 0), W.cell(1, 0), W.cell(0, 1), W.cell(1, 1)]

  defp neighbor_map(graph) do
    graph
    |> Graph.to_adjacency()
    |> Enum.reject(fn {_f, _t, w} -> w == 0.0 end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
  end

  describe "build_contiguity/2" do
    test "rook is the default and shares edges only" do
      graph = Graph.build_contiguity(grid())
      assert neighbor_map(graph) == %{0 => [1, 2], 1 => [0, 3], 2 => [0, 3], 3 => [1, 2]}

      assert Graph.build_contiguity(grid(), rook: true) |> Graph.to_adjacency() ==
               Graph.to_adjacency(graph)
    end

    test "queen also links corner neighbors" do
      graph = Graph.build_contiguity(grid(), rook: false)

      assert neighbor_map(graph) == %{
               0 => [1, 2, 3],
               1 => [0, 2, 3],
               2 => [0, 1, 3],
               3 => [0, 1, 2]
             }
    end

    test "weights are 1.0, the graph is untransformed and ids default to 0..n-1" do
      graph = Graph.build_contiguity(grid())
      assert Graph.n(graph) == 4
      assert Graph.ids(graph) == [0, 1, 2, 3]
      assert Enum.all?(Graph.to_adjacency(graph), fn {_, _, w} -> w == 1.0 end)
      assert Graph.nnz(graph) == 8
    end

    test "islands are zero-weight self-loops" do
      graph = Graph.build_contiguity([W.cell(0, 0), W.cell(1, 0), W.cell(9, 9)])
      assert Graph.isolates(graph) == [2]
      assert Graph.cardinalities(graph) == [1, 1, 0]
      assert {2, 2, 0.0} in Graph.to_adjacency(graph)
    end

    test "custom ids label nodes and keep the input order" do
      graph = Graph.build_contiguity(grid(), ids: ["a", "b", "c", "d"])
      assert Graph.ids(graph) == ["a", "b", "c", "d"]
      assert neighbor_map(graph)["a"] == ["b", "c"]
    end

    test "a single polygon is an island" do
      graph = Graph.build_contiguity([W.cell(0, 0)])
      assert Graph.isolates(graph) == [0]
    end

    test "multipolygons are accepted" do
      multi =
        W.multipolygon([
          [[{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}]],
          [[{5, 5}, {6, 5}, {6, 6}, {5, 6}, {5, 5}]]
        ])

      graph = Graph.build_contiguity([multi, W.cell(1, 0), W.cell(6, 5)])
      assert neighbor_map(graph) == %{0 => [1, 2], 1 => [0], 2 => [0]}
    end

    test "rejects point geometries naming the index; lines need the vertex method" do
      assert_raise ArgumentError, ~r/geometry 1.*Point/, fn ->
        Graph.build_contiguity([W.cell(0, 0), W.point(1, 1)])
      end

      lines = [W.linestring([{0, 0}, {1, 1}]), W.linestring([{1, 1}, {2, 0}])]
      assert Graph.cardinalities(Graph.build_contiguity(lines, rook: false)) == [1, 1]

      assert_raise ArgumentError, ~r/geometry 0.*Polygon/, fn ->
        Graph.build_contiguity(lines, strict: true)
      end
    end

    test "rejects nil, non-binary and malformed entries" do
      assert_raise ArgumentError, ~r/geometry 1.*nil/, fn ->
        Graph.build_contiguity([W.cell(0, 0), nil])
      end

      assert_raise ArgumentError, ~r/geometry 0/, fn -> Graph.build_contiguity([:polygon]) end

      assert_raise ArgumentError, ~r/geometry 0/, fn ->
        Graph.build_contiguity([binary_part(W.cell(0, 0), 0, 20)])
      end
    end

    test "rejects non-finite coordinates" do
      nan =
        <<0::little-float-64, 0::little-64>>
        |> then(fn _ -> <<0x7FF8000000000000::little-64>> end)

      pt = fn x, y -> <<x::little-float-64, y::little-float-64>> end

      nan_ring =
        <<1, 3::little-32, 1::little-32, 4::little-32>> <>
          pt.(0, 0) <> <<nan::binary, 0::little-float-64>> <> pt.(1, 1) <> pt.(0, 0)

      assert_raise ArgumentError, ~r/geometry 0: non-finite/, fn ->
        Graph.build_contiguity([nan_ring])
      end
    end

    test "rejects a non-list or empty input" do
      assert_raise ArgumentError, ~r/list/, fn -> Graph.build_contiguity(:nope) end
      assert_raise ArgumentError, ~r/at least one/, fn -> Graph.build_contiguity([]) end
    end

    test "validates ids and options" do
      assert_raise ArgumentError, ~r/ids/, fn -> Graph.build_contiguity(grid(), ids: [1, 2]) end

      assert_raise ArgumentError, ~r/unique/, fn ->
        Graph.build_contiguity(grid(), ids: [1, 1, 2, 3])
      end

      assert_raise ArgumentError, ~r/rook/, fn -> Graph.build_contiguity(grid(), rook: :yes) end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_contiguity(grid(), bogus: 1)
      end
    end

    test "strict and by_perimeter validate their options" do
      assert %Graph{} = Graph.build_contiguity(grid(), strict: true)

      for key <- [:strict, :by_perimeter] do
        assert_raise ArgumentError, ~r/:#{key} must be a boolean/, fn ->
          Graph.build_contiguity(grid(), [{key, 1}])
        end
      end
    end
  end

  describe "build_block_contiguity/2" do
    test "members of a regime are neighbors, in input order" do
      graph = Graph.build_block_contiguity(["a", "b", "a", "c", "b"])
      assert Graph.ids(graph) == [0, 1, 2, 3, 4]
      assert neighbor_map(graph) == %{0 => [2], 1 => [4], 2 => [0], 4 => [1]}
      assert Graph.isolates(graph) == [3]
    end

    test "custom ids and non-string regimes" do
      graph = Graph.build_block_contiguity([1, 2, 1, 1.0], ids: [:u, :v, :w, :x])
      assert Graph.ids(graph) == [:u, :v, :w, :x]
      assert neighbor_map(graph) == %{u: [:w], w: [:u]}
      assert Graph.isolates(graph) |> Enum.sort() == [:v, :x]
    end

    test "weights are 1.0" do
      graph = Graph.build_block_contiguity([:a, :a, :a])
      assert Graph.nnz(graph) == 6
      assert Enum.all?(Graph.to_adjacency(graph), fn {_, _, w} -> w == 1.0 end)
    end

    test "rejects nil regimes, empty input and bad ids" do
      assert_raise ArgumentError, ~r/regime 1.*nil/, fn ->
        Graph.build_block_contiguity([:a, nil])
      end

      assert_raise ArgumentError, ~r/at least one/, fn -> Graph.build_block_contiguity([]) end
      assert_raise ArgumentError, ~r/list/, fn -> Graph.build_block_contiguity(:a) end

      assert_raise ArgumentError, ~r/ids/, fn ->
        Graph.build_block_contiguity([:a, :b], ids: [1])
      end

      assert_raise ArgumentError, ~r/unique/, fn ->
        Graph.build_block_contiguity([:a, :b], ids: [1, 1])
      end
    end
  end
end
