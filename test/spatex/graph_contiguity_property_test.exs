defmodule Spatex.GraphContiguityPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph
  alias Spatex.TestWKB, as: W

  # A random subset of cells of a 6x6 lattice, in random order. Cells are unit squares, so
  # rook neighbors are at Manhattan distance 1 and queen neighbors at Chebyshev distance 1.
  defp cells_gen do
    # Draw with replacement then dedupe: `uniq_list_of` raises TooManyDuplicatesError when the
    # 36-cell domain is nearly exhausted by a long list.
    {integer(0..5), integer(0..5)}
    |> tuple()
    |> list_of(min_length: 1, max_length: 20)
    |> map(&Enum.uniq/1)
  end

  defp neighbor_sets(graph) do
    graph
    |> Graph.to_adjacency()
    |> Enum.filter(fn {_f, _t, w} -> w != 0.0 end)
    |> MapSet.new(fn {f, t, _w} -> {f, t} end)
  end

  property "contiguity graphs are symmetric, binary and free of self-neighbors" do
    check all(cells <- cells_gen(), rook <- boolean()) do
      graph =
        cells |> Enum.map(fn {x, y} -> W.cell(x, y) end) |> Graph.build_contiguity(rook: rook)

      assert Graph.symmetric?(graph)

      for {f, t, w} <- Graph.to_adjacency(graph) do
        assert w in [0.0, 1.0]
        if f == t, do: assert(w == 0.0)
      end
    end
  end

  property "queen neighbors contain rook neighbors" do
    check all(cells <- cells_gen()) do
      wkbs = Enum.map(cells, fn {x, y} -> W.cell(x, y) end)
      rook = wkbs |> Graph.build_contiguity(rook: true) |> neighbor_sets()
      queen = wkbs |> Graph.build_contiguity(rook: false) |> neighbor_sets()
      assert MapSet.subset?(rook, queen)
    end
  end

  property "lattice cells match brute-force Manhattan and Chebyshev adjacency" do
    check all(cells <- cells_gen(), rook <- boolean()) do
      graph =
        cells |> Enum.map(fn {x, y} -> W.cell(x, y) end) |> Graph.build_contiguity(rook: rook)

      indexed = Enum.with_index(cells)

      expected =
        for {{x1, y1}, i} <- indexed,
            {{x2, y2}, j} <- indexed,
            i != j,
            dx = abs(x1 - x2),
            dy = abs(y1 - y2),
            if(rook, do: dx + dy == 1, else: max(dx, dy) == 1),
            into: MapSet.new(),
            do: {i, j}

      assert neighbor_sets(graph) == expected
    end
  end

  property "block contiguity is symmetric, binary and transitive within regimes" do
    check all(regimes <- list_of(member_of([:a, :b, :c, :d]), min_length: 1, max_length: 15)) do
      graph = Graph.build_block_contiguity(regimes)
      assert Graph.symmetric?(graph)
      pairs = neighbor_sets(graph)
      indexed = Enum.with_index(regimes)

      expected =
        for {r1, i} <- indexed,
            {r2, j} <- indexed,
            i != j,
            r1 == r2,
            into: MapSet.new(),
            do: {i, j}

      assert pairs == expected
    end
  end
end
