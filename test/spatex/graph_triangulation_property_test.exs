defmodule Spatex.GraphTriangulationPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph

  # Distinct integer-grid points jittered off the lattice; collinear draws are filtered out.
  defp points_gen do
    gen all(
          raw <-
            list_of(tuple({integer(0..5000), integer(0..5000)}), min_length: 4, max_length: 60)
        ) do
      raw |> Enum.uniq() |> Enum.map(fn {x, y} -> {x / 100, y / 100 + x / 1.0e6} end)
    end
  end

  defp edge_set(g),
    do: for({f, t, _} <- Graph.to_adjacency(g), f != t, into: MapSet.new(), do: {f, t})

  defp build(pts, method), do: Graph.build_triangulation(pts, method: method)

  property "relative neighborhood is a subgraph of Gabriel, which is a subgraph of Delaunay" do
    check all(pts <- points_gen(), length(pts) >= 4) do
      delaunay = edge_set(build(pts, :delaunay))
      gabriel = edge_set(build(pts, :gabriel))
      rng = edge_set(build(pts, :relative_neighborhood))

      assert MapSet.subset?(gabriel, delaunay)
      assert MapSet.subset?(rng, gabriel)
    end
  end

  property "triangulation graphs are symmetric without self edges" do
    check all(
            pts <- points_gen(),
            method <- member_of([:delaunay, :gabriel, :relative_neighborhood])
          ) do
      g = build(pts, method)
      assert Graph.symmetric?(g)
      assert Enum.all?(Graph.to_adjacency(g), fn {f, t, _} -> f != t end)
    end
  end

  property "a Delaunay graph has at most 3n - 6 edges and connects every node" do
    check all(pts <- points_gen()) do
      g = build(pts, :delaunay)
      n = length(pts)
      assert MapSet.size(edge_set(g)) <= 2 * (3 * n - 6)
      assert Graph.isolates(g) == []
    end
  end
end
