defmodule Spatex.GraphVoronoiOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @clips %{"bounding_box" => :bounding_box, "convex_hull" => :convex_hull, nil => nil}

  test "voronoi rook contiguity matches libpysal for every clip" do
    fx = Reference.load!("weights/voronoi.json")

    for c <- fx["cases"] do
      pts = Enum.map(c["points"], &List.to_tuple/1)

      g =
        Graph.build_triangulation(pts,
          method: :voronoi,
          clip: Map.fetch!(@clips, c["params"]["clip"])
        )

      adj = Graph.to_adjacency(g)

      assert Graph.cardinalities(g) == c["cardinalities"], c["name"]
      assert Enum.map(adj, &elem(&1, 0)) == c["adjacency"]["focal"], c["name"]
      assert Enum.map(adj, &elem(&1, 1)) == c["adjacency"]["neighbor"], c["name"]
      assert Enum.all?(adj, &(elem(&1, 2) == 1.0)), c["name"]
    end
  end
end
