defmodule Spatex.GraphContiguityStrictOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  test "strict, by_perimeter and line contiguity match libpysal" do
    fx = Reference.load!("weights/contiguity_strict.json")

    for set <- fx["sets"], c <- set["cases"] do
      wkbs = Enum.map(set["wkb_hex"], &Base.decode16!(&1, case: :mixed))
      p = c["params"]
      label = "#{set["set"]} #{inspect(p)}"

      g =
        Graph.build_contiguity(wkbs,
          rook: p["rook"],
          strict: p["strict"],
          by_perimeter: p["by_perimeter"]
        )

      adj = Graph.to_adjacency(g)
      assert Graph.cardinalities(g) == c["cardinalities"], label
      assert Enum.map(adj, &elem(&1, 0)) == c["adjacency"]["focal"], label
      assert Enum.map(adj, &elem(&1, 1)) == c["adjacency"]["neighbor"], label
      Reference.assert_close(Enum.map(adj, &elem(&1, 2)), c["adjacency"]["weight"], 1.0e-12)
    end
  end
end
