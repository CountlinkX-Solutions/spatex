defmodule Spatex.GraphIOPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph

  # Random directed graphs on 1..12 integer-id nodes, weights with up to 7 decimals.
  defp graph_gen do
    gen all(
          n <- integer(1..12),
          raw <-
            list_of(
              tuple({integer(0..(n - 1)), integer(0..(n - 1)), integer(1..9_999_999)}),
              max_length: 40
            )
        ) do
      edges =
        raw
        |> Enum.reject(fn {f, t, _} -> f == t end)
        |> Enum.uniq_by(fn {f, t, _} -> {f, t} end)
        |> Enum.map(fn {f, t, w} -> {f, t, w / 1.0e7} end)

      Graph.from_adjacency(edges, ids: Enum.to_list(0..(n - 1)))
    end
  end

  defp pattern(g) do
    for {f, t, w} <- Graph.to_adjacency(g), w != 0.0, do: {f, t}
  end

  property "GAL roundtrip preserves ids and adjacency (weights become 1)" do
    check all(g <- graph_gen()) do
      back = g |> Graph.IO.to_gal() |> IO.iodata_to_binary() |> Graph.IO.parse_gal()
      assert Graph.ids(back) == Graph.ids(g)
      assert pattern(back) == pattern(g)
      assert Graph.isolates(back) == Graph.isolates(g)
    end
  end

  property "GWT roundtrip preserves adjacency and weights to 7 decimals" do
    check all(g <- graph_gen()) do
      back = g |> Graph.IO.to_gwt() |> IO.iodata_to_binary() |> Graph.IO.parse_gwt()
      assert MapSet.new(Graph.ids(back)) == MapSet.new(Graph.ids(g))
      assert pattern(back) == pattern(g)

      expected = Map.new(Graph.to_adjacency(g), fn {f, t, w} -> {{f, t}, w} end)

      for {f, t, w} <- Graph.to_adjacency(back) do
        assert_in_delta w, Map.fetch!(expected, {f, t}), 5.0e-8
      end
    end
  end

  property "string ids roundtrip with ids: :string" do
    check all(g <- graph_gen()) do
      s = Graph.from_adjacency(for({f, t, w} <- Graph.to_adjacency(g), do: {"n#{f}", "n#{t}", w}))
      back = s |> Graph.IO.to_gal() |> IO.iodata_to_binary() |> Graph.IO.parse_gal(ids: :string)
      assert pattern(back) == pattern(s)
    end
  end
end
