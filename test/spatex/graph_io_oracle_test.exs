defmodule Spatex.GraphIOOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @cases ~w(columbus_queen columbus_kernel labeled_island)
  @int_weight_cases ~w(columbus_queen)

  defp triples(fx) do
    ids = fx["ids"]
    at = &Enum.at(ids, &1)

    Enum.zip([fx["focal"], fx["neighbor"], fx["weight"]])
    |> Enum.map(fn {f, t, w} -> {at.(f), at.(t), w} end)
  end

  defp expected_triples(fx, key) do
    rt = fx[key]
    at = &Enum.at(rt["ids"], &1)

    Enum.zip([rt["focal"], rt["neighbor"], rt["weight"]])
    |> Enum.map(fn {f, t, w} -> {at.(f), at.(t), w} end)
  end

  defp assert_same(graph, expected) do
    actual = Graph.to_adjacency(graph)

    assert Enum.map(actual, &{elem(&1, 0), elem(&1, 1)}) ==
             Enum.map(expected, &{elem(&1, 0), elem(&1, 1)})

    Reference.assert_close(
      Enum.map(actual, &elem(&1, 2)),
      Enum.map(expected, &elem(&1, 2)),
      1.0e-12
    )
  end

  defp source_graph(fx), do: Graph.from_adjacency(triples(fx), ids: fx["ids"])

  for name <- @cases do
    describe "fixture #{name}" do
      setup do
        fx = Reference.load!("io/#{unquote(name)}.json")
        {:ok, fx: fx, g: source_graph(fx)}
      end

      test "reads libpysal's GAL file", %{fx: fx} do
        g = Graph.read_gal(Reference.fixture_path("io/#{unquote(name)}.gal"))
        assert Graph.ids(g) == fx["gal_roundtrip"]["ids"]
        assert_same(g, expected_triples(fx, "gal_roundtrip"))
      end

      test "reads libpysal's GWT file", %{fx: fx} do
        g = Graph.read_gwt(Reference.fixture_path("io/#{unquote(name)}.gwt"))
        assert Graph.ids(g) == fx["gwt_roundtrip"]["ids"]
        # GWT is written with 7 decimals.
        actual = Graph.to_adjacency(g)
        expected = expected_triples(fx, "gwt_roundtrip")

        assert Enum.map(actual, &{elem(&1, 0), elem(&1, 1)}) ==
                 Enum.map(expected, &{elem(&1, 0), elem(&1, 1)})

        Reference.assert_close(
          Enum.map(actual, &elem(&1, 2)),
          Enum.map(expected, &elem(&1, 2)),
          1.0e-12
        )
      end

      test "writes GAL byte-identical to libpysal", %{g: g} do
        want = File.read!(Reference.fixture_path("io/#{unquote(name)}.gal"))
        assert IO.iodata_to_binary(Graph.IO.to_gal(g)) == want
      end

      # libpysal writes integer-typed weights (binary contiguity graphs) without decimals
      # (`1`), float weights with `%.7f`. Spatex stores floats only and always writes `%.7f`.
      if name in @int_weight_cases do
        test "writes GWT that libpysal's text parses to the same graph", %{g: g} do
          text = g |> Graph.IO.to_gwt() |> IO.iodata_to_binary()
          want = File.read!(Reference.fixture_path("io/#{unquote(name)}.gwt"))
          assert String.replace(text, ~r/ 1\.0000000$/m, " 1") == want
          assert_same(Graph.IO.parse_gwt(text), Graph.to_adjacency(Graph.IO.parse_gwt(want)))
        end
      else
        test "writes GWT byte-identical to libpysal", %{g: g} do
          want = File.read!(Reference.fixture_path("io/#{unquote(name)}.gwt"))
          assert IO.iodata_to_binary(Graph.IO.to_gwt(g)) == want
        end
      end
    end
  end
end
