defmodule Spatex.GraphKnnOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12
  @fixtures "test/reference/fixtures"
            |> Path.join("weights/{knn,distance_band}_*.json")
            |> Path.wildcard()
            |> Enum.map(&Path.relative_to(&1, "test/reference/fixtures"))

  defp build(points, %{"k" => k, "coincident" => c}),
    do: Graph.build_knn(points, k, coincident: Spatex.Reference.fixture_atom(c))

  defp build(points, %{"threshold" => t, "binary" => true}),
    do: Graph.build_distance_band(points, t)

  defp build(points, %{"threshold" => t, "alpha" => alpha}),
    do: Graph.build_distance_band(points, t, binary: false, alpha: alpha)

  # Sorted neighbor distances per row: the tie-independent view of a KNN graph.
  defp row_distances(adjacency, points) do
    coords = List.to_tuple(Enum.map(points, &List.to_tuple/1))

    adjacency
    |> Enum.group_by(&elem(&1, 0), fn {f, t, _w} ->
      {fx, fy} = elem(coords, f)
      {tx, ty} = elem(coords, t)
      :math.sqrt((fx - tx) * (fx - tx) + (fy - ty) * (fy - ty))
    end)
    |> Enum.sort()
    |> Enum.map(fn {_row, ds} -> Enum.sort(ds) end)
  end

  test "fixtures are present" do
    assert length(@fixtures) == 11
  end

  for rel <- @fixtures do
    describe rel do
      setup do
        {:ok, fx: Reference.load!(unquote(rel))}
      end

      test "every case matches libpysal", %{fx: fx} do
        assert fx["n"] == length(fx["points"])

        for c <- fx["cases"] do
          g = build(fx["points"], c["params"])
          actual = Graph.to_adjacency(g)
          expected = c["adjacency"]
          label = "#{unquote(rel)} #{c["name"]}"

          assert Graph.n(g) == fx["n"], label
          assert Graph.cardinalities(g) == c["cardinalities"], label
          assert Graph.isolates(g) == c["isolates"], label

          if c["ties"] do
            expected_triples =
              Enum.zip([expected["focal"], expected["neighbor"], expected["weight"]])

            Reference.assert_close(
              row_distances(actual, fx["points"]),
              row_distances(expected_triples, fx["points"]),
              @tol
            )
          else
            assert Enum.map(actual, &elem(&1, 0)) == expected["focal"], label
            assert Enum.map(actual, &elem(&1, 1)) == expected["neighbor"], label
            Reference.assert_close(Enum.map(actual, &elem(&1, 2)), expected["weight"], @tol)
          end
        end
      end
    end
  end
end
