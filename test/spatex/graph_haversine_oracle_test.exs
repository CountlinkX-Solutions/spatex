defmodule Spatex.GraphHaversineOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  # libpysal leaves haversine distances in radians; the fixtures hold only unit-free outputs
  # (neighbor sets, kernel weights with data-driven bandwidths), see the generator.
  @tol 1.0e-12
  @fixtures ["weights/haversine_europe.json", "weights/haversine_antimeridian.json"]
  @kernels %{"triangular" => :triangular, "gaussian" => :gaussian, "bisquare" => :bisquare}

  defp points(fx), do: Enum.map(fx["points"], &List.to_tuple/1)

  defp build(pts, %{"kernel" => kernel} = p) do
    Graph.build_kernel(pts,
      kernel: Map.fetch!(@kernels, kernel),
      k: p["k"],
      bandwidth: if(p["bandwidth"] == "adaptive", do: :adaptive),
      metric: :haversine
    )
  end

  defp build(pts, %{"k" => k}), do: Graph.build_knn(pts, k, metric: :haversine)

  for rel <- @fixtures do
    describe rel do
      setup do
        {:ok, fx: Reference.load!(unquote(rel))}
      end

      test "every case matches libpysal", %{fx: fx} do
        pts = points(fx)

        for c <- fx["cases"] do
          g = build(pts, c["params"])
          actual = Graph.to_adjacency(g)
          expected = c["adjacency"]
          label = "#{unquote(rel)} #{c["name"]}"

          assert Graph.cardinalities(g) == c["cardinalities"], label
          assert Graph.isolates(g) == c["isolates"], label
          assert Enum.map(actual, &elem(&1, 0)) == expected["focal"], label
          assert Enum.map(actual, &elem(&1, 1)) == expected["neighbor"], label
          Reference.assert_close(Enum.map(actual, &elem(&1, 2)), expected["weight"], @tol)
        end
      end
    end
  end
end
