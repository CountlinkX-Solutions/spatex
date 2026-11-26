defmodule Spatex.GraphMetricTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  # Madrid, Paris, London, Berlin as {lon, lat}.
  @cities [{-3.70, 40.42}, {2.35, 48.86}, {-0.13, 51.51}, {13.40, 52.52}]

  test "haversine distance band thresholds are kilometres" do
    # Paris - London is about 344 km; every other pair is over 870 km.
    g = Graph.build_distance_band(@cities, 400, metric: :haversine)
    assert Graph.isolates(g) == [0, 3]

    g = Graph.build_distance_band(@cities, 400, metric: :haversine, binary: false)
    [{1, 2, w} | _] = g |> Graph.to_adjacency() |> Enum.filter(&(elem(&1, 0) == 1))
    assert_in_delta 1 / w, 343.5, 1.0
  end

  test "haversine knn and kernels run on the sphere" do
    # Across the antimeridian, 179 and -179 are 2 degrees apart.
    g = Graph.build_knn([{179.0, 0.0}, {-179.0, 0.0}, {175.0, 0.0}], 1, metric: :haversine)
    assert [{0, 1, 1.0} | _] = Graph.to_adjacency(g)

    g = Graph.build_kernel(@cities, kernel: :triangular, bandwidth: 500, metric: :haversine)
    assert [{1, 2, w}] = g |> Graph.to_adjacency() |> Enum.filter(&(elem(&1, 0) == 1))
    assert_in_delta w, 1 - 343.5 / 500, 0.002
  end

  test "out-of-range coordinates and unknown metrics raise" do
    assert_raise ArgumentError, ~r/haversine range/, fn ->
      Graph.build_knn([{0.0, 0.0}, {10.0, 95.0}], 1, metric: :haversine)
    end

    assert_raise ArgumentError, ~r/:metric must be :euclidean or :haversine/, fn ->
      Graph.build_kernel(@cities, k: 2, metric: :manhattan)
    end
  end
end
