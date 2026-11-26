defmodule Spatex.GraphJitterTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  @pts [{0, 0}, {5, 1}, {0, 0}, {3, 3}, {3, 3}, {9, 2}]

  test "coincident: :jitter builds a knn graph over separated points" do
    g = Graph.build_knn(@pts, 2, coincident: :jitter, seed: 4)
    assert Graph.cardinalities(g) == List.duplicate(2, 6)
    adj = Graph.to_adjacency(g)
    # Each coincident partner is the nearest neighbor of the other.
    assert {0, 2, 1.0} in adj and {2, 0, 1.0} in adj and {3, 4, 1.0} in adj
    assert Graph.to_adjacency(Graph.build_knn(@pts, 2, coincident: :jitter, seed: 4)) == adj

    assert_raise ArgumentError, ~r/coincident/, fn -> Graph.build_knn(@pts, 2) end
  end

  test "jittered kernels take their distances from the separated points" do
    g =
      Graph.build_kernel(@pts,
        kernel: :triangular,
        k: 1,
        bandwidth: :adaptive,
        coincident: :jitter
      )

    [{0, 2, w} | _] = g |> Graph.to_adjacency() |> Enum.filter(&(elem(&1, 0) == 0))
    assert w > 0 and w < 1.0e-6
  end

  test "invalid coincident and seed options raise" do
    assert_raise ArgumentError, ~r/:coincident must be one of/, fn ->
      Graph.build_knn(@pts, 2, coincident: :shake)
    end

    assert_raise ArgumentError, ~r/:seed must be a non-negative integer/, fn ->
      Graph.build_kernel(@pts, k: 2, coincident: :jitter, seed: -1)
    end
  end
end
