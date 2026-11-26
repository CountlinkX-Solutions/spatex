defmodule Spatex.GraphKnnPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph

  # Distinct points: random floats on a fine grid, deduplicated.
  defp points_gen(min, max) do
    gen all(
          raw <-
            list_of(tuple({integer(0..2000), integer(0..2000)}), min_length: min, max_length: max)
        ) do
      raw |> Enum.uniq() |> Enum.map(fn {x, y} -> {x / 100, y / 100} end)
    end
  end

  property "KNN rows have exactly k neighbors, never the node itself" do
    check all(
            pts <- points_gen(3, 40),
            k <- integer(1..(length(pts) - 1)),
            length(pts) >= 3
          ) do
      g = Graph.build_knn(pts, k)
      assert Graph.cardinalities(g) == List.duplicate(k, length(pts))
      assert Enum.all?(Graph.to_adjacency(g), fn {f, t, w} -> f != t and w == 1.0 end)
    end
  end

  property "KNN agrees with a brute-force (distance, index) ranking" do
    check all(pts <- points_gen(3, 30), k <- integer(1..(length(pts) - 1)), length(pts) >= 3) do
      indexed = Enum.with_index(pts)

      expected =
        for {{px, py}, i} <- indexed do
          indexed
          |> Enum.reject(fn {_, j} -> j == i end)
          |> Enum.sort_by(fn {{qx, qy}, j} ->
            {(px - qx) * (px - qx) + (py - qy) * (py - qy), j}
          end)
          |> Enum.take(k)
          |> Enum.map(fn {_, j} -> j end)
          |> Enum.sort()
        end

      actual =
        pts
        |> Graph.build_knn(k)
        |> Graph.to_adjacency()
        |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
        |> Enum.sort()
        |> Enum.map(&elem(&1, 1))

      assert actual == expected
    end
  end

  property "binary distance band is symmetric and has no self edges" do
    check all(pts <- points_gen(1, 40), t <- float(min: 0.5, max: 8.0)) do
      g = Graph.build_distance_band(pts, t)
      assert Graph.symmetric?(g)
      assert Enum.all?(Graph.to_adjacency(g), fn {f, t2, w} -> f != t2 or w == 0.0 end)
    end
  end

  property "distance band neighbors are exactly the pairs within the threshold" do
    check all(pts <- points_gen(1, 30), t <- float(min: 0.5, max: 8.0)) do
      indexed = Enum.with_index(pts)

      expected =
        for {{px, py}, i} <- indexed,
            {{qx, qy}, j} <- indexed,
            i != j,
            (px - qx) * (px - qx) + (py - qy) * (py - qy) <= t * t,
            do: {i, j, 1.0}

      actual =
        pts
        |> Graph.build_distance_band(t)
        |> Graph.to_adjacency()
        |> Enum.reject(fn {f, t2, _} -> f == t2 end)

      assert actual == Enum.sort(expected)
    end
  end

  property "inverse-distance weights are positive and symmetric" do
    check all(pts <- points_gen(2, 25), t <- float(min: 1.0, max: 8.0)) do
      g = Graph.build_distance_band(pts, t, binary: false)
      assert Graph.symmetric?(g)
      assert Enum.all?(Graph.to_adjacency(g), fn {_, _, w} -> w >= 0.0 end)
    end
  end
end
