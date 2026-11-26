defmodule Spatex.GraphKernelPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph

  @compact [:triangular, :parabolic, :bisquare, :tricube, :cosine, :boxcar]

  defp points_gen(min, max) do
    gen all(
          raw <-
            list_of(tuple({integer(0..2000), integer(0..2000)}), min_length: min, max_length: max)
        ) do
      raw |> Enum.uniq() |> Enum.map(fn {x, y} -> {x / 100, y / 100} end)
    end
  end

  property "compact kernel weights stay within [0, 1]" do
    check all(
            pts <- points_gen(3, 30),
            kernel <- member_of(@compact),
            h <- float(min: 0.5, max: 15.0),
            length(pts) >= 3
          ) do
      g = Graph.build_kernel(pts, kernel: kernel, bandwidth: h)
      assert Enum.all?(Graph.to_adjacency(g), fn {_, _, w} -> w >= 0.0 and w <= 1.0 end)
    end
  end

  property "decay normalises compact kernel weights into [0, 1]" do
    check all(
            pts <- points_gen(3, 30),
            kernel <- member_of(@compact),
            h <- float(min: 0.5, max: 15.0),
            length(pts) >= 3
          ) do
      g = Graph.build_kernel(pts, kernel: kernel, bandwidth: h, decay: true)
      assert Enum.all?(Graph.to_adjacency(g), fn {_, _, w} -> w >= 0.0 and w <= 1.0 + 1.0e-12 end)
    end
  end

  property "adaptive bandwidth gives each node at least k neighbors for non-boxcar kernels" do
    check all(
            pts <- points_gen(4, 40),
            kernel <- member_of(@compact ++ [:gaussian, :exponential]),
            k <- integer(1..(length(pts) - 1)),
            length(pts) >= 4
          ) do
      g = Graph.build_kernel(pts, kernel: kernel, bandwidth: :adaptive, k: k)
      assert Enum.all?(Graph.cardinalities(g), &(&1 >= k))
    end
  end

  property "fixed bandwidth graphs are symmetric and match a brute-force evaluation" do
    check all(pts <- points_gen(3, 30), h <- float(min: 0.5, max: 10.0), length(pts) >= 3) do
      g = Graph.build_kernel(pts, kernel: :triangular, bandwidth: h)
      assert Graph.symmetric?(g)

      indexed = Enum.with_index(pts)

      expected =
        for {{px, py}, i} <- indexed, {{qx, qy}, j} <- indexed, i != j do
          d = :math.sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy))
          {i, j, 1 - min(d / h, 1.0), d}
        end
        |> Enum.filter(fn {_, _, w, d} -> d <= h and w > 0.0 end)
        |> Enum.map(fn {i, j, w, _} -> {i, j, w} end)

      actual = Enum.reject(Graph.to_adjacency(g), fn {f, t, _} -> f == t end)

      assert Enum.map(actual, &{elem(&1, 0), elem(&1, 1)}) ==
               Enum.map(expected, &{elem(&1, 0), elem(&1, 1)})

      for {{_, _, a}, {_, _, b}} <- Enum.zip(actual, expected) do
        assert_in_delta a, b, 1.0e-12
      end
    end
  end
end
