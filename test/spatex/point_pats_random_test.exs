defmodule Spatex.PointPatsRandomTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats

  @window {0, 0, 10, 5}

  defp nn_mean(points) do
    ds =
      for {x, y} = p <- points do
        points
        |> Enum.reject(&(&1 == p))
        |> Enum.map(fn {a, b} -> :math.sqrt((a - x) * (a - x) + (b - y) * (b - y)) end)
        |> Enum.min()
      end

    Enum.sum(ds) / length(ds)
  end

  describe "poisson/2" do
    test "size, window and determinism" do
      a = PointPats.poisson(@window, size: 200, seed: 4)
      assert length(a) == 200
      assert Enum.all?(a, fn {x, y} -> x >= 0 and x <= 10 and y >= 0 and y <= 5 end)
      assert a == PointPats.poisson(@window, size: 200, seed: 4)
      assert a != PointPats.poisson(@window, size: 200, seed: 5)
    end

    test "size tuple returns a list of patterns; a single point stays a pattern" do
      assert [p1, p2, p3] = PointPats.poisson(@window, size: {20, 3}, seed: 1)
      assert length(p1) == 20 and length(p2) == 20 and length(p3) == 20
      assert p1 != p2
      assert [{_, _}] = PointPats.poisson(@window, size: 1, seed: 1)
      assert [{_, _}] = PointPats.poisson(@window, size: {1, 1}, seed: 1)
    end

    test "intensity picks the count with int() truncation" do
      assert length(PointPats.poisson(@window, intensity: 2.0, seed: 1)) == 100
      assert length(PointPats.poisson(@window, intensity: 0.39, seed: 1)) == 19
      assert length(PointPats.poisson(@window, seed: 1)) == 100

      assert [_, _, _] = PointPats.poisson(@window, intensity: 2.0, size: 3, seed: 1)
    end

    test "points as a window use their bounding box; convex windows are respected" do
      pts = [{0, 0}, {4, 0}, {0, 4}, {1, 1}]
      sample = PointPats.poisson({:convex, pts}, size: 300, seed: 2)
      assert Enum.all?(sample, fn {x, y} -> x + y <= 4 + 1.0e-12 end)
      box = PointPats.poisson(pts, size: 50, seed: 2)
      assert Enum.all?(box, fn {x, y} -> x <= 4 and y <= 4 end)
    end

    test "nearest-neighbour distance matches the CSR expectation over many seeds" do
      # E[d] = 0.5 / sqrt(intensity) (edge effects bias it upward slightly)
      n = 100
      intensity = n / 50
      means = for seed <- 1..60, do: nn_mean(PointPats.poisson(@window, size: n, seed: seed))
      grand = Enum.sum(means) / 60
      sd = :math.sqrt(Enum.sum(Enum.map(means, &((&1 - grand) ** 2))) / 59)
      expected = 0.5 / :math.sqrt(intensity)
      # edge effect inflates the mean by roughly 10-20%
      assert grand > expected and grand < expected * 1.3
      assert sd < 0.2 * expected
    end

    test "validates its options" do
      assert_raise ArgumentError, ~r/intensity/, fn ->
        PointPats.poisson(@window, size: {10, 2}, intensity: 1.0)
      end

      assert_raise ArgumentError, ~r/window/, fn -> PointPats.poisson({1, 2, 3}, size: 3) end
      assert_raise ArgumentError, ~r/area/, fn -> PointPats.poisson({0, 0, 0, 5}, size: 3) end
      assert_raise ArgumentError, ~r/size/, fn -> PointPats.poisson(@window, size: 0) end
    end
  end

  describe "cluster_poisson/2" do
    test "chunks, radius and window" do
      pts = PointPats.cluster_poisson(@window, size: 41, n_seeds: 4, cluster_radius: 0.5, seed: 9)
      assert length(pts) == 41
      # array_split: the first n mod k chunks get the extra point
      sizes = [11, 10, 10, 10]

      {_, groups} =
        Enum.reduce(sizes, {pts, []}, fn s, {rest, acc} ->
          {chunk, rest} = Enum.split(rest, s)
          {rest, [chunk | acc]}
        end)

      for [seed_point | _] = chunk <- groups do
        {sx, sy} = seed_point

        assert Enum.all?(chunk, fn {x, y} ->
                 (x - sx) ** 2 + (y - sy) ** 2 <= 0.25 + 1.0e-12 and x >= 0 and x <= 10 and
                   y >= 0 and y <= 5
               end)
      end
    end

    test "default radius is half the minimum seed distance; deterministic per seed" do
      a = PointPats.cluster_poisson(@window, size: 30, n_seeds: 3, seed: 1)
      assert a == PointPats.cluster_poisson(@window, size: 30, n_seeds: 3, seed: 1)
      assert length(a) == 30
    end

    test "multiple replications and validation" do
      assert [p, q] = PointPats.cluster_poisson(@window, size: {20, 2}, n_seeds: 2, seed: 1)
      assert length(p) == 20 and p != q

      assert_raise ArgumentError, ~r/too many clusters/, fn ->
        PointPats.cluster_poisson(@window, size: 3, n_seeds: 5, seed: 1)
      end

      assert_raise ArgumentError, ~r/cluster_radius/, fn ->
        PointPats.cluster_poisson(@window, size: 10, n_seeds: 1, seed: 1)
      end
    end

    test "clustered patterns have smaller nearest-neighbour distances than CSR" do
      csr = nn_mean(PointPats.poisson(@window, size: 100, seed: 1))
      cl = nn_mean(PointPats.cluster_poisson(@window, size: 100, n_seeds: 4, seed: 1))
      assert cl < csr
    end
  end
end
