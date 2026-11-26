defmodule Spatex.PointPatsPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.PointPats

  defp point_set(min \\ 4) do
    gen_coord = StreamData.integer(0..1000) |> StreamData.map(&(&1 / 10))

    {gen_coord, gen_coord}
    |> StreamData.tuple()
    |> StreamData.list_of(min_length: min, max_length: 40)
    |> StreamData.filter(fn pts -> pts |> Enum.uniq() |> length() >= min end)
  end

  property "G is a non-decreasing CDF starting at 0 and ending at 1" do
    check all(pts <- point_set()) do
      {support, g} = PointPats.g(pts, support: 12)
      assert length(support) == 12 and hd(g) == 0.0
      assert List.last(g) == 1.0
      assert g == Enum.sort(g)
    end
  end

  property "K counts more pairs at larger distances" do
    check all(pts <- point_set()) do
      {_, k} = PointPats.k(pts, support: {0, 50, 10})
      assert k == Enum.sort(k)
      assert hd(k) == 0.0
    end
  end

  property "the hull encloses every point and the minimum circle encloses the hull" do
    check all(pts <- point_set(5)) do
      hull = try_hull(pts)

      if hull do
        n = length(hull)

        for {px, py} <- pts do
          assert Enum.all?(Enum.with_index(hull), fn {{ax, ay}, i} ->
                   {bx, by} = Enum.at(hull, rem(i + 1, n))
                   (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= -1.0e-9
                 end)
        end

        {{cx, cy}, r} = PointPats.minimum_bounding_circle(pts)

        for {x, y} <- pts do
          assert :math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r * (1 + 1.0e-9) + 1.0e-9
        end

        {major, minor, _} = PointPats.ellipse(pts)
        assert major >= minor
      end
    end
  end

  property "the Euclidean median never has a larger total distance than the mean center" do
    check all(pts <- point_set()) do
      m = PointPats.euclidean_median(pts)
      assert PointPats.dtot(m, pts) <= PointPats.dtot(PointPats.mean_center(pts), pts) + 1.0e-9
    end
  end

  property "quadrat counts add up to the number of points" do
    check all(pts <- point_set(), max_runs: 50) do
      xs = Enum.map(pts, &elem(&1, 0))
      ys = Enum.map(pts, &elem(&1, 1))

      if Enum.min(xs) < Enum.max(xs) and Enum.min(ys) < Enum.max(ys) do
        q = PointPats.quadrat_test(pts, nx: 3, ny: 2)
        assert Enum.sum(q.counts) == length(pts)
        h = PointPats.quadrat_test(pts, shape: :hexagon, lh: 25)
        assert Enum.sum(h.counts) == length(pts)
      end
    end
  end

  defp try_hull(pts) do
    PointPats.hull(pts)
  rescue
    ArgumentError -> nil
  end
end
