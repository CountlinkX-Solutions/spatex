defmodule Spatex.PointPatsCentrographyTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.Reference

  @tol 1.0e-9
  @fx Reference.load!("pointpats/centrography.json")["cases"]
  @sets ~w(p10 r30 r100 r500 c200 cross diag tri)
  @skyum_bug ~w(c200 diag r500)

  defp pts(name), do: Enum.map(@fx[name]["points"], fn [x, y] -> {x, y} end)
  defp lst({x, y}), do: [x, y]

  describe "oracle: deterministic centrography" do
    for name <- @sets do
      test "#{name}: centers, spread and bounding box" do
        c = @fx[unquote(name)]
        p = pts(unquote(name))
        w = c["weights"]

        Reference.assert_close(lst(PointPats.mean_center(p)), c["mean_center"], @tol)

        Reference.assert_close(
          lst(PointPats.weighted_mean_center(p, w)),
          c["weighted_mean_center"],
          @tol
        )

        Reference.assert_close(lst(PointPats.manhattan_median(p)), c["manhattan_median"], @tol)
        Reference.assert_close(PointPats.std_distance(p), c["std_distance"], @tol)

        Reference.assert_close(
          PointPats.dtot(PointPats.mean_center(p), p),
          c["dtot_at_mean"],
          @tol
        )

        Reference.assert_close(Tuple.to_list(PointPats.mbr(p)), c["mbr"], @tol)
      end

      test "#{name}: standard deviational ellipses" do
        c = @fx[unquote(name)]
        p = pts(unquote(name))
        e = c["ellipse"]

        Reference.assert_close(Tuple.to_list(PointPats.ellipse(p)), e["crimestat"], @tol)

        Reference.assert_close(
          Tuple.to_list(PointPats.ellipse(p, weights: c["weights"])),
          e["crimestat_weighted"],
          @tol
        )

        for cc <- [true, false], dc <- [true, false] do
          got =
            PointPats.ellipse(p, method: :yuill, crimestat_corr: cc, degfreed_corr: dc)

          Reference.assert_close(
            Tuple.to_list(got),
            e["yuill_#{String.capitalize("#{cc}")}_#{String.capitalize("#{dc}")}"],
            @tol
          )
        end
      end

      test "#{name}: hull, minimum rotated rectangle, bounding circle" do
        c = @fx[unquote(name)]
        p = pts(unquote(name))
        assert_same_cycle(PointPats.hull(p), c["hull"])

        {corners, angle} = PointPats.minimum_rotated_rectangle(p, return_angle: true)
        assert length(corners) == 4
        Reference.assert_close(polygon_area(corners), c["mrr"]["area"], @tol)

        # The cross has four rectangles of equal area; which one GEOS reports is arbitrary.
        if unquote(name) != "cross" do
          Reference.assert_close(angle, c["mrr"]["angle"], @tol)
          expected = c["mrr"]["corners"] |> Enum.map(fn [x, y] -> {x, y} end) |> Enum.sort()

          Reference.assert_close(
            Enum.map(Enum.sort(corners), &lst/1),
            Enum.map(expected, &lst/1),
            @tol
          )
        end

        {center, r} = PointPats.minimum_bounding_circle(p)
        assert PointPats.skyum(p) == {center, r}
        {true_center, true_r} = brute_force_circle(PointPats.hull(p))
        Reference.assert_close(r, true_r, @tol)
        Reference.assert_close(lst(center), lst(true_center), @tol)
        assert Enum.all?(p, fn q -> dist(center, q) <= r * (1 + 1.0e-12) end)

        # Known deviation: pointpats' Skyum loop returns a non-minimal circle (or divides by
        # zero) on some inputs; elsewhere it equals the true minimum circle.
        if unquote(name) in @skyum_bug do
          py = c["mbc"]
          assert py == :nan or match?(%{"error" => _}, py) or py["radius"] > r + 1.0e-6
        else
          Reference.assert_close(lst(center), c["mbc"]["center"], @tol)
          Reference.assert_close(r, c["mbc"]["radius"], @tol)
        end
      end

      test "#{name}: euclidean median reaches the exact optimum (deviation from BFGS)" do
        c = @fx[unquote(name)]
        p = pts(unquote(name))
        m = PointPats.euclidean_median(p)
        # Weiszfeld converges to the exact optimum; pointpats' BFGS stops at gtol 1e-5.
        assert PointPats.dtot(m, p) <= c["euclidean_median"]["dtot"] + 1.0e-9
        scale = max(c["std_distance"], 1.0)
        [ex, ey] = c["euclidean_median"]["xy"]
        {mx, my} = m
        assert abs(mx - ex) <= 1.0e-3 * scale
        assert abs(my - ey) <= 1.0e-3 * scale
      end
    end

    test "probed values on the 10-point set" do
      p = pts("p10")
      assert {51.121, my} = PointPats.mean_center(p)
      assert_in_delta my, 54.683, 1.0e-12

      {major, minor, angle} = PointPats.ellipse(p)
      assert_in_delta major, 53.612878559714375, 1.0e-12
      assert_in_delta minor, 27.133488119689254, 1.0e-12
      assert_in_delta angle, 0.33520089366330835, 1.0e-12
    end

    test "axis-aligned cross takes the den == 0 branch" do
      {major, minor, angle} = PointPats.ellipse(pts("cross"))
      assert_in_delta major, 2.3094010767585034, 1.0e-12
      assert_in_delta minor, 1.1547005383792517, 1.0e-12
      assert angle == 0
    end

    test "the ellipse angle points along the major axis" do
      {_, _, angle} = PointPats.ellipse(pts("diag"))
      assert_in_delta angle, :math.pi() / 4, 0.1
    end

    test "collinear points: spread and ellipse still defined" do
      line = @fx["line"]
      p = Enum.map(line["points"], fn [x, y] -> {x, y} end)
      Reference.assert_close(PointPats.std_distance(p), line["std_distance"], @tol)
      {major, minor, angle} = PointPats.ellipse(p)
      [em, en, ea] = line["ellipse"]
      Reference.assert_close([major, angle], [em, ea], @tol)
      assert_in_delta minor, en, 1.0e-9
    end
  end

  describe "edge cases and validation" do
    test "median of an even pattern averages the middle values" do
      assert PointPats.manhattan_median([{0, 0}, {1, 5}, {2, 1}, {10, 2}]) == {1.5, 1.5}
    end

    test "std_distance uses the raw-moment form" do
      assert_in_delta PointPats.std_distance([{0, 0}, {2, 0}]), 1.0, 1.0e-15
    end

    test "weighted mean center validates weights" do
      assert_raise ArgumentError, ~r/weights/, fn ->
        PointPats.weighted_mean_center([{0, 0}, {1, 1}], [1.0])
      end

      assert_raise ArgumentError, ~r/sum/, fn ->
        PointPats.weighted_mean_center([{0, 0}, {1, 1}], [1.0, -1.0])
      end
    end

    test "ellipse needs three points and a known method" do
      assert_raise ArgumentError, ~r/at least 3/, fn -> PointPats.ellipse([{0, 0}, {1, 2}]) end

      assert_raise ArgumentError, ~r/method/, fn ->
        PointPats.ellipse([{0, 0}, {1, 2}, {3, 1}], method: :other)
      end
    end

    test "hull of collinear points raises (as Qhull does)" do
      assert_raise ArgumentError, ~r/collinear|at least 3/, fn ->
        PointPats.hull(Enum.map(@fx["line"]["points"], fn [x, y] -> {x, y} end))
      end

      assert_raise ArgumentError, fn ->
        PointPats.minimum_rotated_rectangle([{0, 0}, {1, 1}, {2, 2}])
      end
    end

    test "hull drops collinear boundary points and starts at the lowest x" do
      assert PointPats.hull([{0, 0}, {1, 0}, {2, 0}, {1, 1}]) ==
               [{0.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}]
    end

    test "bounding circle of a triangle" do
      assert {{cx, cy}, r} = PointPats.minimum_bounding_circle([{0, 0}, {4, 0}, {2, 1}])
      assert_in_delta cx, 2.0, 1.0e-12
      assert_in_delta cy, 0.0, 1.0e-12
      assert_in_delta r, 2.0, 1.0e-12
    end

    test "rejects empty and non-finite input" do
      assert_raise ArgumentError, fn -> PointPats.mean_center([]) end

      assert_raise ArgumentError, ~r/non-finite/, fn ->
        PointPats.std_distance({[0.0, 1.0], [:nan, 2.0]})
      end
    end

    test "accepts coordinate pairs, lists and packed pairs" do
      assert PointPats.mean_center({[0, 2], [1, 3]}) == {1.0, 2.0}
      assert PointPats.mean_center([[0, 1], [2, 3]]) == {1.0, 2.0}
    end

    test "centrography/1 bundles the summaries" do
      p = pts("p10")
      c = PointPats.centrography(p)
      assert c.mean_center == PointPats.mean_center(p)
      assert c.std_distance == PointPats.std_distance(p)
      assert c.ellipse == PointPats.ellipse(p)
      assert c.mbr == PointPats.mbr(p)
      assert c.hull == PointPats.hull(p)
    end
  end

  # Same vertices in the same cyclic (counter-clockwise) order, whatever the start vertex.
  defp assert_same_cycle(got, expected) do
    expected = Enum.map(expected, fn [x, y] -> {x, y} end)
    assert length(got) == length(expected)
    {start, _} = Enum.min_by(Enum.with_index(expected), fn {{x, y}, _} -> {x, y} end)
    idx = Enum.find_index(expected, &(&1 == start))
    {a, b} = Enum.split(expected, idx)
    rotated = b ++ a
    Reference.assert_close(Enum.map(got, &lst/1), Enum.map(rotated, &lst/1), @tol)
  end

  defp dist({ax, ay}, {bx, by}), do: :math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

  # Smallest circle through 2 or 3 hull vertices that encloses the hull.
  defp brute_force_circle(hull) do
    pairs =
      for a <- hull,
          b <- hull,
          a < b,
          do: {{(elem(a, 0) + elem(b, 0)) / 2, (elem(a, 1) + elem(b, 1)) / 2}, dist(a, b) / 2}

    triples =
      for a <- hull,
          b <- hull,
          c <- hull,
          a < b,
          b < c,
          circle = circumcircle(a, b, c),
          do: circle

    (pairs ++ triples)
    |> Enum.filter(fn {c, r} -> Enum.all?(hull, &(dist(c, &1) <= r * (1 + 1.0e-12))) end)
    |> Enum.min_by(&elem(&1, 1))
  end

  defp circumcircle({ax, ay}, {bx, by}, {cx, cy}) do
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    if d == 0 do
      nil
    else
      a2 = ax * ax + ay * ay
      b2 = bx * bx + by * by
      c2 = cx * cx + cy * cy
      ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
      uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
      {{ux, uy}, dist({ux, uy}, {ax, ay})}
    end
  end

  defp polygon_area(corners) do
    pairs = Enum.zip(corners, tl(corners) ++ [hd(corners)])

    abs(Enum.reduce(pairs, 0.0, fn {{x1, y1}, {x2, y2}}, acc -> acc + (x1 * y2 - x2 * y1) end)) /
      2
  end
end
