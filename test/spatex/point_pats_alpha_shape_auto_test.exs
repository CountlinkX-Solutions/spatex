defmodule Spatex.PointPatsAlphaShapeAutoTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.Reference

  doctest Spatex.PointPats, only: [alpha_shape_auto: 2]

  @fx Reference.load!("pointpats/alpha_shape_auto.json")["cases"]

  # Decodes a hole-free polygon WKB into its open shell.
  defp shell(<<1, 3::little-32, 1::little-32, m::little-32, body::binary>>) do
    size = m * 16
    <<coords::binary-size(size)>> = body
    pts = for <<x::little-float-64, y::little-float-64 <- coords>>, do: [x, y]
    Enum.drop(pts, -1)
  end

  defp area(ring) do
    ring
    |> Enum.zip(tl(ring) ++ [hd(ring)])
    |> Enum.map(fn {[x0, y0], [x1, y1]} -> x0 * y1 - x1 * y0 end)
    |> Enum.sum()
    |> Kernel./(2)
  end

  describe "oracle: libpysal.cg.alpha_shape_auto" do
    @describetag :oracle

    # The search compares `radius < 1 / (1 / radius - EPS)`, which depends on the last bit of
    # the radius, so the oracle stores the results libpysal gives for each vertex order of the
    # triangles; Spatex must equal one of them (usually there is a single one).
    for {c, i} <- Enum.with_index(@fx) do
      test "#{c["name"]} step #{c["step"]} (#{i})" do
        c = Enum.at(@fx, unquote(i))
        pts = Enum.map(c["points"], fn [x, y] -> {x, y} end)

        {wkb, radius, centers} =
          PointPats.alpha_shape_auto(pts, step: c["step"], return_circles: true)

        ring = shell(wkb)
        variant = Enum.find(c["variants"], &(&1["polygon"]["shell"] == ring))

        assert variant,
               "no libpysal variant has this shell (#{length(c["variants"])} variants)"

        Reference.assert_close(radius, variant["radius"], 1.0e-9)
        Reference.assert_close(-area(ring), variant["polygon"]["area"], 1.0e-9)
        assert length(centers) == length(ring)
        Reference.assert_close(Enum.map(centers, &Tuple.to_list/1), variant["centers"], 1.0e-8)
      end
    end
  end

  describe "API" do
    @pts [{0.0, 0.0}, {4.0, 0.3}, {4.2, 4.0}, {0.1, 3.9}, {2.1, 1.9}]

    test "returns a WKB polygon, or it with the radius, or with the radius and circles" do
      wkb = PointPats.alpha_shape_auto(@pts)
      assert is_binary(wkb)
      assert {^wkb, radius} = PointPats.alpha_shape_auto(@pts, return_radius: true)
      assert is_float(radius) and radius > 0

      assert {^wkb, ^radius, centers} = PointPats.alpha_shape_auto(@pts, return_circles: true)
      assert length(centers) == length(shell(wkb))
      assert Enum.all?(centers, fn {x, y} -> is_float(x) and is_float(y) end)
    end

    test "the polygon is clockwise, closed and contains every point as a vertex or inside" do
      ring = @pts |> PointPats.alpha_shape_auto() |> shell()
      assert area(ring) < 0
      assert hd(ring) == Enum.min(ring)
      # the hull of the quadrilateral is the largest the shape can be
      [hull] = PointPats.alpha_shape(@pts, 0.0)
      assert abs(area(ring)) <= abs(area(shell_of_hull(hull))) + 1.0e-12
    end

    defp shell_of_hull(<<1, 3::little-32, _rings::little-32, m::little-32, body::binary>>) do
      size = m * 16
      <<coords::binary-size(size), _::binary>> = body
      for(<<x::little-float-64, y::little-float-64 <- coords>>, do: [x, y]) |> Enum.drop(-1)
    end

    test "a larger step cannot give a tighter shape than step 1 within the same radii" do
      pts =
        for i <- 0..39, do: {:math.sin(i * 0.7) * 3 + i * 0.01, :math.cos(i * 1.3) * 3 - i * 0.02}

      {_, r1} = PointPats.alpha_shape_auto(pts, return_radius: true, step: 1)
      {_, r9} = PointPats.alpha_shape_auto(pts, return_radius: true, step: 9)
      assert r9 >= r1
    end

    test "three points give their triangle" do
      tri = [{0.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}]
      {wkb, radius} = PointPats.alpha_shape_auto(tri, return_radius: true)
      assert length(shell(wkb)) == 3
      assert_in_delta radius, 2.5, 1.0e-12
    end

    test "rejects bad options, too few points and degenerate sets" do
      assert_raise ArgumentError, ~r/step/, fn -> PointPats.alpha_shape_auto(@pts, step: 0) end
      assert_raise ArgumentError, ~r/step/, fn -> PointPats.alpha_shape_auto(@pts, step: 1.5) end

      assert_raise ArgumentError, ~r/return_radius/, fn ->
        PointPats.alpha_shape_auto(@pts, return_radius: :yes)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        PointPats.alpha_shape_auto(@pts, verbose: true)
      end

      assert_raise ArgumentError, ~r/at least 3/, fn ->
        PointPats.alpha_shape_auto([{0.0, 0.0}, {1.0, 1.0}])
      end

      assert_raise ArgumentError, ~r/collinear/, fn ->
        PointPats.alpha_shape_auto([{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}])
      end
    end
  end
end
