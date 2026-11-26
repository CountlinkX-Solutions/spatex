defmodule Spatex.PointPatsAlphaShapeTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.Reference

  doctest Spatex.PointPats, only: [alpha_shape: 2]

  @fx Reference.load!("pointpats/alpha_shape.json")["cases"]

  # Decodes a little-endian polygon WKB into `{shell, holes}` with open rings.
  defp decode(<<1, 3::little-32, n::little-32, rest::binary>>) do
    {rings, <<>>} =
      Enum.map_reduce(1..n, rest, fn _, <<m::little-32, body::binary>> ->
        size = m * 16
        <<coords::binary-size(size), tail::binary>> = body
        pts = for <<x::little-float-64, y::little-float-64 <- coords>>, do: [x, y]
        # the ring is closed in WKB
        {Enum.drop(pts, -1), tail}
      end)

    [shell | holes] = rings
    {shell, holes}
  end

  defp area(ring) do
    ring
    |> Enum.zip(tl(ring) ++ [hd(ring)])
    |> Enum.map(fn {[x0, y0], [x1, y1]} -> x0 * y1 - x1 * y0 end)
    |> Enum.sum()
    |> Kernel./(2)
  end

  defp polygon_area({shell, holes}),
    do: abs(area(shell)) - (holes |> Enum.map(&abs(area(&1))) |> Enum.sum())

  describe "oracle: libpysal.cg.alpha_shape" do
    @describetag :oracle

    for {c, i} <- Enum.with_index(@fx) do
      test "#{c["name"]} #{c["label"]} (#{i})" do
        c = Enum.at(@fx, unquote(i))
        pts = Enum.map(c["points"], fn [x, y] -> {x, y} end)

        got = pts |> PointPats.alpha_shape(c["alpha"]) |> Enum.map(&decode/1)
        want = c["polygons"]

        assert length(got) == length(want)

        for {{shell, holes} = poly, w} <- Enum.zip(got, want) do
          assert shell == w["shell"]
          assert holes == w["holes"]
          assert area(shell) < 0.0
          Reference.assert_close(polygon_area(poly), w["area"], 1.0e-9)
        end
      end
    end
  end

  describe "API" do
    @pts [{0.0, 0.0}, {4.0, 0.3}, {4.2, 4.0}, {0.1, 3.9}, {2.1, 1.9}]

    test "returns a list of WKB polygons, empty when alpha is too large" do
      assert [wkb] = PointPats.alpha_shape(@pts, 0.0)
      assert is_binary(wkb)
      assert {[[x, y] | _], []} = decode(wkb)
      assert {x, y} == {0.0, 0.0}
      assert PointPats.alpha_shape(@pts, 1.0e9) == []
    end

    test "accepts integers, tuples, lists and tensors" do
      base = PointPats.alpha_shape(@pts, 0.1)
      assert PointPats.alpha_shape(Enum.map(@pts, &Tuple.to_list/1), 0.1) == base

      assert PointPats.alpha_shape(Nx.tensor(Enum.map(@pts, &Tuple.to_list/1), type: :f64), 0.1) ==
               base

      assert PointPats.alpha_shape(@pts, 0) == PointPats.alpha_shape(@pts, 0.0)
    end

    test "fewer than four points give their hull whatever alpha is" do
      tri = [{0.0, 0.0}, {4.0, 0.5}, {1.0, 3.0}]
      assert [_] = PointPats.alpha_shape(tri, 50.0)
      assert PointPats.alpha_shape(tri, 50.0) == PointPats.alpha_shape(tri, 0.01)
      assert PointPats.alpha_shape([{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}], 1.0) == []
    end

    test "rejects bad alpha, non-finite points and degenerate sets" do
      assert_raise ArgumentError, ~r/alpha/, fn -> PointPats.alpha_shape(@pts, -1.0) end
      assert_raise ArgumentError, ~r/alpha/, fn -> PointPats.alpha_shape(@pts, :wide) end
      assert_raise ArgumentError, fn -> PointPats.alpha_shape([], 1.0) end

      assert_raise ArgumentError, ~r/collinear/, fn ->
        PointPats.alpha_shape([{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}], 1.0)
      end

      assert_raise ArgumentError, ~r/distinct/, fn ->
        PointPats.alpha_shape([{0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}], 1.0)
      end
    end
  end
end
