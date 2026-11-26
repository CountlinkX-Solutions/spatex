defmodule Spatex.TestWKB do
  @moduledoc false
  # Test-only little-endian WKB encoders for polygons and a few invalid geometries.

  @doc "Polygon from a list of rings, each a list of `{x, y}` (closing vertex included)."
  def polygon(rings), do: <<1, 3::little-32>> <> polygon_body(rings)

  @doc "Multipolygon from a list of polygons, each a list of rings."
  def multipolygon(polygons) do
    body = for rings <- polygons, into: <<>>, do: <<1, 3::little-32>> <> polygon_body(rings)
    <<1, 6::little-32, length(polygons)::little-32>> <> body
  end

  @doc "Axis-aligned rectangle polygon."
  def box(x0, y0, x1, y1),
    do: polygon([[{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}, {x0, y0}]])

  @doc "Unit square whose lower-left corner is `{x, y}`."
  def cell(x, y), do: box(x, y, x + 1, y + 1)

  @doc "A Point (not a valid contiguity input)."
  def point(x, y), do: <<1, 1::little-32, x::little-float-64, y::little-float-64>>

  @doc "A LineString (not a valid contiguity input)."
  def linestring(points) do
    body = for {x, y} <- points, into: <<>>, do: <<x::little-float-64, y::little-float-64>>
    <<1, 2::little-32, length(points)::little-32>> <> body
  end

  defp polygon_body(rings) do
    body =
      for ring <- rings, into: <<>> do
        pts = for {x, y} <- ring, into: <<>>, do: <<x::little-float-64, y::little-float-64>>
        <<length(ring)::little-32>> <> pts
      end

    <<length(rings)::little-32>> <> body
  end
end
