defmodule Spatex.GeoFrameTest do
  # Needs GeoFrame: SPATEX_GEOFRAME_PATH=/path/to/geoframe mix test --only geoframe
  use ExUnit.Case, async: true

  @moduletag :geoframe

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Spatex.Graph
  alias Spatex.Reference
  alias Spatex.TestWKB

  # GeoFrame is reached through atoms so this file compiles without it.
  @gdf GeoFrame.GeoDataFrame
  @gs GeoFrame.GeoSeries

  defp gf(mod, fun, args), do: apply(mod, fun, args)

  defp columbus do
    ds = Reference.load!("datasets/columbus.json")
    wkbs = Enum.map(ds["wkb_hex"], &Base.decode16!(&1, case: :mixed))
    geometry = gf(@gs, :from_wkb!, [wkbs])

    data =
      DataFrame.new(
        crime: ds["attributes"]["CRIME"],
        inc: ds["attributes"]["INC"],
        hoval: ds["attributes"]["HOVAL"]
      )

    {gf(@gdf, :new, [data, geometry]), wkbs, ds}
  end

  defp points_frame(coords, opts \\ []) do
    wkbs = Enum.map(coords, fn {x, y} -> TestWKB.point(x, y) end)
    geometry = gf(@gs, :from_wkb!, [wkbs, opts])
    gf(@gdf, :new, [DataFrame.new(id: Enum.map(coords, fn {x, _} -> "p#{x}" end)), geometry])
  end

  describe "graph/3 contiguity" do
    test "queen and rook equal build_contiguity on the same WKB" do
      {frame, wkbs, _} = columbus()

      for {method, rook} <- [queen: false, rook: true] do
        expected = Graph.build_contiguity(wkbs, rook: rook)

        assert Graph.to_adjacency(Spatex.GeoFrame.graph(frame, method)) ==
                 Graph.to_adjacency(expected)
      end
    end

    test "accepts a bare GeoSeries and the ids option" do
      {frame, wkbs, _} = columbus()
      series = gf(@gdf, :geometry, [frame])
      ids = Enum.map(0..48, &"c#{&1}")
      g = Spatex.GeoFrame.graph(series, :queen, ids: ids)
      assert Graph.ids(g) == ids

      assert Graph.cardinalities(g) ==
               Graph.cardinalities(Graph.build_contiguity(wkbs, rook: false, ids: ids))
    end

    test "ids may name an attribute column" do
      frame = points_frame([{0, 0}, {1, 0}, {3, 0}])
      g = Spatex.GeoFrame.graph(frame, :knn, k: 1, ids: "id")
      assert Graph.ids(g) == ["p0", "p1", "p3"]
    end

    test "polygon-only error for non-polygon geometry" do
      frame = points_frame([{0, 0}, {1, 0}])
      assert_raise ArgumentError, ~r/polygon/i, fn -> Spatex.GeoFrame.graph(frame, :queen) end
    end
  end

  describe "graph/3 point methods" do
    test "knn on polygon centroids equals build_knn on the centroid coordinates" do
      {frame, _, _} = columbus()
      centroids = gf(@gs, :centroid, [gf(@gdf, :geometry, [frame])])
      xs = gf(@gs, :x, [centroids]) |> Series.to_list()
      ys = gf(@gs, :y, [centroids]) |> Series.to_list()

      expected = Graph.build_knn(Enum.zip(xs, ys), 4)
      actual = Spatex.GeoFrame.graph(frame, :knn, k: 4)
      assert Graph.to_adjacency(actual) == Graph.to_adjacency(expected)
    end

    test "distance_band, kernel and triangulation build on points" do
      frame = points_frame([{0, 0}, {1, 0}, {0, 1}, {5, 5}], crs: "EPSG:3857")

      db = Spatex.GeoFrame.graph(frame, :distance_band, threshold: 1.5)
      assert Graph.cardinalities(db) == [2, 2, 2, 0]

      assert %Graph{n: 4} = Spatex.GeoFrame.graph(frame, :kernel, k: 2)
      assert %Graph{n: 4} = Spatex.GeoFrame.graph(frame, :triangulation)
    end

    test "missing required options raise" do
      frame = points_frame([{0, 0}, {1, 0}])
      assert_raise ArgumentError, ~r/:k/, fn -> Spatex.GeoFrame.graph(frame, :knn) end

      assert_raise ArgumentError, ~r/threshold/, fn ->
        Spatex.GeoFrame.graph(frame, :distance_band)
      end

      assert_raise ArgumentError, ~r/unknown method/, fn ->
        Spatex.GeoFrame.graph(frame, :nope)
      end
    end

    test "geographic coordinates are refused for distance methods unless allowed" do
      frame = points_frame([{0, 0}, {1, 0}, {0, 1}], crs: "EPSG:4326")

      assert_raise ArgumentError, ~r/geographic/, fn ->
        Spatex.GeoFrame.graph(frame, :knn, k: 1)
      end

      assert %Graph{} = Spatex.GeoFrame.graph(frame, :knn, k: 1, allow_geographic: true)

      log =
        ExUnit.CaptureLog.capture_log(fn ->
          Spatex.GeoFrame.graph(frame, :knn, k: 1, allow_geographic: :warn)
        end)

      assert log =~ "geographic"
    end
  end

  describe "values/2 and put_results/3" do
    test "values returns the attribute column as a series" do
      {frame, _, ds} = columbus()
      assert Series.to_list(Spatex.GeoFrame.values(frame, "crime")) == ds["attributes"]["CRIME"]

      assert_raise ArgumentError, ~r/geometry/, fn ->
        Spatex.GeoFrame.values(frame, "geometry")
      end
    end

    test "moran_local results come back as columns" do
      {frame, _, _} = columbus()
      g = Spatex.GeoFrame.graph(frame, :queen)

      m =
        Spatex.ESDA.moran_local(Spatex.GeoFrame.values(frame, :crime), g,
          permutations: 99,
          seed: 1
        )

      out = Spatex.GeoFrame.put_results(frame, m)
      names = gf(@gdf, :names, [out])
      for col <- ~w(lisa_is lisa_q lisa_p_sim lisa_cluster crime), do: assert(col in names)
      refute "lisa_id" in names

      assert out |> then(&gf(@gdf, :pull, [&1, "lisa_q"])) |> Series.to_list() == m.q
      assert gf(@gdf, :n_rows, [out]) == 49
      assert gf(@gdf, :geometry, [out]) == gf(@gdf, :geometry, [frame])

      custom = Spatex.GeoFrame.put_results(frame, m, prefix: "x_", crit: 1.0)
      assert "x_cluster" in gf(@gdf, :names, [custom])
    end

    test "classify results add class_yb" do
      {frame, _, _} = columbus()
      c = Spatex.Classify.quantiles(Spatex.GeoFrame.values(frame, :crime), k: 5)
      out = Spatex.GeoFrame.put_results(frame, c)
      assert gf(@gdf, :pull, [out, "class_yb"]) |> Series.to_list() == c.yb
    end

    test "row count mismatch raises" do
      {frame, _, _} = columbus()

      m =
        Spatex.ESDA.moran_local(
          [1, 5, 2, 8],
          Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]),
          permutations: 0
        )

      assert_raise ArgumentError, ~r/rows/, fn -> Spatex.GeoFrame.put_results(frame, m) end
    end

    test "a result without per-site columns raises instead of returning the frame" do
      {frame, _, _} = columbus()

      assert_raise ArgumentError, ~r/no per-site columns/, fn ->
        Spatex.GeoFrame.put_results(frame, %Spatex.ESDA.GLocal{n: 49, permutations: 0})
      end
    end

    test "region results add their labels" do
      {frame, _, _} = columbus()
      g = Spatex.GeoFrame.graph(frame, :queen)
      crime = frame |> Spatex.GeoFrame.values(:crime) |> Series.to_list()
      r = Spatex.Region.skater([crime], g, n_clusters: 3)
      out = Spatex.GeoFrame.put_results(frame, r)
      assert gf(@gdf, :pull, [out, "skater_labels"]) |> Series.to_list() == r.labels
    end
  end
end

defmodule Spatex.GeoFrameAbsentTest do
  use ExUnit.Case, async: true

  test "calls raise a clear error when GeoFrame is not loaded" do
    if Code.ensure_loaded?(GeoFrame.GeoDataFrame) do
      :ok
    else
      assert_raise ArgumentError, ~r/GeoFrame is not available/, fn ->
        Spatex.GeoFrame.graph(%{__struct__: GeoFrame.GeoSeries}, :queen)
      end

      assert_raise ArgumentError, ~r/GeoFrame is not available/, fn ->
        Spatex.GeoFrame.values(%{}, :a)
      end
    end
  end
end
