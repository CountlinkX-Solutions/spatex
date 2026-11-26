defmodule Spatex.GeoFrame do
  @moduledoc """
  Adapter between [GeoFrame](https://github.com/CountlinkX-Solutions/geoframe) and Spatex.

  Spatex does not depend on GeoFrame. This module calls GeoFrame's public API at runtime
  (`GeoFrame.GeoDataFrame` and `GeoFrame.GeoSeries`), and raises an `ArgumentError` saying so
  when GeoFrame is not loaded. GeoFrame is an optional integration that is not on Hex yet: to
  use this module, add a local GeoFrame checkout to your own dependencies, for example
  `{:geoframe, path: "../geoframe"}`. Spatex's own GeoFrame tests and notebooks find the
  checkout through the `SPATEX_GEOFRAME_PATH` environment variable.

  ## Building graphs

  `graph/3` takes a `GeoFrame.GeoDataFrame` or a `GeoFrame.GeoSeries` and a method:

  | Method | Geometry | Required option |
  |---|---|---|
  | `:queen`, `:rook` | polygons or multipolygons (WKB) | - |
  | `:knn` | any; polygons use centroids | `k: pos_integer` |
  | `:distance_band` | any; polygons use centroids | `threshold: number` |
  | `:kernel` | any; polygons use centroids | - (see `Spatex.Graph.build_kernel/2`) |
  | `:triangulation` | any; polygons use centroids | - (see `Spatex.Graph.build_triangulation/2`) |

  Point methods always work on the centroid of each geometry; the centroid of a point is the
  point itself. Remaining options go to the matching `Spatex.Graph` builder. The graph's ids
  are the row positions unless `ids:` is a list of unique ids or the name of an attribute
  column of a `GeoDataFrame`.

  ### Geographic coordinates

  The distance methods are Euclidean, so on longitude/latitude data they measure degrees.
  When the geometry reports a geographic CRS (`GeoSeries.is_geographic/1`), `graph/3` raises
  `ArgumentError` unless `allow_geographic: true` (proceed silently) or
  `allow_geographic: :warn` (proceed and log a warning) is given. Project the frame first
  (`GeoFrame.GeoDataFrame.to_crs/2`) for meaningful distances. Contiguity methods are not
  affected.

  ## Results as columns

  `values/2` pulls an attribute column as an `Explorer.Series`, which every Spatex ESDA
  function accepts. `put_results/3` writes per-site result columns back onto the frame; see
  `Spatex.Explorer.to_dataframe/2` for the columns.

  ## Examples

      frame = GeoFrame.GeoDataFrame.from_wkb!(df, "geometry")
      graph = Spatex.GeoFrame.graph(frame, :queen)
      lisa = Spatex.ESDA.moran_local(Spatex.GeoFrame.values(frame, "crime"), graph, seed: 1)
      Spatex.GeoFrame.put_results(frame, lisa)
      #=> a GeoDataFrame with lisa_is, lisa_q, lisa_p_sim, lisa_cluster, ... columns
  """

  require Logger

  alias Spatex.Graph

  @gdf GeoFrame.GeoDataFrame
  @gs GeoFrame.GeoSeries

  @graph_opts [:allow_geographic]
  @distance_methods [:knn, :distance_band, :kernel, :triangulation]

  @doc """
  Builds a `Spatex.Graph` from a `GeoFrame.GeoDataFrame` or `GeoFrame.GeoSeries`.

  See the module documentation for the methods and options. Raises `ArgumentError` for an
  unknown method, missing required options, geometries a method cannot handle (a non-polygon
  with `:queen`/`:rook`, an empty or null geometry for point methods) or geographic data for
  a distance method.
  """
  @spec graph(struct(), atom(), keyword()) :: Graph.t()
  def graph(frame, method, opts \\ []) when is_atom(method) and is_list(opts) do
    ensure_geoframe!()
    {series, data} = split(frame)
    {gopts, opts} = Keyword.split(opts, @graph_opts)
    {ids, opts} = Keyword.pop(opts, :ids)
    opts = if ids = resolve_ids(ids, data), do: [ids: ids] ++ opts, else: opts

    case method do
      m when m in [:queen, :rook] ->
        contiguity(series, m, opts)

      m when m in @distance_methods ->
        check_geographic!(series, m, gopts[:allow_geographic])
        points(series) |> point_graph(m, opts)

      other ->
        raise ArgumentError,
              "unknown method #{inspect(other)}; expected :queen, :rook, :knn, " <>
                ":distance_band, :kernel or :triangulation"
    end
  end

  @doc """
  The attribute column `column` of a `GeoFrame.GeoDataFrame` as an `Explorer.Series`.

  Raises `ArgumentError` for a geometry column.
  """
  @spec values(struct(), String.t() | atom()) :: struct()
  def values(frame, column) do
    ensure_geoframe!()

    case call(@gdf, :pull, [frame, to_string(column)]) do
      %{__struct__: @gs} ->
        raise ArgumentError, "column #{inspect(to_string(column))} is a geometry column"

      series ->
        series
    end
  end

  @doc """
  Adds the per-site columns of an ESDA or classification result to a `GeoDataFrame`.

  Accepts the results `Spatex.Explorer.to_dataframe/2` accepts, including `Spatex.Region`
  results (a `labels` column). Returns a new frame. Column names are the ones
  `Spatex.Explorer.to_dataframe/2` produces,
  prefixed (the `id` column is not added). The default prefix depends on the result:
  `"lisa_"` for `Spatex.ESDA.MoranLocal`, `"g_"` for `Spatex.ESDA.GLocal`, `"class_"` for
  `Spatex.Classify`, otherwise the snake-cased result module name followed by `_`. Existing
  columns with the same name are replaced.

  ## Options

    * `:prefix` - column name prefix.
    * `:crit` - significance level for LISA cluster labels (default `0.05`).

  Raises `ArgumentError` when any result column's length differs from the frame's rows, or
  when the result has no per-site columns (instead of returning the frame unchanged).
  """
  @spec put_results(struct(), struct(), keyword()) :: struct()
  def put_results(frame, result, opts \\ []) do
    ensure_geoframe!()
    crit = Keyword.get(opts, :crit, 0.05)

    unless is_number(crit),
      do: raise(ArgumentError, ":crit must be a number, got #{inspect(crit)}")

    prefix = Keyword.get(opts, :prefix) || default_prefix(result)
    rows = call(@gdf, :n_rows, [frame])

    columns =
      result |> Spatex.Explorer.columns(crit) |> Enum.reject(fn {name, _} -> name == "id" end)

    if columns == [] do
      raise ArgumentError,
            "result #{inspect(result.__struct__)} has no per-site columns to add to the frame"
    end

    for {name, list} <- columns, length(list) != rows do
      raise ArgumentError,
            "result column #{inspect(name)} has #{length(list)} rows but the frame has " <>
              "#{rows} rows"
    end

    Enum.reduce(columns, frame, fn {name, list}, acc ->
      call(@gdf, :put, [acc, prefix <> name, list])
    end)
  end

  # -- graph builders --------------------------------------------------------------------

  defp contiguity(series, method, opts) do
    wkbs = call(@gs, :to_wkb, [series])
    Graph.build_contiguity(wkbs, [rook: method == :rook] ++ opts)
  end

  defp points(series) do
    centroids = call(@gs, :centroid, [series])
    {call(@gs, :x, [centroids]), call(@gs, :y, [centroids])}
  end

  defp point_graph(points, :knn, opts) do
    {k, opts} = Keyword.pop(opts, :k)

    unless is_integer(k),
      do: raise(ArgumentError, "method :knn requires the :k option (positive integer)")

    Graph.build_knn(points, k, opts)
  end

  defp point_graph(points, :distance_band, opts) do
    {threshold, opts} = Keyword.pop(opts, :threshold)

    unless is_number(threshold),
      do: raise(ArgumentError, "method :distance_band requires the :threshold option (number)")

    Graph.build_distance_band(points, threshold, opts)
  end

  defp point_graph(points, :kernel, opts), do: Graph.build_kernel(points, opts)
  defp point_graph(points, :triangulation, opts), do: Graph.build_triangulation(points, opts)

  # -- helpers ---------------------------------------------------------------------------

  defp split(%{__struct__: @gdf} = frame),
    do: {call(@gdf, :geometry, [frame]), call(@gdf, :data, [frame])}

  defp split(%{__struct__: @gs} = series), do: {series, nil}

  defp split(other) do
    raise ArgumentError,
          "expected a GeoFrame.GeoDataFrame or GeoFrame.GeoSeries, got #{inspect(other, limit: 3)}"
  end

  defp resolve_ids(nil, _data), do: nil
  defp resolve_ids(ids, _data) when is_list(ids), do: ids

  defp resolve_ids(column, nil) when is_binary(column) or is_atom(column) do
    raise ArgumentError,
          "ids: #{inspect(column)} names a column, which needs a GeoDataFrame (got a GeoSeries)"
  end

  defp resolve_ids(column, data) when is_binary(column) or is_atom(column) do
    name = to_string(column)

    unless name in call(Explorer.DataFrame, :names, [data]) do
      raise ArgumentError, "ids: no attribute column #{inspect(name)}"
    end

    data[name] |> then(&call(Explorer.Series, :to_list, [&1]))
  end

  defp resolve_ids(other, _data) do
    raise ArgumentError,
          "ids must be a list of ids or a column name, got #{inspect(other, limit: 3)}"
  end

  defp check_geographic!(series, method, allow) do
    if allow != true and call(@gs, :is_geographic, [series]) == true do
      message =
        "method #{inspect(method)} is Euclidean but the geometry has a geographic CRS " <>
          "(distances would be in degrees); project it first or pass allow_geographic: true"

      case allow do
        :warn -> Logger.warning("Spatex.GeoFrame.graph/3: " <> message)
        _ -> raise ArgumentError, message
      end
    end
  end

  defp default_prefix(%Spatex.ESDA.MoranLocal{}), do: "lisa_"
  defp default_prefix(%Spatex.ESDA.GLocal{}), do: "g_"
  defp default_prefix(%Spatex.Classify{}), do: "class_"

  defp default_prefix(%{__struct__: module}),
    do: module |> Module.split() |> List.last() |> Macro.underscore() |> Kernel.<>("_")

  defp default_prefix(_other), do: "res_"

  defp ensure_geoframe! do
    unless Code.ensure_loaded?(@gdf) do
      raise ArgumentError,
            "GeoFrame is not available: add the :geoframe dependency to use Spatex.GeoFrame"
    end
  end

  # Runtime dispatch keeps GeoFrame and Explorer out of compile-time xref (see Spatex.Input).
  defp call(mod, fun, args), do: apply(mod, fun, args)
end
