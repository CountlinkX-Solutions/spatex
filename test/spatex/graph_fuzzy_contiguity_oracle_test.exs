defmodule Spatex.GraphFuzzyContiguityOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @fixture Reference.load!("weights/contiguity_fuzzy.json")

  # GEOS approximates each buffer arc with 16 chords per quadrant, so its buffers are slightly
  # smaller than exact round buffers: pairs within about 0.15 % below 2 * buffer differ.
  @band_low 0.9985

  defp wkbs(set), do: Enum.map(set["wkb_hex"], &Base.decode16!(&1, case: :lower))

  defp opts(params) do
    for {key, value} <- params, into: [] do
      {String.to_existing_atom(key),
       if(key == "predicate" and value, do: String.to_atom(value), else: value)}
    end
  end

  defp pairs(graph) do
    graph
    |> Graph.to_adjacency()
    |> Enum.reject(fn {f, n, _w} -> f == n end)
    |> Enum.map(fn {f, n, _w} -> [f, n] end)
    |> Enum.sort()
  end

  defp assert_case(set, c) do
    label = "#{set["set"]} #{inspect(c["params"])}"
    options = opts(c["params"])

    if c["error"] do
      assert_raise ArgumentError, fn -> Graph.build_fuzzy_contiguity(wkbs(set), options) end
    else
      graph = Graph.build_fuzzy_contiguity(wkbs(set), options)
      assert pairs(graph) == Enum.sort(c["pairs"]), label
      assert Enum.sort(Graph.isolates(graph)) == Enum.sort(c["isolates"]), label
    end
  end

  for name <- ~w(columbus sids2 synthetic) do
    test "#{name}: neighbor sets match libpysal for every tolerance, buffer and predicate" do
      set = Enum.find(@fixture["sets"], &(&1["set"] == unquote(name)))
      for c <- set["cases"], do: assert_case(set, c)
    end
  end

  describe "near_threshold pairs of polygons" do
    setup do
      set = Enum.find(@fixture["sets"], &(&1["set"] == "near_threshold"))
      graph = Graph.build_fuzzy_contiguity(wkbs(set), buffer: 1.0)
      edges = MapSet.new(pairs(graph))

      rows =
        for {row, k} <- Enum.with_index(set["pairs"]) do
          Map.put(row, "spatex", MapSet.member?(edges, [2 * k, 2 * k + 1]))
        end

      %{set: set, rows: rows}
    end

    test "pairs outside the arc-approximation band match libpysal", %{rows: rows} do
      for r <- rows, r["ratio"] < @band_low or r["ratio"] > 1.0 do
        assert r["spatex"] == r["pysal"], inspect(r)
      end
    end

    test "polygon corners and tips follow the exact distance rule", %{rows: rows} do
      for r <- rows, r["kind"] != "disc" do
        assert r["spatex"] == r["ratio"] <= 1.0, inspect(r)
      end
    end

    test "known deviation: GEOS buffers are inscribed, so only exact neighbors are ever missed",
         %{rows: rows} do
      # libpysal never finds a pair that is not within 2 * buffer ...
      for r <- rows, r["pysal"], do: assert(r["spatex"], inspect(r))
      # ... and misses some pairs within a hair of the threshold (docs/parity.md).
      deviations = Enum.filter(rows, &(&1["spatex"] != &1["pysal"]))
      assert deviations != []
      assert Enum.all?(deviations, &(&1["ratio"] >= @band_low and &1["ratio"] <= 1.0))
    end

    test "the bounding box of a buffer is exact only for axis-aligned geometries", %{set: set} do
      # GEOS draws arcs with chords, so the box of the buffer of a triangle's tip or a disc
      # falls up to 0.12 % short of box + buffer; Spatex grows the box exactly. Same band.
      bbox_case = Enum.at(set["cases"], 1)
      graph = Graph.build_fuzzy_contiguity(wkbs(set), buffer: 1.0, predicate: nil)
      mine = MapSet.new(pairs(graph))
      theirs = MapSet.new(bbox_case["pairs"])

      assert MapSet.subset?(theirs, mine)

      # The box of a disc depends on its centre line, not on the polygon distance, so the
      # ratio band applies to the other shapes only; axis-aligned edges are exact.
      for [f, _n] <- MapSet.difference(mine, theirs) do
        row = Enum.at(set["pairs"], div(f, 2))
        assert row["kind"] != "edge"
        assert row["kind"] == "disc" or (row["ratio"] >= @band_low and row["ratio"] <= 1.0)
      end
    end
  end
end
