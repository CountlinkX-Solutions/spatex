defmodule Spatex.GraphTriangulationOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12
  @fixtures "test/reference/fixtures"
            |> Path.join("weights/triangulation_*.json")
            |> Path.wildcard()
            |> Enum.map(&Path.relative_to(&1, "test/reference/fixtures"))

  # Explicit maps: the atoms may not be loaded yet when this file runs alone.
  @methods Map.new(~w(delaunay gabriel relative_neighborhood)a, &{Atom.to_string(&1), &1})
  @kernels Map.new(
             ~w(gaussian triangular parabolic bisquare tricube cosine boxcar exponential identity)a,
             &{Atom.to_string(&1), &1}
           )

  # `Reference.load!/1` decodes the JSON marker "inf" to :inf.
  defp bandwidth(:inf), do: :infinity
  defp bandwidth(h), do: h

  defp build(points, p) do
    Graph.build_triangulation(points,
      method: Map.fetch!(@methods, p["method"]),
      kernel: Map.fetch!(@kernels, p["kernel"]),
      bandwidth: bandwidth(p["bandwidth"]),
      taper: p["taper"],
      decay: p["decay"],
      coincident: String.to_existing_atom(p["coincident"])
    )
  end

  test "fixtures are present" do
    assert length(@fixtures) == 5
  end

  for rel <- @fixtures, rel != "weights/triangulation_dropped_node.json" do
    describe rel do
      setup do
        {:ok, fx: Reference.load!(unquote(rel))}
      end

      test "every case matches libpysal", %{fx: fx} do
        assert fx["n"] == length(fx["points"])

        for c <- fx["cases"] do
          g = build(fx["points"], c["params"])
          actual = Graph.to_adjacency(g)
          expected = c["adjacency"]
          label = "#{unquote(rel)} #{c["name"]}"

          assert Graph.n(g) == fx["n"], label
          assert Graph.cardinalities(g) == c["cardinalities"], label
          assert Graph.isolates(g) == c["isolates"], label
          assert Enum.map(actual, &elem(&1, 0)) == expected["focal"], label
          assert Enum.map(actual, &elem(&1, 1)) == expected["neighbor"], label
          Reference.assert_close(Enum.map(actual, &elem(&1, 2)), expected["weight"], @tol)
        end
      end
    end
  end

  # Known deviation (test/reference/known_deviations.exs): when a finite bandwidth tapers away
  # every edge of a node, libpysal removes the node from the graph; Spatex keeps it as an island.
  test "libpysal drops a node whose edges are all tapered away, Spatex keeps it as an island" do
    fx = Reference.load!("weights/triangulation_dropped_node.json")
    [c] = fx["cases"]
    assert c["quirk"] == "drops_tapered_node"

    pysal_ids = c["pysal_ids"]
    assert length(pysal_ids) == fx["n"] - 1
    [dropped] = Enum.to_list(0..(fx["n"] - 1)) -- pysal_ids

    # libpysal's positions index its surviving ids; map them back to the input positions.
    pysal =
      c["adjacency"]
      |> Map.take(["focal", "neighbor", "weight"])
      |> then(&Enum.zip([&1["focal"], &1["neighbor"], &1["weight"]]))
      |> Enum.map(fn {f, t, w} -> {Enum.at(pysal_ids, f), Enum.at(pysal_ids, t), w} end)

    g = build(fx["points"], c["params"])
    assert Graph.n(g) == fx["n"]
    assert Graph.isolates(g) == [dropped]

    spatex = g |> Graph.to_adjacency() |> Enum.reject(&(elem(&1, 0) == dropped))
    assert spatex == pysal
  end
end
