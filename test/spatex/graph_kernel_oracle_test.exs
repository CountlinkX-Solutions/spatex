defmodule Spatex.GraphKernelOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12
  @fixtures "test/reference/fixtures"
            |> Path.join("weights/kernel_*.json")
            |> Path.wildcard()
            |> Enum.map(&Path.relative_to(&1, "test/reference/fixtures"))

  # Explicit maps: the atoms may not be loaded yet when this file runs alone.
  @kernels Map.new(
             ~w(gaussian triangular parabolic bisquare tricube cosine boxcar exponential identity)a,
             &{Atom.to_string(&1), &1}
           )
  defp kernel(name), do: Map.fetch!(@kernels, name)
  defp coincident("raise"), do: :raise
  defp coincident("clique"), do: :clique

  defp bandwidth(nil), do: nil
  defp bandwidth("adaptive"), do: :adaptive
  defp bandwidth(h), do: h

  defp build(points, %{"threshold" => t} = p) do
    Graph.build_distance_band(points, t,
      binary: false,
      kernel: kernel(p["kernel"]),
      bandwidth: bandwidth(p["bandwidth"]),
      taper: p["taper"],
      decay: p["decay"]
    )
  end

  defp build(points, p) do
    Graph.build_kernel(points,
      kernel: kernel(p["kernel"]),
      bandwidth: bandwidth(p["bandwidth"]),
      k: p["k"],
      taper: p["taper"],
      decay: p["decay"],
      coincident: coincident(p["coincident"])
    )
  end

  test "fixtures are present" do
    assert length(@fixtures) == 6
  end

  for rel <- @fixtures do
    describe rel do
      setup do
        {:ok, fx: Reference.load!(unquote(rel))}
      end

      test "every case matches libpysal", %{fx: fx} do
        assert fx["n"] == length(fx["points"])

        for c <- fx["cases"], c["quirk"] == nil do
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

  # Known deviation (test/reference/known_deviations.exs): libpysal's fixed-bandwidth sparse
  # path calls eliminate_zeros(), which also drops coincident pairs; Spatex keeps them.
  test "libpysal drops coincident pairs on its fixed-bandwidth fast path, Spatex keeps them" do
    fx = Reference.load!("weights/kernel_coincident.json")
    [c] = Enum.filter(fx["cases"], &(&1["quirk"] == "fast_path_drops_coincident"))

    pysal =
      Enum.zip([c["adjacency"]["focal"], c["adjacency"]["neighbor"], c["adjacency"]["weight"]])

    refute Enum.any?(pysal, &match?({0, 1, _}, &1))

    spatex = Graph.to_adjacency(build(fx["points"], c["params"]))
    assert {0, 1, 1.0} in spatex and {1, 0, 1.0} in spatex

    without_pair = Enum.reject(spatex, fn {f, t, _} -> {f, t} in [{0, 1}, {1, 0}] end)

    assert Enum.map(without_pair, &{elem(&1, 0), elem(&1, 1)}) ==
             Enum.map(pysal, &{elem(&1, 0), elem(&1, 1)})
  end
end
