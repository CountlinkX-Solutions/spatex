defmodule Spatex.GraphContiguityOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-12
  @fixtures ~w(columbus sids2 mexico synthetic islands labeled)

  for name <- @fixtures, {mode, rook} <- [{"queen", false}, {"rook", true}] do
    test "#{name} #{mode}: exact adjacency, order and weights match libpysal" do
      fx = Reference.load!("weights/contiguity_#{unquote(name)}.json")
      ids = fx["ids"]
      wkbs = Enum.map(fx["wkb_hex"], &Base.decode16!(&1, case: :lower))
      expected = fx[unquote(mode)]
      by_pos = fn positions -> Enum.map(positions, &Enum.at(ids, &1)) end

      graph = Graph.build_contiguity(wkbs, rook: unquote(rook), ids: ids)
      actual = Graph.to_adjacency(graph)

      assert Graph.ids(graph) == ids
      assert Enum.map(actual, &elem(&1, 0)) == by_pos.(expected["focal"])
      assert Enum.map(actual, &elem(&1, 1)) == by_pos.(expected["neighbor"])
      Reference.assert_close(Enum.map(actual, &elem(&1, 2)), expected["weight"], @tol)
      assert Graph.isolates(graph) == by_pos.(expected["isolates"])
    end
  end

  test "default ids are 0..n-1 and the default is rook" do
    fx = Reference.load!("weights/contiguity_columbus.json")
    wkbs = Enum.map(fx["wkb_hex"], &Base.decode16!(&1, case: :lower))
    graph = Graph.build_contiguity(wkbs)
    assert Graph.ids(graph) == Enum.to_list(0..48)

    actual = Graph.to_adjacency(graph)
    assert Enum.map(actual, &elem(&1, 0)) == fx["rook"]["focal"]
    assert Enum.map(actual, &elem(&1, 1)) == fx["rook"]["neighbor"]
  end

  describe "block contiguity" do
    @cases Reference.load!("weights/block_contiguity.json")["cases"]

    for %{"name" => name} <- @cases do
      test "#{name}: same neighbor sets as libpysal, compared by id" do
        fx = Enum.find(@cases, &(&1["name"] == unquote(name)))
        graph = Graph.build_block_contiguity(fx["regimes"], ids: fx["ids"])

        actual =
          graph
          |> Graph.to_adjacency()
          |> Enum.reject(fn {_f, _t, w} -> w == 0.0 end)
          |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
          |> Map.new(fn {k, v} -> {k, Enum.sort(v)} end)

        expected =
          fx["neighbors"] |> Enum.reject(fn {_k, v} -> v == [] end) |> Map.new()

        assert actual == Map.new(expected, fn {k, v} -> {k, Enum.sort(v)} end)
        assert Enum.sort(Graph.isolates(graph)) == Enum.sort(fx["isolates"])
      end

      test "#{name}: Spatex keeps input order where PySAL reorders (known deviation)" do
        fx = Enum.find(@cases, &(&1["name"] == unquote(name)))
        graph = Graph.build_block_contiguity(fx["regimes"], ids: fx["ids"])
        assert Graph.ids(graph) == fx["ids"]
        assert Enum.sort(fx["pysal_unique_ids"]) == Enum.sort(fx["ids"])
      end
    end

    test "PySAL reorders ids by regime for interleaved input" do
      abacb = Enum.find(@cases, &(&1["name"] == "abacb"))
      assert abacb["pysal_unique_ids"] == ["0", "2", "1", "4", "3"]
    end
  end
end
