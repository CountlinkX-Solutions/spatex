defmodule Spatex.GraphPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph

  # Random directed graphs on 1..12 nodes with positive weights and no duplicate edges.
  defp graph_gen do
    gen all(
          n <- integer(1..12),
          raw <-
            list_of(
              tuple({integer(0..(n - 1)), integer(0..(n - 1)), float(min: 0.1, max: 10.0)}),
              max_length: 40
            ),
          y <- list_of(float(min: -100.0, max: 100.0), length: n)
        ) do
      edges = Enum.uniq_by(raw, fn {f, t, _} -> {f, t} end)
      {Graph.from_adjacency(edges, ids: Enum.to_list(0..(n - 1))), edges, y}
    end
  end

  defp row_sums(g) do
    g
    |> Graph.to_adjacency()
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 2))
    |> Map.new(fn {k, ws} -> {k, Enum.sum(ws)} end)
  end

  property "R rows sum to 1 except isolates, which sum to 0" do
    check all({g, _edges, _y} <- graph_gen()) do
      sums = row_sums(Graph.transform(g, :r))
      isolates = MapSet.new(Graph.isolates(g))

      for {id, sum} <- sums do
        if MapSet.member?(isolates, id),
          do: assert(sum == 0.0),
          else: assert_in_delta(sum, 1.0, 1.0e-12)
      end
    end
  end

  property "B weights are in {0, 1}" do
    check all({g, _edges, _y} <- graph_gen()) do
      for {_f, _t, w} <- Graph.to_adjacency(Graph.transform(g, :b)), do: assert(w in [0.0, 1.0])
    end
  end

  property "D weights sum to 1 when there is any edge" do
    check all({g, edges, _y} <- graph_gen(), edges != []) do
      assert_in_delta(Graph.s0(Graph.transform(g, :d)), 1.0, 1.0e-12)
    end
  end

  property "V weights sum to n when there is any edge" do
    check all({g, edges, _y} <- graph_gen(), edges != []) do
      assert_in_delta(Graph.s0(Graph.transform(g, :v)), Graph.n(g), 1.0e-9)
    end
  end

  property "lag of a constant vector under R is that constant for non-isolates, 0 for isolates" do
    check all({g, _edges, _y} <- graph_gen(), c <- float(min: -50.0, max: 50.0)) do
      r = Graph.transform(g, :r)
      lag = Graph.lag(r, List.duplicate(c, Graph.n(r)))
      isolates = MapSet.new(Graph.isolates(g))

      for {id, value} <- Enum.zip(Graph.ids(g), lag) do
        if MapSet.member?(isolates, id),
          do: assert(value == 0.0),
          else: assert_in_delta(value, c, 1.0e-9)
      end
    end
  end

  property "lag is linear: W(y + z) == Wy + Wz" do
    check all({g, _e, y} <- graph_gen(), k <- float(min: -3.0, max: 3.0)) do
      z = Enum.map(y, &(&1 * k + 1.0))
      sum = Graph.lag(g, Enum.zip_with(y, z, &(&1 + &2)))
      parts = Enum.zip_with(Graph.lag(g, y), Graph.lag(g, z), &(&1 + &2))
      for {a, b} <- Enum.zip(sum, parts), do: assert_in_delta(a, b, 1.0e-8)
    end
  end

  property "to_adjacency |> from_adjacency roundtrips" do
    check all({g, _edges, _y} <- graph_gen()) do
      adjacency = Graph.to_adjacency(g)
      again = Graph.from_adjacency(adjacency, ids: Graph.ids(g))
      assert Graph.to_adjacency(again) == adjacency
      assert Graph.ids(again) == Graph.ids(g)
    end
  end

  property "a symmetrised graph is symmetric and S1 = 2 * sum of squares for it" do
    check all({_g, edges, _y} <- graph_gen()) do
      pairs =
        Enum.uniq_by(edges ++ Enum.map(edges, fn {f, t, w} -> {t, f, w} end), fn {f, t, _} ->
          {f, t}
        end)

      # make weights symmetric: take the weight of the smaller (f, t) orientation
      weight = Map.new(edges, fn {f, t, w} -> {{min(f, t), max(f, t)}, w} end)
      sym = Enum.map(pairs, fn {f, t, _} -> {f, t, weight[{min(f, t), max(f, t)}]} end)
      ids = sym |> Enum.flat_map(fn {f, t, _} -> [f, t] end) |> Enum.uniq()
      g = Graph.from_adjacency(sym, ids: ids)
      assert Graph.symmetric?(g)
      # for symmetric W: s1 = 2 * sum w^2
      assert_in_delta(Graph.s1(g), 2.0 * Enum.sum(for {_, _, w} <- sym, do: w * w), 1.0e-8)
    end
  end
end
