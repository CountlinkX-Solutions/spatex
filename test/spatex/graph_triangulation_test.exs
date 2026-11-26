defmodule Spatex.GraphTriangulationTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  # General position: no four points are cocircular.
  @pts [{0, 0}, {4, 0.3}, {1.1, 3}, {5.2, 3.9}, {2.3, 1.4}, {0.2, 5.1}]

  defp edges(g), do: for({f, t, _} <- Graph.to_adjacency(g), f != t, do: {f, t})

  describe "build_triangulation/2" do
    test "defaults to the Delaunay graph with unit weights" do
      g = Graph.build_triangulation(@pts)
      assert %Graph{n: 6, ids: [0, 1, 2, 3, 4, 5], transform: :o} = g
      assert Graph.symmetric?(g)
      assert Enum.all?(Graph.to_adjacency(g), fn {_, _, w} -> w == 1.0 end)
      assert Graph.isolates(g) == []
    end

    test "a single triangle is complete under every method but loses edges to the lune test" do
      tri = [{0, 0}, {4, 0}, {1, 3}]

      assert Graph.cardinalities(Graph.build_triangulation(tri)) == [2, 2, 2]
      assert Graph.cardinalities(Graph.build_triangulation(tri, method: :gabriel)) == [2, 2, 2]
      # RNG: |p1-p2| = sqrt(18) is the longest side, so vertex 0 lies in its lune and prunes it.
      rng = Graph.build_triangulation(tri, method: :relative_neighborhood)
      assert edges(rng) == [{0, 1}, {0, 2}, {1, 0}, {2, 0}]
    end

    test "weights come from the kernel on edge lengths, tapered by a finite bandwidth" do
      tri = [{0, 0}, {3, 0}, {0, 4}]
      g = Graph.build_triangulation(tri, kernel: :triangular, bandwidth: 5.0)
      # Edge lengths 3, 4 and 5 (the hypotenuse is tapered: d == h gives weight 0).
      assert Graph.to_adjacency(g) == [
               {0, 1, 1 - 3 / 5},
               {0, 2, 1 - 4 / 5},
               {1, 0, 1 - 3 / 5},
               {2, 0, 1 - 4 / 5}
             ]
    end

    test "nodes whose edges are all tapered away stay in the graph as islands" do
      pts = [{0, 0}, {1, 0}, {0.4, 0.9}, {9, 9.5}]
      g = Graph.build_triangulation(pts, bandwidth: 2.0)
      assert Graph.n(g) == 4
      assert Graph.isolates(g) == [3]
    end

    test "labels nodes with :ids" do
      g = Graph.build_triangulation(@pts, ids: [:a, :b, :c, :d, :e, :f])
      assert Graph.ids(g) == [:a, :b, :c, :d, :e, :f]
      assert Enum.all?(Graph.to_adjacency(g), fn {f, t, _} -> is_atom(f) and is_atom(t) end)
    end

    test "coincident points raise by default and connect as a clique on request" do
      pts = @pts ++ [{2.3, 1.4}]
      assert_raise ArgumentError, ~r/coincident/, fn -> Graph.build_triangulation(pts) end

      g = Graph.build_triangulation(pts, coincident: :clique)
      assert {4, 6, 1.0} in Graph.to_adjacency(g)
      assert {6, 4, 1.0} in Graph.to_adjacency(g)
      assert Enum.at(Graph.cardinalities(g), 4) == Enum.at(Graph.cardinalities(g), 6)
    end

    test "voronoi keeps Delaunay edges with a shared cell edge and validates its options" do
      # A 3 x 2 lattice: cocircular squares, so the cells meet the diagonals at a point only.
      lattice = for y <- 0..1, x <- 0..2, do: {x, y}
      g = Graph.build_triangulation(lattice, method: :voronoi)
      assert Graph.cardinalities(g) == [2, 3, 2, 2, 3, 2]
      assert Graph.n(Graph.build_triangulation(@pts, method: :voronoi, clip: nil)) == 6

      assert_raise ArgumentError, ~r/rook: true/, fn ->
        Graph.build_triangulation(@pts, method: :voronoi, rook: false)
      end

      assert_raise ArgumentError, ~r/:clip/, fn ->
        Graph.build_triangulation(@pts, method: :voronoi, clip: :alpha_shape)
      end
    end

    test "rejects degenerate and invalid input" do
      assert_raise ArgumentError, ~r/at least 3/, fn ->
        Graph.build_triangulation([{0, 0}, {1, 1}])
      end

      assert_raise ArgumentError, ~r/collinear/, fn ->
        Graph.build_triangulation([{0, 0}, {1, 1}, {2, 2}, {3, 3}])
      end

      assert_raise ArgumentError, ~r/method/, fn ->
        Graph.build_triangulation(@pts, method: :alpha)
      end

      assert_raise ArgumentError, ~r/bandwidth/, fn ->
        Graph.build_triangulation(@pts, bandwidth: -1)
      end

      assert_raise ArgumentError, ~r/kernel/, fn ->
        Graph.build_triangulation(@pts, kernel: :nope)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_triangulation(@pts, k: 3)
      end

      assert_raise ArgumentError, ~r/coincident/, fn ->
        Graph.build_triangulation(@pts, coincident: :jitter)
      end
    end
  end
end
