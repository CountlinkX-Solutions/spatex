defmodule Spatex.NativeGuardsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Dynamics
  alias Spatex.Graph
  alias Spatex.PointPats

  # Every guard must raise ArgumentError *before* allocating; the inputs are tiny and the
  # limits are lowered so that nothing large is ever built.

  @pts for i <- 0..9, do: {i * 1.0, rem(i * 7, 5) * 1.0}

  describe "quadrat grid (max_cells)" do
    @square [{0.0, 0.0}, {100.0, 100.0}, {50.0, 20.0}]

    test "rectangle grid over the limit names the cell count" do
      assert_raise ArgumentError, ~r/16 cells.*max_cells/s, fn ->
        PointPats.quadrat_test(@square, nx: 4, ny: 4, max_cells: 10)
      end

      assert %{num_cells: 16} = PointPats.quadrat_test(@square, nx: 4, ny: 4, max_cells: 16)
    end

    test "cell-size grid and hexagon grid are checked from the bounding box" do
      assert_raise ArgumentError, ~r/cells.*max_cells/s, fn ->
        PointPats.quadrat_test(@square, rectangle_width: 1, rectangle_height: 1, max_cells: 10)
      end

      assert_raise ArgumentError, ~r/cells.*max_cells/s, fn ->
        PointPats.quadrat_test(@square, shape: :hexagon, lh: 1, max_cells: 10)
      end
    end

    test "the default lh of 10 on metre coordinates cannot allocate ~1e9 cells" do
      pts = [{0.0, 0.0}, {500_000.0, 500_000.0}, {1.0, 2.0}]

      assert_raise ArgumentError, ~r/max_cells/, fn ->
        PointPats.quadrat_test(pts, shape: :hexagon)
      end

      assert_raise ArgumentError, ~r/max_cells/, fn ->
        PointPats.quadrat_test(pts, rectangle_width: 1.0e-3, rectangle_height: 1.0e-3)
      end
    end

    test "max_cells must be a positive integer" do
      for bad <- [0, -1, 1.5, "10", nil] do
        assert_raise ArgumentError, ~r/max_cells/, fn ->
          PointPats.quadrat_test(@square, max_cells: bad)
        end
      end
    end
  end

  describe "weights builders (max_edges)" do
    test "default kernel without bandwidth or k builds all pairs and is guarded" do
      assert_raise ArgumentError, ~r/90 edges.*max_edges.*(k:|bandwidth:)/s, fn ->
        Graph.build_kernel(@pts, max_edges: 50)
      end

      assert %Graph{n: 10} = Graph.build_kernel(@pts, max_edges: 90)
    end

    test "non-compact kernel with taper: false is guarded" do
      assert_raise ArgumentError, ~r/max_edges/, fn ->
        Graph.build_kernel(@pts, bandwidth: 2.0, taper: false, max_edges: 50)
      end
    end

    test "fixed bandwidth that reaches everything is guarded" do
      assert_raise ArgumentError, ~r/max_edges.*bandwidth/s, fn ->
        Graph.build_kernel(@pts, kernel: :triangular, bandwidth: 1.0e6, max_edges: 20)
      end
    end

    test "k neighbors and knn are bounded by n * k" do
      assert_raise ArgumentError, ~r/50 edges.*max_edges/s, fn ->
        Graph.build_kernel(@pts, k: 5, max_edges: 20)
      end

      assert_raise ArgumentError, ~r/50 edges.*max_edges/s, fn ->
        Graph.build_knn(@pts, 5, max_edges: 20)
      end

      assert %Graph{} = Graph.build_knn(@pts, 5, max_edges: 50)
    end

    test "distance bands, binary or kernel weighted, are guarded" do
      assert_raise ArgumentError, ~r/max_edges.*threshold/s, fn ->
        Graph.build_distance_band(@pts, 1.0e6, max_edges: 20)
      end

      assert_raise ArgumentError, ~r/max_edges/, fn ->
        Graph.build_distance_band(@pts, 1.0e6, binary: false, kernel: :gaussian, max_edges: 20)
      end
    end

    test "max_edges must be a positive integer" do
      for bad <- [0, -5, 1.5, "x"] do
        assert_raise ArgumentError, ~r/max_edges/, fn ->
          Graph.build_kernel(@pts, bandwidth: 2.0, max_edges: bad)
        end
      end
    end
  end

  describe "Markov class counts" do
    test "markov rejects more than 1000 distinct classes" do
      data = [Enum.to_list(0..1000), Enum.to_list(1000..0//-1)]

      assert_raise ArgumentError, ~r/1001 classes.*1000/s, fn -> Dynamics.markov(data) end
    end

    test "discrete spatial markov rejects more than 1000 distinct classes" do
      n = 1001
      g = Graph.build_knn(for(i <- 1..n, do: {i * 1.0, 0.0}), 1)
      y = for i <- 0..(n - 1), do: [i, rem(i + 1, n)]

      assert_raise ArgumentError, ~r/1001 classes.*1000/s, fn ->
        Dynamics.spatial_markov(y, g, discrete: true)
      end
    end

    test "k or m above the cap and an oversized m * k * k table are rejected" do
      g = Graph.build_knn(for(i <- 1..6, do: {i * 1.0, 0.0}), 1)
      y = for i <- 1..6, do: [i * 1.0, i * 2.0]

      assert_raise ArgumentError, ~r/classes/, fn ->
        Dynamics.spatial_markov(y, g, k: 1001)
      end

      assert_raise ArgumentError, ~r/m \* k \* k/, fn ->
        Dynamics.spatial_markov(y, g, k: 400, m: 400)
      end
    end
  end

  describe "Kendall tau (Knight's algorithm)" do
    # The textbook O(n^2) definition of tau-b with giddy's pair bookkeeping.
    defp naive(x, y) do
      pairs =
        for {{a, b}, i} <- Enum.with_index(Enum.zip(x, y)),
            {{c, d}, j} <- Enum.with_index(Enum.zip(x, y)),
            i < j,
            do: {a - c, b - d}

      count = fn f -> Enum.count(pairs, f) end
      c = count.(fn {dx, dy} -> dx * dy > 0 end)
      d = count.(fn {dx, dy} -> dx * dy < 0 end)
      ey = count.(fn {dx, dy} -> dx == 0 and dy != 0 end)
      ex = count.(fn {dx, dy} -> dy == 0 and dx != 0 end)
      {c, d, ex, ey}
    end

    property "matches the pairwise definition on random data with ties" do
      # Leading {0, 0} and {6, 4} guarantee at least two distinct values in each variable.
      check all(pairs <- list_of({integer(0..6), integer(0..4)}, max_length: 38)) do
        {xs, ys} = Enum.unzip([{0, 0}, {6, 4} | pairs])
        t = Dynamics.kendall_tau(xs, ys)
        {c, d, ex, ey} = naive(xs, ys)

        assert {t.concordant, t.discordant, t.extra_x, t.extra_y} ==
                 {c / 1, d / 1, ex / 1, ey / 1}

        assert_in_delta t.tau, (c - d) / :math.sqrt((c + d + ex) * (c + d + ey)), 1.0e-12
      end
    end

    test "handles a hundred thousand points quickly" do
      xs = for i <- 1..100_000, do: rem(i * 7919, 5003)
      ys = for x <- xs, do: x + rem(x * 31, 97)
      assert Dynamics.kendall_tau(xs, ys).tau > 0.9
    end
  end
end
