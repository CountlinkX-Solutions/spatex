defmodule Spatex.GraphKernelTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  @line [{0, 0}, {1, 0}, {2, 0}, {3.5, 0}, {10, 0}]

  defp weight(g, f, t) do
    Enum.find_value(Graph.to_adjacency(g), fn
      {^f, ^t, w} -> w
      _ -> nil
    end)
  end

  describe "build_kernel/2" do
    test "gaussian with a fixed bandwidth tapers beyond it" do
      g = Graph.build_kernel(@line, bandwidth: 1.0)
      assert %Graph{n: 5, ids: [0, 1, 2, 3, 4], transform: :o} = g
      assert_in_delta weight(g, 0, 1), :math.exp(-0.5) / :math.sqrt(2 * :math.pi()), 1.0e-15
      assert weight(g, 0, 2) == nil
      assert Graph.isolates(g) == [3, 4]
    end

    test "defaults to the gaussian kernel and the 25th-percentile bandwidth" do
      assert Graph.to_adjacency(Graph.build_kernel(@line)) ==
               Graph.to_adjacency(Graph.build_kernel(@line, kernel: :gaussian, bandwidth: 1.5))
    end

    test "adaptive bandwidth uses each node's k-th neighbor distance" do
      g = Graph.build_kernel(@line, kernel: :triangular, bandwidth: :adaptive, k: 2)
      h = 2.0 * 1.0000001
      assert_in_delta weight(g, 0, 1), 1 - 1 / h, 1.0e-12
      assert_in_delta weight(g, 0, 2), 1 - 2 / h, 1.0e-12
      assert Graph.cardinalities(g) == [2, 2, 2, 2, 2]
    end

    test "k without a bandwidth uses the largest k-nearest distance" do
      g = Graph.build_kernel(@line, kernel: :triangular, k: 2)
      # largest 2-NN distance: point 4 -> 3 (6.5) and 2 (8.0)
      assert_in_delta weight(g, 0, 1), 1 - 1 / 8, 1.0e-12
    end

    test "kernel aliases and the other libpysal kernels" do
      for k <- [:triangular, :parabolic, :bisquare, :tricube, :cosine, :boxcar, :exponential] do
        assert %Graph{} = Graph.build_kernel(@line, kernel: k, bandwidth: 2.0)
      end

      assert Graph.to_adjacency(Graph.build_kernel(@line, kernel: :epanechnikov, bandwidth: 2.0)) ==
               Graph.to_adjacency(Graph.build_kernel(@line, kernel: :parabolic, bandwidth: 2.0))

      assert Graph.to_adjacency(Graph.build_kernel(@line, kernel: :uniform, bandwidth: 2.5)) ==
               Graph.to_adjacency(Graph.build_kernel(@line, kernel: :boxcar, bandwidth: 2.5))

      id = Graph.build_kernel(@line, kernel: :identity, bandwidth: 2.0)
      assert weight(id, 0, 2) == 2.0
    end

    test "taper: false keeps every pair for the gaussian kernel" do
      g = Graph.build_kernel(@line, bandwidth: 1.0, taper: false)
      assert Graph.nnz(g) == 20
    end

    test "decay rescales the kernel to 1 at distance zero" do
      pts = [{0, 0}, {0, 0}]
      g = Graph.build_kernel(pts, bandwidth: 1.0, decay: true)
      assert weight(g, 0, 1) == 1.0
    end

    test "coincident points raise with k and form cliques on request" do
      pts = [{0, 0}, {0, 0}, {1, 0}, {3, 0}]

      assert_raise ArgumentError, ~r/coincident/, fn ->
        Graph.build_kernel(pts, kernel: :triangular, k: 1)
      end

      g =
        Graph.build_kernel(pts,
          kernel: :triangular,
          bandwidth: :adaptive,
          k: 1,
          coincident: :clique
        )

      assert weight(g, 0, 1) == 1.0
    end

    test "coincident pairs are kept for a fixed bandwidth (libpysal drops them)" do
      g = Graph.build_kernel([{0, 0}, {0, 0}, {1, 0}], kernel: :triangular, bandwidth: 2.0)
      assert weight(g, 0, 1) == 1.0
    end

    test "labels nodes with :ids" do
      g = Graph.build_kernel(@line, bandwidth: 1.0, ids: ~w(a b c d e)a)
      assert {:a, :b} == g |> Graph.to_adjacency() |> hd() |> then(&{elem(&1, 0), elem(&1, 1)})
    end

    test "validates its arguments" do
      assert_raise ArgumentError, ~r/kernel/, fn -> Graph.build_kernel(@line, kernel: :nope) end
      assert_raise ArgumentError, ~r/bandwidth/, fn -> Graph.build_kernel(@line, bandwidth: 0) end

      assert_raise ArgumentError, ~r/bandwidth/, fn ->
        Graph.build_kernel(@line, bandwidth: -2.0)
      end

      assert_raise ArgumentError, ~r/:auto|auto/, fn ->
        Graph.build_kernel(@line, bandwidth: :auto)
      end

      assert_raise ArgumentError, ~r/adaptive.*k|k.*adaptive/, fn ->
        Graph.build_kernel(@line, bandwidth: :adaptive)
      end

      assert_raise ArgumentError, ~r/k must be/, fn -> Graph.build_kernel(@line, k: 0) end
      assert_raise ArgumentError, ~r/at most 4/, fn -> Graph.build_kernel(@line, k: 5) end
      assert_raise ArgumentError, ~r/taper/, fn -> Graph.build_kernel(@line, taper: 2) end
      assert_raise ArgumentError, ~r/decay/, fn -> Graph.build_kernel(@line, decay: :yes) end

      assert_raise ArgumentError, ~r/identity/, fn ->
        Graph.build_kernel(@line, kernel: :identity, bandwidth: 1.0, decay: true)
      end

      assert_raise ArgumentError, ~r/:coincident/, fn ->
        Graph.build_kernel(@line, coincident: :shake)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Graph.build_kernel(@line, bogus: 1)
      end

      assert_raise ArgumentError, ~r/ids/, fn -> Graph.build_kernel(@line, ids: [1]) end
    end
  end

  describe "build_distance_band/3 with kernel weights" do
    test "applies the kernel to the band distances" do
      g =
        Graph.build_distance_band(@line, 1.5,
          binary: false,
          kernel: :triangular,
          bandwidth: 3.0
        )

      assert_in_delta weight(g, 1, 2), 1 - 1 / 3, 1.0e-12
      assert_in_delta weight(g, 2, 3), 0.5, 1.0e-12
      assert Graph.isolates(g) == [4]
    end

    test "default bandwidth is the 25th percentile of the band distances" do
      g = Graph.build_distance_band(@line, 1.5, binary: false, kernel: :gaussian)
      assert_in_delta weight(g, 0, 1), :math.exp(-0.5) / :math.sqrt(2 * :math.pi()), 1.0e-15
    end

    test "kernel options need binary: false" do
      assert_raise ArgumentError, ~r/binary: false/, fn ->
        Graph.build_distance_band(@line, 1.5, kernel: :gaussian)
      end

      assert_raise ArgumentError, ~r/kernel/, fn ->
        Graph.build_distance_band(@line, 1.5, binary: false, bandwidth: 2.0)
      end

      assert_raise ArgumentError, ~r/kernel/, fn ->
        Graph.build_distance_band(@line, 1.5, binary: false, kernel: :nope)
      end
    end
  end
end
