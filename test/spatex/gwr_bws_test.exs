defmodule Spatex.GWRBwsTest do
  use ExUnit.Case, async: true

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, SpatialVariability}

  defp sample(n \\ 6) do
    pts = for r <- 0..(n - 1), c <- 0..(n - 1), do: {c + 0.07 * rem(r * 5 + c, 4), r + 0.05 * c}
    x = for {px, _} <- pts, do: :math.sin(px)

    y =
      for {{px, py}, a} <- Enum.zip(pts, x),
          do: 1.0 + 0.2 * px * a + 0.1 * py + :math.cos(px * py)

    {pts, y, x}
  end

  describe "bandwidth_intervals/2" do
    test "keeps the bandwidths whose cumulative Akaike weight reaches the level" do
      # score gaps 0, 1, 2, 10 -> Akaike weights ~ .505, .306, .186, .003
      history = [{10, 100.0}, {20, 101.0}, {30, 102.0}, {40, 110.0}]
      b = %Bandwidth{history: history}
      # cumulative weights .505, .811, .997, 1: the entries below the level, plus one
      assert GWR.bandwidth_intervals(b, level: 0.5) == {10, 10}
      assert GWR.bandwidth_intervals(b, level: 0.8) == {10, 20}
      assert GWR.bandwidth_intervals(b, level: 0.9) == {10, 30}
      assert GWR.bandwidth_intervals(b) == {10, 30}
      assert GWR.bandwidth_intervals(b, level: 0.999) == {10, 40}
    end

    test "a single evaluation gives a point interval" do
      assert GWR.bandwidth_intervals(%Bandwidth{history: [{7, 1.0}]}) == {7, 7}
    end

    test "rejects invalid levels, empty histories and other inputs" do
      b = %Bandwidth{history: [{10, 1.0}]}
      assert_raise ArgumentError, ~r/level/, fn -> GWR.bandwidth_intervals(b, level: 1.5) end
      assert_raise ArgumentError, ~r/level/, fn -> GWR.bandwidth_intervals(b, level: 0) end
      assert_raise ArgumentError, ~r/unknown option/, fn -> GWR.bandwidth_intervals(b, lvl: 1) end

      assert_raise ArgumentError, ~r/history/, fn ->
        GWR.bandwidth_intervals(%Bandwidth{history: []})
      end

      assert_raise ArgumentError, ~r/Bandwidth|MGWRResult/, fn ->
        GWR.bandwidth_intervals(:nope)
      end
    end

    test "works on a real golden-section search and rejects the scipy one" do
      {pts, y, x} = sample()
      found = GWR.select_bandwidth(pts, y, [x], fixed: true, kernel: :gaussian)
      {lo, hi} = GWR.bandwidth_intervals(found)
      assert lo <= hi
      assert lo >= found.history |> Enum.map(&elem(&1, 0)) |> Enum.min()

      scipy = GWR.select_bandwidth(pts, y, [x], fixed: true, kernel: :gaussian, search: :scipy)
      assert_raise ArgumentError, ~r/history/, fn -> GWR.bandwidth_intervals(scipy) end
    end

    test "MGWR needs the search histories of the backfitting" do
      {pts, y, x} = sample(7)
      r = GWR.mgwr(pts, y, [x], fixed: true, standardize: true, inference: false)
      assert length(r.sel_hist) >= r.k
      intervals = GWR.bandwidth_intervals(r)
      assert length(intervals) == r.k
      assert Enum.all?(intervals, fn {lo, hi} -> lo <= hi end)

      assert_raise ArgumentError, ~r/search histories/, fn ->
        GWR.bandwidth_intervals(%{r | sel_hist: []})
      end
    end
  end

  describe "spatial_variability/4" do
    test "is reproducible for a seed, records a drawn seed and keeps the draws" do
      {pts, y, x} = sample()
      opts = [fixed: true, kernel: :gaussian, permutations: 5]
      a = GWR.spatial_variability(pts, y, [x], opts ++ [seed: 3])
      b = GWR.spatial_variability(pts, y, [x], opts ++ [seed: 3])
      assert %SpatialVariability{k: 2, permutations: 5, seed: 3} = a
      assert a.p_values == b.p_values and a.simulated_sd == b.simulated_sd
      assert length(a.simulated_sd) == 5 and Enum.all?(a.simulated_sd, &(length(&1) == 2))
      assert Enum.all?(a.p_values, &(&1 >= 0 and &1 <= 1))

      drawn = GWR.spatial_variability(pts, y, [x], opts)
      assert is_integer(drawn.seed)
      again = GWR.spatial_variability(pts, y, [x], opts ++ [seed: drawn.seed])
      assert again.p_values == drawn.p_values
    end

    test "keep_simulations: false drops the draws without changing the p-values" do
      {pts, y, x} = sample()
      opts = [fixed: true, kernel: :gaussian, permutations: 4, seed: 9]
      kept = GWR.spatial_variability(pts, y, [x], opts)
      dropped = GWR.spatial_variability(pts, y, [x], opts ++ [keep_simulations: false])
      assert dropped.simulated_sd == nil
      assert dropped.p_values == kept.p_values
    end

    test "keep_simulations: :tensor returns an {permutations, k} tensor" do
      {pts, y, x} = sample()
      opts = [fixed: true, kernel: :gaussian, permutations: 4, seed: 9]
      r = GWR.spatial_variability(pts, y, [x], opts ++ [keep_simulations: :tensor])
      assert Nx.shape(r.simulated_sd) == {4, 2}
      assert Nx.type(r.simulated_sd) == {:f, 64}
    end

    test "the observed statistic is the standard deviation of the fitted surfaces" do
      {pts, y, x} = sample()
      b = GWR.select_bandwidth(pts, y, [x], fixed: true, kernel: :gaussian)
      fit = GWR.fit(pts, y, [x], fixed: true, kernel: :gaussian, bw: b.bw)

      want =
        fit.params
        |> Enum.zip()
        |> Enum.map(fn col ->
          vals = Tuple.to_list(col)
          m = Enum.sum(vals) / length(vals)
          :math.sqrt(Enum.sum(for v <- vals, do: (v - m) * (v - m)) / length(vals))
        end)

      r = GWR.spatial_variability(pts, y, [x], fixed: true, kernel: :gaussian, permutations: 2)
      assert r.bw == b.bw

      for {a, e} <- Enum.zip(r.observed_sd, want), do: assert_in_delta(a, e, 1.0e-10)
    end

    test "validates options and the size guard" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/permutations/, fn ->
        GWR.spatial_variability(pts, y, [x], permutations: 0)
      end

      assert_raise ArgumentError, ~r/seed/, fn ->
        GWR.spatial_variability(pts, y, [x], seed: -1)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        GWR.spatial_variability(pts, y, [x], n_iters: 3)
      end

      assert_raise ArgumentError, ~r/max_n/, fn ->
        GWR.spatial_variability(pts, y, [x], max_n: 10)
      end

      assert_raise ArgumentError, ~r/keep_simulations/, fn ->
        GWR.spatial_variability(pts, y, [x], keep_simulations: :maybe)
      end
    end
  end
end
