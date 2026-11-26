defmodule Spatex.GWRHatSearchTest do
  use ExUnit.Case, async: true

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, Result}

  # 9 x 9 jittered lattice (n = 81) with a slope that grows to the east.
  defp sample do
    :rand.seed(:exsss, {5, 6, 7})

    pts =
      for r <- 0..8, c <- 0..8 do
        {c + :rand.uniform() * 0.4, r + :rand.uniform() * 0.4}
      end

    x1 = for _ <- pts, do: :rand.uniform() * 4 - 2

    y =
      for {{px, py}, a} <- Enum.zip(pts, x1),
          do: 1 + 0.3 * py + (0.5 + 0.2 * px) * a + :rand.uniform() * 0.1

    {pts, y, [x1]}
  end

  describe "hat_matrix option of fit/4" do
    test "is off by default" do
      {pts, y, x} = sample()
      assert %Result{hat_matrix: nil} = GWR.fit(pts, y, x, bw: 30)
    end

    test "true returns S as a list of rows consistent with the fit" do
      {pts, y, x} = sample()
      r = GWR.fit(pts, y, x, bw: 30, hat_matrix: true)
      assert length(r.hat_matrix) == 81
      assert Enum.all?(r.hat_matrix, &(length(&1) == 81))

      diag = for {row, i} <- Enum.with_index(r.hat_matrix), do: Enum.at(row, i)
      assert_close(diag, r.influ)
      assert_close(Enum.sum(diag), r.tr_s)

      # S y reproduces the fitted values.
      fitted =
        for row <- r.hat_matrix,
            do: row |> Enum.zip(y) |> Enum.map(fn {s, v} -> s * v end) |> Enum.sum()

      assert_close(fitted, r.predy)
    end

    test ":tensor returns an {n, n} f64 tensor with the same values" do
      {pts, y, x} = sample()
      rows = GWR.fit(pts, y, x, bw: 30, hat_matrix: true).hat_matrix
      t = GWR.fit(pts, y, x, bw: 30, hat_matrix: :tensor).hat_matrix
      assert Nx.shape(t) == {81, 81}
      assert Nx.type(t) == {:f, 64}
      assert Nx.to_list(t) == rows
    end

    test "false and the guard" do
      {pts, y, x} = sample()
      assert GWR.fit(pts, y, x, bw: 30, hat_matrix: false).hat_matrix == nil

      assert_raise ArgumentError, ~r/max_cells/, fn ->
        GWR.fit(pts, y, x, bw: 30, hat_matrix: true, max_cells: 6_000)
      end

      # The guard only concerns the hat matrix.
      assert %Result{hat_matrix: nil} = GWR.fit(pts, y, x, bw: 30, max_cells: 6_000)
    end

    test "rejects invalid values and is not a predict option" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/hat_matrix/, fn ->
        GWR.fit(pts, y, x, bw: 30, hat_matrix: :yes)
      end

      assert_raise ArgumentError, ~r/max_cells/, fn ->
        GWR.fit(pts, y, x, bw: 30, hat_matrix: true, max_cells: 0)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        GWR.predict(pts, y, x, pts, x, bw: 30, hat_matrix: true)
      end
    end
  end

  describe "search: :scipy" do
    test "returns a float count with no history and the optimizer fields" do
      {pts, y, x} = sample()
      b = GWR.select_bandwidth(pts, y, x, search: :scipy)
      assert %Bandwidth{history: [], nfev: nfev, converged: true} = b
      assert is_float(b.bw) and b.bw >= 44 and b.bw <= 81
      assert nfev > 3
      assert b.score == GWR.fit(pts, y, x, bw: b.bw).aicc
    end

    test "golden section and interval searches leave the optimizer fields empty" do
      {pts, y, x} = sample()
      b = GWR.select_bandwidth(pts, y, x)
      assert %Bandwidth{nfev: nil, converged: nil} = b
    end

    test "validates its options" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/distinct/, fn ->
        GWR.select_bandwidth(pts, y, x, search: :scipy, bw_min: 50, bw_max: 50)
      end

      assert_raise ArgumentError, ~r/scipy/, fn ->
        GWR.select_bandwidth(pts, y, x, search: :scipy, include_global: true)
      end

      assert_raise ArgumentError, ~r/search/, fn ->
        GWR.select_bandwidth(pts, y, x, search: :simplex)
      end
    end
  end

  describe "x_glob" do
    test "only moves the default adaptive bounds" do
      {pts, y, x} = sample()
      glob = [for(_ <- pts, do: :rand.uniform())]
      a = GWR.select_bandwidth(pts, y, x, search: :scipy)
      b = GWR.select_bandwidth(pts, y, x, search: :scipy, x_glob: glob)
      # 40 + 2 * (1 + 1) = 44 versus 40 + 2 * (1 + 1 + 1) = 46 neighbors: a different search.
      assert a.bw != b.bw
      assert b.bw >= 46

      # Explicit bounds make it irrelevant.
      opts = [search: :scipy, bw_min: 50, bw_max: 70]

      assert GWR.select_bandwidth(pts, y, x, opts) ==
               GWR.select_bandwidth(pts, y, x, [x_glob: glob] ++ opts)
    end

    test "accepts the shapes of x and checks the length" do
      {pts, y, x} = sample()
      col = for(_ <- pts, do: :rand.uniform())
      one = GWR.select_bandwidth(pts, y, x, x_glob: col)
      assert one == GWR.select_bandwidth(pts, y, x, x_glob: [col])
      assert one == GWR.select_bandwidth(pts, y, x, x_glob: %{"a" => col})
      assert one == GWR.select_bandwidth(pts, y, x, x_glob: Nx.tensor(Enum.map(col, &[&1])))

      assert_raise ArgumentError, ~r/x_glob/, fn ->
        GWR.select_bandwidth(pts, y, x, x_glob: [Enum.take(col, 10)])
      end
    end
  end

  defp assert_close(a, b) when is_list(a) do
    assert length(a) == length(b)
    Enum.zip(a, b) |> Enum.each(fn {u, v} -> assert_close(u, v) end)
  end

  defp assert_close(a, b), do: assert(abs(a - b) <= 1.0e-9 * (1 + abs(b)))
end
