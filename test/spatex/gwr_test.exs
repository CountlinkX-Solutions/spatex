defmodule Spatex.GWRTest do
  use ExUnit.Case, async: true

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, Result}

  doctest Spatex.GWR

  # 7 x 7 jittered lattice with a slope that grows to the east.
  defp sample do
    :rand.seed(:exsss, {11, 22, 33})

    pts =
      for r <- 0..6, c <- 0..6 do
        {c + :rand.uniform() * 0.4, r + :rand.uniform() * 0.4}
      end

    x = for _ <- pts, do: :rand.uniform() * 4 - 2
    y = for {{px, py}, xv} <- Enum.zip(pts, x), do: 1 + 0.3 * py + (0.5 + 0.2 * px) * xv
    {pts, y, x}
  end

  describe "fit/4" do
    test "returns the result struct with consistent shapes" do
      {pts, y, x} = sample()
      r = GWR.fit(pts, y, [x], bw: 20)

      assert %Result{n: 49, k: 2, bw: 20, kernel: :bisquare, fixed: false} = r
      assert length(r.params) == 49 and Enum.all?(r.params, &(length(&1) == 2))
      assert length(r.bse) == 49 and length(r.tvalues) == 49
      assert length(r.predy) == 49 and length(r.resid) == 49
      assert r.tr_s > 2 and r.tr_s < 49
      assert r.enp == r.tr_s
      assert length(r.adj_alpha) == 3
      assert r.warnings == []
    end

    test "reports a fixed distance bandwidth as given and an adaptive one as a count" do
      {pts, y, x} = sample()
      assert GWR.fit(pts, y, [x], bw: 2.5, fixed: true).bw == 2.5
      assert GWR.fit(pts, y, [x], bw: 20.0).bw == 20
    end

    test "an infinite bandwidth is the global OLS fit" do
      {pts, y, x} = sample()
      r = GWR.fit(pts, y, [x], bw: :infinity)
      ols = Spatex.Regression.ols(y, [x], nonspat_diag: false)

      [b0, b1] = ols.betas
      assert Enum.all?(r.params, fn [a, b] -> abs(a - b0) < 1.0e-9 and abs(b - b1) < 1.0e-9 end)
      assert_in_delta r.tr_s, 2.0, 1.0e-9
      assert_in_delta r.r2, ols.r2, 1.0e-9
    end

    test "kernels, fixed and adaptive bandwidths all fit" do
      {pts, y, x} = sample()

      for kernel <- [:bisquare, :gaussian, :exponential] do
        assert %Result{} = GWR.fit(pts, y, [x], bw: 15, kernel: kernel)
        assert %Result{fixed: true} = GWR.fit(pts, y, [x], bw: 3.0, kernel: kernel, fixed: true)
      end
    end

    test "x may be a single column or a name => column map" do
      {pts, y, x} = sample()
      a = GWR.fit(pts, y, x, bw: 20)
      b = GWR.fit(pts, y, [x], bw: 20)
      c = GWR.fit(pts, y, %{"x" => x}, bw: 20)
      assert a.params == b.params and b.params == c.params
    end

    test "results follow the observations when rows are permuted" do
      {pts, y, x} = sample()
      order = Enum.sort_by(0..48, &:erlang.phash2({&1, 7}))
      pick = fn list -> Enum.map(order, &Enum.at(list, &1)) end

      a = GWR.fit(pts, y, [x], bw: 18, kernel: :gaussian)
      b = GWR.fit(pick.(pts), pick.(y), [pick.(x)], bw: 18, kernel: :gaussian)

      for {pa, pb} <- Enum.zip(pick.(a.params), b.params), {u, v} <- Enum.zip(pa, pb) do
        assert_in_delta u, v, 1.0e-10
      end

      assert_in_delta a.aicc, b.aicc, 1.0e-9
    end

    test "a fractional adaptive bandwidth is truncated with a warning" do
      {pts, y, x} = sample()
      r = GWR.fit(pts, y, [x], bw: 20.7)
      assert [warning] = r.warnings
      assert warning =~ "truncated"
      assert r.params == GWR.fit(pts, y, [x], bw: 20).params
    end

    test "constant columns are dropped with a warning and constant: false keeps the design" do
      {pts, y, x} = sample()
      ones = List.duplicate(3.0, 49)
      r = GWR.fit(pts, y, [x, ones], bw: 20)
      assert r.k == 2
      assert Enum.any?(r.warnings, &(&1 =~ "constant"))

      r2 = GWR.fit(pts, y, [ones, x], bw: 20, constant: false)
      assert r2.k == 2
    end

    test "filter_tvals zeroes insignificant t-values" do
      {pts, y, x} = sample()
      r = GWR.fit(pts, y, [x], bw: 25)
      filtered = GWR.filter_tvals(r, critical_t: 2.0)

      for {row, frow} <- Enum.zip(r.tvalues, filtered), {t, f} <- Enum.zip(row, frow) do
        if abs(t) < 2.0, do: assert(f == 0.0), else: assert(f == t)
      end

      assert GWR.critical_tval(r) > 1.9
      assert GWR.critical_tval(r, 0.05) < GWR.critical_tval(r, 0.01)
    end

    test "rejects invalid input" do
      {pts, y, x} = sample()
      assert_raise ArgumentError, ~r/bw/, fn -> GWR.fit(pts, y, [x], []) end
      assert_raise ArgumentError, ~r/bw/, fn -> GWR.fit(pts, y, [x], bw: 1) end
      assert_raise ArgumentError, ~r/bw/, fn -> GWR.fit(pts, y, [x], bw: 500) end
      assert_raise ArgumentError, ~r/bw/, fn -> GWR.fit(pts, y, [x], bw: -1.0, fixed: true) end
      assert_raise ArgumentError, ~r/kernel/, fn -> GWR.fit(pts, y, [x], bw: 20, kernel: :tri) end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        GWR.fit(pts, y, [x], bw: 20, foo: 1)
      end

      assert_raise ArgumentError, ~r/length/, fn -> GWR.fit(pts, tl(y), [x], bw: 20) end
      assert_raise ArgumentError, ~r/max_n/, fn -> GWR.fit(pts, y, [x], bw: 20, max_n: 10) end
    end
  end

  describe "select_bandwidth/4" do
    test "finds an adaptive bandwidth inside explicit bounds" do
      {pts, y, x} = sample()
      b = GWR.select_bandwidth(pts, y, [x], bw_min: 10, bw_max: 49)
      assert %Bandwidth{criterion: :aicc, fixed: false} = b
      assert is_integer(b.bw) or b.bw == trunc(b.bw)
      assert b.bw >= 10 and b.bw <= 49
      assert is_float(b.score) and b.history != []
    end

    test "a small sample without bounds raises a clear error" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/n >= 40 \+ 2k/, fn ->
        GWR.select_bandwidth(Enum.take(pts, 30), Enum.take(y, 30), [Enum.take(x, 30)], [])
      end
    end

    test "criteria and the interval search" do
      {pts, y, x} = sample()

      for criterion <- [:aicc, :aic, :bic, :cv] do
        b =
          GWR.select_bandwidth(pts, y, [x],
            fixed: true,
            kernel: :gaussian,
            criterion: criterion
          )

        assert b.bw > 0 and b.criterion == criterion
      end

      b =
        GWR.select_bandwidth(pts, y, [x],
          fixed: true,
          kernel: :gaussian,
          search: :interval,
          bw_min: 1.0,
          bw_max: 4.0,
          interval: 0.5
        )

      assert length(b.history) == 7

      assert_raise ArgumentError, ~r/interval/, fn ->
        GWR.select_bandwidth(pts, y, [x],
          fixed: true,
          search: :interval,
          bw_min: 1.0,
          bw_max: 4.0
        )
      end
    end

    test "include_global also evaluates the global model" do
      {pts, _y, x} = sample()
      y = for {xv, i} <- Enum.with_index(x), do: 1.0 + 2.0 * xv + 0.1 * :math.sin(i)
      b = GWR.select_bandwidth(pts, y, [x], bw_min: 10, include_global: true)
      assert {:infinity, _} = List.last(b.history)
      assert b.bw == :infinity or is_integer(b.bw)
    end

    test "rejects invalid options" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/criterion/, fn ->
        GWR.select_bandwidth(pts, y, [x], criterion: :nope)
      end

      assert_raise ArgumentError, ~r/include_global/, fn ->
        GWR.select_bandwidth(pts, y, [x], bw_max: 40, include_global: true)
      end
    end
  end
end
