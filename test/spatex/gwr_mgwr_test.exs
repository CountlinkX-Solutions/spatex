defmodule Spatex.MGWRTest do
  use ExUnit.Case, async: true

  alias Spatex.GWR
  alias Spatex.GWR.MGWRResult

  # 8 x 8 jittered lattice, one regional and one near-global effect.
  defp sample do
    :rand.seed(:exsss, {5, 6, 7})

    pts = for r <- 0..7, c <- 0..7, do: {c + :rand.uniform() * 0.4, r + :rand.uniform() * 0.4}
    x1 = for _ <- pts, do: :rand.normal()
    x2 = for _ <- pts, do: :rand.normal()

    y =
      for {{px, _py}, a, b} <- Enum.zip([pts, x1, x2]) do
        1.0 + (0.3 + 0.2 * px) * a + 0.8 * b + 0.2 * :rand.normal()
      end

    {pts, y, [x1, x2]}
  end

  defp standardize(col) do
    n = length(col)
    mean = Enum.sum(col) / n
    sd = :math.sqrt(Enum.sum(Enum.map(col, &((&1 - mean) * (&1 - mean)))) / n)
    Enum.map(col, &((&1 - mean) / sd))
  end

  test "returns the result struct with consistent shapes" do
    {pts, y, x} = sample()
    r = GWR.mgwr(pts, standardize(y), Enum.map(x, &standardize/1), fixed: true)

    assert %MGWRResult{n: 64, k: 3, fixed: true, kernel: :bisquare, converged: true} = r
    assert length(r.bws) == 3 and length(hd(r.bws_history)) == 3
    assert length(r.params) == 64 and Enum.all?(r.params, &(length(&1) == 3))
    assert length(r.bse) == 64 and length(r.tvalues) == 64
    assert length(r.enp_j) == 3
    assert_in_delta r.tr_s, Enum.sum(r.enp_j), 1.0e-12
    assert_in_delta r.sigma2, r.resid_ss / (64 - r.tr_s), 1.0e-12
    assert r.influ == nil and r.std_res == nil and r.cooks_d == nil
    assert Enum.all?(r.enp_j, &(&1 >= 0.9))
  end

  test "standardize: true equals fitting pre-standardized data" do
    {pts, y, x} = sample()
    a = GWR.mgwr(pts, standardize(y), Enum.map(x, &standardize/1), fixed: true)
    b = GWR.mgwr(pts, y, x, fixed: true, standardize: true)
    assert a.bws == b.bws
    assert_in_delta a.aicc, b.aicc, 1.0e-9
    assert a.params == b.params
  end

  test "inference: false skips the hat-matrix pass" do
    {pts, y, x} = sample()
    ys = standardize(y)
    xs = Enum.map(x, &standardize/1)
    full = GWR.mgwr(pts, ys, xs, fixed: true)
    lite = GWR.mgwr(pts, ys, xs, fixed: true, inference: false)

    assert lite.params == full.params
    assert lite.bws == full.bws
    assert lite.bse == nil and lite.tvalues == nil and lite.enp_j == nil
    assert lite.aicc == nil and lite.tr_s == nil and lite.critical_tval == nil
    assert_in_delta lite.resid_ss, full.resid_ss, 1.0e-12
    assert_in_delta lite.r2, full.r2, 1.0e-12
  end

  test "inference does not depend on the chunk size" do
    {pts, y, x} = sample()
    ys = standardize(y)
    xs = Enum.map(x, &standardize/1)
    a = GWR.mgwr(pts, ys, xs, fixed: true, chunk_size: 64)
    b = GWR.mgwr(pts, ys, xs, fixed: true, chunk_size: 7)

    for {ra, rb} <- Enum.zip(a.bse, b.bse), {u, v} <- Enum.zip(ra, rb) do
      assert_in_delta u, v, 1.0e-10
    end

    for {u, v} <- Enum.zip(a.enp_j, b.enp_j), do: assert_in_delta(u, v, 1.0e-10)
  end

  test "the n guard refuses inference on large samples with a clear message" do
    {pts, y, x} = sample()

    assert_raise ArgumentError, ~r/max_n.*inference: false/s, fn ->
      GWR.mgwr(pts, y, x, fixed: true, max_n: 30)
    end

    assert %MGWRResult{bse: nil} = GWR.mgwr(pts, y, x, fixed: true, max_n: 30, inference: false)
  end

  test "multi_bw_min and multi_bw_max bound each covariate (constant first)" do
    {pts, y, x} = sample()

    r =
      GWR.mgwr(pts, standardize(y), Enum.map(x, &standardize/1),
        fixed: true,
        multi_bw_min: [1.0, 1.0, 1.0],
        multi_bw_max: [3.0, 3.0, 3.0]
      )

    assert Enum.all?(r.bws, &(&1 >= 1.0 and &1 <= 3.0))
  end

  test "the outer iteration limit reports non-convergence" do
    {pts, y, x} = sample()
    r = GWR.mgwr(pts, standardize(y), Enum.map(x, &standardize/1), fixed: true, max_iter_multi: 1)
    assert r.iterations == 1
    assert Enum.any?(r.warnings, &(&1 =~ "converge"))
  end

  test "rejects invalid input" do
    {pts, y, x} = sample()
    assert_raise ArgumentError, ~r/family/, fn -> GWR.mgwr(pts, y, x, family: :poisson) end

    assert_raise ArgumentError, ~r/multi_bw_min/, fn ->
      GWR.mgwr(pts, y, x, multi_bw_min: [1.0, 2.0])
    end

    assert_raise ArgumentError, ~r/unknown option/, fn -> GWR.mgwr(pts, y, x, bw: 3) end
    assert_raise ArgumentError, ~r/tol_multi/, fn -> GWR.mgwr(pts, y, x, tol_multi: 0) end

    assert_raise ArgumentError, ~r/n >= 40 \+ 2k/, fn ->
      GWR.mgwr(Enum.take(pts, 30), Enum.take(y, 30), Enum.map(x, &Enum.take(&1, 30)), [])
    end
  end
end
