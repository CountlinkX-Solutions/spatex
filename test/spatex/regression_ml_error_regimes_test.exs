defmodule Spatex.RegressionMLErrorRegimesTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Reference, Regression}

  # Two disconnected rings of ten observations: the regime subgraphs are the rings themselves.
  @n 20
  @y [
    3.0,
    1.5,
    4.0,
    1.0,
    5.5,
    9.0,
    2.5,
    6.0,
    4.5,
    2.0,
    7.0,
    3.5,
    8.0,
    1.2,
    5.1,
    2.2,
    6.4,
    3.3,
    4.8,
    0.7
  ]
  @x1 [
    1.0,
    2.0,
    0.5,
    4.0,
    3.0,
    2.0,
    6.0,
    5.0,
    1.5,
    3.5,
    2.5,
    4.5,
    7.0,
    0.2,
    3.1,
    5.6,
    1.1,
    2.9,
    4.2,
    6.3
  ]
  @x2 [
    0.3,
    1.7,
    2.2,
    0.9,
    3.8,
    1.1,
    2.7,
    4.4,
    0.6,
    3.0,
    1.9,
    2.4,
    0.8,
    5.0,
    1.3,
    2.6,
    3.7,
    0.4,
    4.1,
    1.6
  ]
  @regimes List.duplicate(:a, 10) ++ List.duplicate(:b, 10)

  defp ring(offset) do
    for i <- 0..9, j <- [rem(i + 1, 10), rem(i + 9, 10)], do: {offset + i, offset + j}
  end

  defp graph do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ [{9, 10}, {10, 9}, {0, 19}, {19, 0}])

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
    |> Graph.transform(:r)
  end

  defp ring_graph(offset) do
    {f, t} = ring(offset) |> Enum.map(fn {a, b} -> {a - offset, b - offset} end) |> Enum.unzip()

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..9))
    |> Graph.transform(:r)
  end

  defp fit(opts \\ []), do: Regression.ml_error_regimes(@y, [@x1, @x2], @regimes, graph(), opts)

  test "the default is the pooled fit with one lambda, in spreg's coefficient order" do
    r = fit()
    assert r.mode == :pooled and r.regimes == [:a, :b]

    assert r.name_x ==
             ["a_CONSTANT", "a_var_1", "a_var_2", "b_CONSTANT", "b_var_1", "b_var_2", "lambda"]

    assert length(r.betas) == 7 and {r.kr, r.kf, r.nr} == {3, 1, 2}
    # k does not count lambda (as ml_error/4); the information criteria use it
    assert r.k == 6
    assert r.lam == List.last(r.betas) and r.lams == nil and r.multi == nil
    assert length(r.vm) == 7 and length(r.vm1) == 2
    assert length(r.chow.regi) == 3 and elem(r.chow.joint, 1) >= 0.0
    assert_in_delta r.utu, Enum.sum(Enum.map(r.u, &(&1 * &1))), 1.0e-9
    assert_in_delta r.aic, 2 * 6 - 2 * r.logll, 1.0e-9
  end

  test "the lagged design is the lag of the original columns expanded by regime" do
    # With a graph whose regimes touch, W regimes(X) differs from regimes(W X): the pooled fit
    # must equal the base error model fed spreg's lagged design (checked via the likelihood at
    # lambda 0, where both agree, and a distinct value at lambda 0.4 on the border rows).
    r = Regression.ml_error_regimes_at(@y, [@x1, @x2], @regimes, graph(), 0.4)
    naive_cols = regime_design()
    naive = Regression.ml_error_at(@y, naive_cols, graph(), 0.4, constant: false)
    assert r.lam == 0.4
    refute_in_delta r.logll, naive.logll, 1.0e-6

    at0 = Regression.ml_error_regimes_at(@y, [@x1, @x2], @regimes, graph(), 0.0)
    base0 = Regression.ml_error_at(@y, naive_cols, graph(), 0.0, constant: false)
    Reference.assert_close(at0.betas, base0.betas, 1.0e-10)
  end

  defp regime_design do
    z = fn col, l -> Enum.zip_with(col, @regimes, &if(&2 == l, do: &1, else: 0.0)) end
    one = List.duplicate(1.0, @n)
    for l <- [:a, :b], c <- [one, @x1, @x2], do: z.(c, l)
  end

  test "on disconnected regimes the pooled fit is the base fit on the regime design" do
    # two rings without edges between them: W regimes(X) = regimes(W X)
    {f, t} = Enum.unzip(ring(0) ++ ring(10))

    g =
      f
      |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
      |> Graph.transform(:r)

    r = Regression.ml_error_regimes(@y, [@x1, @x2], @regimes, g)
    base = Regression.ml_error(@y, regime_design(), g, constant: false)
    Reference.assert_close(r.betas, base.betas, 1.0e-12)
    assert r.logll == base.logll
  end

  test "global columns follow the regime blocks and lambda is the last global" do
    r = fit(cols2regi: [true, false], constant_regi: :one)

    assert r.name_x ==
             ["a_var_1", "b_var_1", "_Global_CONSTANT", "_Global_var_2", "lambda"]

    assert {r.kr, r.kf} == {1, 3}
    assert Enum.map(r.chow.regi, &elem(&1, 0)) == ["var_1"]
  end

  test "regime_err_sep fits one error model per regime on its own subgraph" do
    r = fit(regime_err_sep: true)
    assert r.mode == :separate and r.ml == nil and r.lam == nil
    assert Enum.sort(Map.keys(r.multi)) == [:a, :b]

    for {label, offset, cols} <- [{:a, 0, 0..9}, {:b, 10, 10..19}] do
      take = fn v -> Enum.slice(v, cols) end
      sub = Regression.ml_error(take.(@y), [take.(@x1), take.(@x2)], ring_graph(offset))
      assert_in_delta r.multi[label].lam, sub.lam, 1.0e-12
      Reference.assert_close(r.multi[label].betas, sub.betas, 1.0e-12)
      assert r.multi[label].name_x == Enum.map(sub.name_x, &"#{label}_#{&1}")
    end

    assert r.betas == r.multi.a.betas ++ r.multi.b.betas
    assert r.lams == %{a: r.multi.a.lam, b: r.multi.b.lam}
    assert length(r.vm) == 8 and Enum.at(Enum.at(r.vm, 0), 4) == 0.0
    assert {r.kr, r.kf} == {4, 0}
    assert List.last(r.chow.regi) |> elem(0) == "lambda"
    assert r.logll == nil and r.aic == nil
  end

  test "the methods agree on a row-standardized graph" do
    full = fit()
    lu = fit(method: :lu)
    assert_in_delta full.lam, lu.lam, 1.0e-7
    assert_in_delta fit(method: :ord).lam, full.lam, 1.0e-6
  end

  test "a column constant within a regime is dropped from the separate fits with a warning" do
    flag = Enum.map(@regimes, &if(&1 == :a, do: 1.0, else: 0.0))
    # the dummy is constant within each regime
    r =
      Regression.ml_error_regimes(@y, [@x1, flag], @regimes, graph(),
        regime_err_sep: true,
        name_x: ["x1", "flag"]
      )

    assert r.name_x == ["a_CONSTANT", "a_x1", "a_lambda", "b_CONSTANT", "b_x1", "b_lambda"]
    assert r.kr == 3
    assert Enum.any?(r.warnings, &String.contains?(&1, "removed for being constant"))
  end

  test "regime_err_sep needs every column by regime and a constant per regime" do
    for opts <- [[cols2regi: [true, false]], [constant_regi: :one]] do
      assert_raise ArgumentError, ~r/must vary across regimes/, fn ->
        fit([regime_err_sep: true] ++ opts)
      end
    end
  end

  test "bad options and inputs raise" do
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(slx_lags: 1) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(regime_lag_sep: true) end
    assert_raise ArgumentError, ~r/constant_regi/, fn -> fit(constant_regi: :few) end
    assert_raise ArgumentError, ~r/cols2regi/, fn -> fit(cols2regi: [true]) end

    assert_raise ArgumentError, ~r/regime_err_sep must be a boolean/, fn ->
      fit(regime_err_sep: 1)
    end

    assert_raise ArgumentError, ~r/at least one column/, fn ->
      fit(cols2regi: [false, false], constant_regi: :one)
    end

    assert_raise ArgumentError, ~r/graph has/, fn ->
      Regression.ml_error_regimes(@y, [@x1], @regimes, ring_graph(0))
    end

    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.ml_error_regimes(@y, [@x1], List.duplicate(1, @n), graph())
    end

    assert_raise ArgumentError, ~r/one value per regime/, fn ->
      Regression.ml_error_regimes_at(@y, [@x1, @x2], @regimes, graph(), [0.1],
        regime_err_sep: true
      )
    end
  end
end
