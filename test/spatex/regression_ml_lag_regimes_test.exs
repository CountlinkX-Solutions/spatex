defmodule Spatex.RegressionMLLagRegimesTest do
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

  defp fit(opts \\ []), do: Regression.ml_lag_regimes(@y, [@x1, @x2], @regimes, graph(), opts)

  test "the default is the pooled fit with one rho, in spreg's coefficient order" do
    r = fit()
    assert r.mode == :pooled and r.regimes == [:a, :b]

    assert r.name_x ==
             [
               "a_CONSTANT",
               "a_var_1",
               "a_var_2",
               "b_CONSTANT",
               "b_var_1",
               "b_var_2",
               "_Global_W_dep_var"
             ]

    assert length(r.betas) == 7 and {r.kr, r.kf, r.nr, r.k} == {3, 1, 2, 7}
    assert r.rho == List.last(r.betas) and r.rhos == nil and r.multi == nil
    assert r.ml.n == @n and length(r.vm) == 7 and length(r.vm1) == 8
    assert Enum.map(r.impacts.simple, & &1.name) == ["a_var_1", "a_var_2", "b_var_1", "b_var_2"]
    assert length(r.chow.regi) == 3 and elem(r.chow.joint, 1) >= 0.0
    assert_in_delta r.utu, Enum.sum(Enum.map(r.u, &(&1 * &1))), 1.0e-9
  end

  test "one regime-wide design equals the base fit with every column crossed with the regime" do
    r = fit()
    z = fn col, l -> Enum.zip_with(col, @regimes, &if(&2 == l, do: &1, else: 0.0)) end
    one = List.duplicate(1.0, @n)
    cols = for l <- [:a, :b], c <- [one, @x1, @x2], do: z.(c, l)
    base = Regression.ml_lag(@y, cols, graph(), constant: false)
    Reference.assert_close(r.betas, base.betas, 1.0e-12)
    assert r.logll == base.logll
  end

  test "global columns follow the regime blocks and rho is the last global" do
    r = fit(cols2regi: [true, false], constant_regi: :one)

    assert r.name_x == [
             "a_var_1",
             "b_var_1",
             "_Global_CONSTANT",
             "_Global_var_2",
             "_Global_W_dep_var"
           ]

    assert {r.kr, r.kf} == {1, 3}
    assert Enum.map(r.chow.regi, &elem(&1, 0)) == ["var_1"]
    # impacts skip the constant and rho
    assert Enum.map(r.impacts.simple, & &1.name) == ["a_var_1", "b_var_1", "_Global_var_2"]
  end

  test "regime_lag_sep fits one lag model per regime on its own subgraph" do
    r = fit(regime_lag_sep: true)
    assert r.mode == :separate and r.ml == nil and r.rho == nil
    assert Enum.sort(Map.keys(r.multi)) == [:a, :b]

    for {label, offset, cols} <- [{:a, 0, 0..9}, {:b, 10, 10..19}] do
      take = fn v -> Enum.slice(v, cols) end
      sub = Regression.ml_lag(take.(@y), [take.(@x1), take.(@x2)], ring_graph(offset))
      assert_in_delta r.multi[label].rho, sub.rho, 1.0e-12
      Reference.assert_close(r.multi[label].betas, sub.betas, 1.0e-12)
      assert r.multi[label].name_x == Enum.map(sub.name_x, &"#{label}_#{&1}")
    end

    assert r.betas == r.multi.a.betas ++ r.multi.b.betas
    assert r.rhos == %{a: r.multi.a.rho, b: r.multi.b.rho}
    assert length(r.vm) == 8 and Enum.at(Enum.at(r.vm, 0), 4) == 0.0
    assert {r.kr, r.kf} == {4, 0}
    assert List.last(r.chow.regi) |> elem(0) == "W_dep_var"
    assert r.logll == nil and r.aic == nil
    assert Enum.map(r.multi.a.impacts.simple, & &1.name) == ["a_var_1", "a_var_2"]
  end

  test "spat_impacts: nil reports no multipliers and :all the three methods" do
    assert fit(spat_impacts: nil).sp_multipliers == nil
    assert Enum.sort(Map.keys(fit(spat_impacts: :all).sp_multipliers)) == [:full, :power, :simple]
  end

  test "the methods agree on a row-standardized graph" do
    full = fit()
    lu = fit(method: :lu)
    assert_in_delta full.rho, lu.rho, 1.0e-7
    assert_in_delta fit(method: :ord).rho, full.rho, 1.0e-6
  end

  test "a column constant within a regime is dropped from the separate fits with a warning" do
    flag = Enum.map(@regimes, &if(&1 == :a, do: 1.0, else: 0.0))
    # the dummy is constant within each regime
    r =
      Regression.ml_lag_regimes(@y, [@x1, flag], @regimes, graph(),
        regime_lag_sep: true,
        name_x: ["x1", "flag"]
      )

    assert r.name_x == ["a_CONSTANT", "a_x1", "a_W_dep_var", "b_CONSTANT", "b_x1", "b_W_dep_var"]
    assert r.kr == 3
    assert Enum.any?(r.warnings, &String.contains?(&1, "removed for being constant"))
  end

  test "regime_lag_sep needs every column by regime and a constant per regime" do
    for opts <- [[cols2regi: [true, false]], [constant_regi: :one]] do
      assert_raise ArgumentError, ~r/must vary by regimes/, fn ->
        fit([regime_lag_sep: true] ++ opts)
      end
    end
  end

  test "bad options and inputs raise" do
    assert_raise ArgumentError, ~r/no :regime_err_sep/, fn -> fit(regime_err_sep: true) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(slx_lags: 1) end
    assert_raise ArgumentError, ~r/constant_regi/, fn -> fit(constant_regi: :few) end
    assert_raise ArgumentError, ~r/cols2regi/, fn -> fit(cols2regi: [true]) end

    assert_raise ArgumentError, ~r/at least one column/, fn ->
      fit(cols2regi: [false, false], constant_regi: :one)
    end

    assert_raise ArgumentError, ~r/graph has/, fn ->
      Regression.ml_lag_regimes(@y, [@x1], @regimes, ring_graph(0))
    end

    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.ml_lag_regimes(@y, [@x1], List.duplicate(1, @n), graph())
    end

    assert_raise ArgumentError, ~r/one value per regime/, fn ->
      Regression.ml_lag_regimes_at(@y, [@x1, @x2], @regimes, graph(), [0.1], regime_lag_sep: true)
    end
  end
end
