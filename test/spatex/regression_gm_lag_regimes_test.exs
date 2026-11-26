defmodule Spatex.RegressionGMLagRegimesTest do
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
  @yend [
    2.0,
    1.0,
    3.5,
    0.5,
    4.0,
    6.5,
    1.0,
    5.0,
    3.0,
    2.5,
    5.5,
    3.0,
    6.0,
    0.8,
    4.4,
    1.9,
    5.2,
    2.1,
    3.6,
    0.3
  ]
  @q [
    1.0,
    0.0,
    3.0,
    1.0,
    3.5,
    6.0,
    0.5,
    4.0,
    2.5,
    3.0,
    4.0,
    2.0,
    5.5,
    0.4,
    3.2,
    1.0,
    4.6,
    1.5,
    2.8,
    0.1
  ]
  @regimes List.duplicate(:a, 10) ++ List.duplicate(:b, 10)

  defp ring(offset) do
    for i <- 0..9, j <- [rem(i + 1, 10), rem(i + 9, 10)], do: {offset + i, offset + j}
  end

  defp graph(transform \\ :r) do
    {f, t} = Enum.unzip(ring(0) ++ ring(10))

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
    |> Graph.transform(transform)
  end

  defp connected do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ [{9, 10}, {10, 9}, {0, 19}, {19, 0}])

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
    |> Graph.transform(:r)
  end

  defp fit(opts \\ [], g \\ graph()),
    do:
      Regression.gm_lag_regimes(@y, [@x1, @x2], @regimes, g, Keyword.put_new(opts, :robust, nil))

  defp take(list, a, b), do: Enum.slice(list, a, b - a)

  test "the default is the pooled fit with one rho, in spreg's coefficient order" do
    r = fit()
    assert r.mode == :pooled and r.regimes == [:a, :b]
    assert r.name_x == ["a_CONSTANT", "a_var_1", "a_var_2", "b_CONSTANT", "b_var_1", "b_var_2"]
    assert r.name_yend == ["_Global_W_dep_var"]
    assert r.name_q == ["a_W_var_1", "a_W_var_2", "b_W_var_1", "b_W_var_2"]
    assert length(r.betas) == 7 and {r.kr, r.kf, r.kryd, r.nr} == {3, 1, 0, 2}
    assert r.rho == List.last(r.betas) and r.rhos == nil
    assert {r.regime_lag_sep, r.regime_err_sep, r.robust} == {false, false, nil}
    assert r.pr2_e != nil and elem(r.ak_test, 1) > 0.0
    assert r.tsls.n == @n
    assert Map.keys(r.impacts) == [:simple]
    assert Enum.map(r.impacts.simple, & &1.name) == ["a_var_1", "a_var_2", "b_var_1", "b_var_2"]
  end

  test "regime_lag_sep fits one lag model per regime on its own subgraph" do
    r = fit(regime_lag_sep: true)
    assert r.mode == :separate and r.regime_err_sep and r.tsls == nil
    assert Enum.sort(Map.keys(r.multi)) == [:a, :b]
    assert Enum.any?(r.warnings, &String.contains?(&1, "regime_err_sep set to True"))

    for {label, [from, to]} <- [a: [0, 10], b: [10, 20]] do
      sub =
        Regression.gm_lag(
          take(@y, from, to),
          [take(@x1, from, to), take(@x2, from, to)],
          ring_graph()
        )

      assert_in_delta r.multi[label].rho, sub.rho, 1.0e-12
      Reference.assert_close(r.multi[label].betas, sub.betas, 1.0e-12)
      assert r.multi[label].name_x == Enum.map(sub.name_x, &"#{label}_#{&1}")
    end

    assert r.betas == r.multi.a.betas ++ r.multi.b.betas
    assert r.rhos == %{a: r.multi.a.rho, b: r.multi.b.rho} and r.rho == r.multi.b.rho
    assert length(r.vm) == 8 and Enum.at(Enum.at(r.vm, 0), 4) == 0.0
    assert {r.kr, r.kf, r.kryd} == {4, 0, 0}
    assert r.names == r.multi.a.name_x ++ r.multi.b.name_x
  end

  defp ring_graph do
    {f, t} = Enum.unzip(ring(0))

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..9))
    |> Graph.transform(:r)
  end

  test "regime_lag_sep with a global column pools the fit on the union of the regime subgraphs" do
    r = fit(regime_lag_sep: true, constant_regi: :one)
    assert r.mode == :pooled and r.tsls != nil
    assert r.name_yend == ["a_W_dep_var", "b_W_dep_var"]
    assert Map.keys(r.rhos) |> Enum.sort() == [:a, :b]
    assert r.rho == r.rhos.b and r.rhos.a == Enum.at(r.betas, 5)
    assert {r.kr, r.kf, r.kryd} == {3, 1, 1}
  end

  test "regime_lag_sep drops the edges between regimes, as spreg's w_regimes does" do
    # the connected graph restricted to each regime is the ring: same fits as the disconnected one
    pooled = fit([regime_lag_sep: true, constant_regi: :one], connected())
    isolated = fit([regime_lag_sep: true, constant_regi: :one], graph())
    Reference.assert_close(pooled.betas, isolated.betas, 1.0e-12)

    separate = fit([regime_lag_sep: true], connected())
    Reference.assert_close(separate.betas, fit(regime_lag_sep: true).betas, 1.0e-12)

    # without regime_lag_sep the cross-regime edges enter Wy
    refute fit([], connected()).betas == fit().betas
  end

  test "a regime subgraph is row-standardized again only when the graph was" do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ [{9, 10}, {10, 9}])
    ids = Enum.to_list(0..(@n - 1))
    ones = List.duplicate(1.0, length(f))
    binary = Graph.from_arrays(f, t, ones, ids: ids)
    row = Graph.transform(binary, :r)

    a = fit([regime_lag_sep: true], binary)
    b = fit([regime_lag_sep: true], row)
    assert a.betas != b.betas

    assert_in_delta a.multi.a.rho,
                    Regression.gm_lag(
                      take(@y, 0, 10),
                      [take(@x1, 0, 10), take(@x2, 0, 10)],
                      ring_binary()
                    ).rho,
                    1.0e-12
  end

  defp ring_binary do
    {f, t} = Enum.unzip(ring(0))
    Graph.from_arrays(f, t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..9))
  end

  test "regime_err_sep alone is ignored with spreg's warning" do
    r = fit(regime_err_sep: true)
    assert r.mode == :pooled and r.regime_err_sep == false
    assert Enum.any?(r.warnings, &String.contains?(&1, "Groupwise heteroskedasticity"))
    Reference.assert_close(r.betas, fit().betas, 1.0e-12)
  end

  test "HAC disables the separate regressions and the spatial diagnostics" do
    ids = Enum.to_list(0..(@n - 1))
    gwk = Graph.from_arrays(ids, ids, List.duplicate(1.0, @n), ids: ids)
    r = fit(regime_lag_sep: true, robust: :hac, gwk: gwk)
    assert r.mode == :pooled and r.regime_lag_sep and r.regime_err_sep == false
    assert r.ak_test == nil and r.robust == :hac

    assert Enum.any?(
             r.warnings,
             &String.contains?(&1, "Error by regimes is not available for HAC")
           )

    assert Enum.any?(r.warnings, &String.contains?(&1, "Spatial diagnostics are not available"))
  end

  test "an extra endogenous variable, w_lags and lag_q change the instruments" do
    base = fit(yend: [@yend], q: [@q])
    assert base.name_yend == ["a_yend_1", "b_yend_1", "_Global_W_dep_var"]
    assert base.name_q == ~w(a_q_1 a_W_var_1 a_W_var_2 a_W_q_1 b_q_1 b_W_var_1 b_W_var_2 b_W_q_1)
    assert {base.kr, base.kf, base.kryd} == {4, 1, 1}

    two = fit(yend: [@yend], q: [@q], w_lags: 2, lag_q: false)
    assert "a_W2_var_2" in two.name_q and "a_W_q_1" not in two.name_q
    refute two.betas == base.betas
  end

  test "spat_impacts and spat_diag select the reported diagnostics" do
    r = fit(spat_impacts: :all, spat_diag: false)
    assert r.ak_test == nil
    assert Enum.sort(Map.keys(r.sp_multipliers)) == [:full, :power, :simple]
    assert r.power_terms != nil
    assert fit(spat_impacts: nil).impacts == nil
  end

  test "spreg_compatible only changes the Chow tests, and only with global x and regime-wise yend" do
    assert fit(spreg_compatible: true).chow == fit().chow

    opts = [regime_lag_sep: true, constant_regi: :one]
    right = fit(opts)
    legacy = fit([spreg_compatible: true] ++ opts)
    assert right.betas == legacy.betas and right.vm == legacy.vm
    assert right.chow.joint != legacy.chow.joint
    assert hd(right.chow.regi) == hd(legacy.chow.regi)
    assert_raise ArgumentError, ~r/spreg_compatible/, fn -> fit(spreg_compatible: :yes) end
  end

  test "a regime split that leaves islands warns" do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ [{9, 10}, {10, 9}, {0, 19}, {19, 0}])
    ids = Enum.to_list(0..(@n - 1))
    g = f |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: ids) |> Graph.transform(:r)
    regimes = List.duplicate(:a, 9) ++ [:b] ++ List.duplicate(:a, 5) ++ List.duplicate(:b, 5)

    r =
      Regression.gm_lag_regimes(@y, [@x1], regimes, g,
        robust: nil,
        regime_lag_sep: true,
        constant_regi: :one
      )

    assert Enum.any?(r.warnings, &String.contains?(&1, "islands for regime"))
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/one label per observation/, fn ->
      Regression.gm_lag_regimes(@y, [@x1], [:a], graph())
    end

    assert_raise ArgumentError, ~r/two distinct/, fn ->
      Regression.gm_lag_regimes(@y, [@x1], List.duplicate(:a, @n), graph())
    end

    assert_raise ArgumentError, ~r/graph must be a Spatex.Graph/, fn ->
      Regression.gm_lag_regimes(@y, [@x1], @regimes, :nope)
    end

    assert_raise ArgumentError, ~r/:cols2regi/, fn -> fit(cols2regi: [true]) end
    assert_raise ArgumentError, ~r/:constant_regi/, fn -> fit(constant_regi: :none) end
    assert_raise ArgumentError, ~r/robust/, fn -> fit(robust: :ogmm) end
    assert_raise ArgumentError, ~r/gwk/, fn -> fit(robust: :hac) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(slx_lags: 1) end
    assert_raise ArgumentError, ~r/w_lags/, fn -> fit(w_lags: 0) end
    assert_raise ArgumentError, ~r/spat_impacts/, fn -> fit(spat_impacts: :bogus) end
    assert_raise ArgumentError, ~r/q needs/, fn -> fit(q: [@q]) end

    assert_raise ArgumentError, ~r/at least one/, fn ->
      fit(constant_regi: :one, cols2regi: [false, false])
    end
  end

  test "regimes with too few observations for the variables are rejected" do
    assert_raise ArgumentError, ~r/not enough observations/, fn ->
      Regression.gm_lag_regimes(@y, [@x1, @x2], Enum.to_list(1..@n), graph())
    end

    small = List.duplicate(:a, 17) ++ [:b, :b, :b]

    assert_raise ArgumentError, ~r/less observations than variables in regime b/, fn ->
      Regression.gm_lag_regimes(@y, [@x1, @x2], small, graph(), regime_lag_sep: true, robust: nil)
    end
  end
end
