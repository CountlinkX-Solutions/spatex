defmodule Spatex.RegressionGMErrorRegimesTest do
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

  defp graph(extra \\ []) do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ extra)

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
    |> Graph.transform(:r)
  end

  defp ring_graph do
    {f, t} = Enum.unzip(ring(0))

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..9))
    |> Graph.transform(:r)
  end

  defp err(opts \\ [], g \\ graph()),
    do: Regression.gm_error_regimes(@y, [@x1, @x2], @regimes, g, opts)

  defp combo(opts \\ [], g \\ graph()),
    do: Regression.gm_combo_regimes(@y, [@x1, @x2], @regimes, g, opts)

  defp endog(opts), do: err([yend: [@yend], q: [@q]] ++ opts)

  defp half(list, :a), do: Enum.slice(list, 0, 10)
  defp half(list, :b), do: Enum.slice(list, 10, 10)

  test "the default is the pooled fit with one lambda, in spreg's coefficient order" do
    r = err()
    assert r.mode == :pooled and r.regimes == [:a, :b]
    assert r.name_x == ~w(a_CONSTANT a_var_1 a_var_2 b_CONSTANT b_var_1 b_var_2 lambda)
    assert r.names == r.name_x and r.name_yend == nil and r.name_q == nil
    assert length(r.betas) == 7 and length(r.vm) == 6 and length(r.std_err) == 6
    assert {r.kr, r.kf, r.kryd, r.nr} == {3, 1, 0, 2}
    assert r.lam == List.last(r.betas) and r.lams == nil and r.multi == nil
    assert {r.regime_err_sep, r.constant_regi} == {false, :many}
    assert length(r.chow.regi) == 3 and elem(r.chow.joint, 1) >= 0.0

    w_u = Graph.lag(graph(), r.u)
    Reference.assert_close(r.e_filtered, Enum.zip_with(r.u, w_u, &(&1 - r.lam * &2)), 1.0e-12)

    assert Enum.zip_with(@y, r.predy, &(&1 - &2))
           |> Enum.zip(r.u)
           |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-12 end)
  end

  test "regime_err_sep fits one gm_error per regime on its own subgraph" do
    r = err(regime_err_sep: true)
    assert r.mode == :separate and r.regime_err_sep and r.lam == nil
    assert {r.kr, r.kf, r.kryd} == {3, 0, 0}

    for label <- [:a, :b] do
      sub =
        Regression.gm_error(
          half(@y, label),
          [half(@x1, label), half(@x2, label)],
          ring_graph()
        )

      Reference.assert_close(r.multi[label].betas, sub.betas, 1.0e-12)
      assert r.lams[label] == sub.lam
    end

    assert r.betas == r.multi.a.betas ++ r.multi.b.betas
    assert length(r.vm) == 6 and Enum.at(Enum.at(r.vm, 0), 3) == 0.0

    assert r.name_x ==
             Enum.flat_map(
               [:a, :b],
               &Enum.map(~w(CONSTANT var_1 var_2 lambda), fn n -> "#{&1}_#{n}" end)
             )

    assert r.names == r.name_x
  end

  test "regime_err_sep with the endogenous model keeps the instruments by regime" do
    r = endog(regime_err_sep: true)
    assert r.mode == :separate
    assert r.name_yend == ["a_yend_1", "b_yend_1"] and length(r.name_q) == 2

    sub =
      Regression.gm_error(
        half(@y, :b),
        [half(@x1, :b), half(@x2, :b)],
        ring_graph(),
        yend: [half(@yend, :b)],
        q: [half(@q, :b)]
      )

    Reference.assert_close(r.multi.b.betas, sub.betas, 1.0e-12)

    assert r.names ==
             ~w(a_CONSTANT a_var_1 a_var_2 a_yend_1 a_lambda b_CONSTANT b_var_1 b_var_2 b_yend_1 b_lambda)
  end

  test "the endogenous model lays out x and yend blocks and its Chow layout" do
    r = endog([])
    assert r.name_x == ~w(a_CONSTANT a_var_1 a_var_2 b_CONSTANT b_var_1 b_var_2)
    assert r.name_yend == ["a_yend_1", "b_yend_1"] and r.name_q == ["a_q_1", "b_q_1"]
    assert r.names == r.name_x ++ r.name_yend ++ ["lambda"]
    assert {r.kr, r.kf, r.kryd} == {4, 1, 1}
  end

  test "the combo adds W y, its spatial instruments and the reduced-form prediction" do
    r = combo()
    assert r.mode == :pooled and r.rho == Enum.at(r.betas, -2) and r.rhos == nil
    assert r.name_yend == ["_Global_W_dep_var"]
    assert r.name_q == ["a_W_var_1", "a_W_var_2", "b_W_var_1", "b_W_var_2"]
    assert {r.kr, r.kf, r.kryd} == {3, 2, 0}
    assert length(r.predy_e) == @n and abs(r.rho) < 1 == (r.pr2_e != nil)
    assert r.names == r.name_x ++ r.name_yend ++ ["lambda"]
  end

  test "regime_lag_sep fits one gm_combo per regime and forces regime_err_sep on" do
    r = combo(regime_lag_sep: true)
    assert r.mode == :separate and r.regime_err_sep and r.regime_lag_sep
    assert r.rho == nil and Enum.sort(Map.keys(r.rhos)) == [:a, :b]
    assert r.name_yend == ["a_W_dep_var", "b_W_dep_var"]
    assert Enum.any?(r.warnings, &String.contains?(&1, "regime_err_sep set to True"))
    assert Enum.any?(r.warnings, &String.contains?(&1, "so setting cols2regi = 'all'"))

    sub =
      Regression.gm_combo(
        half(@y, :a),
        [half(@x1, :a), half(@x2, :a)],
        ring_graph()
      )

    Reference.assert_close(r.multi.a.betas, sub.betas, 1.0e-12)
    assert r.rhos.a == sub.rho
    assert length(r.predy_e) == @n
  end

  test "regime_err_sep alone is switched off in the combo, with spreg's warning" do
    r = combo(regime_err_sep: true)
    assert r.mode == :pooled and r.regime_err_sep == false
    assert Enum.any?(r.warnings, &String.contains?(&1, "regime_err_sep set to False"))
    Reference.assert_close(r.betas, combo().betas, 1.0e-12)
  end

  test "regime_lag_sep resets a partial cols2regi, silently for an all-true list" do
    mixed = combo(regime_lag_sep: true, cols2regi: [true, false])
    assert mixed.cols2regi == [true, true]
    assert Enum.any?(mixed.warnings, &String.contains?(&1, "so setting cols2regi = 'all'"))

    quiet = combo(regime_lag_sep: true, cols2regi: [true, true])
    refute Enum.any?(quiet.warnings, &String.contains?(&1, "so setting cols2regi"))
  end

  test "separate regressions need every coefficient by regime, as spreg" do
    assert_raise ArgumentError, ~r/All coefficients must vary/, fn ->
      err(regime_err_sep: true, cols2regi: [true, false])
    end

    assert_raise ArgumentError, ~r/All coefficients must vary/, fn ->
      err(regime_err_sep: true, constant_regi: :one)
    end

    assert_raise ArgumentError, ~r/All coefficients must vary/, fn ->
      combo(regime_lag_sep: true, constant_regi: :one)
    end
  end

  test "a column constant within a regime is dropped in the separate fits, with a warning" do
    flat = Enum.map(Enum.with_index(@x2), fn {v, i} -> if i < 10, do: v, else: 1.0 end)

    r = Regression.gm_error_regimes(@y, [@x1, flat], @regimes, graph(), regime_err_sep: true)
    assert Enum.any?(r.warnings, &String.contains?(&1, "removed for being constant"))
    assert r.name_x == ~w(a_CONSTANT a_var_1 a_lambda b_CONSTANT b_var_1 b_lambda)
    assert {r.kr, length(r.betas)} == {2, 6}
  end

  test "the estimator is validated; Het and Hom have their own test module" do
    assert err(estimator: :kp98).mode == :pooled
    assert err(estimator: :het).estimator == :het
    assert combo(estimator: :hom).estimator == :hom
    assert_raise ArgumentError, ~r/:estimator/, fn -> err(estimator: :gls) end
  end

  test "spreg_compatible only changes the Chow tests, and only with global x and regime-wise yend" do
    assert err(spreg_compatible: true).chow == err().chow
    assert combo(spreg_compatible: true).chow == combo().chow

    right = endog(constant_regi: :one)
    legacy = endog(constant_regi: :one, spreg_compatible: true)
    assert right.betas == legacy.betas and right.vm == legacy.vm
    assert right.chow.joint != legacy.chow.joint
    assert_raise ArgumentError, ~r/spreg_compatible/, fn -> err(spreg_compatible: :yes) end
  end

  test "a regime split that leaves islands warns" do
    g = graph([{9, 10}, {10, 9}, {0, 19}, {19, 0}])
    regimes = List.duplicate(:a, 9) ++ [:b] ++ List.duplicate(:a, 5) ++ List.duplicate(:b, 5)

    r = Regression.gm_error_regimes(@y, [@x1], regimes, g, regime_err_sep: true)
    assert Enum.any?(r.warnings, &String.contains?(&1, "islands for regime"))
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/one label per observation/, fn ->
      Regression.gm_error_regimes(@y, [@x1], [:a], graph())
    end

    assert_raise ArgumentError, ~r/two distinct/, fn ->
      Regression.gm_error_regimes(@y, [@x1], List.duplicate(:a, @n), graph())
    end

    assert_raise ArgumentError, ~r/graph must be a Spatex.Graph/, fn ->
      Regression.gm_error_regimes(@y, [@x1], @regimes, :nope)
    end

    assert_raise ArgumentError, ~r/:cols2regi/, fn -> err(cols2regi: [true]) end
    assert_raise ArgumentError, ~r/:constant_regi/, fn -> err(constant_regi: :none) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> err(robust: nil) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> err(w_lags: 2) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> combo(slx_lags: 1) end
    assert_raise ArgumentError, ~r/w_lags/, fn -> combo(w_lags: 0) end
    assert_raise ArgumentError, ~r/regime_lag_sep/, fn -> combo(regime_lag_sep: 1) end
    assert_raise ArgumentError, ~r/q needs a matching yend/, fn -> err(q: [@q]) end
  end
end
