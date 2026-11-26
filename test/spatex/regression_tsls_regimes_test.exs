defmodule Spatex.RegressionTSLSRegimesTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}

  @y [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 2.0, 7.0, 3.5]
  @x [[1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 3.5, 2.5, 4.5]]
  @yend [[2.0, 1.0, 3.5, 0.5, 4.0, 6.5, 1.0, 5.0, 3.0, 2.5, 5.5, 3.0]]
  @q [
    [1.0, 0.0, 3.0, 1.0, 3.5, 6.0, 0.5, 4.0, 2.5, 3.0, 4.0, 2.0],
    [0.5, 2.5, 1.0, 3.0, 2.0, 1.0, 4.5, 0.0, 3.5, 1.5, 2.5, 0.5]
  ]
  @regimes [:a, :a, :a, :a, :a, :a, :b, :b, :b, :b, :b, :b]

  defp fit(opts \\ []), do: Regression.tsls_regimes(@y, @x, @yend, @q, @regimes, opts)

  test "the default is the pooled fit (spreg: regime_err_sep false) in spreg's coefficient order" do
    r = fit()
    assert r.mode == :pooled and r.regimes == [:a, :b]
    assert r.name_x == ["a_CONSTANT", "a_var_1", "b_CONSTANT", "b_var_1"]
    assert r.name_yend == ["a_yend_1", "b_yend_1"]
    assert r.name_q == ["a_q_1", "a_q_2", "b_q_1", "b_q_2"]
    assert r.names == r.name_x ++ r.name_yend
    assert length(r.betas) == 6 and {r.kr, r.kf, r.kryd, r.nr} == {3, 0, 1, 2}
    assert r.tsls.n == 12
  end

  test "separate fits give block-diagonal covariance and one 2SLS per regime" do
    r = fit(regime_err_sep: true)
    assert r.mode == :separate and r.tsls == nil
    assert Enum.sort(Map.keys(r.multi)) == [:a, :b]
    assert r.names == ["a_CONSTANT", "a_var_1", "a_yend_1", "b_CONSTANT", "b_var_1", "b_yend_1"]
    assert Enum.at(Enum.at(r.vm, 0), 3) == 0.0
    assert {r.kr, r.kf, r.kryd} == {3, 0, 0}
    assert r.multi.a.name_x == ["a_CONSTANT", "a_var_1", "a_yend_1"]
  end

  test "regime_err_sep is ignored unless every column varies by regime with a constant per regime" do
    assert fit(regime_err_sep: true, constant_regi: :one).mode == :pooled
    assert fit(regime_err_sep: true, cols2regi: [true, false]).mode == :pooled
  end

  test "global columns follow the regime-specific ones, exogenous before endogenous" do
    r = fit(cols2regi: [true, false])
    assert r.name_x == ["a_CONSTANT", "a_var_1", "b_CONSTANT", "b_var_1"]
    assert r.name_yend == ["_Global_yend_1"]
    assert {r.kr, r.kf, r.kryd} == {2, 1, 0}

    r = fit(cols2regi: [false, true], constant_regi: :one)
    assert r.name_x == ["_Global_CONSTANT", "_Global_var_1"]
    assert r.name_yend == ["a_yend_1", "b_yend_1"]
    assert {r.kr, r.kf, r.kryd} == {1, 2, 1}
  end

  test "spreg_compatible only changes the Chow tests, and only with global x and regime-wise yend" do
    plain = fit()
    assert fit(spreg_compatible: true).chow == plain.chow

    opts = [constant_regi: :one]
    right = fit(opts)
    legacy = fit([spreg_compatible: true] ++ opts)
    assert right.betas == legacy.betas and right.vm == legacy.vm
    assert right.chow.joint != legacy.chow.joint
    assert Enum.map(right.chow.regi, &elem(&1, 0)) == ["var_1", "yend_1"]

    # the exogenous test is unaffected; only the endogenous one points at other coefficients
    assert hd(right.chow.regi) == hd(legacy.chow.regi)
    assert_raise ArgumentError, ~r/spreg_compatible/, fn -> fit(spreg_compatible: :yes) end
  end

  test "HAC with regime_err_sep falls back to the pooled fit with a warning" do
    ids = Enum.to_list(0..11)
    gwk = Graph.from_arrays(ids, ids, List.duplicate(1.0, 12), ids: ids)
    r = fit(regime_err_sep: true, robust: :hac, gwk: gwk)
    assert r.mode == :pooled and r.robust == :hac
    assert Enum.any?(r.warnings, &String.contains?(&1, "not available for HAC"))
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/one label per observation/, fn ->
      Regression.tsls_regimes(@y, @x, @yend, @q, [:a])
    end

    assert_raise ArgumentError, ~r/two distinct/, fn ->
      Regression.tsls_regimes(@y, @x, @yend, @q, List.duplicate(:a, 12))
    end

    assert_raise ArgumentError, ~r/:cols2regi/, fn -> fit(cols2regi: [true]) end
    assert_raise ArgumentError, ~r/:constant_regi/, fn -> fit(constant_regi: :none) end
    assert_raise ArgumentError, ~r/:robust/, fn -> fit(robust: :ogmm) end
    assert_raise ArgumentError, ~r/gwk/, fn -> fit(robust: :hac) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(slx_lags: 1) end

    assert_raise ArgumentError, ~r/at least one/, fn ->
      fit(constant_regi: :one, cols2regi: [false, false])
    end
  end

  test "regimes with too few observations for the variables are rejected" do
    assert_raise ArgumentError, ~r/not enough observations/, fn ->
      Regression.tsls_regimes(@y, @x, @yend, @q, Enum.to_list(1..12))
    end
  end
end
