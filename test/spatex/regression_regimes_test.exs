defmodule Spatex.RegressionRegimesTest do
  use ExUnit.Case, async: true

  alias Spatex.Regression

  @y [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5]
  @x [[1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5]]
  @regimes [:a, :a, :a, :a, :a, :b, :b, :b, :b, :b]

  test "pooled and separate fits give the same betas, Chow tests differ by the variance" do
    sep = Regression.ols_regimes(@y, @x, @regimes)
    pool = Regression.ols_regimes(@y, @x, @regimes, regime_err_sep: false)
    assert sep.mode == :separate and pool.mode == :pooled
    for {a, b} <- Enum.zip(sep.betas, pool.betas), do: assert_in_delta(a, b, 1.0e-10)
    assert sep.regimes == [:a, :b] and Enum.sort(Map.keys(sep.multi)) == [:a, :b]
    assert [{"CONSTANT", _, _}, {"var_1", _, _}] = sep.chow.regi
  end

  test "a column constant within a regime is dropped from the separate fits" do
    x = [[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 6.0, 5.0, 1.5, 2.5], hd(@x)]
    r = Regression.ols_regimes(@y, x, @regimes)
    assert r.name_x == ["a_CONSTANT", "a_var_2", "b_CONSTANT", "b_var_2"]

    assert Enum.any?(
             r.warnings,
             &String.contains?(&1, "constant for at least one of the regimes")
           )
  end

  test "regimes can be an Explorer series or an Nx vector" do
    lists = Regression.ols_regimes(@y, @x, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    codes = Nx.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert Regression.ols_regimes(@y, @x, codes) == lists

    names = Explorer.Series.from_list(~w(a a a a a b b b b b))
    assert Regression.ols_regimes(@y, @x, names).betas == lists.betas
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/one label per observation/, fn ->
      Regression.ols_regimes(@y, @x, [:a])
    end

    assert_raise ArgumentError, ~r/two distinct/, fn ->
      Regression.ols_regimes(@y, @x, List.duplicate(:a, 10))
    end

    assert_raise ArgumentError, ~r/:cols2regi/, fn ->
      Regression.ols_regimes(@y, @x, @regimes, cols2regi: [true, false])
    end

    assert_raise ArgumentError, ~r/:robust/, fn ->
      Regression.ols_regimes(@y, @x, @regimes, robust: :hac)
    end
  end
end
