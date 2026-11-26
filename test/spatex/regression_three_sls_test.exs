defmodule Spatex.RegressionThreeSlsTest do
  use ExUnit.Case, async: true

  alias Spatex.Regression

  @y1 [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
  @y2 [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
  @x [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
  @e [0.3, 1.1, 0.9, 2.0, 1.4, 0.2, 2.5, 1.9, 0.6, 1.0, 0.1, 1.2]
  @q1 [1.2, 0.4, 2.2, 0.8, 3.1, 1.7, 0.9, 2.8, 1.1, 0.5, 2.6, 1.9]
  @q2 [0.2, 1.9, 0.7, 2.4, 1.0, 0.6, 2.9, 0.3, 1.8, 2.2, 0.8, 1.4]

  defp fit(opts \\ []) do
    Regression.three_sls([@y1, @y2], [[@x], [@x]], [[@e], [@e]], [[@q1, @q2], [@q1, @q2]], opts)
  end

  test "names, shapes and the Chow tests of equally sized equations" do
    r = fit(name_bigy: ["a", "b"], name_bigyend: [["e1"], ["e2"]])
    assert r.name_y == ["a", "b"]
    assert r.name_x == [["Constant_1", "var_1"], ["Constant_2", "var_1"]]
    assert r.name_yend == [["e1"], ["e2"]]
    assert r.name_q == [["var_1", "var_2"], ["var_1", "var_2"]]
    assert r.k == [3, 3] and r.n == 12 and r.n_eq == 2
    assert length(r.varb) == 6 and length(r.chow) == 3
    assert [[_, _], [_, _]] = r.sig
  end

  test "the 3SLS covariance is symmetric and the correlation has a unit diagonal" do
    r = fit()
    assert_in_delta Enum.at(Enum.at(r.corr, 1), 1), 1.0, 1.0e-12
    assert_in_delta Enum.at(Enum.at(r.sig, 0), 1), Enum.at(Enum.at(r.sig, 1), 0), 1.0e-12
  end

  test "arguments are validated" do
    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.three_sls([@y1], [[@x]], [[@e]], [[@q1]])
    end

    assert_raise ArgumentError, ~r/same number/, fn ->
      Regression.three_sls([@y1, @y2], [[@x], [@x]], [[@e]], [[@q1], [@q1]])
    end

    assert_raise ArgumentError, ~r/fewer instruments/, fn ->
      Regression.three_sls([@y1, @y2], [[@x], [@x]], [[@e, @q2], [@e]], [[@q1], [@q1]])
    end

    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(regimes: [1, 2]) end
    assert_raise ArgumentError, ~r/regimes/, fn -> fit(regimes: [1, 2]) end
  end
end
