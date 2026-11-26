defmodule Spatex.RegressionGMKKPTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}
  alias Spatex.Regression.PanelGMKKP

  defp ring(n) do
    f = Enum.flat_map(0..(n - 1), &[&1, &1])
    t = Enum.flat_map(0..(n - 1), &[rem(&1 + 1, n), rem(&1 + n - 1, n)])
    Graph.from_arrays(f, t, List.duplicate(0.5, 2 * n))
  end

  @y [
    [1.0, 2.0, 1.5, 1.9],
    [3.0, 2.5, 4.0, 3.1],
    [2.0, 1.0, 1.5, 1.7],
    [5.0, 4.0, 4.5, 5.2],
    [0.5, 1.5, 1.0, 0.7],
    [2.5, 3.5, 3.0, 3.3],
    [1.2, 0.8, 1.9, 1.4],
    [3.3, 3.9, 2.8, 3.0]
  ]
  @x1 [
    [0.2, 0.8, 0.5, 0.9],
    [1.1, 0.9, 1.6, 1.0],
    [0.4, 0.1, 0.3, 0.6],
    [2.0, 1.7, 1.8, 2.2],
    [0.0, 0.6, 0.2, 0.1],
    [0.9, 1.4, 1.2, 1.3],
    [0.3, 0.2, 0.8, 0.5],
    [1.5, 1.9, 1.1, 1.4]
  ]

  test "fits a panel and names the coefficients as spreg" do
    r = Regression.gm_kkp(@y, [@x1], ring(8), name_x: ["x1"])
    assert %PanelGMKKP{n: 8, t: 4, time_effects: false, full_weights: false} = r
    assert r.name_x == ["CONSTANT", "x1", "lambda", "sigma2_v", "sigma2_1"]
    assert length(r.betas) == 5 and length(r.vm) == 2 and length(r.std_err) == 2
    assert length(r.z_stat) == 2 and length(r.u) == 32 and length(r.e_filtered) == 32
    assert [b0, b1, lam, sv, s1] = r.betas
    assert is_float(b0) and is_float(b1)
    assert {lam, sv, s1} == {r.lam, r.sig2_v, r.sig2_1}
    assert abs(lam) <= 0.99 and sv >= 0 and s1 >= 0
    assert_in_delta r.theta, 1 - :math.sqrt(sv / s1), 1.0e-12
  end

  test "time effects add one dummy per period after the first" do
    r = Regression.gm_kkp(@y, [@x1], ring(8), time_effects: true, name_x: ["x1"])

    assert r.name_x ==
             ["CONSTANT", "x1", "time_2", "time_3", "time_4", "lambda", "sigma2_v", "sigma2_1"]

    assert length(r.vm) == 5
  end

  test "full_weights changes the estimates and is recorded" do
    a = Regression.gm_kkp(@y, [@x1], ring(8))
    b = Regression.gm_kkp(@y, [@x1], ring(8), full_weights: true)
    assert b.full_weights
    refute a.lam == b.lam
  end

  test "a zero moment scale raises instead of returning spreg's silent start" do
    y = List.duplicate(List.duplicate(0.0, 3), 5)
    x = [[0.1, 0.5, 0.2], [0.9, 0.3, 0.7], [0.4, 0.4, 0.1], [0.2, 0.8, 0.6], [0.7, 0.1, 0.3]]

    assert_raise ArgumentError, ~r/zero scale/, fn ->
      Regression.gm_kkp(y, [x], ring(5), name_x: ["x"])
    end
  end

  test "regimes are not supported and say so" do
    assert_raise ArgumentError, ~r/regimes are not supported/, fn ->
      Regression.gm_kkp(@y, [@x1], ring(8), regimes: List.duplicate(1, 8))
    end
  end

  test "options and inputs are validated" do
    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Regression.gm_kkp(@y, [@x1], ring(8), epsilon: 1.0e-7)
    end

    assert_raise ArgumentError, ~r/full_weights must be a boolean/, fn ->
      Regression.gm_kkp(@y, [@x1], ring(8), full_weights: 1)
    end

    assert_raise ArgumentError, ~r/time_effects must be a boolean/, fn ->
      Regression.gm_kkp(@y, [@x1], ring(8), time_effects: nil)
    end

    assert_raise ArgumentError, ~r/expected 8 rows/, fn ->
      Regression.gm_kkp(Enum.take(@y, 7), [@x1], ring(8))
    end

    assert_raise ArgumentError, ~r/graph must be a Spatex.Graph/, fn ->
      Regression.gm_kkp(@y, [@x1], :not_a_graph)
    end
  end
end
