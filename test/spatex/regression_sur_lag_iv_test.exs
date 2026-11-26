defmodule Spatex.RegressionSurLagIvTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}

  @y1 [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
  @y2 [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
  @x [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]
  @e [0.3, 1.1, 0.9, 2.0, 1.4, 0.2, 2.5, 1.9, 0.6, 1.0, 0.1, 1.2]
  @q1 [1.2, 0.4, 2.2, 0.8, 3.1, 1.7, 0.9, 2.8, 1.1, 0.5, 2.6, 1.9]

  defp graph do
    # a ring of 12 areas, row-standardized
    focal = for i <- 0..11, _ <- 1..2, do: i
    nbr = for i <- 0..11, d <- [1, 11], do: rem(i + d, 12)

    Graph.from_arrays(focal, nbr, List.duplicate(0.5, 24), ids: Enum.to_list(0..11))
  end

  defp fit(opts \\ []) do
    Regression.sur_lag_iv([@y1, @y2], [[@x], [@x]], graph(), opts)
  end

  test "names, shapes, rho and the joint test with spat_diag" do
    r = fit(name_bigy: ["a", "b"], spat_diag: true)
    assert r.name_y == ["a", "b"]
    assert r.name_x == [["Constant_1", "var_1"], ["Constant_2", "var_1"]]
    assert r.name_yend == [["W_a"], ["W_b"]]
    assert r.name_q == [["W_var_1"], ["W_var_1"]]
    assert r.k == [3, 3] and r.n == 12 and r.n_eq == 2
    assert length(r.rho) == 2 and length(r.chow) == 3
    assert {stat, 2, p} = r.joinrho
    assert stat >= 0 and p >= 0 and p <= 1
    assert fit().joinrho == nil
  end

  test "w_lags and lag_q build the instrument names as spreg does" do
    opts = [
      yends: [[@e], [@e]],
      qs: [[@q1], [@q1]],
      name_bigy: ["a", "b"],
      name_bigq: [["z"], ["z"]]
    ]

    r = fit([w_lags: 2, name_bigyend: [["e"], ["e"]]] ++ opts)
    assert hd(r.name_yend) == ["e", "W_a"]
    assert hd(r.name_q) == ["z", "W_var_1", "WW_var_1", "W_z", "WW_z"]

    r1 = fit([w_lags: 2, lag_q: false] ++ opts)
    assert hd(r1.name_q) == ["z", "W_var_1", "WW_var_1"]
    assert r1.k == [4, 4]
  end

  test "arguments are validated" do
    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.sur_lag_iv([@y1], [[@x]], graph())
    end

    assert_raise ArgumentError, ~r/instruments/, fn ->
      fit(yends: [[@e], [@e]])
    end

    assert_raise ArgumentError, ~r/w_lags/, fn -> fit(w_lags: 0) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(regimes: [1, 2]) end
    assert_raise ArgumentError, ~r/regimes/, fn -> fit(regimes: [1, 2]) end
  end
end
