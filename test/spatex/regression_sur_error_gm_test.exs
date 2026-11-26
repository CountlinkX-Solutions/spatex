defmodule Spatex.RegressionSurErrorGmTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}

  @y1 [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5, 2.0, 7.0]
  @y2 [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5, 1.0, 4.5]
  @x [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5, 3.5, 4.5]

  # a ring of 12 areas, row-standardized
  defp ring do
    focal = Enum.flat_map(0..11, &[&1, &1])
    neighbor = Enum.flat_map(0..11, &[rem(&1 + 1, 12), rem(&1 + 11, 12)])
    Graph.from_arrays(focal, neighbor) |> Graph.transform(:r)
  end

  defp fit(opts \\ []), do: Regression.sur_error_gm([@y1, @y2], [[@x], [@x]], ring(), opts)

  test "shapes, names and the Chow tests of equally sized equations" do
    r = fit(name_bigy: ["a", "b"])
    assert r.name_y == ["a", "b"]
    assert r.name_x == [["Constant_1", "var_1"], ["Constant_2", "var_1"]]
    assert r.k == [2, 2] and r.n == 12 and r.n_eq == 2
    assert length(r.lam) == 2 and length(r.varb) == 4 and length(r.chow) == 2
    assert [[_, _], [_, _]] = r.sig
    assert Enum.all?(r.lam, &(&1 > -0.99 and &1 < 0.99))
  end

  test "lambda is the KP98 estimate of the OLS residuals of each equation (gm_error/4)" do
    r = fit()
    [l1, l2] = r.lam
    assert_in_delta l1, Regression.gm_error(@y1, [@x], ring()).lam, 1.0e-9
    assert_in_delta l2, Regression.gm_error(@y2, [@x], ring()).lam, 1.0e-9
  end

  test "with lambda zero the estimator is SUR" do
    r = Regression.sur_error_gm_at([@y1, @y2], [[@x], [@x]], ring(), [0.0, 0.0], [])
    s = Regression.sur([@y1, @y2], [[@x], [@x]])
    assert_in_delta hd(hd(r.b_sur)), hd(hd(s.b_sur)), 1.0e-9
    assert_in_delta hd(hd(r.varb)), hd(hd(s.varb)), 1.0e-9
  end

  test "arguments are validated" do
    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.sur_error_gm([@y1], [[@x]], ring())
    end

    assert_raise ArgumentError, ~r/as many/, fn ->
      Regression.sur_error_gm([@y1, @y2], [[@x]], ring())
    end

    assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
      Regression.sur_error_gm([@y1, @y2], [[@x], [@x]], nil)
    end

    assert_raise ArgumentError, ~r/nodes/, fn ->
      Regression.sur_error_gm([@y1, @y2], [[@x], [@x]], Graph.from_arrays([0, 1], [1, 0]))
    end

    assert_raise ArgumentError, ~r/regimes/, fn -> fit(regimes: [1, 2]) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(bogus: 1) end
  end
end
