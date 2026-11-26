defmodule Spatex.RegressionSlxTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}

  @y [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0]
  @x [[1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0], [2.0, 1.0, 3.0, 2.5, 1.0, 4.0, 2.0, 3.5]]

  defp ring do
    n = 8
    f = Enum.flat_map(0..(n - 1), &[&1, &1])
    t = Enum.flat_map(0..(n - 1), &[rem(&1 + 1, n), rem(&1 + n - 1, n)])
    Graph.from_arrays(f, t, List.duplicate(0.5, 2 * n))
  end

  test "SLX columns are W x lags of the non-constant regressors" do
    r = Regression.ols(@y, @x, graph: ring(), slx_lags: 2, slx_vars: [true, false])
    assert r.name_x == ["CONSTANT", "var_1", "var_2", "W_var_1", "W2_var_1"]
    assert length(r.betas) == 5
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/spatial weights/, fn -> Regression.ols(@y, @x, slx_lags: 1) end

    assert_raise ArgumentError, ~r/:slx_vars must be :all or 2 booleans/, fn ->
      Regression.ols(@y, @x, graph: ring(), slx_lags: 1, slx_vars: [true])
    end

    assert_raise ArgumentError, ~r/selects no column/, fn ->
      Regression.ml_error(@y, @x, ring(), slx_lags: 1, slx_vars: [false, false])
    end

    assert_raise ArgumentError, ~r/:slx_lags must be a non-negative integer/, fn ->
      Regression.gm_lag(@y, @x, ring(), slx_lags: -1)
    end

    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Regression.gm_combo(@y, @x, ring(), slx_lags: 1)
    end
  end
end
