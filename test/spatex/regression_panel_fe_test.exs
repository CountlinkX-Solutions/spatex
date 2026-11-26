defmodule Spatex.RegressionPanelFETest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}

  defp ring(n) do
    f = Enum.flat_map(0..(n - 1), &[&1, &1])
    t = Enum.flat_map(0..(n - 1), &[rem(&1 + 1, n), rem(&1 + n - 1, n)])
    Graph.from_arrays(f, t, List.duplicate(0.5, 2 * n))
  end

  @y [
    [1.0, 2.0, 1.5],
    [3.0, 2.5, 4.0],
    [2.0, 1.0, 1.5],
    [5.0, 4.0, 4.5],
    [0.5, 1.5, 1.0],
    [2.5, 3.5, 3.0]
  ]
  @x1 [
    [0.2, 0.8, 0.5],
    [1.1, 0.9, 1.6],
    [0.4, 0.1, 0.3],
    [2.0, 1.7, 1.8],
    [0.0, 0.6, 0.2],
    [0.9, 1.4, 1.2]
  ]

  test "fits both models and names the coefficients" do
    lag = Regression.ml_lag_fe(@y, [@x1], ring(6), name_x: ["x1"])
    assert lag.model == :lag and lag.name_x == ["x1", "W_y"] and length(lag.fixed_effects) == 6
    err = Regression.ml_error_fe(@y, %{"x1" => @x1}, ring(6))
    assert err.model == :error and err.name_x == ["x1", "lambda"] and length(err.u) == 18
  end

  test "the random-effects lag model adds an intercept and estimates phi" do
    r = Regression.ml_lag_re(@y, [@x1], ring(6), name_x: ["x1"])
    assert r.name_x == ["CONSTANT", "x1", "W_y", "phi"]
    assert r.phi > 0 and r.phi < 1 and length(r.betas) == 4
  end

  test "panels can be Explorer data frames (one column per period) or {n, t} tensors" do
    frame = fn m ->
      [t1, t2, t3] = m |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
      Explorer.DataFrame.new(t1: t1, t2: t2, t3: t3)
    end

    lists = Regression.ml_lag_fe(@y, [@x1], ring(6))
    assert Regression.ml_lag_fe(frame.(@y), [Nx.tensor(@x1, type: :f64)], ring(6)) == lists
  end

  test "inputs are validated" do
    constant = Enum.map(@x1, fn row -> List.duplicate(hd(row), 3) end)

    assert_raise ArgumentError, ~r/time-invariant/, fn ->
      Regression.ml_lag_fe(@y, [@x1, constant], ring(6))
    end

    assert_raise ArgumentError, ~r/expected 6 rows/, fn ->
      Regression.ml_lag_fe(Enum.take(@y, 5), [@x1], ring(6))
    end

    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Regression.ml_error_fe(@y, [@x1], ring(6), spat_impacts: :simple)
    end
  end
end
