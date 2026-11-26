defmodule Spatex.RegressionPanelREErrorTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Regression}
  alias Spatex.Regression.PanelREError

  defp ring(n) do
    f = Enum.flat_map(0..(n - 1), &[&1, &1])
    t = Enum.flat_map(0..(n - 1), &[rem(&1 + 1, n), rem(&1 + n - 1, n)])
    Graph.from_arrays(f, t, List.duplicate(0.5, 2 * n))
  end

  defp chain(edges, n),
    do:
      Graph.from_arrays(
        Enum.map(edges, &elem(&1, 0)),
        Enum.map(edges, &elem(&1, 1)),
        List.duplicate(1.0, length(edges)),
        ids: Enum.to_list(0..(n - 1))
      )

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

  test "fits a symmetric panel and names the coefficients as spreg" do
    r = Regression.ml_error_re(@y, [@x1], ring(6), name_x: ["x1"])
    assert %PanelREError{n: 6, t: 3} = r
    assert r.name_x == ["CONSTANT", "x1", "lambda", "sigma2_u"]
    assert length(r.betas) == 4 and length(r.vm) == 4 and length(r.u) == 18
    assert r.lam > -1 and r.lam < 1 and r.phi >= 0
    assert_in_delta r.sig2_u, r.phi * r.sig2, 1.0e-12
    assert_in_delta Enum.at(r.betas, 3), r.sig2_u, 1.0e-12
  end

  test "a complex spectrum is rejected" do
    cycle = chain([{0, 1}, {1, 2}, {2, 0}], 3)
    y = [[1.0, 2.0], [0.5, 1.0], [2.0, 0.2]]
    x = [[0.3, 0.1], [1.0, 0.7], [0.2, 0.9]]

    assert_raise ArgumentError, ~r/complex eigenvalues/, fn ->
      Regression.ml_error_re(y, [x], cycle)
    end
  end

  test "a non-symmetric W with repeated eigenvalues is rejected" do
    nilpotent = chain([{0, 1}, {1, 2}, {2, 3}], 4)
    y = [[1.0, 2.0], [0.5, 1.0], [2.0, 0.2], [1.1, 0.3]]
    x = [[0.3, 0.1], [1.0, 0.7], [0.2, 0.9], [0.5, 0.4]]

    assert_raise ArgumentError, ~r/repeated eigenvalues/, fn ->
      Regression.ml_error_re(y, [x], nilpotent)
    end
  end

  test "options and inputs are validated" do
    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Regression.ml_error_re(@y, [@x1], ring(6), spat_impacts: :simple)
    end

    assert_raise ArgumentError, ~r/time_effects must be a boolean/, fn ->
      Regression.ml_error_re(@y, [@x1], ring(6), time_effects: 1)
    end

    assert_raise ArgumentError, ~r/expected 6 rows/, fn ->
      Regression.ml_error_re(Enum.take(@y, 5), [@x1], ring(6))
    end
  end
end
