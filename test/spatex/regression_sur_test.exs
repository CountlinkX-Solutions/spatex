defmodule Spatex.RegressionSurTest do
  use ExUnit.Case, async: true

  alias Spatex.Regression

  @y1 [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 4.5, 3.5]
  @y2 [2.0, 2.5, 3.5, 1.5, 4.0, 6.5, 2.0, 5.0, 3.0, 3.5]
  @x [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 1.5, 2.5]
  @z [0.3, 1.1, 0.9, 2.0, 1.4, 0.2, 2.5, 1.9, 0.6, 1.0]

  test "names, shapes and the Chow tests of equally sized equations" do
    r = Regression.sur([@y1, @y2], [[@x], [@z]], name_bigy: ["a", "b"], name_bigx: [["x"], ["z"]])
    assert r.name_x == [["Constant_1", "x"], ["Constant_2", "z"]]
    assert length(r.varb) == 4 and length(r.chow) == 2 and r.iterations == 1
    assert {_, 1, _} = r.lr_test
    assert r.lm_error == nil
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/at least two y vectors/, fn ->
      Regression.sur([@y1], [[@x]])
    end

    assert_raise ArgumentError, ~r/needs the :graph/, fn ->
      Regression.sur([@y1, @y2], [[@x], [@z]], spat_diag: true)
    end
  end
end
