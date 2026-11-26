defmodule Spatex.RegressionSurErrorMlTest do
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

  defp fit(opts), do: Regression.sur_error_ml([@y1, @y2], [[@x], [@x]], ring(), opts)

  test "shapes, names, the diagnostics and their options" do
    r = fit(name_bigy: ["a", "b"])
    assert r.name_y == ["a", "b"]
    assert r.name_x == [["Constant_1", "var_1"], ["Constant_2", "var_1"]]
    assert r.k == [2, 2] and r.n == 12 and r.n_eq == 2
    assert length(r.lam) == 2 and length(r.varb) == 4 and length(r.chow) == 2
    assert Enum.all?(r.lam, &(&1 > -1.0 and &1 < 1.0))
    assert {_, 1, _} = r.lr_test
    # off by default: spat_diag and vm
    assert r.lr_lambda == nil and r.vm == nil and r.lam_setp == nil
    assert r.lam_test == nil and r.join_lam == nil

    full = fit(spat_diag: true, vm: true)
    assert {_, 2, _} = full.lr_lambda
    assert length(full.vm) == 5 and length(full.lam_setp) == 2
    assert {_, 1, _} = full.lam_test
    assert {_, 2, _} = full.join_lam
    # vm without spat_diag has no joint test
    assert fit(vm: true).join_lam == nil
    assert fit(nonspat_diag: false).lr_test == nil
  end

  test "the reported likelihoods and tests are consistent" do
    r = fit(spat_diag: true, vm: true)
    {stat, _, _} = r.lr_test
    assert_in_delta stat, 2 * (r.llik_sur_err - r.llik_err), 1.0e-9
    {stat, _, _} = r.lr_lambda
    assert_in_delta stat, 2 * (r.llik_sur_err - r.llik), 1.0e-9
    # the lambdas of the single-equation ML error fits start the search
    assert [_, _] = r.lam_ols
    # a scripted path replays the SUR steps: one round gives one search
    one =
      Regression.sur_error_ml_at([@y1, @y2], [[@x], [@x]], ring(), [[0.0, 0.0], [0.1, 0.2]], [])

    assert one.lam == [0.1, 0.2] and one.iterations == 1
  end

  test "binary weights, where spreg's sparse LU fails, give a finite local optimum" do
    focal = Enum.flat_map(0..11, &[&1, &1])
    neighbor = Enum.flat_map(0..11, &[rem(&1 + 1, 12), rem(&1 + 11, 12)])
    g = Graph.from_arrays(focal, neighbor)
    r = Regression.sur_error_ml([@y1, @y2], [[@x], [@x]], g, [])
    assert Enum.all?(r.lam, &(is_float(&1) and &1 >= -1.0 and &1 <= 1.0))
    assert is_float(r.llik_sur_err) and r.llik_sur_err == r.llik_sur_err
  end

  test "arguments are validated" do
    assert_raise ArgumentError, ~r/at least two/, fn ->
      Regression.sur_error_ml([@y1], [[@x]], ring())
    end

    assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
      Regression.sur_error_ml([@y1, @y2], [[@x], [@x]], nil)
    end

    assert_raise ArgumentError, ~r/nodes/, fn ->
      Regression.sur_error_ml([@y1, @y2], [[@x], [@x]], Graph.from_arrays([0, 1], [1, 0]))
    end

    assert_raise ArgumentError, ~r/regimes/, fn -> fit(regimes: [1, 2]) end
    assert_raise ArgumentError, ~r/unknown option/, fn -> fit(bogus: 1) end
    assert_raise ArgumentError, ~r/epsilon/, fn -> fit(epsilon: 0) end
    assert_raise ArgumentError, ~r/boolean/, fn -> fit(vm: 1) end
  end
end
