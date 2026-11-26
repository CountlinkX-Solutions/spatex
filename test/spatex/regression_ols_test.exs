defmodule Spatex.RegressionOLSTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph
  alias Spatex.Regression
  alias Spatex.Regression.OLS

  doctest Spatex.Regression

  @y [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 5.0, 7.0]
  @x1 [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 7.5, 3.5]
  @x2 [0.3, 1.1, 0.2, 0.9, 0.5, 1.7, 0.8, 1.9, 0.1, 1.2]

  defp path(n) do
    focal = Enum.flat_map(0..(n - 2), &[&1, &1 + 1])
    neighbor = Enum.flat_map(0..(n - 2), &[&1 + 1, &1])
    n |> then(fn _ -> Graph.from_arrays(focal, neighbor, nil, ids: Enum.to_list(0..(n - 1))) end)
  end

  defp call(mod, fun, args), do: apply(mod, fun, args)

  # Adds a trend so generated columns are never constant.
  defp spread(values, step),
    do: values |> Enum.with_index() |> Enum.map(fn {v, i} -> v + i * step end)

  describe "input forms" do
    test "list of columns, map, tensor-like and names agree" do
      a = Regression.ols(@y, [@x1, @x2], name_x: ["A", "B"])
      b = Regression.ols(@y, %{"A" => @x1, "B" => @x2})
      assert a.betas == b.betas
      assert a.name_x == ["CONSTANT", "A", "B"]
      assert b.name_x == ["CONSTANT", "A", "B"]
      assert a.k == 3 and a.n == 10
    end

    test "a single column list is accepted as a bare vector" do
      r = Regression.ols(@y, [@x1])
      assert r.k == 2
    end

    test "a bare numeric vector is one regressor" do
      assert Regression.ols(@y, @x1).betas == Regression.ols(@y, [@x1]).betas
    end

    test "explorer dataframe columns" do
      if Code.ensure_loaded?(Explorer.DataFrame) do
        df = call(Explorer.DataFrame, :new, [%{"A" => @x1, "B" => @x2}])
        assert Regression.ols(@y, df).betas == Regression.ols(@y, [@x1, @x2]).betas
      end
    end

    test "nx tensor {n, k}" do
      if Code.ensure_loaded?(Nx) do
        t = call(Nx, :tensor, [Enum.zip_with([@x1, @x2], & &1), [type: :f64]])
        assert Regression.ols(@y, t).betas == Regression.ols(@y, [@x1, @x2]).betas
      end
    end
  end

  describe "options" do
    test "constant: false fits through the origin" do
      r = Regression.ols(@y, [@x1], constant: false)
      assert r.k == 1 and r.name_x == ["var_1"]
      [x] = [@x1]
      beta = Enum.sum(Enum.zip_with(x, @y, &(&1 * &2))) / Enum.sum(Enum.map(x, &(&1 * &1)))
      assert_in_delta hd(r.betas), beta, 1.0e-12
    end

    test "sig2n_k: false scales by n" do
      a = Regression.ols(@y, [@x1])
      b = Regression.ols(@y, [@x1], sig2n_k: false)
      assert_in_delta a.sig2, a.utu / 8, 1.0e-12
      assert_in_delta b.sig2, b.utu / 10, 1.0e-12
    end

    test "nonspat_diag: false leaves diagnostics nil" do
      r = Regression.ols(@y, [@x1], nonspat_diag: false)
      assert r.f_stat == nil and r.jarque_bera == nil and r.logll == nil
      assert is_float(r.r2)
    end

    test "white test is opt-in and off under robust errors" do
      assert Regression.ols(@y, [@x1, @x2]).white == nil
      assert %{wh: _} = Regression.ols(@y, [@x1, @x2], white_test: true).white
      r = Regression.ols(@y, [@x1, @x2], white_test: true, robust: :white)
      assert r.white == nil
      assert Enum.any?(r.warnings, &(&1 =~ "White test not available"))
    end

    test "spatial diagnostics fill LM tests and Moran" do
      r = Regression.ols(@y, [@x1], graph: path(10), spat_diag: true, moran: true)
      assert {lm, p} = r.lm_error
      assert is_float(lm) and p >= 0.0 and p <= 1.0
      assert {_i, _z, _p} = r.moran_res
      assert r.rlm_lag != nil and r.lm_sarma != nil
    end

    test "vif lists nil for the constant" do
      r = Regression.ols(@y, [@x1, @x2], vif: true)
      assert [nil, {v1, t1}, {_v2, _t2}] = r.vif
      assert_in_delta v1 * t1, 1.0, 1.0e-12
    end
  end

  describe "errors" do
    test "spat_diag and moran require a graph" do
      assert_raise ArgumentError, ~r/graph/, fn -> Regression.ols(@y, [@x1], spat_diag: true) end
      assert_raise ArgumentError, ~r/graph/, fn -> Regression.ols(@y, [@x1], moran: true) end
    end

    test "graph size must match n" do
      assert_raise ArgumentError, ~r/nodes/, fn ->
        Regression.ols(@y, [@x1], graph: path(5), spat_diag: true)
      end
    end

    test "HAC needs a kernel graph with unit diagonal" do
      assert_raise ArgumentError, ~r/gwk/, fn -> Regression.ols(@y, [@x1], robust: :hac) end

      assert_raise ArgumentError, ~r/diagonal/, fn ->
        Regression.ols(@y, [@x1], robust: :hac, gwk: path(10))
      end
    end

    test "length mismatches, non-finite values, singular designs" do
      assert_raise ArgumentError, ~r/length/, fn -> Regression.ols(@y, [Enum.take(@x1, 5)]) end

      assert_raise ArgumentError, fn ->
        Regression.ols(@y, [List.replace_at(@x1, 0, :nan)])
      end

      assert_raise ArgumentError, ~r/singular/, fn -> Regression.ols(@y, [@x1, @x1]) end
      assert_raise ArgumentError, ~r/robust/, fn -> Regression.ols(@y, [@x1], robust: :foo) end
      assert_raise ArgumentError, ~r/unknown/, fn -> Regression.ols(@y, [@x1], bogus: 1) end
    end

    test "all-constant regressors leave nothing to fit" do
      assert_raise ArgumentError, ~r/constant/, fn ->
        Regression.ols(@y, [List.duplicate(2.0, 10)])
      end
    end

    test "more columns than rows" do
      assert_raise ArgumentError, fn ->
        Regression.ols([1.0, 2.0], [[1.0, 2.0], [3.0, 5.0]])
      end
    end
  end

  describe "summary/1" do
    test "prints the headline statistics" do
      text = @y |> Regression.ols([@x1, @x2], name_y: "Y") |> Regression.summary()
      assert text =~ "ORDINARY LEAST SQUARES"
      assert text =~ "CONSTANT"
      assert text =~ "R-squared"
      assert text =~ "Jarque-Bera"
    end

    test "includes spatial diagnostics when present" do
      text =
        @y
        |> Regression.ols([@x1], graph: path(10), spat_diag: true, moran: true)
        |> Regression.summary()

      assert text =~ "DIAGNOSTICS FOR SPATIAL DEPENDENCE"
      assert text =~ "Lagrange Multiplier (lag)"
    end
  end

  describe "properties" do
    property "residuals are orthogonal to X and fitted + residual = y" do
      check all(
              n <- integer(8..25),
              y0 <- list_of(float(min: -50.0, max: 50.0), length: n),
              a0 <- list_of(float(min: -10.0, max: 10.0), length: n),
              b0 <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 40
            ) do
        {y, a, b} = {spread(y0, 0.11), spread(a0, 0.37), spread(b0, 0.53)}

        r =
          try do
            Regression.ols(y, [a, b], nonspat_diag: false)
          rescue
            e in ArgumentError ->
              if e.message =~ "singular", do: nil, else: reraise(e, __STACKTRACE__)
          end

        if r do
          scale = 1.0 + Enum.sum(Enum.map(y, &abs/1))

          for col <- [List.duplicate(1.0, n), a, b] do
            dot = Enum.zip_with(col, r.u, &(&1 * &2)) |> Enum.sum()
            assert abs(dot) <= 1.0e-6 * scale * (1.0 + Enum.sum(Enum.map(col, &abs/1)))
          end

          y_back = Enum.zip_with(r.predy, r.u, &(&1 + &2))
          for {p, q} <- Enum.zip(y_back, y), do: assert_in_delta(p, q, 1.0e-9)
        end
      end
    end

    property "r2 lies in [0, 1] with a constant and ar2 <= r2" do
      check all(
              n <- integer(8..25),
              y0 <- list_of(float(min: -50.0, max: 50.0), length: n),
              a0 <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 40
            ) do
        {y, a} = {spread(y0, 0.11), spread(a0, 0.37)}
        r = Regression.ols(y, [a], nonspat_diag: false)
        assert r.r2 >= -1.0e-9 and r.r2 <= 1.0 + 1.0e-9
        assert r.ar2 <= r.r2 + 1.0e-12
        assert %OLS{} = r
      end
    end
  end
end
