defmodule Spatex.RegressionGMTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph
  alias Spatex.Regression
  alias Spatex.Regression.{GMCombo, GMError, GMLag, TSLS}

  # Row-standardized rook lattice.
  defp lattice(side) do
    cells = for r <- 0..(side - 1), c <- 0..(side - 1), do: {r, c}
    index = cells |> Enum.with_index() |> Map.new()

    edges =
      for {r, c} = a <- cells,
          b <- [{r + 1, c}, {r - 1, c}, {r, c + 1}, {r, c - 1}],
          Map.has_key?(index, b),
          do: {index[a], index[b]}

    {f, t} = Enum.unzip(edges)
    Graph.transform(Graph.from_arrays(f, t, List.duplicate(1.0, length(f))), :r)
  end

  defp data(n, seed) do
    :rand.seed(:exsss, {seed, 2, 3})
    x1 = for _ <- 1..n, do: :rand.normal()
    x2 = for _ <- 1..n, do: :rand.normal()
    q = for _ <- 1..n, do: :rand.normal()
    yend = for {a, b} <- Enum.zip(x1, q), do: 0.5 * a + 0.8 * b + :rand.normal() * 0.3
    y = for {a, b} <- Enum.zip(x1, x2), do: 1.0 + 2.0 * a - b + :rand.normal()
    %{y: y, x: [x1, x2], yend: [yend], q: [q]}
  end

  describe "tsls/5" do
    test "returns the documented shape" do
      d = data(40, 1)

      m =
        Regression.tsls(d.y, d.x, d.yend, d.q, name_x: ["A", "B"], name_yend: ["E"], name_y: "Y")

      assert %TSLS{n: 40, k: 4, robust: nil} = m
      assert m.name_x == ["CONSTANT", "A", "B", "E"]
      assert m.name_yend == ["E"] and m.name_q == ["q_1"]
      assert length(m.betas) == 4 and length(m.vm) == 4 and length(hd(m.vm)) == 4
      assert length(m.z_stat) == 4 and match?({_, _}, hd(m.z_stat))
      assert_in_delta m.sig2, m.utu / 40, 1.0e-12

      m2 = Regression.tsls(d.y, d.x, d.yend, d.q, sig2n_k: true)
      assert_in_delta m2.sig2, m.utu / 36, 1.0e-12
    end

    test "a just-identified model with an exogenous instrument equals the IV formula" do
      d = data(40, 2)
      [x1, _] = d.x
      m = Regression.tsls(d.y, [x1], d.yend, d.q)
      assert length(m.betas) == 3
      # residuals orthogonal to the instruments (just identified: H'u = 0)
      [q] = d.q
      assert_in_delta Enum.sum(m.u), 0.0, 1.0e-9

      assert_in_delta Enum.zip_reduce([q, m.u], 0.0, fn [a, b], acc -> acc + a * b end),
                      0.0,
                      1.0e-9
    end

    test "robust variances" do
      d = data(40, 3)
      plain = Regression.tsls(d.y, d.x, d.yend, d.q)
      white = Regression.tsls(d.y, d.x, d.yend, d.q, robust: :white)
      assert white.betas == plain.betas and white.vm != plain.vm and white.robust == :white

      kernel =
        Graph.from_arrays(Enum.to_list(0..39), Enum.to_list(0..39), List.duplicate(1.0, 40))

      hac = Regression.tsls(d.y, d.x, d.yend, d.q, robust: :hac, gwk: kernel)

      for {a, b} <-
            Enum.zip(
              List.flatten(hac.vm),
              List.flatten(Regression.tsls(d.y, d.x, d.yend, d.q, robust: :white).vm)
            ) do
        assert_in_delta a, b, 1.0e-9
      end
    end

    test "raises on invalid input" do
      d = data(40, 4)

      assert_raise ArgumentError, ~r/robust: :hac requires/, fn ->
        Regression.tsls(d.y, d.x, d.yend, d.q, robust: :hac)
      end

      assert_raise ArgumentError, ~r/yend has no columns/, fn ->
        Regression.tsls(d.y, d.x, [], d.q)
      end

      assert_raise ArgumentError, ~r/q has no columns/, fn ->
        Regression.tsls(d.y, d.x, d.yend, [])
      end

      assert_raise ArgumentError, ~r/under-identified/, fn ->
        Regression.tsls(d.y, d.x, d.yend ++ d.yend, d.q)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Regression.tsls(d.y, d.x, d.yend, d.q, bogus: 1)
      end

      assert_raise ArgumentError, ~r/robust must be/, fn ->
        Regression.tsls(d.y, d.x, d.yend, d.q, robust: :x)
      end

      assert_raise ArgumentError, ~r/name_q has 2 names for 1 columns/, fn ->
        Regression.tsls(d.y, d.x, d.yend, d.q, name_q: ["a", "b"])
      end
    end
  end

  describe "gm_lag/4" do
    test "returns the documented shape" do
      d = data(36, 5)
      m = Regression.gm_lag(d.y, d.x, lattice(6), name_x: ["A", "B"], name_y: "Y")

      assert %GMLag{n: 36, k: 4, w_lags: 1, lag_q: true, robust: nil} = m
      assert m.name_x == ["CONSTANT", "A", "B", "W_Y"]
      assert length(m.betas) == 4 and List.last(m.betas) == m.rho
      assert length(m.vm) == 4 and length(m.z_stat) == 4
      assert {ak, p} = m.ak_test
      assert ak >= 0 and p > 0 and p <= 1
      assert Map.keys(m.sp_multipliers) == [:simple]
      assert [%{name: "A"}, %{name: "B"}] = m.impacts.simple
      assert_in_delta m.sig2, Enum.sum(Enum.map(m.u, &(&1 * &1))) / 36, 1.0e-12

      for {y, p, u} <- Enum.zip([d.y, m.predy, m.u]), do: assert_in_delta(y, p + u, 1.0e-10)
      assert m.pr2_e >= 0 and m.pr2 >= 0
    end

    test "is the 2SLS of y on x and W y with the lags of x as instruments" do
      d = data(36, 6)
      g = lattice(6)
      m = Regression.gm_lag(d.y, d.x, g, spat_diag: false)
      wy = Graph.lag(g, d.y)
      wx = Enum.map(d.x, &Graph.lag(g, &1))
      t = Regression.tsls(d.y, d.x, [wy], wx)

      for {a, b} <- Enum.zip(m.betas, t.betas), do: assert_in_delta(a, b, 1.0e-9)

      for {a, b} <- Enum.zip(List.flatten(m.vm), List.flatten(t.vm)),
          do: assert_in_delta(a, b, 1.0e-9)
    end

    test "options: w_lags, robust, impacts, endogenous variables" do
      d = data(36, 7)
      g = lattice(6)
      base = Regression.gm_lag(d.y, d.x, g)
      two = Regression.gm_lag(d.y, d.x, g, w_lags: 2)
      assert two.w_lags == 2 and abs(two.rho - base.rho) > 1.0e-9

      white = Regression.gm_lag(d.y, d.x, g, robust: :white)
      assert white.betas == base.betas and white.vm != base.vm and white.ak_test == base.ak_test

      kernel = lattice(6) |> Graph.transform(:b)

      kernel =
        Graph.from_arrays(Enum.to_list(0..35), Enum.to_list(0..35), List.duplicate(1.0, 36))
        |> then(fn _ -> kernel end)

      assert Regression.gm_lag(d.y, d.x, g, robust: :hac, gwk: add_diagonal(kernel)).ak_test ==
               nil

      all = Regression.gm_lag(d.y, d.x, g, spat_impacts: :all)
      assert Enum.sort(Map.keys(all.sp_multipliers)) == [:full, :power, :simple]
      assert Regression.gm_lag(d.y, d.x, g, spat_impacts: nil).sp_multipliers == nil
      assert Regression.gm_lag(d.y, d.x, g, spat_diag: false).ak_test == nil

      endog = Regression.gm_lag(d.y, d.x, g, yend: d.yend, q: d.q, name_yend: ["E"])
      assert endog.k == 5 and endog.name_x == ["CONSTANT", "var_1", "var_2", "E", "W_dep_var"]
      assert endog.name_yend == ["E"] and endog.name_q == ["q_1"]
      assert length(hd(endog.impacts.simple) |> Map.keys()) == 4
    end

    test "constant: false uses the columns as given" do
      d = data(36, 8)
      [x1, x2] = d.x
      m = Regression.gm_lag(d.y, [Enum.map(x1, &(&1 + 5)), x2], lattice(6), constant: false)
      assert m.k == 3 and m.name_x == ["var_1", "var_2", "W_dep_var"]
      assert length(m.impacts.simple) == 2
    end

    test "raises on invalid input" do
      d = data(36, 9)
      g = lattice(6)

      assert_raise ArgumentError, ~r/graph has 36 nodes but y has 35/, fn ->
        Regression.gm_lag(tl(d.y), Enum.map(d.x, &tl/1), g)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> Regression.gm_lag(d.y, d.x, :nope) end
      assert_raise ArgumentError, ~r/w_lags/, fn -> Regression.gm_lag(d.y, d.x, g, w_lags: 0) end

      assert_raise ArgumentError, ~r/spat_impacts/, fn ->
        Regression.gm_lag(d.y, d.x, g, spat_impacts: :x)
      end

      assert_raise ArgumentError, ~r/requires a kernel/, fn ->
        Regression.gm_lag(d.y, d.x, g, robust: :hac)
      end

      assert_raise ArgumentError, ~r/q needs a matching yend/, fn ->
        Regression.gm_lag(d.y, d.x, g, q: d.q)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Regression.gm_lag(d.y, d.x, g, method: :full)
      end

      assert_raise ArgumentError, ~r/gwk has 4 nodes/, fn ->
        Regression.gm_lag(d.y, d.x, g,
          robust: :hac,
          gwk: Graph.from_arrays([0, 1, 2, 3], [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
        )
      end
    end

    test "rho outside (-1, 1) warns, or raises with hard_bound" do
      # y built to make the 2SLS lag coefficient explode: y is a deterministic function of W y.
      g = lattice(6)
      x1 = for i <- 1..36, do: :math.sin(i * 1.0)
      y = Enum.map(1..36, fn i -> :math.cos(i * 0.7) * 3.0 end)
      m = Regression.gm_lag(y, [x1], g, spat_diag: false, spat_impacts: nil)

      if abs(m.rho) >= 1 do
        assert Enum.any?(m.warnings, &String.contains?(&1, "outside the boundary"))
        assert m.pr2_e == nil and Enum.all?(m.predy_e, &(&1 == 0.0))

        assert_raise ArgumentError, ~r/outside the maximum/, fn ->
          Regression.gm_lag(y, [x1], g, hard_bound: true, spat_diag: false)
        end
      else
        assert m.pr2_e != nil
      end
    end
  end

  defp add_diagonal(g) do
    adjacency = Graph.to_adjacency(g)
    f = Enum.map(adjacency, &elem(&1, 0))
    t = Enum.map(adjacency, &elem(&1, 1))
    w = Enum.map(adjacency, &elem(&1, 2))
    n = g.n

    Graph.from_arrays(
      f ++ Enum.to_list(0..(n - 1)),
      t ++ Enum.to_list(0..(n - 1)),
      w ++ List.duplicate(1.0, n)
    )
  end

  describe "gm_error/4" do
    test "kp98 shape: lambda has no standard error" do
      d = data(36, 10)
      m = Regression.gm_error(d.y, d.x, lattice(6), name_x: ["A", "B"])

      assert %GMError{estimator: :kp98, n: 36, k: 3, iteration: nil, a1: nil} = m
      assert m.name_x == ["CONSTANT", "A", "B", "lambda"]
      assert length(m.betas) == 4 and List.last(m.betas) == m.lam
      assert length(m.vm) == 3 and length(m.std_err) == 3 and length(m.z_stat) == 3
      assert m.lam > -0.99 and m.lam < 0.99 and length(m.lambdas) == 1
      assert is_float(m.sig2) and m.step1c == nil
      for {y, p, u} <- Enum.zip([d.y, m.predy, m.u]), do: assert_in_delta(y, p + u, 1.0e-10)
    end

    test "het and hom shapes" do
      d = data(36, 11)
      g = lattice(6)
      het = Regression.gm_error(d.y, d.x, g, estimator: :het, max_iter: 3, step1c: true)
      assert %GMError{estimator: :het, sig2: nil, step1c: true, max_iter: 3} = het
      assert length(het.vm) == 4 and length(het.std_err) == 4 and length(het.z_stat) == 4
      assert het.iteration in 1..3 and is_binary(het.iter_stop)
      assert length(het.lambdas) == 2 + het.iteration

      hom = Regression.gm_error(d.y, d.x, g, estimator: :hom, a1: :hom)
      assert %GMError{estimator: :hom, a1: :hom, iteration: 1} = hom
      assert hom.iter_stop == "Maximum number of iterations reached."
      assert is_float(hom.sig2) and length(hom.vm) == 4
      assert length(hom.lambdas) == 2

      for a1 <- [:hom_sc, :hom, :het] do
        assert %GMError{a1: ^a1} = Regression.gm_error(d.y, d.x, g, estimator: :hom, a1: a1)
      end
    end

    test "with yend and q the Endog variants are fitted" do
      d = data(36, 12)
      g = lattice(6)

      for est <- [:kp98, :het, :hom] do
        m =
          Regression.gm_error(d.y, d.x, g, yend: d.yend, q: d.q, estimator: est, name_yend: ["E"])

        assert m.k == 4 and m.name_x == ["CONSTANT", "var_1", "var_2", "E", "lambda"]
        assert m.name_yend == ["E"] and m.name_q == ["q_1"]
        assert length(m.betas) == 5
      end

      assert_raise ArgumentError, ~r/yend needs instruments/, fn ->
        Regression.gm_error(d.y, d.x, g, yend: d.yend)
      end
    end

    test "the hook with lambda = 0 reproduces the OLS coefficients" do
      d = data(36, 13)
      g = lattice(6)
      ols = Regression.ols(d.y, d.x)
      kp = Regression.gm_error_at(d.y, d.x, g, [0.0])
      assert kp.lam == 0.0

      for {a, b} <- Enum.zip(Enum.take(kp.betas, 3), ols.betas), do: assert_in_delta(a, b, 1.0e-9)

      het = Regression.gm_error_at(d.y, d.x, g, [0.0, 0.0], estimator: :het)

      for {a, b} <- Enum.zip(Enum.take(het.betas, 3), ols.betas),
          do: assert_in_delta(a, b, 1.0e-9)
    end

    test "hooks consume lambdas in call order and reject a short list" do
      d = data(36, 14)
      g = lattice(6)

      given = [0.1, 0.2, 0.3, 0.4, 0.5]
      m = Regression.gm_error_at(d.y, d.x, g, given, estimator: :het, max_iter: 3, step1c: true)
      assert m.lambdas == given and m.lam == 0.5 and m.iteration == 3

      assert_raise ArgumentError, ~r/not enough lambdas/, fn ->
        Regression.gm_error_at(d.y, d.x, g, [0.1], estimator: :hom)
      end
    end

    test "raises on invalid input" do
      d = data(36, 15)
      g = lattice(6)

      assert_raise ArgumentError, ~r/estimator must be/, fn ->
        Regression.gm_error(d.y, d.x, g, estimator: :ml)
      end

      assert_raise ArgumentError, ~r/a1 must be/, fn ->
        Regression.gm_error(d.y, d.x, g, estimator: :hom, a1: :x)
      end

      assert_raise ArgumentError, ~r/max_iter/, fn ->
        Regression.gm_error(d.y, d.x, g, max_iter: 0)
      end

      assert_raise ArgumentError, ~r/epsilon/, fn ->
        Regression.gm_error(d.y, d.x, g, epsilon: -1)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> Regression.gm_error(d.y, d.x, nil) end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Regression.gm_error(d.y, d.x, g, spat_diag: true)
      end

      assert_raise ArgumentError, ~r/step1c must be a boolean/, fn ->
        Regression.gm_error(d.y, d.x, g, step1c: 1)
      end

      assert_raise ArgumentError, ~r/graph has 36 nodes/, fn ->
        Regression.gm_error(tl(d.y), Enum.map(d.x, &tl/1), g)
      end
    end

    test "hard_bound raises when lambda hits the bound" do
      d = data(36, 16)
      g = lattice(6)

      assert_raise ArgumentError, ~r/outside the bounds/, fn ->
        Regression.gm_error_at(d.y, d.x, g, [0.995], hard_bound: true)
      end

      m = Regression.gm_error_at(d.y, d.x, g, [0.995])
      assert Enum.any?(m.warnings, &String.contains?(&1, "lambda"))
    end
  end

  describe "gm_combo/4" do
    test "kp98, het and hom shapes" do
      d = data(36, 17)
      g = lattice(6)

      for est <- [:kp98, :het, :hom] do
        m = Regression.gm_combo(d.y, d.x, g, estimator: est, name_y: "Y")
        assert %GMCombo{estimator: ^est, n: 36, k: 4, w_lags: 1, lag_q: true} = m
        assert m.name_x == ["CONSTANT", "var_1", "var_2", "W_Y", "lambda"]

        assert length(m.betas) == 5 and Enum.at(m.betas, 3) == m.rho and
                 List.last(m.betas) == m.lam

        assert length(m.predy_e) == 36 and length(m.e_pred) == 36
        assert m.pr2_e >= 0
        assert length(m.std_err) == if(est == :kp98, do: 4, else: 5)
      end
    end

    test "w_lags, lag_q and endogenous variables" do
      d = data(36, 18)
      g = lattice(6)
      one = Regression.gm_combo(d.y, d.x, g)
      two = Regression.gm_combo(d.y, d.x, g, w_lags: 2)
      assert two.w_lags == 2 and abs(one.rho - two.rho) > 1.0e-9
      m = Regression.gm_combo(d.y, d.x, g, yend: d.yend, q: d.q, lag_q: false, name_yend: ["E"])
      assert m.lag_q == false and m.k == 5 and m.name_yend == ["E"]

      assert_raise ArgumentError, ~r/w_lags/, fn ->
        Regression.gm_combo(d.y, d.x, g, w_lags: 0)
      end
    end

    test "with lambda fixed at zero the combo is the spatial 2SLS lag model" do
      d = data(36, 19)
      g = lattice(6)
      lag = Regression.gm_lag(d.y, d.x, g, spat_diag: false)
      combo = Regression.gm_combo_at(d.y, d.x, g, [0.0])

      for {a, b} <- Enum.zip(Enum.take(combo.betas, 4), lag.betas),
          do: assert_in_delta(a, b, 1.0e-9)

      assert_in_delta combo.rho, lag.rho, 1.0e-9
      assert_in_delta combo.sig2, lag.sig2, 1.0e-9
    end
  end

  describe "properties" do
    property "GM_Lag residuals are y minus the fitted values on random lattices" do
      check all(seed <- integer(1..10_000), side <- member_of([4, 5, 6]), max_runs: 15) do
        n = side * side
        d = data(n, seed)
        m = Regression.gm_lag(d.y, d.x, lattice(side), spat_impacts: nil)
        assert length(m.u) == n

        for {y, p, u} <- Enum.zip([d.y, m.predy, m.u]), do: assert_in_delta(y, p + u, 1.0e-8)
        assert_in_delta m.sig2, Enum.sum(Enum.map(m.u, &(&1 * &1))) / n, 1.0e-9
      end
    end

    property "GM_Error lambda stays in the optimizer's bounds and the covariance is symmetric" do
      check all(seed <- integer(1..10_000), est <- member_of([:kp98, :het, :hom]), max_runs: 15) do
        d = data(36, seed)
        m = Regression.gm_error(d.y, d.x, lattice(6), estimator: est)
        assert m.lam >= -0.99 and m.lam <= 0.99

        for {row, i} <- Enum.with_index(m.vm), {v, j} <- Enum.with_index(row) do
          assert_in_delta v, m.vm |> Enum.at(j) |> Enum.at(i), 1.0e-8 * max(1.0, abs(v))
        end

        assert Enum.all?(m.std_err, &(&1 > 0))
      end
    end
  end
end
