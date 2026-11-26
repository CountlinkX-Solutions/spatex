defmodule Spatex.RegressionMLTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph
  alias Spatex.Regression
  alias Spatex.Regression.{MLError, MLLag}

  doctest Spatex.Regression

  # 4 x 4 rook lattice, row-standardized.
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
    y = for {a, b} <- Enum.zip(x1, x2), do: 1.0 + 2.0 * a - b + :rand.normal()
    {y, [x1, x2]}
  end

  describe "ml_lag/4" do
    test "returns the documented shape" do
      g = lattice(5)
      {y, x} = data(25, 1)
      m = Regression.ml_lag(y, x, g, name_x: ["A", "B"], name_y: "Y")

      assert %MLLag{n: 25, k: 4, method: :full, converged: true} = m
      assert m.name_x == ["CONSTANT", "A", "B", "W_Y"]
      assert m.name_y == "Y"
      assert length(m.betas) == 4 and List.last(m.betas) == m.rho
      assert m.rho > -1 and m.rho < 1
      assert length(m.vm) == 4 and length(hd(m.vm)) == 4
      assert length(m.vm1) == 5
      assert length(m.u) == 25 and length(m.predy_e) == 25
      assert m.sig2 == m.utu / 25
      assert Map.keys(m.sp_multipliers) == [:simple]
      assert [%{name: "A"}, %{name: "B"}] = m.impacts.simple
      assert m.iterations > 3
    end

    test "residuals are consistent: predy = y - u and u = e0 - rho e1" do
      g = lattice(5)
      {y, x} = data(25, 2)
      m = Regression.ml_lag(y, x, g)

      Enum.zip([y, m.predy, m.u])
      |> Enum.each(fn {a, p, u} -> assert_in_delta a, p + u, 1.0e-10 end)

      assert m.e_pred
             |> Enum.zip(Enum.zip(y, m.predy_e))
             |> Enum.all?(fn {e, {a, p}} -> abs(e - (a - p)) < 1.0e-12 end)
    end

    test "spat_impacts variants" do
      g = lattice(5)
      {y, x} = data(25, 3)
      all = Regression.ml_lag(y, x, g, spat_impacts: :all)
      assert Enum.sort(Map.keys(all.sp_multipliers)) == [:full, :power, :simple]
      assert is_integer(all.power_terms) and all.power_terms >= 2
      {adi_f, _, _} = all.sp_multipliers.full
      {adi_p, _, _} = all.sp_multipliers.power
      # Both approximate tr((I - rho W)^-1) / n; the power series stops early on bipartite graphs
      # (odd traces vanish), exactly as spreg's does.
      assert_in_delta adi_f, adi_p, 1.0e-3

      none = Regression.ml_lag(y, x, g, spat_impacts: nil)
      assert none.sp_multipliers == nil and none.impacts == nil
    end

    test "constant: false uses the columns as given" do
      g = lattice(5)
      {y, [x1, x2]} = data(25, 4)
      m = Regression.ml_lag(y, [Enum.map(x1, &(&1 + 5)), x2], g, constant: false)
      assert m.k == 3 and m.name_x == ["var_1", "var_2", "W_dep_var"]
      assert length(m.impacts.simple) == 2
    end

    test "ord and full agree on a symmetric row-standardized lattice" do
      g = lattice(5)
      {y, x} = data(25, 5)
      full = Regression.ml_lag(y, x, g)
      ord = Regression.ml_lag(y, x, g, method: :ord)
      lu = Regression.ml_lag(y, x, g, method: :lu)
      assert_in_delta ord.rho, full.rho, 1.0e-6
      assert_in_delta lu.rho, full.rho, 1.0e-9
      assert_in_delta hd(ord.std_err), hd(full.std_err), 1.0e-5
      assert ord.method == :ord and lu.method == :lu
    end

    test "raises on invalid input" do
      g = lattice(5)
      {y, x} = data(25, 6)

      assert_raise ArgumentError, ~r/graph has 25 nodes but y has 24/, fn ->
        Regression.ml_lag(tl(y), Enum.map(x, &tl/1), g)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> Regression.ml_lag(y, x, :nope) end

      assert_raise ArgumentError, ~r/method must be/, fn ->
        Regression.ml_lag(y, x, g, method: :qr)
      end

      assert_raise ArgumentError, ~r/epsilon/, fn -> Regression.ml_lag(y, x, g, epsilon: 0) end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Regression.ml_lag(y, x, g, bogus: 1)
      end

      assert_raise ArgumentError, ~r/spat_impacts/, fn ->
        Regression.ml_lag(y, x, g, spat_impacts: :x)
      end
    end

    test "dense-size guard names the limit and the way out" do
      g = lattice(5)
      {y, x} = data(25, 7)

      assert_raise ArgumentError, ~r/exceeds max_dense_n = 10.*Raise :max_dense_n/, fn ->
        Regression.ml_lag(y, x, g, max_dense_n: 10)
      end

      assert %MLLag{} = Regression.ml_lag(y, x, g, max_dense_n: 25)
    end

    test "graph with a nonzero diagonal is rejected" do
      g = Graph.from_arrays([0, 1, 2, 3], [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
      y = [1.0, 2.0, 4.0, 3.0]

      assert_raise ArgumentError, ~r/zero diagonal/, fn ->
        Regression.ml_lag(y, [[1.0, 3.0, 2.0, 5.0]], g)
      end
    end

    test "singular design raises" do
      g = lattice(5)
      {y, [x1, _]} = data(25, 8)

      assert_raise ArgumentError, ~r/singular/, fn ->
        Regression.ml_lag(y, [x1, Enum.map(x1, &(&1 * 2.0))], g)
      end
    end

    test "all-NaN objective is reported as non-convergence, not a crash" do
      # |rho| search stays inside (-1, 1); an unrowstandardized dense graph makes det(I - rho W)
      # negative for rho above 1/lambda_max, giving NaN objective values mid-search.
      g = lattice(5) |> Graph.transform(:b)
      {y, x} = data(25, 9)
      m = Regression.ml_lag(y, x, g)
      assert is_boolean(m.converged)
      assert m.rho > -1 and m.rho < 1
    end

    property "estimates are invariant to the order of the observations" do
      check all(seed <- integer(1..1000), max_runs: 6) do
        n = 16
        g = lattice(4)
        {y, x} = data(n, seed)
        perm = Enum.shuffle(0..(n - 1))
        inv = perm |> Enum.with_index() |> Map.new()

        {f, t, w} =
          Graph.to_adjacency(g)
          |> Enum.reduce({[], [], []}, fn {a, b, c}, {fs, ts, ws} ->
            {[inv[a] | fs], [inv[b] | ts], [c | ws]}
          end)

        gp = Graph.from_arrays(f, t, w, ids: Enum.to_list(0..(n - 1)))
        pick = fn col -> Enum.map(perm, &Enum.at(col, &1)) end

        a = Regression.ml_lag(y, x, g)
        b = Regression.ml_lag(pick.(y), Enum.map(x, pick), gp)
        # Reordering observations changes floating-point summation order, which can flip a late
        # Brent comparison; rho is only pinned to the search tolerance (epsilon 1e-7), while
        # the log-likelihood is flat at the optimum and stays equal to 1e-9.
        assert_in_delta a.rho, b.rho, 1.0e-6
        assert_in_delta a.logll, b.logll, 1.0e-9
      end
    end
  end

  describe "ml_error/4" do
    test "returns the documented shape" do
      g = lattice(5)
      {y, x} = data(25, 11)
      m = Regression.ml_error(y, x, g, name_x: ["A", "B"])

      assert %MLError{n: 25, k: 3, method: :full, converged: true} = m
      assert m.name_x == ["CONSTANT", "A", "B", "lambda"]
      assert List.last(m.betas) == m.lam
      assert length(m.vm) == 4
      # beta and lambda are uncorrelated.
      assert Enum.take(List.last(m.vm), 3) == [0.0, 0.0, 0.0]
      assert length(m.vm1) == 2
      assert m.sig2 > 0
      assert length(m.e_filtered) == 25
    end

    test "aic and schwarz count k without lambda" do
      g = lattice(5)
      {y, x} = data(25, 12)
      m = Regression.ml_error(y, x, g)
      assert_in_delta m.aic, 2 * 3 - 2 * m.logll, 1.0e-10
      assert_in_delta m.schwarz, 3 * :math.log(25) - 2 * m.logll, 1.0e-10
    end

    test "u is the unfiltered residual: u = y - predy" do
      g = lattice(5)
      {y, x} = data(25, 13)
      m = Regression.ml_error(y, x, g)
      for {a, p, u} <- Enum.zip([y, m.predy, m.u]), do: assert_in_delta(a, p + u, 1.0e-10)
    end

    test "ord matches full on a symmetric row-standardized lattice and rejects bad options" do
      g = lattice(5)
      {y, x} = data(25, 14)
      full = Regression.ml_error(y, x, g)
      ord = Regression.ml_error(y, x, g, method: :ord)
      assert_in_delta ord.lam, full.lam, 1.0e-6

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Regression.ml_error(y, x, g, spat_impacts: :simple)
      end

      assert_raise ArgumentError, ~r/max_dense_n/, fn ->
        Regression.ml_error(y, x, g, max_dense_n: 3)
      end
    end

    test "at-parameter hook evaluates at the given lambda" do
      g = lattice(5)
      {y, x} = data(25, 15)
      m = Regression.ml_error_at(y, x, g, 0.25)
      assert m.lam == 0.25 and m.iterations == 1
      l = Regression.ml_lag_at(y, x, g, -0.1)
      assert l.rho == -0.1 and l.iterations == 1
    end
  end
end
