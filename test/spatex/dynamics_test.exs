defmodule Spatex.DynamicsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Dynamics
  alias Spatex.Dynamics.{LisaMarkov, Markov, SpatialMarkov, SpatialTau, Tau, Theta}
  alias Spatex.Graph

  doctest Spatex.Dynamics

  # path graph 0-1-2-3-4-5, row-standardized
  defp path(n) do
    pairs = for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair
    {f, nb} = Enum.unzip(pairs)
    f |> Graph.from_arrays(nb, nil, ids: Enum.to_list(0..(n - 1))) |> Graph.transform(:r)
  end

  defp panel(n, t) do
    for i <- 0..(n - 1),
        do: for(s <- 0..(t - 1), do: rem(i * 7 + s * 5 + div(i * s, 3), 11) + 1.0)
  end

  describe "markov/2 with class ids" do
    test "counts transitions between consecutive periods" do
      m = Dynamics.markov([[0, 1, 1], [1, 0, 0]])
      assert %Markov{k: 2, classes: [0, 1]} = m
      assert m.transitions == [[1.0, 1.0], [1.0, 1.0]]
      assert m.p == [[0.5, 0.5], [0.5, 0.5]]
      assert m.steady_state == [0.5, 0.5]
      assert m.sojourn_time == [2.0, 2.0]
    end

    test "classes are the sorted distinct labels, or the classes: option" do
      m = Dynamics.markov([[10, 30, 30], [30, 10, 20]])
      assert m.classes == [10, 20, 30]
      m = Dynamics.markov([[1, 1], [1, 1]], classes: [0, 1, 2])
      assert m.k == 3 and m.transitions == [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    end

    test "empty rows stay zero by default and become absorbing with fill_empty_classes" do
      ids = [[0, 1, 1], [1, 0, 0], [2, 2, 3]]
      m = Dynamics.markov(ids)
      assert Enum.at(m.p, 3) == [0.0, 0.0, 0.0, 0.0]
      # the lazily computed summaries always fill, like giddy
      assert Enum.at(m.sojourn_time, 3) == :infinity
      f = Dynamics.markov(ids, fill_empty_classes: true)
      assert Enum.at(f.p, 3) == [0.0, 0.0, 0.0, 1.0]
      assert f.absorbing_states == [3]
    end

    test "a reducible chain has one stationary distribution per recurrent class" do
      m = Dynamics.markov([[0, 1, 0, 1], [1, 0, 0, 0], [2, 2, 2, 2]])
      assert m.recurrent_classes == [[0, 1], [2]]
      assert [[a, b, +0.0], [+0.0, +0.0, 1.0]] = m.steady_state
      assert_in_delta a + b, 1.0, 1.0e-12
      assert Enum.at(Enum.at(m.mfpt, 0), 2) == :infinity
    end

    test "k: classifies every period into quantiles" do
      y = for i <- 1..20, do: [i * 1.0, i * 2.0, 25.0 - i]
      m = Dynamics.markov(y, k: 4)
      assert m.k == 4 and m.classes == [0, 1, 2, 3]
      assert m.transitions |> List.flatten() |> Enum.sum() == 40.0
    end

    test "rejects malformed input" do
      assert_raise ArgumentError, ~r/at least two/, fn -> Dynamics.markov([[0], [1]]) end
      assert_raise ArgumentError, ~r/same length/, fn -> Dynamics.markov([[0, 1], [1]]) end
      assert_raise ArgumentError, ~r/integer class ids/, fn -> Dynamics.markov([[0.5, 1.0]]) end

      assert_raise ArgumentError, ~r/not in classes/, fn ->
        Dynamics.markov([[0, 3]], classes: [0, 1])
      end

      assert_raise ArgumentError, ~r/:k must be/, fn -> Dynamics.markov([[1.0, 2.0]], k: 0) end
      assert_raise ArgumentError, fn -> Dynamics.markov([[0, 1]], bogus: 1) end
      assert_raise ArgumentError, ~r/matrix/, fn -> Dynamics.markov(:nope) end
    end
  end

  describe "spatial_markov/3" do
    test "conditional counts partition the transition counts" do
      y = panel(6, 5)
      sm = Dynamics.spatial_markov(y, path(6), k: 3, m: 2, fill_empty_classes: true)
      assert %SpatialMarkov{k: 3, m: 2} = sm
      assert length(sm.t) == sm.m
      summed = sm.t |> Enum.map(&List.flatten/1) |> Enum.zip_with(&Enum.sum/1)
      assert summed == List.flatten(sm.transitions)
      assert length(sm.chi2) == sm.m and sm.x2_dof == 3 * 2 * 2
      assert sm.x2_rpvalue == nil and sm.x2_realizations == nil
    end

    test "same seed gives the same permutation test" do
      y = panel(8, 6)
      opts = [k: 3, m: 3, permutations: 49, seed: 5, fill_empty_classes: true]
      a = Dynamics.spatial_markov(y, path(8), opts)
      b = Dynamics.spatial_markov(y, path(8), opts)
      assert a.x2_realizations == b.x2_realizations and a.x2_rpvalue == b.x2_rpvalue
      assert a.x2_rpvalue > 0.0 and a.x2_rpvalue <= 1.0 and a.seed == 5
    end

    test "conditional steady states need fill_empty_classes when a row is empty" do
      y = panel(6, 3)
      sm = Dynamics.spatial_markov(y, path(6), k: 3, m: 3)
      empty? = fn p -> Enum.any?(p, fn row -> Enum.sum(row) == 0.0 end) end
      for {p, s} <- Enum.zip(sm.p_cond, sm.s_cond), do: assert(s == nil == empty?.(p))
    end

    test "discrete input uses the categorical lag" do
      ids = for i <- 0..5, do: for(s <- 0..3, do: rem(i + s, 3))
      sm = Dynamics.spatial_markov(ids, path(6), discrete: true, fill_empty_classes: true)
      assert sm.k == 3 and sm.m == 3 and sm.cutoffs == nil
      assert length(sm.lclass_ids) == 6
    end

    test "rejects mismatched graphs and bad options" do
      y = panel(6, 3)
      assert_raise ArgumentError, ~r/6 rows/, fn -> Dynamics.spatial_markov(y, path(5)) end
      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> Dynamics.spatial_markov(y, :g) end
      assert_raise ArgumentError, fn -> Dynamics.spatial_markov(y, path(6), k: 0) end
      assert_raise ArgumentError, fn -> Dynamics.spatial_markov(y, path(6), permutations: -1) end

      assert_raise ArgumentError, ~r/cutoffs/, fn ->
        Dynamics.spatial_markov(y, path(6), cutoffs: [3, 1])
      end
    end
  end

  describe "lisa_markov/3" do
    test "quadrants follow Moran_Local per period" do
      y = panel(6, 4)
      lm = Dynamics.lisa_markov(y, path(6))
      assert %LisaMarkov{classes: [1, 2, 3, 4], k: 4} = lm

      cols = y |> Enum.zip_with(& &1)

      expected =
        cols
        |> Enum.map(&Spatex.ESDA.moran_local(&1, path(6), permutations: 0).q)
        |> Enum.zip_with(& &1)

      assert lm.q == expected
      assert lm.p_values == nil and lm.significant_moves == nil
      assert length(lm.move_types) == 6 and length(hd(lm.move_types)) == 3
      assert Enum.all?(List.flatten(lm.move_types), &(&1 in 1..16))
      {stat, _p, dof} = lm.chi_2
      assert is_float(stat) and dof == 9
    end

    test "permutations add p-values and significant moves, reproducibly" do
      y = panel(8, 4)
      a = Dynamics.lisa_markov(y, path(8), permutations: 99, seed: 3)
      b = Dynamics.lisa_markov(y, path(8), permutations: 99, seed: 3)
      assert a.p_values == b.p_values and a.significant_moves == b.significant_moves
      assert Enum.all?(List.flatten(a.significant_moves), &(&1 in 1..64))
    end

    test "rejects bad input" do
      assert_raise ArgumentError, fn -> Dynamics.lisa_markov(panel(6, 3), path(5)) end

      assert_raise ArgumentError, ~r/significance_level/, fn ->
        Dynamics.lisa_markov(panel(6, 3), path(6), significance_level: 2.0)
      end
    end
  end

  describe "rank mobility" do
    test "kendall_tau handles ties like tau-b" do
      assert %Tau{tau: t} = Dynamics.kendall_tau([1, 2, 3, 4], [2, 4, 6, 8])
      assert_in_delta t, 1.0, 1.0e-14

      t = Dynamics.kendall_tau([1, 1, 2], [1, 2, 3])
      assert {t.concordant, t.discordant, t.extra_x, t.extra_y} == {2.0, 0.0, 0.0, 1.0}
      assert_in_delta t.tau, 2 / :math.sqrt(6), 1.0e-14

      assert_raise ArgumentError, fn -> Dynamics.kendall_tau([1, 2], [1]) end
    end

    test "spatial_tau compares neighbor pairs and tests by permutation" do
      x = for i <- 0..29, do: i * 1.0
      y = for i <- 0..29, do: rem(i * 7, 11) * 1.0
      r = Dynamics.spatial_tau(x, y, path(30), permutations: 99, seed: 1)
      assert %SpatialTau{pairs_spatial: 29} = r
      assert length(r.taus) == 99 and r.tau_spatial_psim > 0.0
      assert r == Dynamics.spatial_tau(x, y, path(30), permutations: 99, seed: 1)
      none = Dynamics.spatial_tau(x, y, path(30))
      assert none.taus == nil and none.tau_spatial_psim == nil
    end

    test "theta is one when every regime moves in one direction" do
      y = [[1, 4], [2, 3], [3, 2], [4, 1]]
      r = Dynamics.theta(y, [0, 0, 1, 1], permutations: 0)
      assert %Theta{theta: [1.0], total: [8.0], max_total: 8.0} = r
      assert r.pvalue_left == nil
      assert Dynamics.theta(y, [:a, :b, :a, :b], permutations: 0).theta == [0.5]
    end

    test "theta permutation p-values lie in [0, 1] and are seeded" do
      y = for i <- 0..19, do: for(s <- 0..3, do: rem(i * 13 + s * 5, 17) * 1.0)
      regime = for i <- 0..19, do: rem(i, 3)
      a = Dynamics.theta(y, regime, permutations: 99, seed: 2)
      assert a == Dynamics.theta(y, regime, permutations: 99, seed: 2)
      assert Enum.all?(a.pvalue_left ++ a.pvalue_right, &(&1 >= 0.0 and &1 <= 1.0))
    end
  end

  describe "properties" do
    defp class_panel do
      gen all(
            n <- integer(2..12),
            t <- integer(2..6),
            k <- integer(2..4),
            ids <- list_of(list_of(integer(0..(k - 1)), length: t), length: n)
          ) do
        ids
      end
    end

    property "rows of p sum to one or zero and transitions count every move" do
      check all(ids <- class_panel()) do
        m = Dynamics.markov(ids)

        assert Enum.all?(m.p, fn row ->
                 abs(Enum.sum(row) - 1.0) < 1.0e-12 or Enum.sum(row) == 0.0
               end)

        n = length(ids)
        t = length(hd(ids))
        assert Enum.sum(List.flatten(m.transitions)) == n * (t - 1) * 1.0
      end
    end

    property "steady states are stationary" do
      check all(ids <- class_panel()) do
        m = Dynamics.markov(ids, fill_empty_classes: true)
        rows = if is_float(hd(m.steady_state)), do: [m.steady_state], else: m.steady_state

        for pi <- rows do
          assert_in_delta Enum.sum(pi), 1.0, 1.0e-9
          k = m.k

          pij = fn i, j -> m.p |> Enum.at(i) |> Enum.at(j) end

          next =
            for j <- 0..(k - 1),
                do: Enum.sum(for i <- 0..(k - 1), do: Enum.at(pi, i) * pij.(i, j))

          for {a, b} <- Enum.zip(next, pi), do: assert_in_delta(a, b, 1.0e-9)
        end
      end
    end
  end
end
