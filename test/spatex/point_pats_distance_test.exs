defmodule Spatex.PointPatsDistanceTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.PointPats.DistanceTest
  alias Spatex.Reference

  @tol 1.0e-9
  @dist Reference.load!("pointpats/distance.json")["cases"]
  @supports %{
    "default" => nil,
    "int5" => 5,
    "tuple1" => {25},
    "tuple3" => {0, 30, 7},
    "array" => [0.0, 2.5, 5.0, 10.0, 20.0, 40.0]
  }

  defp pts(list), do: Enum.map(list, fn [x, y] -> {x, y} end)

  # Public results carry :infinity; the fixtures use :inf.
  defp norm(list),
    do:
      Enum.map(list, fn
        :infinity -> :inf
        :neg_infinity -> :neg_inf
        v -> v
      end)

  defp check({support, stat}, expected) do
    Reference.assert_close(support, expected["support"], @tol)
    Reference.assert_close(norm(stat), expected["statistic"], @tol)
  end

  describe "oracle: G, K, L, F, J on stored patterns" do
    for {case_name, _} <- @dist, {sup_name, sup} <- @supports do
      test "#{case_name} / support #{sup_name}" do
        c = @dist[unquote(case_name)]
        s = c["supports"][unquote(sup_name)]
        p = pts(c["points"])
        e = pts(c["empty_points"])
        sup = unquote(Macro.escape(sup))

        check(PointPats.g(p, support: sup), s["g"])
        check(PointPats.k(p, support: sup), s["k"])
        check(PointPats.l(p, support: sup), s["l"])
        check(PointPats.l(p, support: sup, linearized: true), s["l_linearized"])
        check(PointPats.f(p, support: sup, empty_points: e), s["f"])
        check(PointPats.j(p, support: sup, empty_points: e), s["j_truncated"])
        check(PointPats.j(p, support: sup, empty_points: e, truncate: false), s["j_untruncated"])
      end
    end

    test "R30 probed values" do
      c = @dist["r30"]
      p = pts(c["points"])
      {_, g} = PointPats.g(p)
      assert length(g) == 20
      assert hd(g) == 0.0 and List.last(g) == 1.0
      {sup, _} = PointPats.k(p, support: {0, 30, 7})
      assert sup == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    end

    test "duplicated point lands in the first bin but G(0) stays 0" do
      c = @dist["r30_dup"]
      {sup, g} = PointPats.g(pts(c["points"]))
      assert hd(g) == 0.0
      assert Enum.at(g, 1) > 0.0
      assert length(sup) == 20
    end

    test "J is truncated at the first infinite value" do
      c = @dist["r30"]
      p = pts(c["points"])
      e = pts(c["empty_points"])
      {sup, j} = PointPats.j(p, empty_points: e)
      assert length(sup) == length(j)
      assert length(j) < 20
      assert Enum.all?(j, &is_float/1)
    end
  end

  describe "simulation envelopes" do
    @sim Reference.load!("pointpats/simulation.json")["cases"]

    for name <- ~w(r100 c200), fun <- ~w(g k l) do
      test "#{name} #{fun}_test matches the observed statistic and the simulation distribution" do
        c = @sim[unquote(name)]
        py = c[unquote(fun)]
        p = pts(c["points"])
        fun = String.to_existing_atom(unquote(fun) <> "_test")

        r = apply(PointPats, fun, [p, [permutations: 999, seed: 3, keep_simulations: true]])
        assert %DistanceTest{} = r
        Reference.assert_close(r.support, py["support"], @tol)
        Reference.assert_close(r.statistic, py["statistic"], @tol)
        assert length(r.simulations) == 999
        assert Enum.all?(r.simulations, &(length(&1) == length(r.support)))

        # per-support mean of the simulations within ~6 standard errors of pointpats'
        cols = transpose(r.simulations)

        for {col, i} <- Enum.with_index(cols) do
          mean = Enum.sum(col) / 999
          band = 6 * Enum.at(py["sim_sd"], i) * :math.sqrt(2 / 999) + 1.0e-9

          assert abs(mean - Enum.at(py["sim_mean"], i)) <= band,
                 "support #{i}: mean #{mean} vs #{Enum.at(py["sim_mean"], i)} (band #{band})"
        end

        # p-values agree up to Monte Carlo error of both runs
        r.pvalue
        |> Enum.zip(py["pvalue"])
        |> Enum.each(fn {a, b} -> assert abs(a - b) <= 0.12 end)

        assert Enum.at(r.pvalue, 0) == 0.0
        assert Enum.all?(r.pvalue, &(&1 >= 0.0 and &1 <= 0.5))
        assert length(r.envelope_low) == length(r.support)
        assert Enum.zip(r.envelope_low, r.envelope_high) |> Enum.all?(fn {l, h} -> l <= h end)
      end
    end

    for name <- ~w(r100 c200) do
      test "#{name} f_test: support exact, simulations statistically close" do
        c = @sim[unquote(name)]
        py = c["f"]
        p = pts(c["points"])
        r = PointPats.f_test(p, permutations: 199, seed: 5, keep_simulations: true)
        Reference.assert_close(r.support, py["support"], @tol)
        # F uses 1000 random empty-space points in both implementations
        Enum.zip(r.statistic, py["statistic"])
        |> Enum.each(fn {a, b} -> assert abs(a - b) <= 0.12 end)

        cols = transpose(r.simulations)

        for {col, i} <- Enum.with_index(cols) do
          mean = Enum.sum(col) / 199
          assert abs(mean - Enum.at(py["sim_mean"], i)) <= 0.1
        end
      end

      test "#{name} j_test: untruncated shape and the J(0) = 1 invariant" do
        c = @sim[unquote(name)]
        py = c["j_untruncated"]
        p = pts(c["points"])

        r = PointPats.j_test(p, permutations: 49, seed: 5, truncate: false)
        Reference.assert_close(r.support, py["support"], @tol)
        assert length(r.statistic) == length(py["statistic"])
        assert hd(r.statistic) == 1.0
        assert hd(r.pvalue) == 0.0
        assert r.simulations == nil

        t = PointPats.j_test(p, permutations: 49, seed: 5)
        assert length(t.support) <= length(r.support)
        assert length(t.support) == length(t.statistic)
        assert length(t.support) == length(t.pvalue)
        assert Enum.all?(t.statistic, &is_float/1)
      end
    end

    test "determinism: the same seed reproduces, a different seed does not" do
      p = pts(@sim["r100"]["points"])
      a = PointPats.k_test(p, permutations: 99, seed: 11, keep_simulations: true)
      b = PointPats.k_test(p, permutations: 99, seed: 11, keep_simulations: true)
      c = PointPats.k_test(p, permutations: 99, seed: 12, keep_simulations: true)
      assert a == b
      assert a.simulations != c.simulations
    end

    test "zero simulations give NaN p-values and no envelope" do
      p = pts(@sim["r100"]["points"])
      r = PointPats.g_test(p, permutations: 0)
      assert Enum.all?(r.pvalue, &(&1 == :nan))
      assert r.permutations == 0
      assert r.simulations == nil
    end

    test "convex hull and explicit window are accepted" do
      p = pts(@sim["r100"]["points"])
      assert %DistanceTest{} = PointPats.k_test(p, permutations: 19, hull: :convex)
      assert %DistanceTest{} = PointPats.l_test(p, permutations: 19, hull: {0, 0, 100, 100})

      assert_raise ArgumentError, ~r/hull/, fn -> PointPats.k_test(p, hull: :alpha) end
    end

    test "p-values of a CSR pattern are roughly uniform over seeds" do
      p = pts(@sim["r100"]["points"])

      mins =
        for seed <- 1..40 do
          r = PointPats.g_test(p, permutations: 99, seed: seed, support: 6)
          Enum.at(r.pvalue, 3)
        end

      # min(p, 1-p) of a uniform p has mean 0.25
      mean = Enum.sum(mins) / 40
      assert mean > 0.1 and mean < 0.4
    end
  end

  describe "validation" do
    test "needs at least two points" do
      assert_raise ArgumentError, ~r/at least 2/, fn -> PointPats.g([{0, 0}]) end
    end

    test "rejects bad supports and options" do
      p = pts(@dist["r30"]["points"])
      assert_raise ArgumentError, ~r/support/, fn -> PointPats.k(p, support: [3.0, 1.0]) end
      assert_raise ArgumentError, ~r/support/, fn -> PointPats.k(p, support: 0) end
      assert_raise ArgumentError, ~r/support/, fn -> PointPats.k(p, support: {1, 2, 3, 4}) end

      assert_raise ArgumentError, ~r/permutations/, fn ->
        PointPats.k_test(p, permutations: -1)
      end

      assert_raise ArgumentError, ~r/empty_points/, fn -> PointPats.f(p, empty_points: []) end
    end

    test "F draws seeded empty-space points when none are given" do
      p = pts(@dist["r30"]["points"])
      assert PointPats.f(p, seed: 1) == PointPats.f(p, seed: 1)
      assert PointPats.f(p, seed: 1) != PointPats.f(p, seed: 2)
      {_, f} = PointPats.f(p, seed: 1)
      assert hd(f) == 0.0 and List.last(f) == 1.0
    end
  end

  defp transpose(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
end
