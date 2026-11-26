defmodule Spatex.InequalityTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Graph
  alias Spatex.Inequality
  alias Spatex.Inequality.{SpatialGini, TheilD}

  doctest Spatex.Inequality

  defp path(n) do
    pairs = for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair
    {f, nb} = Enum.unzip(pairs)
    Graph.from_arrays(f, nb, nil, ids: Enum.to_list(0..(n - 1)))
  end

  describe "gini/1" do
    test "known distributions" do
      assert Inequality.gini([1, 1, 1, 1]) == 0.0
      assert_in_delta Inequality.gini([0, 0, 0, 1]), 0.75, 1.0e-15
      assert_in_delta Inequality.gini([4.0, 1.0, 3.0, 2.0]), 0.25, 1.0e-15
    end

    test "all-zero values give :nan, bad input raises" do
      assert Inequality.gini([0, 0]) == :nan
      assert_raise ArgumentError, fn -> Inequality.gini([]) end
      assert_raise ArgumentError, fn -> Inequality.gini([1, :nan]) end
      assert_raise ArgumentError, fn -> Inequality.gini(:nope) end
    end

    property "lies in [0, 1) for non-negative values and is scale invariant" do
      check all(xs <- list_of(float(min: 0.1, max: 100.0), min_length: 2, max_length: 30)) do
        g = Inequality.gini(xs)
        assert g >= -1.0e-12 and g < 1.0
        assert_in_delta Inequality.gini(Enum.map(xs, &(&1 * 3.0))), g, 1.0e-9
      end
    end
  end

  describe "theil/1" do
    test "vector gives a float, matrix one value per column" do
      assert_in_delta Inequality.theil([1, 3]),
                      0.25 * :math.log(0.5) + 0.75 * :math.log(1.5),
                      1.0e-15

      assert Inequality.theil([2, 2, 2]) |> abs() < 1.0e-15
      assert [a, b] = Inequality.theil([[1, 0], [3, 4]])
      assert_in_delta a, Inequality.theil([1, 3]), 1.0e-15
      assert_in_delta b, :math.log(2), 1.0e-12
    end

    test "rejects ragged matrices" do
      assert_raise ArgumentError, fn -> Inequality.theil([[1, 2], [3]]) end
    end
  end

  describe "theil_d/3" do
    test "between plus within equals the total" do
      y = [1, 2, 3, 4, 5, 9]
      d = Inequality.theil_d(y, [:a, :a, :a, :b, :b, :b])
      assert %TheilD{bg_pvalue: nil, permutations: 0} = d
      assert_in_delta d.t, d.bg + d.wg, 1.0e-15
      assert d.bg > 0 and d.wg > 0
    end

    test "matrix input returns lists" do
      d = Inequality.theil_d([[1, 2], [2, 3], [3, 1], [4, 1]], [0, 0, 1, 1])
      assert length(d.t) == 2 and length(d.bg) == 2 and length(d.wg) == 2
    end

    test "permutation test is seeded and its p-value valid" do
      y = for i <- 0..23, do: 1.0 + rem(i * 7, 13)
      part = for i <- 0..23, do: div(i, 6)
      a = Inequality.theil_d(y, part, permutations: 99, seed: 3)
      assert a == Inequality.theil_d(y, part, permutations: 99, seed: 3)
      assert a.bg_pvalue > 0.0 and a.bg_pvalue <= 1.0 and a.seed == 3
    end

    test "rejects a partition of the wrong length" do
      assert_raise ArgumentError, ~r/partition/, fn -> Inequality.theil_d([1, 2, 3], [0, 1]) end
    end
  end

  describe "spatial_gini/3" do
    test "splits inequality over neighbor and distant pairs" do
      s = Inequality.spatial_gini([1, 2, 4, 8], path(4), permutations: 0)
      assert %SpatialGini{} = s
      assert_in_delta s.wg, 14.0, 1.0e-12
      assert_in_delta s.wcg, 32.0, 1.0e-12
      assert_in_delta s.dtotal, 46.0, 1.0e-12
      assert_in_delta s.polarization, 32 / 14, 1.0e-12
      assert s.p_sim == nil and s.wcgp == nil
    end

    test "permutations are seeded" do
      x = for i <- 0..19, do: 1.0 + rem(i * 5, 9)
      opts = [permutations: 99, seed: 7]
      a = Inequality.spatial_gini(x, path(20), opts)
      assert a == Inequality.spatial_gini(x, path(20), opts)
      assert a.p_sim > 0.0 and length(a.wcgp) == 99 and a.seed == 7
      lean = Inequality.spatial_gini(x, path(20), opts ++ [keep_simulations: false])
      assert lean.wcgp == nil and lean.p_sim == a.p_sim
    end

    test "default is PySAL's 99 permutations" do
      x = for i <- 0..9, do: 1.0 + i
      assert Inequality.spatial_gini(x, path(10)).permutations == 99
    end

    test "rejects mismatched graphs" do
      assert_raise ArgumentError, fn -> Inequality.spatial_gini([1, 2, 3], path(4)) end
      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> Inequality.spatial_gini([1, 2], :g) end
    end
  end
end
