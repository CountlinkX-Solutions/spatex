defmodule Spatex.InequalityIndicesTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Inequality
  alias Spatex.Inequality.{Atkinson, Schutz}

  describe "atkinson/2" do
    test "matches the inequality docstring" do
      assert %Atkinson{a: a, ede: ede, epsilon: 0.5} =
               Inequality.atkinson([10, 20, 30, 40, 50], 0.5)

      assert_in_delta a, 0.06315, 5.0e-6
      assert_in_delta ede, 28.1054, 5.0e-5

      one = Inequality.atkinson([10, 20, 30, 40, 50], 1)
      assert_in_delta one.a, 0.13161, 5.0e-6
      assert_in_delta one.ede, 26.05171, 5.0e-6
      assert one.epsilon == 1
    end

    test "no aversion or no inequality gives zero" do
      assert_in_delta Inequality.atkinson([3, 1, 8], 0).a, 0.0, 1.0e-15
      assert_in_delta Inequality.atkinson([4, 4, 4], 2).a, 0.0, 1.0e-15
    end

    test "rejects non-positive values, bad epsilon and empty input" do
      assert_raise ArgumentError, ~r/positive/, fn -> Inequality.atkinson([1, 0], 0.5) end
      assert_raise ArgumentError, ~r/positive/, fn -> Inequality.atkinson([1, -2], 0.5) end
      assert_raise ArgumentError, ~r/epsilon/, fn -> Inequality.atkinson([1, 2], -0.1) end
      assert_raise ArgumentError, ~r/epsilon/, fn -> Inequality.atkinson([1, 2], :high) end
      assert_raise ArgumentError, ~r/empty/, fn -> Inequality.atkinson([], 0.5) end
    end

    property "lies in [0, 1) and is scale invariant" do
      check all(
              xs <- list_of(float(min: 0.5, max: 100.0), min_length: 2, max_length: 20),
              eps <- float(min: 0.0, max: 4.0)
            ) do
        a = Inequality.atkinson(xs, eps).a
        assert a >= -1.0e-12 and a < 1.0
        scaled = Inequality.atkinson(Enum.map(xs, &(&1 * 7.0)), eps).a
        assert_in_delta scaled, a, 1.0e-9
      end
    end
  end

  describe "schutz/1" do
    test "matches the inequality docstring" do
      assert %Schutz{distance: d, intersection_point: ip, coefficient: c} =
               Inequality.schutz([1000, 2000, 1500, 3000, 2500])

      assert_in_delta d, 0.15, 1.0e-12
      assert_in_delta ip, 0.6, 1.0e-12
      assert_in_delta c, 7.5, 1.0e-12
    end

    test "all-zero and empty input raise" do
      assert_raise ArgumentError, ~r/zero/, fn -> Inequality.schutz([0, 0]) end
      assert_raise ArgumentError, ~r/empty/, fn -> Inequality.schutz([]) end
    end
  end

  describe "lorenz_curve/1" do
    test "returns the population and value shares with the origin prepended" do
      {pop, share} = Inequality.lorenz_curve([3, 1, 2, 4])
      assert pop == [0.0, 0.25, 0.5, 0.75, 1.0]
      for {s, e} <- Enum.zip(share, [0.0, 0.1, 0.3, 0.6, 1.0]), do: assert_in_delta(s, e, 1.0e-15)
    end

    test "empty input raises" do
      assert_raise ArgumentError, ~r/empty/, fn -> Inequality.lorenz_curve([]) end
    end
  end

  describe "wolfson/1" do
    test "matches the inequality docstring" do
      income = [20_000, 25_000, 27_000, 30_000, 35_000, 45_000, 60_000, 75_000, 80_000, 120_000]
      assert_in_delta Inequality.wolfson(income), 0.2013, 5.0e-5
      assert_in_delta Inequality.wolfson([6, 6, 8, 8, 10, 10, 12, 12]), 0.0833, 5.0e-5
    end

    test "a single value has no bipolarization; bad input raises" do
      assert_in_delta Inequality.wolfson([7]), 0.0, 1.0e-15
      assert_raise ArgumentError, ~r/empty/, fn -> Inequality.wolfson([]) end
      assert_raise ArgumentError, fn -> Inequality.wolfson([1, :nan]) end
    end
  end
end
