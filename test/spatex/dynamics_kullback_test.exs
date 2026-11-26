defmodule Spatex.DynamicsKullbackTest do
  use ExUnit.Case, async: true

  alias Spatex.Dynamics
  alias Spatex.Dynamics.Kullback

  @a [[10, 5], [4, 12]]

  describe "kullback/2" do
    test "identical strata are homogeneous" do
      assert %Kullback{statistic: s, dof: 2, p_value: p} = Dynamics.kullback([@a, @a])
      assert_in_delta s, 0.0, 1.0e-12
      assert_in_delta p, 1.0, 1.0e-12
    end

    test "different strata are not" do
      k = Dynamics.kullback([@a, [[6, 9], [8, 3]]])
      assert k.dof == 2 and k.statistic > 0.0 and k.p_value < 0.05
    end

    test "one stratum has no degrees of freedom" do
      k = Dynamics.kullback([@a])
      assert k.dof == 0 and k.p_value == :nan
    end

    test "giddy_compatible reproduces giddy's column totals" do
      k = Dynamics.kullback([@a, @a], giddy_compatible: true)
      assert_in_delta k.statistic, -0.5170275647985818, 1.0e-9
    end

    test "a spatial markov result supplies its conditional count matrices" do
      y = for i <- 0..11, do: for(s <- 0..5, do: 1.0 + rem(i * 7 + s * 5 + div(i * s, 3), 11))
      g = path(12)
      sm = Dynamics.spatial_markov(y, g, k: 3, m: 2)
      assert Dynamics.kullback(sm) == Dynamics.kullback(sm.t)
    end

    test "accepts a rank-3 tensor" do
      t = Nx.tensor([@a, [[6, 9], [8, 3]]])
      assert Dynamics.kullback(t) == Dynamics.kullback([@a, [[6, 9], [8, 3]]])
    end

    test "rejects malformed input and options" do
      assert_raise ArgumentError, ~r/strata/, fn -> Dynamics.kullback([]) end

      assert_raise ArgumentError, ~r/square/, fn ->
        Dynamics.kullback([[[1, 2, 3], [4, 5, 6]]])
      end

      assert_raise ArgumentError, ~r/same/, fn -> Dynamics.kullback([@a, [[1]]]) end

      assert_raise ArgumentError, ~r/non-negative/, fn ->
        Dynamics.kullback([[[1, -2], [3, 4]]])
      end

      assert_raise ArgumentError, ~r/giddy/, fn -> Dynamics.kullback([@a], giddy: true) end

      assert_raise ArgumentError, ~r/giddy_compatible/, fn ->
        Dynamics.kullback([@a], giddy_compatible: 1)
      end

      assert_raise ArgumentError, fn -> Dynamics.kullback(:nope) end
    end
  end

  defp path(n) do
    pairs = for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair
    {f, nb} = Enum.unzip(pairs)

    f
    |> Spatex.Graph.from_arrays(nb, nil, ids: Enum.to_list(0..(n - 1)))
    |> Spatex.Graph.transform(:r)
  end
end
