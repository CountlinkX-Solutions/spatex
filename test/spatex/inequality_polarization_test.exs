defmodule Spatex.InequalityPolarizationTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph
  alias Spatex.Inequality
  alias Spatex.Inequality.Polarization

  defp path(n) do
    pairs = for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair
    {f, nb} = Enum.unzip(pairs)
    Graph.from_arrays(f, nb, nil, ids: Enum.to_list(0..(n - 1)))
  end

  @clustered [1, 1, 1, 9, 9, 9]

  describe "polarization/3 statistic" do
    test "clustered values give 1 and alternating values 0" do
      assert %Polarization{statistic: 1.0, n_i_components: 2, p_value: nil, sim: nil} =
               Inequality.polarization(@clustered, path(6), permutations: 0)

      alt = Inequality.polarization([1, 9, 1, 9, 1, 9], path(6), permutations: 0)
      assert alt.statistic == 0.0 and alt.n_i_components == 6
    end

    test "reports the component counts and labels" do
      r = Inequality.polarization(@clustered, path(6), permutations: 0)
      assert {r.n, r.n_a_components, r.n_g_components} == {6, 2, 1}
      assert r.i_labels == [0, 0, 0, 1, 1, 1]
      assert r.a_labels == [0, 0, 0, 1, 1, 1]
      assert r.g_labels == [0, 0, 0, 0, 0, 0]
    end

    test "islands are graph components and k follows the larger count" do
      g = Graph.from_arrays([0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4], nil, ids: Enum.to_list(0..5))
      r = Inequality.polarization([1, 2, 8, 9, 1, 2], g, permutations: 0)
      assert r.n_g_components == 3 and r.g_labels == [0, 0, 0, 1, 2, 2]
      assert_in_delta r.statistic, 1 - (4 - 3) / 3, 1.0e-15
    end

    test "explicit bins are right-closed and leave the lowest edge without a group" do
      r = Inequality.polarization([1, 2, 3, 4], path(4), bins: [1, 5], permutations: 0)
      assert r.a_labels == [nil, 0, 0, 0] and r.i_labels == [nil, 0, 0, 0]
      assert r.n_a_components == 1 and r.statistic == 1.0
    end

    test "the graph's weights do not matter, only its structure" do
      weighted = Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1], [5.0, 0.1, 3.0, 2.0])
      binary = Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1], nil)
      a = Inequality.polarization([1, 1, 9], weighted, k: 2, permutations: 0)

      assert a.statistic ==
               Inequality.polarization([1, 1, 9], binary, k: 2, permutations: 0).statistic
    end
  end

  describe "polarization/3 permutations" do
    test "defaults to 999 permutations, kept as a list, with a recorded seed" do
      r = Inequality.polarization(Enum.to_list(1..20), path(20), k: 3)
      assert r.permutations == 999 and length(r.sim) == 999
      assert is_integer(r.seed) and r.p_value > 0.0 and r.p_value <= 1.0
    end

    test "a seed reproduces the run; keep_simulations false drops the draws" do
      opts = [k: 3, permutations: 49, seed: 7]
      x = Enum.to_list(1..20)
      a = Inequality.polarization(x, path(20), opts)
      assert a == Inequality.polarization(x, path(20), opts)
      assert a.seed == 7
      lean = Inequality.polarization(x, path(20), opts ++ [keep_simulations: false])
      assert lean.sim == nil and lean.p_value == a.p_value
    end

    test "a sorted ramp is as clustered as possible" do
      r = Inequality.polarization(Enum.to_list(1..30), path(30), k: 3, permutations: 99, seed: 1)
      assert r.statistic == 1.0 and r.p_value == 0.01
      assert Enum.all?(r.sim, &(&1 <= 1.0))
    end

    test "keep_simulations: :tensor returns a {permutations} f64 tensor" do
      x = Enum.to_list(1..20)
      opts = [k: 3, permutations: 19, seed: 2]
      list = Inequality.polarization(x, path(20), opts)
      t = Inequality.polarization(x, path(20), opts ++ [keep_simulations: :tensor])
      assert %Nx.Tensor{} = t.sim
      assert Nx.type(t.sim) == {:f, 64} and Nx.shape(t.sim) == {19}
      assert Nx.to_list(t.sim) == list.sim
    end
  end

  describe "polarization/3 validation" do
    test "rejects bad options and arguments" do
      g = path(6)

      assert_raise ArgumentError, ~r/color/, fn ->
        Inequality.polarization(@clustered, g, color: 1)
      end

      assert_raise ArgumentError, ~r/:k/, fn -> Inequality.polarization(@clustered, g, k: 0) end
      assert_raise ArgumentError, ~r/:k/, fn -> Inequality.polarization(@clustered, g, k: 7) end
      assert_raise ArgumentError, ~r/:k/, fn -> Inequality.polarization(@clustered, g, k: 2.5) end

      assert_raise ArgumentError, ~r/:bins/, fn ->
        Inequality.polarization(@clustered, g, bins: [3])
      end

      assert_raise ArgumentError, ~r/bins/, fn ->
        Inequality.polarization(@clustered, g, bins: [5, 1])
      end

      assert_raise ArgumentError, ~r/outside/, fn ->
        Inequality.polarization(@clustered, g, bins: [2, 8])
      end

      assert_raise ArgumentError, ~r/:permutations/, fn ->
        Inequality.polarization(@clustered, g, permutations: -1)
      end

      assert_raise ArgumentError, ~r/:seed/, fn ->
        Inequality.polarization(@clustered, g, seed: -1)
      end

      assert_raise ArgumentError, ~r/keep_simulations/, fn ->
        Inequality.polarization(@clustered, g, keep_simulations: :all)
      end

      assert_raise ArgumentError, ~r/Spatex.Graph/, fn ->
        Inequality.polarization(@clustered, :g)
      end

      assert_raise ArgumentError, fn -> Inequality.polarization([1, 2], g) end
    end
  end
end
