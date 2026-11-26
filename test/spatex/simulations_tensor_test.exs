defmodule Spatex.SimulationsTensorTest do
  use ExUnit.Case, async: true

  alias Spatex.{Dynamics, Graph, Inequality, PointPats}

  @x [1, 2, 3, 4, 5]
  @y [2, 1, 4, 3, 5]

  # Path 0-1-2-3 plus island 4.
  defp island_path,
    do: Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2], ids: Enum.to_list(0..4))

  defp path(n) do
    src = Enum.flat_map(0..(n - 2), &[&1, &1 + 1])
    dst = Enum.flat_map(0..(n - 2), &[&1 + 1, &1])
    Graph.from_arrays(src, dst)
  end

  defp assert_same(tensor, list, shape) do
    assert %Nx.Tensor{} = tensor
    assert Nx.type(tensor) == {:f, 64}
    assert Nx.shape(tensor) == shape
    assert Nx.to_list(tensor) == list
  end

  describe "Dynamics" do
    test "local tau (neighbor and neighborhood) returns {n, permutations} tensors" do
      opts = [permutations: 19, seed: 1]

      for fun <- [:tau_local_neighbor, :tau_local_neighborhood] do
        list = apply(Dynamics, fun, [@x, @y, island_path(), opts])

        tensor =
          apply(Dynamics, fun, [@x, @y, island_path(), opts ++ [keep_simulations: :tensor]])

        assert_same(tensor.sim, list.sim, {5, 19})
        assert tensor.p_sim == list.p_sim
      end
    end

    test "tau_regional returns a {permutations, k, k} tensor" do
      regime = [:b, :b, :a, :a, :a]
      opts = [permutations: 19, seed: 3]
      list = Dynamics.tau_regional(@x, @y, regime, opts)
      tensor = Dynamics.tau_regional(@x, @y, regime, opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {19, 2, 2})
      assert tensor.p_sim == list.p_sim
    end

    test ":tensor with permutations: 0 leaves sim nil" do
      assert %{sim: nil} =
               Dynamics.tau_local_neighbor(@x, @y, island_path(), keep_simulations: :tensor)

      assert %{sim: nil} =
               Dynamics.tau_regional(@x, @y, [:b, :b, :a, :a, :a], keep_simulations: :tensor)
    end

    test "invalid values raise naming the three accepted values" do
      assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
        Dynamics.tau_local_neighbor(@x, @y, island_path(), keep_simulations: :list)
      end

      assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
        Dynamics.tau_regional(@x, @y, [1, 1, 2, 2, 2], keep_simulations: 1)
      end
    end
  end

  describe "Inequality" do
    test "spatial_gini returns {permutations} tensors for wcgp and polarization_sim" do
      x = for i <- 0..19, do: 1.0 + rem(i * 5, 9)
      opts = [permutations: 49, seed: 7]
      list = Inequality.spatial_gini(x, path(20), opts)
      tensor = Inequality.spatial_gini(x, path(20), opts ++ [keep_simulations: :tensor])
      assert_same(tensor.wcgp, list.wcgp, {49})
      assert_same(tensor.polarization_sim, list.polarization_sim, {49})
      assert tensor.p_sim == list.p_sim
    end

    test "permutations: 0 and invalid values" do
      assert %{wcgp: nil} =
               Inequality.spatial_gini([1, 2, 4, 8], path(4),
                 permutations: 0,
                 keep_simulations: :tensor
               )

      assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
        Inequality.spatial_gini([1, 2, 4, 8], path(4), keep_simulations: :list)
      end
    end
  end

  describe "PointPats" do
    defp points do
      for i <- 1..25, do: {rem(i * 37, 101) / 10, rem(i * 53, 97) / 10}
    end

    test "distance tests return a {permutations, support} tensor" do
      for fun <- [:g_test, :f_test, :j_test] do
        opts = [permutations: 19, seed: 5]
        list = apply(PointPats, fun, [points(), opts ++ [keep_simulations: true]])
        tensor = apply(PointPats, fun, [points(), opts ++ [keep_simulations: :tensor]])
        assert_same(tensor.simulations, list.simulations, {19, length(list.support)})
      end
    end

    test "false and permutations: 0 leave simulations nil; invalid values raise" do
      assert %{simulations: nil} = PointPats.g_test(points(), permutations: 19, seed: 5)

      assert %{simulations: nil} =
               PointPats.g_test(points(), permutations: 0, keep_simulations: :tensor)

      assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
        PointPats.g_test(points(), permutations: 19, keep_simulations: :list)
      end
    end
  end
end
