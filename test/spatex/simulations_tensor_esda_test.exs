defmodule Spatex.SimulationsTensorESDATest do
  use ExUnit.Case, async: true

  alias Spatex.ESDA
  alias Spatex.Graph

  # Path 0-1-2-3-4 plus island 5.
  defp graph do
    Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], nil,
      ids: Enum.to_list(0..5)
    )
  end

  @y [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]
  @x [2.0, 1.0, 7.0, 3.0, 6.0, 5.0]
  @b [1, 1, 0, 1, 1, 0]
  @perms 19
  @opts [permutations: @perms, seed: 11]

  # `Nx.to_list/1` decodes NaN and infinity to the same atoms the list form uses.
  defp assert_same(tensor, list, shape) do
    assert %Nx.Tensor{} = tensor
    assert Nx.type(tensor) == {:f, 64}
    assert Nx.shape(tensor) == shape
    assert Nx.to_list(tensor) == list
  end

  describe "global statistics" do
    for {fun, args} <- [
          geary: [@y],
          getis_ord_g: [@y],
          moran: [@y],
          moran_bv: [@x, @y]
        ] do
      test "#{fun} returns a {permutations} tensor equal to the list form" do
        args = unquote(Macro.escape(args)) ++ [graph()]
        list = apply(ESDA, unquote(fun), args ++ [@opts ++ [keep_simulations: true]])
        tensor = apply(ESDA, unquote(fun), args ++ [@opts ++ [keep_simulations: :tensor]])
        assert_same(tensor.sim, list.sim, {@perms})
        assert tensor.p_sim == list.p_sim
      end
    end

    test "moran_rate returns a tensor" do
      events = [1, 4, 2, 6, 3, 5]
      base = [10, 20, 15, 30, 12, 25]
      list = ESDA.moran_rate(events, base, graph(), @opts)
      tensor = ESDA.moran_rate(events, base, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {@perms})
    end

    test "join_counts returns one {permutations} tensor per simulated vector" do
      list = ESDA.join_counts(@b, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.join_counts(@b, graph(), @opts ++ [keep_simulations: :tensor])

      for key <- ~w(sim_bb sim_bw sim_chi2 sim_autocurr_pos sim_autocurr_neg)a do
        assert_same(Map.fetch!(tensor, key), Map.fetch!(list, key), {@perms})
      end

      assert tensor.p_sim_bb == list.p_sim_bb
    end

    test ":tensor with permutations: 0 leaves sim nil" do
      assert %{sim: nil} = ESDA.moran(@y, graph(), permutations: 0, keep_simulations: :tensor)
    end
  end

  describe "local statistics" do
    test "moran_local and moran_local_bv return {n, permutations} tensors" do
      list = ESDA.moran_local(@y, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.moran_local(@y, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {6, @perms})
      assert tensor.p_sim == list.p_sim
      assert tensor.ei_sim == list.ei_sim

      list = ESDA.moran_local_bv(@x, @y, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.moran_local_bv(@x, @y, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {6, @perms})
    end

    test "g_local returns a {n, permutations} tensor" do
      list = ESDA.g_local(@y, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.g_local(@y, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {6, @perms})
      assert tensor.eg_sim == list.eg_sim
    end

    test "geary_local and geary_local_mv return {n, permutations} tensors" do
      list = ESDA.geary_local(@x, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.geary_local(@x, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {6, @perms})

      vars = [@x, @y]
      list = ESDA.geary_local_mv(vars, graph(), @opts ++ [keep_simulations: true])
      tensor = ESDA.geary_local_mv(vars, graph(), @opts ++ [keep_simulations: :tensor])
      assert_same(tensor.sim, list.sim, {6, @perms})
    end

    test "local join counts (uni-, bi- and multivariate) return {n, permutations} tensors" do
      z = [1, 0, 1, 1, 0, 1]

      for run <- [
            &ESDA.join_counts_local(@b, graph(), &1),
            &ESDA.join_counts_local_bv(@b, z, graph(), &1),
            &ESDA.join_counts_local_mv([@b, z], graph(), &1)
          ] do
        list = run.(@opts ++ [keep_simulations: true])
        tensor = run.(@opts ++ [keep_simulations: :tensor])
        assert_same(tensor.sim, list.sim, {6, @perms})
      end
    end

    test ":tensor with permutations: 0 leaves sim nil" do
      assert %{sim: nil} =
               ESDA.moran_local(@y, graph(), permutations: 0, keep_simulations: :tensor)

      assert %{sim: nil} =
               ESDA.geary_local(@x, graph(), permutations: 0, keep_simulations: :tensor)
    end
  end

  describe "validation" do
    test "values other than true, false and :tensor raise naming the three" do
      for bad <- [:list, "tensor", nil, 1] do
        assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
          ESDA.moran(@y, graph(), keep_simulations: bad)
        end

        assert_raise ArgumentError, ~r/:keep_simulations must be true, false or :tensor/, fn ->
          ESDA.moran_local(@y, graph(), keep_simulations: bad)
        end
      end
    end

    test ":tensor without Nx raises naming the optional :nx dependency" do
      assert_raise ArgumentError, ~r/:nx/, fn ->
        Spatex.Input.keep_simulations!(:tensor, false)
      end

      assert Spatex.Input.keep_simulations!(:tensor, true) == :tensor
      assert Spatex.Input.keep_simulations!(true, false) == true
    end
  end
end
