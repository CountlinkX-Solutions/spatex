defmodule Spatex.DynamicsLocalTauTest do
  use ExUnit.Case, async: true

  alias Spatex.Dynamics
  alias Spatex.Graph

  @x [1, 2, 3, 4, 5]
  @y [2, 1, 4, 3, 5]

  # Path 0-1-2-3 plus island 4.
  defp path,
    do: Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2], ids: Enum.to_list(0..4))

  test "islands are :nan with sign 1, and simulations follow keep_simulations" do
    t = Dynamics.tau_local_neighbor(@x, @y, path(), permutations: 19, seed: 1)
    assert List.last(t.tau_ln) == :nan and List.last(t.p_sim) == :nan
    assert t.sign == [-1, 1, 1, -1, 1]
    assert length(t.sim) == 5 and length(hd(t.sim)) == 19

    h = Dynamics.tau_local_neighborhood(@x, @y, path(), permutations: 19, keep_simulations: false)
    assert h.sim == nil and length(h.p_sim) == 5
    assert Dynamics.tau_local_neighbor(@x, @y, path()).p_sim == nil
  end

  test "tau_regional keeps the regime labels and validates them" do
    r = Dynamics.tau_regional(@x, @y, [:b, :b, :a, :a, :a])
    assert r.regimes == [:a, :b] and r.p_sim == nil
    assert_raise ArgumentError, ~r/regime/, fn -> Dynamics.tau_regional(@x, @y, [1, 2]) end
    assert_raise ArgumentError, ~r/length/, fn -> Dynamics.tau_local(@x, [1, 2]) end
  end

  describe "lisa_spillover/3" do
    test "is nil without permutations and validates the quadrant" do
      y = for i <- 1..5, do: [i * 1.0, 6.0 - i, i * 2.0]
      lm = Dynamics.lisa_markov(y, path())
      assert Dynamics.lisa_spillover(lm, path()) == nil

      assert_raise ArgumentError, ~r/:quadrant/, fn ->
        Dynamics.lisa_spillover(lm, path(), quadrant: 5)
      end
    end

    test "a cluster that grows into a new site spills over" do
      g = path()

      lm = %Spatex.Dynamics.LisaMarkov{
        q: [[1, 1], [3, 1], [3, 3], [3, 3], [3, 3]],
        classes: [1, 2, 3, 4],
        p_values: List.duplicate([0.01, 0.01], 5),
        significance_level: 0.05,
        permutations: 99
      }

      # Period 0: site 0 is significant, component {0, 1}; period 1: sites 0 and 1, component
      # {0, 1, 2}. Site 2 is new and joins the old component.
      assert %{components: comps, spill_over: spill} = Dynamics.lisa_spillover(lm, g)
      assert Enum.map(comps, &hd/1) == [1, 1, 0, 0, 0]
      assert spill == [[0], [0], [1], [0], [0]]
    end
  end
end
