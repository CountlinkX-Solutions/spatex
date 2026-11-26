defmodule Spatex.ESDAMoranTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.ESDA
  alias Spatex.ESDA.{Moran, MoranBV, MoranRate}
  alias Spatex.Graph

  doctest Spatex.ESDA

  defp path(n, islands \\ 0) do
    focal = Enum.flat_map(0..(n - 2)//1, &[&1, &1 + 1])
    neighbor = Enum.flat_map(0..(n - 2)//1, &[&1 + 1, &1])
    Graph.from_arrays(focal, neighbor, nil, ids: Enum.to_list(0..(n + islands - 1)))
  end

  @y [1.0, 3.0, 2.0, 8.0, 5.0, 9.0, 4.0, 7.0]

  describe "moran/3 analytic" do
    test "matches the esda island probe" do
      g = path(3, 1)
      m = ESDA.moran([1, 2, 4, 7], g, permutations: 0)
      assert_in_delta m.i, 0.2857142857142857, 1.0e-12
      assert_in_delta m.ei, -1 / 3, 1.0e-12
      assert_in_delta m.vi_norm, 0.2222222222222222, 1.0e-12
      assert m.n == 4
      assert m.permutations == 0
    end

    test "permutations: 0 disables every simulation field" do
      m = ESDA.moran(@y, path(8), permutations: 0)
      assert %Moran{sim: nil, p_sim: nil, ei_sim: nil, se_i_sim: nil} = m
      assert %Moran{vi_sim: nil, z_sim: nil, p_z_sim: nil, seed: nil} = m
    end

    test "non-finite results surface as atoms" do
      m = ESDA.moran([1, 5, 2], path(3), permutations: 0)
      assert m.vi_rand == :nan
      assert m.p_rand == :nan
      assert is_float(m.vi_norm)
    end

    test "two_tailed: false halves the analytic p-values" do
      two = ESDA.moran(@y, path(8), permutations: 0)
      one = ESDA.moran(@y, path(8), permutations: 0, two_tailed: false)
      assert_in_delta two.p_norm, 2 * one.p_norm, 1.0e-12
      assert_in_delta two.p_rand, 2 * one.p_rand, 1.0e-12
    end

    test "default transform is row-standardization; :b differs" do
      g = path(8)
      r = ESDA.moran(@y, g, permutations: 0)
      assert r.i == ESDA.moran(@y, Graph.transform(g, :r), permutations: 0).i
      b = ESDA.moran(@y, g, permutations: 0, transform: :b)
      refute b.i == r.i
    end

    test "accepts packed binaries" do
      {:f64, bin} = {:f64, for(v <- @y, into: <<>>, do: <<v::float-64-native>>)}

      assert ESDA.moran({:f64, bin}, path(8), permutations: 0).i ==
               ESDA.moran(@y, path(8), permutations: 0).i
    end
  end

  describe "moran/3 permutations" do
    test "same seed gives an identical simulation, other seeds differ" do
      g = path(8)
      a = ESDA.moran(@y, g, permutations: 199, seed: 7)
      b = ESDA.moran(@y, g, permutations: 199, seed: 7)
      c = ESDA.moran(@y, g, permutations: 199, seed: 8)
      assert a == b
      assert length(a.sim) == 199
      refute a.sim == c.sim
      assert a.seed == 7 and a.permutations == 199
    end

    test "seed: nil draws a seed, stores it, and reproduces with it" do
      g = path(8)
      a = ESDA.moran(@y, g, permutations: 99)
      assert is_integer(a.seed) and a.seed >= 0
      assert ESDA.moran(@y, g, permutations: 99, seed: a.seed) == a
    end

    test "keep_simulations: false drops sim but keeps the summary" do
      g = path(8)
      kept = ESDA.moran(@y, g, permutations: 99, seed: 1)
      dropped = ESDA.moran(@y, g, permutations: 99, seed: 1, keep_simulations: false)
      assert dropped.sim == nil
      assert %{dropped | sim: kept.sim} == kept
    end

    property "p_sim stays within [1/(P+1), 1] and sim has P entries" do
      check all(
              y <- StreamData.list_of(StreamData.integer(-20..20), length: 9),
              Enum.uniq(y) != [hd(y)],
              perms <- StreamData.member_of([9, 49, 99]),
              seed <- StreamData.integer(0..1000),
              max_runs: 25
            ) do
        m = ESDA.moran(y, path(9), permutations: perms, seed: seed)
        assert m.p_sim >= 1 / (perms + 1) - 1.0e-12 and m.p_sim <= 1.0
        assert length(m.sim) == perms
      end
    end
  end

  describe "moran/3 validation" do
    test "rejects bad input" do
      g = path(4)
      assert_raise ArgumentError, ~r/length/, fn -> ESDA.moran([1, 2, 3], g) end
      assert_raise ArgumentError, ~r/constant/, fn -> ESDA.moran([2, 2, 2, 2], g) end

      assert_raise ArgumentError, ~r/non-finite|expected a number/, fn ->
        ESDA.moran([1, 2, :nan, 4], g)
      end

      assert_raise ArgumentError, ~r/permutations/, fn ->
        ESDA.moran([1, 2, 3, 4], g, permutations: -1)
      end

      assert_raise ArgumentError, ~r/seed/, fn -> ESDA.moran([1, 2, 3, 4], g, seed: -3) end

      assert_raise ArgumentError, ~r/transform/, fn ->
        ESDA.moran([1, 2, 3, 4], g, transform: :x)
      end

      assert_raise ArgumentError, fn -> ESDA.moran([1, 2, 3, 4], g, bogus: 1) end
      assert_raise ArgumentError, ~r/Spatex.Graph/, fn -> ESDA.moran([1, 2, 3, 4], :nope) end
    end

    test "a graph with only islands is rejected" do
      g = Graph.from_arrays([0, 1, 2], [0, 1, 2], [0, 0, 0])
      assert_raise ArgumentError, ~r/no weights/, fn -> ESDA.moran([1, 2, 3], g) end
    end
  end

  describe "moran_bv/4" do
    test "x with itself equals the row-standardized univariate I" do
      g = path(8)
      bv = ESDA.moran_bv(@y, @y, g, permutations: 0)
      assert %MoranBV{permutations: 0, sim: nil, p_sim: nil} = bv
      assert_in_delta bv.i, ESDA.moran(@y, g, permutations: 0).i, 1.0e-12
    end

    test "is deterministic per seed and reports the simulation summary" do
      g = path(8)
      x = [2.0, 4.0, 1.0, 9.0, 3.0, 3.5, 8.0, 0.5]
      a = ESDA.moran_bv(x, @y, g, permutations: 99, seed: 3)
      assert a == ESDA.moran_bv(x, @y, g, permutations: 99, seed: 3)
      assert length(a.sim) == 99
      assert a.p_sim >= 0.01 and a.p_sim <= 1.0
      assert is_float(a.z_sim) and is_float(a.p_z_sim)
    end

    test "validates inputs" do
      g = path(4)

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.moran_bv([1, 1, 1, 1], [1, 2, 3, 4], g)
      end

      assert_raise ArgumentError, ~r/length/, fn -> ESDA.moran_bv([1, 2], [1, 2, 3, 4], g) end
    end
  end

  describe "moran_rate/4 and assuncao_rate/2" do
    test "EB standardization matches the hand computation" do
      [a, b] = ESDA.assuncao_rate([1, 4], [10, 20])
      assert_in_delta a, -0.7559289460184543, 1.0e-12
      assert_in_delta b, 0.365148371670111, 1.0e-12
    end

    test "unadjusted rates are e / b and feed plain Moran" do
      g = path(6)
      e = [1, 4, 2, 9, 5, 7]
      b = [10, 20, 15, 30, 12, 14]
      r = ESDA.moran_rate(e, b, g, adjusted: false, permutations: 0)
      assert %MoranRate{adjusted: false} = r
      assert_in_delta hd(r.y), 0.1, 1.0e-12

      plain = ESDA.moran(Enum.zip_with(e, b, &(&1 / &2)), g, permutations: 0)
      assert_in_delta r.i, plain.i, 1.0e-12
    end

    test "adjusted is the default and uses the EB rates" do
      g = path(6)
      e = [1, 4, 2, 9, 5, 7]
      b = [10, 20, 15, 30, 12, 14]
      r = ESDA.moran_rate(e, b, g, permutations: 49, seed: 1)
      assert r.adjusted
      assert r.y == ESDA.assuncao_rate(e, b)
      assert length(r.sim) == 49
    end

    test "validates events and base" do
      g = path(3)
      assert_raise ArgumentError, ~r/base/, fn -> ESDA.moran_rate([1, 2, 3], [1, 0, 3], g) end
      assert_raise ArgumentError, ~r/events/, fn -> ESDA.moran_rate([1, -2, 3], [1, 2, 3], g) end
      assert_raise ArgumentError, ~r/length/, fn -> ESDA.moran_rate([1, 2], [1, 2, 3], g) end
    end
  end
end
