defmodule Spatex.ESDAMoranLocalOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.ESDA
  alias Spatex.ESDA.MoranLocal
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9
  @perm_seed 20_240_601
  @crit 0.05
  # P(|Z| > 4) for a standard normal, used to budget the multiple comparisons below.
  @tail4 6.334e-5

  defp graph(fx) do
    g = fx["graph"]
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  defp run(c, g, opts) do
    base = [
      transform: Spatex.Reference.fixture_atom(c["transform"]),
      geoda_quads: c["geoda_quads"],
      alternative: Spatex.Reference.fixture_atom(c["alternative"])
    ]

    if Map.has_key?(c, "x"),
      do: ESDA.moran_local_bv(c["x"], c["y"], g, [keep_simulations: true] ++ base ++ opts),
      else: ESDA.moran_local(c["y"], g, [keep_simulations: true] ++ base ++ opts)
  end

  # Two independent P-permutation estimates of the same p differ with sd sqrt(2 p (1-p) / P).
  # With N sites compared, the expected number beyond 4 sd is N * 6.3e-5; we allow the Poisson
  # upper bound ceil(mu + 3 sqrt(mu)) of them (1 for N = 49) and none beyond 6 sd.
  defp assert_p_sims_close(actual, expected, perms, label) do
    sds =
      for {a, e} <- Enum.zip(actual, expected) do
        sd = :math.sqrt(2 * max(e * (1 - e), 1.0e-4) / perms) + 1 / perms
        {abs(a - e) / sd, a, e}
      end

    mu = length(sds) * @tail4
    allowed = ceil(mu + 3 * :math.sqrt(mu))
    over4 = Enum.count(sds, fn {d, _, _} -> d > 4 end)

    assert over4 <= allowed,
           "#{label}: #{over4} sites beyond 4 sd (allowed #{allowed}): #{inspect(sds)}"

    assert Enum.all?(sds, fn {d, _, _} -> d <= 6 end),
           "#{label}: a site beyond 6 sd #{inspect(sds)}"
  end

  # Labels must agree wherever PySAL's p_sim is more than 3 sd away from the threshold.
  defp assert_labels_agree(m, c, perms) do
    sd = :math.sqrt(2 * @crit * (1 - @crit) / perms)
    mine = MoranLocal.cluster_labels(m, @crit)

    for {{a, e}, p} <- Enum.zip(Enum.zip(mine, c["labels"]), c["perm"]["p_sim"]),
        abs(p - @crit) > 3 * sd do
      assert a == e, "#{c["label"]}: label #{a} vs PySAL #{e} (PySAL p_sim #{p})"
    end
  end

  for file <-
        ~w(local_moran_columbus_queen local_moran_bv_columbus local_moran_islands local_moran_islands_many local_moran_lattice_islands) do
    describe "#{file}" do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx)}
      end

      test "Is, quadrants, z, lag and analytic moments match PySAL to 1e-9", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          m = run(c, g, permutations: 0)
          a = c["analytic"]
          mine = %{"is" => m.is, "q" => m.q, "z" => m.z, "lag" => m.lag}

          mine =
            if m.ei,
              do:
                Map.merge(mine, %{"ei" => m.ei, "vi" => m.vi, "eic" => m.ei_c, "vic" => m.vi_c}),
              else: mine

          try do
            Reference.assert_close(mine, Map.take(a, Map.keys(mine)), @tol)
          rescue
            e in ExUnit.AssertionError ->
              reraise %{e | message: "#{c["label"]}: #{e.message}"}, __STACKTRACE__
          end
        end
      end

      test "permutation inference agrees with PySAL statistically", %{fx: fx, g: g} do
        # Tiny graphs are skipped here: their reference distribution has a handful of discrete
        # values, so PySAL's p_sim depends on floating-point ties (numba fastmath) and two
        # correct implementations legitimately differ. They are still checked analytically.
        for c <- fx["cases"], c["perm_comparable"] do
          perms = c["permutations"]
          m = run(c, g, permutations: perms, seed: @perm_seed)
          p = c["perm"]

          assert_p_sims_close(m.p_sim, p["p_sim"], perms, c["label"])
          assert_labels_agree(m, c, perms)

          for {a, e, se} <-
                Enum.zip([m.ei_sim, p["ei_sim"], p["sei_sim"]]) do
            assert abs(a - e) <= 4 * :math.sqrt(2) * se / :math.sqrt(perms) + 1.0e-12
          end

          for {a, e} <- Enum.zip(m.se_i_sim, p["sei_sim"]) do
            assert abs(a - e) <= 0.05 * e + 1.0e-12
          end

          # Islands: PySAL's z_sim is NaN exactly where ours is.
          assert Enum.map(m.z_sim, &(&1 == :nan)) == Enum.map(p["z_sim"], &(&1 == :nan))
          assert length(m.sim) == m.n
        end
      end
    end
  end

  describe "known deviation: constant y" do
    test "PySAL yields NaN, Spatex raises" do
      fx = Reference.load!("esda/local_moran_constant.json")
      assert Enum.all?(fx["pysal"]["is"], &(&1 == :nan))

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.moran_local(fx["y"], graph(fx), permutations: 0)
      end
    end
  end
end
