defmodule Spatex.ESDAMoranOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.ESDA
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9
  @analytic_keys ~w(i ei vi_norm sei_norm z_norm p_norm vi_rand sei_rand z_rand p_rand)
  @perm_seed 20_240_601

  defp graph(fx) do
    g = fx["graph"]
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  defp transform(t), do: Spatex.Reference.fixture_atom(t)

  defp analytic(m) do
    %{
      "i" => m.i,
      "ei" => m.ei,
      "vi_norm" => m.vi_norm,
      "sei_norm" => m.se_i_norm,
      "z_norm" => m.z_norm,
      "p_norm" => m.p_norm,
      "vi_rand" => m.vi_rand,
      "sei_rand" => m.se_i_rand,
      "z_rand" => m.z_rand,
      "p_rand" => m.p_rand
    }
  end

  # Statistical comparison of a P-permutation pseudo p-value with PySAL's own estimate: two
  # independent binomial estimates differ by at most 4 sqrt(2 p (1 - p) / P) (+ 1/P rounding).
  defp assert_p_sim_close(actual, expected, perms) do
    band = 4 * :math.sqrt(2 * expected * (1 - expected) / perms) + 1 / perms

    assert abs(actual - expected) <= band,
           "p_sim #{actual} vs PySAL #{expected} (band #{band})"
  end

  defp assert_sim_summary_close(m, perm, perms) do
    assert_p_sim_close(m.p_sim, perm["p_sim"], perms)
    # Means of P draws: difference of two estimates, 4 sigma.
    band = 4 * :math.sqrt(2) * perm["sei_sim"] / :math.sqrt(perms)
    assert abs(m.ei_sim - perm["ei_sim"]) <= band
    assert abs(m.se_i_sim - perm["sei_sim"]) <= 0.05 * perm["sei_sim"]
    assert abs(m.z_sim - perm["z_sim"]) <= 0.06 * abs(perm["z_sim"]) + 0.1
    assert length(m.sim) == perms
  end

  for file <-
        ~w(moran_columbus_queen moran_lattice moran_islands moran_islands_many moran_asymmetric) do
    describe "#{file}" do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx)}
      end

      test "analytic quantities match PySAL to 1e-9", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          m =
            ESDA.moran(c["y"], g,
              permutations: 0,
              transform: transform(c["transform"]),
              two_tailed: c["two_tailed"]
            )

          try do
            Reference.assert_close(analytic(m), Map.take(c["analytic"], @analytic_keys), @tol)
          rescue
            e in ExUnit.AssertionError ->
              reraise %{e | message: "#{c["label"]}: #{e.message}"}, __STACKTRACE__
          end
        end
      end

      test "permutation inference agrees with PySAL statistically", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          perms = c["permutations"]

          m =
            ESDA.moran(c["y"], g,
              permutations: perms,
              seed: @perm_seed,
              transform: transform(c["transform"]),
              two_tailed: c["two_tailed"]
            )

          assert_sim_summary_close(m, c["perm"], perms)

          # The simulated mean is close to the analytic expectation.
          assert abs(m.ei_sim - m.ei) <= 4 * m.se_i_sim / :math.sqrt(perms),
                 "#{c["label"]}: ei_sim #{m.ei_sim} vs ei #{m.ei}"
        end
      end
    end
  end

  for file <- ~w(moran_bv_columbus moran_bv_lattice) do
    describe "#{file}" do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx)}
      end

      test "I matches PySAL and the permutation test agrees statistically", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          perms = c["permutations"]
          t = transform(c["transform"])

          none = ESDA.moran_bv(c["x"], c["y"], g, permutations: 0, transform: t)
          Reference.assert_close(none.i, c["analytic"]["i"], @tol)

          m =
            ESDA.moran_bv(c["x"], c["y"], g, permutations: perms, seed: @perm_seed, transform: t)

          assert m.i == none.i
          assert_p_sim_close(m.p_sim, c["perm"]["p_sim"], perms)
          band = 4 * :math.sqrt(2) * c["perm"]["sei_sim"] / :math.sqrt(perms)
          assert abs(m.ei_sim - c["perm"]["ei_sim"]) <= band
          assert abs(m.se_i_sim - c["perm"]["sei_sim"]) <= 0.05 * c["perm"]["sei_sim"]
        end
      end
    end
  end

  describe "moran_rate_columbus" do
    setup do
      fx = Reference.load!("esda/moran_rate_columbus.json")
      {:ok, fx: fx, g: graph(fx)}
    end

    test "EB standardization matches esda.smoothing.assuncao_rate", %{fx: fx} do
      for c <- fx["assuncao"] do
        Reference.assert_close(ESDA.assuncao_rate(c["e"], c["b"]), c["y"], @tol)
      end
    end

    test "analytic quantities and rates match PySAL", %{fx: fx, g: g} do
      for c <- fx["cases"] do
        m = ESDA.moran_rate(c["e"], c["b"], g, adjusted: c["adjusted"], permutations: 0)
        Reference.assert_close(m.y, c["y"], @tol)
        Reference.assert_close(analytic(m), Map.take(c["analytic"], @analytic_keys), @tol)
      end
    end

    test "permutation inference agrees statistically", %{fx: fx, g: g} do
      for c <- fx["cases"] do
        perms = c["permutations"]

        m =
          ESDA.moran_rate(c["e"], c["b"], g,
            adjusted: c["adjusted"],
            permutations: perms,
            seed: @perm_seed
          )

        assert_sim_summary_close(m, c["perm"], perms)
      end
    end
  end

  describe "known deviation: constant y" do
    test "PySAL yields NaN, Spatex raises" do
      fx = Reference.load!("esda/moran_constant.json")
      assert fx["pysal"]["i"] == :nan
      assert fx["pysal"]["z_norm"] == :nan

      assert_raise ArgumentError, ~r/constant/, fn ->
        ESDA.moran(fx["y"], graph(fx), permutations: 0)
      end
    end
  end
end
