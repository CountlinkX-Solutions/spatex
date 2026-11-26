defmodule Spatex.ESDAGLocalOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.ESDA
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9
  @perm_seed 20_240_601
  # P(|Z| > 4) for a standard normal, used to budget the multiple comparisons below.
  @tail4 6.334e-5

  defp graph(fx) do
    g = fx["graph"]
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  defp star("infer"), do: nil
  defp star(other), do: other

  defp run(c, g, opts) do
    g =
      if c["input_transform"] == "o",
        do: g,
        else: Graph.transform(g, String.to_atom(c["input_transform"]))

    ESDA.g_local(
      c["y"],
      g,
      [
        transform: String.to_existing_atom(c["transform"]),
        star: star(c["star_arg"]),
        alternative: c["alternative"] |> String.replace("-", "_") |> String.to_existing_atom(),
        island_weight: c["island_weight"],
        keep_simulations: true
      ] ++ opts
    )
  end

  # Two independent P-permutation estimates of the same p differ with sd sqrt(2 p (1-p) / P).
  # With N sites compared, the expected number beyond 4 sd is N * 6.3e-5; we allow the Poisson
  # upper bound ceil(mu + 3 sqrt(mu)) of them and none beyond 6 sd.
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

  for file <-
        ~w(g_local_columbus_queen g_local_columbus_band g_local_columbus_selfweight g_local_islands g_local_lattice_islands) do
    describe "#{file}" do
      setup do
        fx = Reference.load!("esda/#{unquote(file)}.json")
        {:ok, fx: fx, g: graph(fx)}
      end

      test "Gs, EGs, VGs, Zs and p_norm match PySAL to 1e-9", %{fx: fx, g: g} do
        for c <- fx["cases"] do
          m = run(c, g, permutations: 0)
          assert m.star == c["star"], "#{c["label"]}: star flag"

          mine = %{
            "gs" => m.gs,
            "egs" => m.egs,
            "vgs" => m.vgs,
            "zs" => m.zs,
            "p_norm" => m.p_norm
          }

          try do
            Reference.assert_close(mine, Map.take(c["analytic"], Map.keys(mine)), @tol)
          rescue
            e in ExUnit.AssertionError ->
              reraise %{e | message: "#{c["label"]}: #{e.message}"}, __STACKTRACE__
          end
        end
      end

      test "permutation inference agrees with PySAL statistically", %{fx: fx, g: g} do
        # Tiny graphs are analytic-only: their reference distribution has a handful of discrete
        # values, so PySAL's p_sim depends on floating-point ties (numba fastmath).
        for c <- fx["cases"], c["perm_comparable"] do
          perms = c["permutations"]
          m = run(c, g, permutations: perms, seed: @perm_seed)
          p = c["perm"]

          if c["alternative"] in ["two-sided", "folded"] do
            # These p-values hinge on the sampled median/mean of a reference distribution shared by
            # all sites, so they are noisier near the center than a binomial proportion (differences
            # up to 0.042 over five seeds at P = 9999); compare with an absolute bound instead.
            for {a, e} <- Enum.zip(m.p_sim, p["p_sim"]) do
              assert abs(a - e) <= 0.07, "#{c["label"]}: #{a} vs #{e}"
            end
          else
            assert_p_sims_close(m.p_sim, p["p_sim"], perms, c["label"])
          end

          for {a, e, se} <- Enum.zip([m.eg_sim, p["eg_sim"], p["seg_sim"]]) do
            assert abs(a - e) <= 4 * :math.sqrt(2) * se / :math.sqrt(perms) + 1.0e-12
          end

          for {a, e} <- Enum.zip(m.se_g_sim, p["seg_sim"]) do
            assert abs(a - e) <= 0.05 * e + 1.0e-12
          end

          assert Enum.map(m.z_sim, &(&1 == :nan)) == Enum.map(p["z_sim"], &(&1 == :nan))
          assert length(m.sim) == m.n
        end
      end
    end
  end

  describe "esda probes (columbus queen, CRIME, R)" do
    setup do
      fx = Reference.load!("esda/g_local_columbus_queen.json")
      {:ok, fx: fx, g: graph(fx)}
    end

    test "first sites", %{fx: fx, g: g} do
      y = hd(fx["cases"])["y"]
      off = ESDA.g_local(y, g, permutations: 0)
      on = ESDA.g_local(y, g, permutations: 0, star: true)
      assert_close_list(Enum.take(off.gs, 3), [0.01449019, 0.01541655, 0.01739635])
      assert_close_list(Enum.take(off.zs, 3), [-0.65603764, -0.55683105, -0.3475563])
      assert_close_list(Enum.take(on.gs, 3), [0.01261722, 0.01416685, 0.01722799])
      assert_close_list(Enum.take(on.zs, 3), [-0.8097987, -0.64872863, -0.33055009])
    end

    defp assert_close_list(actual, expected) do
      for {a, e} <- Enum.zip(actual, expected), do: assert_in_delta(a, e, 1.0e-7)
    end
  end

  describe "fdr" do
    test "reproduces esda.fdr bit for bit" do
      for c <- Reference.load!("esda/fdr.json")["cases"] do
        assert ESDA.fdr(c["p"], c["alpha"]) == c["cutoff"], "#{c["label"]} alpha #{c["alpha"]}"
      end
    end
  end
end
