defmodule Spatex.GWRBwsOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, SpatialVariability}
  alias Spatex.Reference

  @fixture "regression/gwr_bws.json"
  @levels [0.95, 0.9, 0.5, 0.99]

  describe "bandwidth_intervals/2 for GWR (mgwr 2.2.1 GWRResults.get_bws_intervals)" do
    for {key, _} <- Reference.load!(@fixture)["intervals"] do
      @key key
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["intervals"][@key]
        ds = fx["datasets"][c["dataset"]]

        # The selector history is the one mgwr recorded: the interval follows from it exactly.
        selector = %Bandwidth{history: pairs(c["history"]), bw: c["bw"]}

        for level <- @levels do
          [lo, hi] = c["intervals"][to_string(level)]
          assert {got_lo, got_hi} = GWR.bandwidth_intervals(selector, level: level)
          assert_bounds([got_lo, got_hi], [lo, hi], c["options"]["fixed"])
        end

        # And from a search run by Spatex (whose history is mgwr's to 1e-9).
        found = GWR.select_bandwidth(ds["coords"], ds["y"], ds["x"], search_opts(c["options"]))
        [lo, hi] = c["intervals"]["0.95"]
        {got_lo, got_hi} = GWR.bandwidth_intervals(found)
        assert_bounds([got_lo, got_hi], [lo, hi], c["options"]["fixed"])
      end
    end
  end

  describe "bandwidth_intervals/2 for MGWR (mgwr 2.2.1 MGWRResults.get_bws_intervals)" do
    for {key, _} <- Reference.load!(@fixture)["mgwr_intervals"] do
      @key key
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["mgwr_intervals"][@key]
        ds = fx["datasets"][c["dataset"]]
        opts = [fixed: c["options"]["fixed"] || false]
        r = GWR.mgwr(ds["coords"], c["y"], c["x"], opts)

        # The searches of the backfitting are mgwr's `sel_hist`, entry by entry.
        assert length(r.sel_hist) == length(c["sel_hist"])

        for {got, want} <- Enum.zip(r.sel_hist, c["sel_hist"]) do
          assert length(got) == length(want)
          Reference.assert_close(Enum.map(got, &Tuple.to_list/1), want, 1.0e-9)
        end

        for level <- @levels do
          want = c["intervals"][to_string(level)]
          got = r |> GWR.bandwidth_intervals(level: level) |> Enum.map(&Tuple.to_list/1)
          assert length(got) == r.k
          assert_bounds(List.flatten(got), List.flatten(want), c["options"]["fixed"])
        end
      end
    end
  end

  describe "spatial_variability/4 (mgwr 2.2.1 GWRResults.spatial_variability)" do
    for {key, _} <- Reference.load!(@fixture)["variability"] do
      @key key
      @tag :statistical
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["variability"][@key]
        ds = fx["datasets"][c["dataset"]]
        perms = c["n_iters"]

        r =
          GWR.spatial_variability(
            ds["coords"],
            ds["y"],
            ds["x"],
            search_opts(c["options"]) ++ [permutations: perms, seed: 2026]
          )

        assert %SpatialVariability{permutations: ^perms, seed: 2026} = r
        assert r.bw == c["bw"] or abs(r.bw - c["bw"]) < 1.0e-6 * c["bw"]

        # The observed statistic is deterministic.
        Reference.assert_close(r.observed_sd, c["observed_sd"], 1.0e-8)
        assert length(r.p_values) == length(c["p_values"])

        # The draws differ (numpy versus ChaCha8): compare within Monte Carlo error.
        for {{p, pm}, j} <- Enum.with_index(Enum.zip(r.p_values, c["p_values"])) do
          avg = (p + pm) / 2
          band = 4 * :math.sqrt(2 * avg * (1 - avg) / perms) + 3 / perms
          assert abs(p - pm) <= band, "column #{j}: p #{p} vs mgwr #{pm} (band #{band})"
        end

        mgwr_cols = transpose(c["sds"])
        sim_cols = transpose(r.simulated_sd)

        for {sim, ref} <- Enum.zip(sim_cols, mgwr_cols) do
          mean = fn v -> Enum.sum(v) / length(v) end
          sd = :math.sqrt(Enum.sum(for v <- ref, do: (v - mean.(ref)) ** 2) / (length(ref) - 1))
          assert abs(mean.(sim) - mean.(ref)) <= 4 * sd * :math.sqrt(2 / perms)
        end
      end
    end
  end

  # Adaptive bounds are exact. Fixed searches record each probe and its two-decimal rounding with
  # equal scores; ties decide which one bounds (known_deviations.exs), so allow the rounding step.
  defp assert_bounds(got, want, fixed) do
    tol = if fixed, do: 0.005 + 1.0e-9, else: 1.0e-9

    for {g, w} <- Enum.zip(got, want) do
      assert abs(g - w) <= tol, "bound #{g} vs mgwr #{w}"
    end
  end

  defp transpose(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

  defp pairs(history), do: Enum.map(history, fn [b, s] -> {b, s} end)

  defp search_opts(o) do
    [
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      criterion: String.to_atom(String.downcase(o["criterion"] || "aicc"))
    ]
    |> put(:search, o["search_method"] && String.to_atom(o["search_method"]))
    |> put(:bw_min, o["bw_min"])
    |> put(:bw_max, o["bw_max"])
    |> put(:interval, o["interval"])
  end

  defp put(opts, _key, nil), do: opts
  defp put(opts, key, value), do: Keyword.put(opts, key, value)
end
