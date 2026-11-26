defmodule Spatex.ESDALocalGearyJoinsLoshOracleTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

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

  defp tag(label, fun) do
    fun.()
  rescue
    e in ExUnit.AssertionError ->
      reraise %{e | message: "#{label}: #{e.message}"}, __STACKTRACE__
  end

  # Two independent P-permutation estimates of the same p differ with sd sqrt(2 p (1-p) / P).
  # We allow the Poisson upper bound of sites beyond 4 sd and none beyond 6 sd.
  #
  # The two-sided p-value mirrors percentiles and counts both tails, so it is about twice as
  # volatile as a binomial proportion (`scale`) and wildly so near the median (p close to 1):
  # sites with p above `max_p` are not compared for `:two_sided`.
  defp assert_p_sims_close(actual, expected, perms, label, {max_p, scale} \\ {1.0, 1}) do
    pairs =
      Enum.reject(Enum.zip(actual, expected), fn {_a, e} -> e == :nan or e > max_p end)

    sds =
      for {a, e} <- pairs do
        sd = scale * :math.sqrt(2 * max(e * (1 - e), 1.0e-4) / perms) + 1 / perms
        {abs(a - e) / sd, a, e}
      end

    mu = length(sds) * @tail4
    allowed = ceil(mu + 3 * :math.sqrt(mu))
    over4 = Enum.count(sds, fn {d, _, _} -> d > 4 end)
    assert over4 <= allowed, "#{label}: #{over4} sites beyond 4 sd: #{inspect(sds)}"

    assert Enum.all?(sds, fn {d, _, _} -> d <= 6 end),
           "#{label}: a site beyond 6 sd #{inspect(sds)}"
  end

  defp band(c), do: if(alt(c) == :two_sided, do: {0.8, 2}, else: {1.0, 1})
  defp alt(c), do: c["alternative"] |> String.replace("-", "_") |> Reference.fixture_atom()

  describe "geary_local" do
    for file <- ~w(local_geary_columbus_queen local_geary_lattice local_geary_islands) do
      test "#{file}: localG matches PySAL to 1e-9" do
        fx = Reference.load!("esda/#{unquote(file)}.json")

        for c <- fx["cases"] do
          m = ESDA.geary_local(c["x"], graph(fx), permutations: 0)
          tag(c["label"], fn -> Reference.assert_close(m.local_g, c["local_g"], @tol) end)
        end
      end

      test "#{file}: p_sim and labels agree with PySAL statistically" do
        fx = Reference.load!("esda/#{unquote(file)}.json")

        for c <- fx["cases"] do
          perms = c["permutations"]

          m =
            ESDA.geary_local(c["x"], graph(fx),
              permutations: perms,
              seed: @perm_seed,
              alternative: alt(c),
              labels: true,
              sig: c["sig"],
              keep_simulations: true
            )

          assert_p_sims_close(m.p_sim, c["p_sim"], perms, c["label"], band(c))
          assert length(m.sim) == m.n

          # A site's label is decided by p_sim <= sig: compare away from the threshold.
          sd = :math.sqrt(2 * c["sig"] * (1 - c["sig"]) / perms)

          for {{mine, theirs}, p} <- Enum.zip(Enum.zip(m.labs, c["labs"]), c["p_sim"]),
              abs(p - c["sig"]) > 4 * sd do
            assert (mine || :nan) == theirs, "#{c["label"]}: label #{mine} vs #{theirs} (p #{p})"
          end
        end
      end
    end

    test "known deviation: PySAL drops islands from localG and misaligns it with p_sim" do
      fx = Reference.load!("esda/local_geary_islands.json")
      assert fx["pysal_default"]["local_g_length"] == 25
      assert fx["pysal_default"]["p_sim_length"] == 27

      m =
        ESDA.geary_local(fx["pysal_default"]["x"], graph(fx),
          permutations: 99,
          seed: 1,
          keep_simulations: true
        )

      assert length(m.local_g) == 27 and length(m.p_sim) == 27
      assert Enum.take(m.local_g, -2) == [0.0, 0.0]
    end
  end

  describe "join_counts_local" do
    for file <- ~w(local_joins_columbus_queen local_joins_lattice local_joins_islands) do
      test "#{file}: counts match PySAL exactly and p_sim statistically" do
        fx = Reference.load!("esda/#{unquote(file)}.json")

        for c <- fx["cases"] do
          g = graph(fx)
          counts = ESDA.join_counts_local(c["y"], g, permutations: 0).ljc
          tag(c["label"], fn -> Reference.assert_close(counts, c["ljc"], @tol) end)

          perms = c["permutations"]

          m =
            ESDA.join_counts_local(c["y"], g,
              permutations: perms,
              seed: @perm_seed,
              alternative: alt(c),
              keep_simulations: true
            )

          assert Enum.map(m.p_sim, &(&1 == :nan)) == Enum.map(c["p_sim"], &(&1 == :nan))
          assert_p_sims_close(m.p_sim, c["p_sim"], perms, c["label"], band(c))
        end
      end
    end

    test "known deviations: PySAL crashes on islands, counts an island as its own join and ignores permutations=" do
      fx = Reference.load!("esda/local_joins_islands.json")
      assert fx["pysal_graph_crash"] == "IndexError"

      # An island with y = 1 is its own (self, self) pair in PySAL's adjacency list: LJC 1.
      y = fx["pysal_island_selfpair"]["y"]
      assert Enum.take(fx["pysal_island_selfpair"]["ljc"], -2) == [1.0, 1.0]

      assert Enum.take(ESDA.join_counts_local(y, graph(fx), permutations: 0).ljc, -2) == [
               0.0,
               0.0
             ]

      # fit() ignores the constructor's permutations: Spatex honors its single option.
      assert fx["pysal_constructor_permutations"] == %{
               "constructor" => 49,
               "permutations_used" => 999
             }

      m = ESDA.join_counts_local(y, graph(fx), permutations: 49, seed: 1, keep_simulations: true)
      assert m.permutations == 49 and Enum.all?(m.sim, &(length(&1) == 49))
    end
  end

  describe "losh" do
    for file <- ~w(losh_columbus_queen losh_lattice) do
      test "#{file}: Hi, ylag, yresid, VarHi and p-values match PySAL to 1e-9" do
        fx = Reference.load!("esda/#{unquote(file)}.json")

        for c <- fx["cases"] do
          g = if c["label"] =~ "/R", do: Graph.transform(graph(fx), :r), else: graph(fx)
          inference = if c["inference"], do: :chi2

          capture_log(fn ->
            m = ESDA.losh(c["y"], g, a: c["a"], inference: inference)

            mine = %{"hi" => m.hi, "ylag" => m.ylag, "yresid" => m.yresid, "var_hi" => m.var_hi}
            expected = Map.take(c, Map.keys(mine))

            mine =
              if Map.has_key?(c, "pval"),
                do: Map.put(mine, "pval", m.pval),
                else:
                  (
                    assert m.pval == nil
                    mine
                  )

            expected =
              if Map.has_key?(c, "pval"), do: Map.put(expected, "pval", c["pval"]), else: expected

            tag(c["label"], fn -> Reference.assert_close(mine, expected, @tol) end)
          end)
        end
      end
    end

    test "known deviation: PySAL gives all-NaN Hi with any island, Spatex the island-free values" do
      fx = Reference.load!("esda/losh_islands.json")
      assert Enum.all?(fx["pysal_islands"]["hi"], &(&1 == :nan))
      assert Enum.all?(fx["pysal_islands"]["pval"], &(&1 == :nan))

      m = ESDA.losh(fx["y"], graph(fx), inference: :chi2)
      sub = fx["subgraph_case"]

      Reference.assert_close(
        %{
          "hi" => Enum.take(m.hi, 5),
          "var_hi" => Enum.take(m.var_hi, 5),
          "pval" => Enum.take(m.pval, 5)
        },
        %{"hi" => sub["hi"], "var_hi" => sub["var_hi"], "pval" => sub["pval"]},
        @tol
      )

      assert List.last(m.hi) == :nan
    end
  end
end
