defmodule Spatex.ESDALocalMultivariateOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.ESDA
  alias Spatex.Graph
  alias Spatex.Reference

  @perm_seed 20_240_601
  # P(|Z| > 4) for a standard normal, used to budget the multiple comparisons below.
  @tail4 6.334e-5

  defp graph(fx) do
    g = fx["graph"]
    ids = Enum.to_list(0..(length(g["ids"]) - 1))
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: ids)
  end

  # As in Spatex.ESDALocalGearyJoinsLoshOracleTest: two independent P-permutation estimates of
  # the same p differ with sd sqrt(2 p (1 - p) / P); allow the Poisson upper bound of sites
  # beyond 4 sd and none beyond 6 sd.
  defp assert_p_sims_close(actual, expected, perms, label) do
    pairs = Enum.reject(Enum.zip(actual, expected), fn {_a, e} -> e in [nil, :nan] end)

    sds =
      for {a, e} <- pairs do
        sd = :math.sqrt(2 * max(e * (1 - e), 1.0e-4) / perms) + 1 / perms
        {abs(a - e) / sd, a, e}
      end

    mu = length(sds) * @tail4
    allowed = ceil(mu + 3 * :math.sqrt(mu))
    over4 = Enum.count(sds, fn {d, _, _} -> d > 4 end)
    assert over4 <= allowed, "#{label}: #{over4} sites beyond 4 sd: #{inspect(sds)}"
    assert Enum.all?(sds, fn {d, _, _} -> d <= 6 end), "#{label}: a site beyond 6 sd"
  end

  describe "island_weight" do
    test "geary_local p_sim with island_weight 1 agrees with esda on the islands Graph" do
      fx = Reference.load!("esda/local_multivariate_islands.json")
      c = fx["island_weight"]

      m =
        ESDA.geary_local(c["x"], graph(fx),
          permutations: c["permutations"],
          seed: @perm_seed,
          island_weight: c["island_weight"]
        )

      assert_p_sims_close(m.p_sim, c["geary_p_sim"], c["permutations"], "geary island_weight")
      # Islands: observed 0 against a non-negative reference, the smallest p-value.
      assert Enum.take(m.p_sim, -2) == [1 / 1000, 1 / 1000]
    end
  end

  describe "geary_local_mv" do
    for file <- ~w(local_multivariate_columbus local_multivariate_islands) do
      test "#{file}: localG matches esda to 1e-9 and p_sim statistically" do
        fx = Reference.load!("esda/#{unquote(file)}.json")

        for c <- fx["geary_mv"] do
          m =
            ESDA.geary_local_mv(c["variables"], graph(fx),
              permutations: c["permutations"],
              seed: @perm_seed
            )

          Reference.assert_close(m.local_g, c["local_g"], 1.0e-9)
          assert_p_sims_close(m.p_sim, c["p_sim"], c["permutations"], c["label"])
        end
      end
    end

    test "islands get localG 0 whether or not esda drops them" do
      fx = Reference.load!("esda/local_multivariate_islands.json")
      [c] = fx["geary_mv"]
      m = ESDA.geary_local_mv(c["variables"], graph(fx), permutations: 0)
      Reference.assert_close(m.local_g, fx["geary_mv_dropped"]["local_g"], 1.0e-9)
      assert Enum.take(m.local_g, -2) == [0.0, 0.0]
    end
  end

  describe "join_counts_local_bv and join_counts_local_mv" do
    setup do
      {:ok, fx: Reference.load!("esda/local_multivariate_columbus.json")}
    end

    defp assert_joins(m, c) do
      assert m.ljc == c["ljc"], c["label"]
      nan? = fn v -> v in [nil, :nan] end
      assert Enum.map(m.p_sim, nan?) == Enum.map(c["p_sim"], nan?), c["label"]
      assert_p_sims_close(m.p_sim, c["p_sim"], c["permutations"], c["label"])
    end

    test "bivariate CLC and BJC match esda", %{fx: fx} do
      for c <- fx["joins_bv"] do
        case_atom = c["case"] |> String.downcase() |> Reference.fixture_atom()

        m =
          ESDA.join_counts_local_bv(c["x"], c["z"], graph(fx),
            case: case_atom,
            permutations: c["permutations"],
            seed: @perm_seed
          )

        assert_joins(m, c)
      end
    end

    test "multivariate co-location matches esda", %{fx: fx} do
      for c <- fx["joins_mv"] do
        m =
          ESDA.join_counts_local_mv(c["variables"], graph(fx),
            permutations: c["permutations"],
            seed: @perm_seed
          )

        assert_joins(m, c)
      end
    end
  end
end
