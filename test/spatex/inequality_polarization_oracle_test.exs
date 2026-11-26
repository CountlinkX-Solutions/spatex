defmodule Spatex.InequalityPolarizationOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Inequality
  alias Spatex.Reference

  defp graph(fx, name) do
    g = fx["graphs"][name]

    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
      ids: Enum.to_list(0..(length(g["ids"]) - 1))
    )
  end

  defp opts(c), do: for({key, v} <- c["opts"], do: {String.to_existing_atom(key), v})

  setup_all do
    {:ok, fx: Reference.load!("dynamics/polarization.json")}
  end

  test "statistic, counts and labels match inequality.polarization.S", %{fx: fx} do
    for c <- fx["cases"] do
      r = Inequality.polarization(c["y"], graph(fx, c["graph"]), [permutations: 0] ++ opts(c))
      l = c["label"]
      Reference.assert_close(r.statistic, c["statistic"], 1.0e-12)

      assert {r.n_a_components, r.n_g_components, r.n_i_components} ==
               {c["n_a_components"], c["n_g_components"], c["n_i_components"]},
             l

      # PySAL marks observations without a group -1, Spatex nil
      assert r.i_labels == Enum.map(c["i_labels"], &if(&1 < 0, do: nil, else: &1)), l
      assert r.a_labels == c["a_labels"], l
      assert r.g_labels == c["g_labels"], l
    end
  end

  test "the permutation null matches PySAL's statistically", %{fx: fx} do
    perms = 9999

    for c <- fx["cases"] do
      r =
        Inequality.polarization(
          c["y"],
          graph(fx, c["graph"]),
          [permutations: perms, seed: 31] ++ opts(c)
        )

      l = c["label"]
      expected = c["p_value"]
      sd = :math.sqrt(2 * max(expected * (1 - expected), 1.0e-4) / perms) + 1 / perms
      assert abs(r.p_value - expected) <= 5 * sd, "#{l}: #{r.p_value} vs #{expected}"
      mean = Enum.sum(r.sim) / perms
      se = max(c["sim_std"], 1.0e-3) / :math.sqrt(perms)
      assert abs(mean - c["sim_mean"]) <= 6 * se * :math.sqrt(2) + 1.0e-9, l
      sd = :math.sqrt(Enum.sum(for v <- r.sim, do: (v - mean) * (v - mean)) / perms)
      assert abs(sd - c["sim_std"]) <= 0.05 * c["sim_std"] + 1.0e-9, l
    end
  end

  test "the docstring lattice is fully polarized", %{fx: fx} do
    d = fx["docstring_lattice"]

    pairs =
      for r <- 0..39,
          c <- 0..39,
          {dr, dc} <- [{0, 1}, {1, 0}],
          r + dr < 40 and c + dc < 40,
          do: {r * 40 + c, (r + dr) * 40 + c + dc}

    {f, nb} = pairs |> Enum.flat_map(fn {a, b} -> [{a, b}, {b, a}] end) |> Enum.unzip()
    g = Graph.from_arrays(f, nb, nil, ids: Enum.to_list(0..1599))
    r = Inequality.polarization(Enum.to_list(0..1599), g, k: 2, permutations: 99, seed: 123)
    assert r.statistic == d["statistic"]

    assert {r.n_a_components, r.n_g_components, r.n_i_components} ==
             {d["n_a_components"], d["n_g_components"], d["n_i_components"]}

    assert_in_delta r.p_value, d["p_value"], 0.05
  end

  test "invalid arguments raise where PySAL raises ValueError", %{fx: fx} do
    g = graph(fx, "mexico")
    y = Enum.at(fx["cases"], 0)["y"]

    assert fx["errors"] == %{
             "bins_out_of_range" => "ValueError",
             "k0" => "ValueError",
             "k_above_n" => "ValueError"
           }

    assert_raise ArgumentError, fn -> Inequality.polarization(y, g, k: 0) end
    assert_raise ArgumentError, fn -> Inequality.polarization(y, g, k: 99) end
    assert_raise ArgumentError, fn -> Inequality.polarization(y, g, bins: [9000.0, 20_000.0]) end
  end
end
