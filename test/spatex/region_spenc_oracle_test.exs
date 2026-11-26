defmodule Spatex.RegionSpencOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  import Spatex.RegionHelper

  alias Spatex.Graph
  alias Spatex.Reference
  alias Spatex.Region

  setup_all do
    %{fx: Reference.load!("region/spenc.json")}
  end

  defp columns(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

  defp graph_with_weights(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  # The attributes and graph of a case, as the generator used them.
  defp input(fx, name) do
    c = fx["columbus"]

    {x, graph} =
      cond do
        name =~ "weighted" -> {c["x"], c["weighted_graph"]}
        name =~ ~r/linear|polynomial|sigmoid|cosine/ -> {c["x_positive"], c["graph"]}
        name =~ "precomputed" -> {c["precomputed"], c["graph"]}
        true -> {c["x"], c["graph"]}
      end

    {columns(x), graph_with_weights(graph)}
  end

  defp opts(c) do
    p = c["params"]

    base =
      for {key, atom} <- [
            {"gamma", :gamma},
            {"degree", :degree},
            {"coef0", :coef0},
            {"n_neighbors", :n_neighbors}
          ],
          Map.has_key?(p, key),
          do: {atom, p[key]}

    base =
      if p["n_clusters"] == "inf",
        do: [{:n_clusters, :infinity} | base],
        else: [{:n_clusters, p["n_clusters"]} | base]

    base =
      if p["affinity"], do: [{:affinity, String.to_atom(p["affinity"])} | base], else: base

    base =
      if p["assign_labels"],
        do: [{:assign_labels, String.to_atom(p["assign_labels"])} | base],
        else: base

    fit = c["fit_kwargs"]
    base = if fit["floor"], do: [{:floor, fit["floor"]} | base], else: base
    if fit["grid_resolution"], do: [{:grid_resolution, fit["grid_resolution"]} | base], else: base
  end

  defp deterministic_cases(fx), do: Enum.filter(fx["cases"], &Map.has_key?(&1, "affinity"))

  defp align_signs(mine, theirs) do
    mine_cols = columns(mine)
    their_cols = columns(theirs)

    mine_cols
    |> Enum.zip(their_cols)
    |> Enum.map(fn {m, t} ->
      dot = m |> Enum.zip(t) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()
      if dot < 0, do: Enum.map(m, &(-&1)), else: m
    end)
    |> columns()
  end

  test "the affinity matrix equals spopt's to 1.0e-12", %{fx: fx} do
    for c <- deterministic_cases(fx) do
      {x, g} = input(fx, c["name"])
      r = Region.spenc(x, g, [keep_affinity: true, seed: 1] ++ opts(c))

      Reference.assert_close(
        List.flatten(r.affinity_matrix),
        List.flatten(c["affinity"]),
        1.0e-12
      )
    end
  end

  test "the Laplacian eigenvalues equal spopt's eigsh values (of -L, reversed)", %{fx: fx} do
    for c <- deterministic_cases(fx) do
      {x, g} = input(fx, c["name"])
      r = Region.spenc(x, g, [seed: 1] ++ opts(c))
      expected = c["eigenvalues"] |> Enum.map(&(-&1)) |> Enum.reverse()
      Reference.assert_close(r.eigenvalues, expected, 1.0e-8)
    end
  end

  test "the embedding equals spopt's up to the sign of each vector", %{fx: fx} do
    for c <- deterministic_cases(fx) do
      {x, g} = input(fx, c["name"])
      r = Region.spenc(x, g, [seed: 1] ++ opts(c))
      mine = align_signs(r.embedding, c["embedding"])
      Reference.assert_close(List.flatten(mine), List.flatten(c["embedding"]), 1.0e-6)
    end
  end

  test "spopt's sign normalization is reproduced (largest entry positive)", %{fx: fx} do
    for c <- deterministic_cases(fx) do
      {x, g} = input(fx, c["name"])
      r = Region.spenc(x, g, [seed: 1] ++ opts(c))

      for col <- columns(r.embedding) do
        big = Enum.max_by(col, &abs/1)
        assert big > 0
      end
    end
  end

  test "discretize labels equal sklearn's for the start row each seed draws", %{fx: fx} do
    for c <- fx["cases"], Map.has_key?(c, "by_start_row") do
      {x, g} = input(fx, c["name"])

      rows =
        for seed <- 0..149 do
          r = Region.spenc(x, g, [seed: seed] ++ opts(c))
          assert r.labels == Enum.at(c["by_start_row"], r.start_row), "#{c["name"]} seed #{seed}"
          r.start_row
        end

      assert length(Enum.uniq(rows)) > 20
    end
  end

  test "hierarchical labels equal spopt's, and its failures are Spatex's", %{fx: fx} do
    for c <- fx["cases"], hierarchical?(c) do
      {x, g} = input(fx, c["name"])

      if c["error"] do
        assert_raise ArgumentError, fn -> Region.spenc(x, g, opts(c)) end
      else
        assert Region.spenc(x, g, opts(c)).labels == c["labels"], c["name"]
      end
    end
  end

  defp hierarchical?(c),
    do: c["params"]["assign_labels"] == "hierarchical" or c["params"]["n_clusters"] == "inf"

  describe "statistical agreement of the randomized label assignment" do
    test "discretize partitions follow the exact start-row distribution", %{fx: fx} do
      case4 = Enum.find(fx["cases"], &(&1["name"] == "rbf_k4"))
      {x, g} = input(fx, "rbf_k4")
      n = length(case4["by_start_row"])

      # spopt's partition distribution is the uniform mixture over the n start rows.
      exact = case4["by_start_row"] |> Enum.frequencies() |> Map.new(fn {k, v} -> {k, v / n} end)

      # Every partition spopt produced over 100 runs is one of those.
      spopt = fx["columbus_runs"]["discretize_runs"]
      assert Enum.all?(spopt, &Map.has_key?(exact, &1))

      runs = for seed <- 0..399, do: Region.spenc(x, g, [seed: seed] ++ opts(case4)).labels
      assert Enum.all?(runs, &Map.has_key?(exact, &1))

      empirical = runs |> Enum.frequencies() |> Map.new(fn {k, v} -> {k, v / 400} end)
      keys = Enum.uniq(Map.keys(exact) ++ Map.keys(empirical))
      tv = Enum.sum(for k <- keys, do: abs(Map.get(exact, k, 0) - Map.get(empirical, k, 0))) / 2
      assert tv < 0.1, "total variation distance #{tv}"
    end

    test "both strategies recover the planted partition like spopt", %{fx: fx} do
      pl = fx["planted"]
      g = graph_with_weights(pl["graph"])
      x = columns(pl["x"])

      for {assign, key} <- [discretize: "discretize_runs", kmeans: "kmeans_runs"] do
        spopt_hits = Enum.count(pl[key], &(ari(pl["planted"], &1) == 1.0))

        hits =
          Enum.count(0..99, fn seed ->
            r = Region.spenc(x, g, n_clusters: 4, gamma: 0.5, assign_labels: assign, seed: seed)
            ari(pl["planted"], r.labels) == 1.0 and Region.contiguous?(r.labels, g)
          end)

        assert spopt_hits == 100
        assert hits >= 95, "#{assign}: #{hits} of 100 runs recovered the planted partition"
      end
    end

    test "kmeans partitions resemble spopt's on Columbus", %{fx: fx} do
      runs = fx["columbus_runs"]
      {x, g} = input(fx, "rbf_k4")
      spopt = runs["kmeans_runs"]
      mode = spopt |> Enum.frequencies() |> Enum.max_by(&elem(&1, 1)) |> elem(0)
      spopt_ari = mean(Enum.map(spopt, &ari(mode, &1)))

      mine =
        for seed <- 0..199,
            do:
              Region.spenc(x, g, n_clusters: 4, gamma: 0.5, assign_labels: :kmeans, seed: seed).labels

      mine_ari = mean(Enum.map(mine, &ari(mode, &1)))

      assert abs(mine_ari - spopt_ari) < 0.1,
             "mean ARI to spopt's mode: #{mine_ari} vs #{spopt_ari}"
    end
  end

  defp mean(xs), do: Enum.sum(xs) / length(xs)
end
