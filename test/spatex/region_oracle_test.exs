defmodule Spatex.RegionOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  import Spatex.RegionHelper

  alias Spatex.Reference
  alias Spatex.Region

  @tol 1.0e-9

  # -- SKATER ------------------------------------------------------------------------------

  describe "skater/3 against spopt SpanningForest" do
    setup do
      %{fx: Reference.load!("region/skater.json")}
    end

    test "columbus cases reproduce the labels, MSF and score exactly", %{fx: fx} do
      g = graph(fx["columbus"]["graph"])
      x = columns(fx["columbus"]["x"])

      for c <- fx["cases"] do
        opts =
          [n_clusters: c["k"], floor: c["floor"]] ++
            if c["dissimilarity"] == "euclidean", do: [dissimilarity: :euclidean], else: []

        r = Region.skater(x, g, opts ++ [trace: c["name"] == "columbus_k5"])
        assert r.labels == c["labels"], c["name"]
        assert r.n_regions == length(Enum.uniq(c["labels"])), c["name"]

        # sklearn's euclidean_distances uses the x^2 + y^2 - 2xy expansion, whose rounding noise
        # can reorder near-equal links; the forest is only compared for the exact manhattan metric
        if c["dissimilarity"] != "euclidean" do
          assert Enum.map(r.msf_edges, &Tuple.to_list/1) == c["msf_edges"], c["name"]
        end

        Reference.assert_close(r.score, c["score"], @tol)

        assert length(r.warnings) == length(c["warnings"]), c["name"]
      end
    end

    test "the trace reproduces each pruning step", %{fx: fx} do
      g = graph(fx["columbus"]["graph"])
      x = columns(fx["columbus"]["x"])
      c = Enum.find(fx["cases"], &(&1["name"] == "columbus_k5"))
      r = Region.skater(x, g, n_clusters: 5, trace: true)
      assert length(r.trace) == length(c["trace"])

      for {mine, theirs} <- Enum.zip(r.trace, c["trace"]) do
        assert mine.labels == theirs["labels"]
        assert mine.in_node == theirs["in"]
        assert mine.out_node == theirs["out"]

        case theirs["score"] do
          :inf -> assert mine.score == :infinity
          s -> Reference.assert_close(mine.score, s, @tol)
        end
      end
    end

    test "the floor stops early with a warning like spopt", %{fx: fx} do
      g = graph(fx["columbus"]["graph"])
      x = columns(fx["columbus"]["x"])
      c = Enum.find(fx["cases"], &(&1["name"] == "columbus_k20_floor5_early_stop"))
      r = Region.skater(x, g, n_clusters: 20, floor: 5)
      assert r.labels == c["labels"]
      assert r.n_regions == 8
      assert [warning] = r.warnings
      assert warning == hd(c["warnings"])
    end

    test "islands: :increase adds every component, :ignore counts them", %{fx: fx} do
      isl = fx["islands"]
      g = graph(isl["graph"])
      x = columns(isl["x"])
      by_name = Map.new(isl["cases"], &{&1["name"], &1})

      inc = Region.skater(x, g, n_clusters: 3, islands: :increase)
      assert inc.labels == by_name["islands_increase_k3"]["labels"]
      assert inc.n_regions == 6
      assert [_] = inc.warnings

      ign = Region.skater(x, g, n_clusters: 3, islands: :ignore)
      assert ign.labels == by_name["islands_ignore_k3"]["labels"]

      assert_raise ArgumentError, ~r/Islands must be larger than the quorum/, fn ->
        Region.skater(x, g, n_clusters: 3, floor: 2)
      end
    end

    test "zero_dissimilarity: :drop reproduces spopt's removal of identical neighbors", %{fx: fx} do
      d = fx["duplicates"]
      g = graph(d["graph"])
      x = columns(d["x"])
      pysal = d["case"]["labels"]
      assert Region.skater(x, g, n_clusters: 3, zero_dissimilarity: :drop).labels == pysal
    end

    test "default keeps identical neighbors together (documented deviation)", %{fx: fx} do
      d = fx["duplicates"]
      g = graph(d["graph"])
      x = columns(d["x"])
      r = Region.skater(x, g, n_clusters: 3)
      assert r.n_regions == 3
      assert r.warnings == []
      assert Region.contiguous?(r.labels, g)
      # nodes 0 and 1 (value 1,1) and 3, 4 (value 3,3) stay together
      [l0, l1, _l2, l3, l4 | _] = r.labels
      assert l0 == l1
      assert l3 == l4
    end
  end

  # -- Ward --------------------------------------------------------------------------------

  describe "ward_spatial/3 against sklearn" do
    setup do
      %{fx: Reference.load!("region/ward.json")}
    end

    test "columbus labels, tree and distances", %{fx: fx} do
      g = graph(fx["columbus"]["graph"])
      x = columns(fx["columbus"]["x"])

      for c <- fx["cases"] do
        r = Region.ward_spatial(x, g, n_clusters: c["k"])
        assert r.labels == c["labels"], c["name"]
        assert Enum.map(r.children, &Tuple.to_list/1) == c["children"], c["name"]
        Reference.assert_close(r.distances, c["distances"], 1.0e-12)
        assert r.n_connected_components == 1
        assert r.n_leaves == 49
      end
    end

    test "a disconnected graph is completed in attribute space", %{fx: fx} do
      d = fx["disconnected"]
      g = graph(d["graph"])
      x = columns(d["x"])

      for c <- d["cases"] do
        r = Region.ward_spatial(x, g, n_clusters: c["k"])
        assert r.labels == c["labels"], c["name"]
        assert r.n_connected_components == c["n_connected_components"]
        assert Enum.map(r.children, &Tuple.to_list/1) == c["children"], c["name"]
        Reference.assert_close(r.distances, c["distances"], 1.0e-12)
        assert Enum.any?(r.warnings, &(&1 =~ "connected components"))
      end
    end

    test "large k stops the tree early and numbers regions by head", %{fx: fx} do
      b = fx["lattice250"]
      g = graph(b["graph"])
      x = columns(b["x"])
      [partial, full] = b["cases"]

      r = Region.ward_spatial(x, g, n_clusters: 120)
      assert r.labels == partial["labels"]
      assert length(r.children) == 250 - 120

      r = Region.ward_spatial(x, g, n_clusters: 8)
      assert r.labels == full["labels"]
      assert Enum.map(r.children, &Tuple.to_list/1) == full["children"]
    end
  end

  # -- Region-K-Means ----------------------------------------------------------------------

  describe "region_kmeans/3 against spopt" do
    test "given PySAL's seeds the labels, centroids and iterations are identical" do
      fx = Reference.load!("region/region_kmeans.json")
      g = graph(fx["graph"])
      x = columns(fx["x"])

      for c <- fx["cases"] do
        r = Region.region_kmeans(x, g, n_clusters: 5, seeds: c["seeds"])
        assert r.labels == c["labels"], "seed #{c["seed"]}"
        assert r.iters == c["iters"]
        assert r.seeds == c["seeds"]
        Reference.assert_close(r.centroids, c["centroids"], 1.0e-12)
      end
    end
  end

  # -- objective functions -----------------------------------------------------------------

  describe "objective/3 against spopt" do
    test "pairwise and center objectives" do
      fx = Reference.load!("region/objective.json")
      x = columns(fx["x"])

      for c <- fx["cases"], metric <- [:manhattan, :euclidean, :sqeuclidean] do
        for kind <- [:pairwise, :center] do
          expected = c["#{kind}_#{metric}"]
          actual = Region.objective(x, c["labels"], kind: kind, metric: metric)
          Reference.assert_close(actual, expected, @tol)
        end
      end
    end
  end

  # -- AZP ---------------------------------------------------------------------------------

  describe "azp/3 against brute force and spopt" do
    test "tiny lattice: best of 30 seeds is the global optimum and the median matches spopt" do
      fx = Reference.load!("region/azp.json")["lattice"]
      g = graph(fx["graph"])
      x = fx["x"]

      results = for s <- 0..29, do: Region.azp([x], g, n_clusters: fx["k"], seed: s)

      for r <- results do
        assert r.n_regions == 3
        assert Region.contiguous?(r.labels, g)
        Reference.assert_close(r.objective, pairwise_l1(x, r.labels), 1.0e-9)
        assert r.objective >= fx["brute_force_optimum"] - 1.0e-9
      end

      best = results |> Enum.map(& &1.objective) |> Enum.min()
      Reference.assert_close(best, fx["brute_force_optimum"], 1.0e-9)
      assert median(Enum.map(results, & &1.objective)) <= median(fx["pysal_objectives"]) * 1.05
    end

    test "columbus k=5: valid, never worse than the initial solution, close to spopt" do
      fx = Reference.load!("region/azp.json")["columbus"]
      g = graph(fx["graph"])
      x = columns(fx["x"])

      results =
        for s <- 0..29 do
          r = Region.azp(x, g, n_clusters: 5, seed: s)
          assert r.n_regions == 5
          assert Region.contiguous?(r.labels, g)

          assert r.objective <=
                   Region.objective(x, r.initial_labels, kind: :pairwise, metric: :euclidean) +
                     1.0e-9

          r
        end

      assert median(Enum.map(results, & &1.objective)) <= median(fx["pysal_objectives"]) * 1.05
    end
  end

  # -- Max-p -------------------------------------------------------------------------------

  describe "max_p/3 against brute force and spopt" do
    test "tiny lattice: valid, p within [spopt p, brute-force p*], objective above the optimum" do
      fx = Reference.load!("region/maxp.json")["lattice"]
      g = graph(fx["graph"])
      pop = fx["pop"]
      pysal_p = fx["pysal"] |> Enum.map(& &1["p"]) |> Enum.min()

      best_p =
        fx["brute_force_by_p"] |> Map.keys() |> Enum.map(&String.to_integer/1) |> Enum.max()

      for s <- 0..9 do
        r =
          Region.max_p([fx["x"]], g,
            threshold_variable: pop,
            threshold: fx["threshold"],
            seed: s,
            max_iterations_construction: 20
          )

        assert r.p >= pysal_p and r.p <= best_p
        assert length(Enum.uniq(r.labels)) == r.p
        assert Region.contiguous?(r.labels, g)

        for reg <- regions(r.labels) do
          assert Enum.sum(Enum.map(reg, &Enum.at(pop, &1))) >= fx["threshold"]
        end

        optimum = fx["brute_force_by_p"]["#{r.p}"]
        assert r.objective >= optimum - 1.0e-9
        Reference.assert_close(r.objective, pairwise_l1(fx["x"], r.labels), 1.0e-9)
      end
    end

    test "columbus: at least spopt's p, thresholds met" do
      fx = Reference.load!("region/maxp.json")["columbus"]
      g = graph(fx["graph"])
      x = columns(fx["x"])
      pysal_p = fx["pysal"] |> Enum.map(& &1["p"]) |> Enum.min()

      for s <- 0..4 do
        r =
          Region.max_p(x, g,
            threshold_variable: List.duplicate(1.0, 49),
            threshold: fx["threshold"],
            seed: s
          )

        assert r.p >= pysal_p
        assert Region.contiguous?(r.labels, g)
        assert regions(r.labels) |> Enum.all?(&(length(&1) >= 8))
      end
    end

    test "the construction phase with a fixed order matches spopt (top_n = 1)" do
      fx = Reference.load!("region/maxp.json")["columbus"]
      g = graph(fx["graph"])
      x = columns(fx["x"])
      ones = List.duplicate(1.0, 49)

      for c <- fx["construction"] do
        {labels, p} = Region.maxp_construct(x, g, ones, fx["threshold"], c["order"], 1, 0)
        assert p == c["p"]
        assert labels == c["labels"]
      end
    end
  end

  defp columns(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

  defp median(list) do
    sorted = Enum.sort(list)
    n = length(sorted)

    if rem(n, 2) == 1,
      do: Enum.at(sorted, div(n, 2)),
      else: (Enum.at(sorted, div(n, 2) - 1) + Enum.at(sorted, div(n, 2))) / 2
  end
end
