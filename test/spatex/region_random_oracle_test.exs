defmodule Spatex.RegionRandomOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle
  @moduletag :statistical

  import Spatex.RegionHelper

  alias Spatex.Reference
  alias Spatex.Region

  # spopt draws numpy's global generator, so partitions are never compared: the summaries of
  # many runs are (see test/reference/generate_region_random.py and known_deviations.exs).
  @fixture "region/random_region.json"
  @runs 1500
  # two independent means differ by at most 5 combined standard errors (plus rounding slack)
  @z 5.0

  setup_all do
    fx = Reference.load!(@fixture)

    {:ok,
     fx: fx, g: lattice(fx["lattice"]["rows"], fx["lattice"]["cols"]), cols: fx["lattice"]["cols"]}
  end

  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp sd(xs) do
    m = mean(xs)
    :math.sqrt(Enum.sum(Enum.map(xs, &((&1 - m) * (&1 - m)))) / (length(xs) - 1))
  end

  defp compactness(regions, cols) do
    total =
      Enum.sum(
        for members <- regions,
            {a, i} <- Enum.with_index(members),
            b <- Enum.drop(members, i + 1) do
          abs(div(a, cols) - div(b, cols)) + abs(rem(a, cols) - rem(b, cols))
        end
      )

    total / (regions |> List.flatten() |> length())
  end

  defp scalars(r, cols) do
    sizes = Enum.map(r.regions, &length/1)

    %{
      "n_regions" => r.n_regions,
      "first_size" => hd(sizes),
      "last_size" => List.last(sizes),
      "max_size" => Enum.max(sizes),
      "min_size" => Enum.min(sizes),
      "compactness" => compactness(r.regions, cols)
    }
  end

  defp run_case(name, fx, g, cols, base_opts) do
    c = fx["cases"][name]
    spec = c["spec"]

    opts =
      [maxiter: spec["maxiter"], permutations: @runs, seed: 20_260_515] ++
        if(spec["num_regions"], do: [num_regions: spec["num_regions"]], else: []) ++
        if(spec["cardinality"], do: [cardinality: spec["cardinality"]], else: []) ++
        if(spec["compact"], do: [compact: true], else: []) ++
        if(spec["max_swaps"], do: [max_swaps: spec["max_swaps"]], else: []) ++ base_opts

    arg = if spec["contiguity"], do: g, else: spec["n"]
    result = Region.random_regions(arg, opts)

    feasible = result.feasible_solutions
    assert length(feasible) / @runs >= c["feasible_share"] - 0.05, "#{name}: feasible share"
    per_run = Enum.map(feasible, &scalars(&1, cols))

    for {stat, ref} <- c["stats"],
        stat != "compactness" or spec["contiguity"] do
      xs = Enum.map(per_run, & &1[stat])
      se = :math.sqrt(sd(xs) ** 2 / length(xs) + ref["sd"] ** 2 / ref["runs"])
      band = @z * se + 1.0e-9

      assert abs(mean(xs) - ref["mean"]) <= band,
             "#{name}.#{stat}: spatex #{mean(xs)} vs spopt #{ref["mean"]} (band #{band})"

      # the spread agrees too (loose: 20 % plus the sampling error of an sd)
      assert abs(sd(xs) - ref["sd"]) <= 0.2 * ref["sd"] + 3 * se + 1.0e-9,
             "#{name}.#{stat}: sd #{sd(xs)} vs #{ref["sd"]}"
    end

    per_run
  end

  for name <- ~w(free_n20 regions4_n20 cardinality_n20 contig_regions4 contig_cardinality
                 contig_cardinality_compact contig_free) do
    test "#{name} summaries match spopt", %{fx: fx, g: g, cols: cols} do
      run_case(unquote(name), fx, g, cols, [])
    end
  end

  test "compact growth lowers the within-region distances in both implementations", %{
    fx: fx,
    g: g,
    cols: cols
  } do
    free = run_case("contig_cardinality", fx, g, cols, [])
    compact = run_case("contig_cardinality_compact", fx, g, cols, [])

    assert mean(Enum.map(compact, & &1["compactness"])) <
             mean(Enum.map(free, & &1["compactness"]))

    assert fx["cases"]["contig_cardinality_compact"]["stats"]["compactness"]["mean"] <
             fx["cases"]["contig_cardinality"]["stats"]["compactness"]["mean"]
  end
end
