defmodule Spatex.RegionPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Spatex.RegionHelper

  alias Spatex.Region

  defp lattice_and_values do
    gen all(
          rows <- integer(2..5),
          cols <- integer(2..5),
          values <- list_of(float(min: -10.0, max: 10.0), length: rows * cols)
        ) do
      {rows, cols, values}
    end
  end

  property "every method returns contiguous partitions with the requested number of regions" do
    check all(
            {rows, cols, values} <- lattice_and_values(),
            k <- integer(1..4),
            seed <- integer(0..1000),
            max_runs: 40
          ) do
      g = lattice(rows, cols)
      n = rows * cols
      k = min(k, n)
      x = [values]

      for labels <- [
            Region.skater(x, g, n_clusters: k).labels,
            Region.ward_spatial(x, g, n_clusters: k).labels,
            Region.region_kmeans(x, g, n_clusters: k, seed: seed).labels,
            Region.azp(x, g, n_clusters: k, seed: seed).labels
          ] do
        assert length(labels) == n
        assert length(Enum.uniq(labels)) == k
        assert Region.contiguous?(labels, g)
      end
    end
  end

  property "azp never worsens its initial objective and is a strict local optimum" do
    check all(
            {rows, cols, values} <- lattice_and_values(),
            seed <- integer(0..1000),
            max_runs: 30
          ) do
      g = lattice(rows, cols)
      x = [values]
      k = min(3, rows * cols)
      r = Region.azp(x, g, n_clusters: k, seed: seed, metric: :manhattan)
      initial = Region.objective(x, r.initial_labels)
      assert r.objective <= initial + 1.0e-9
      assert_in_delta r.objective, Region.objective(x, r.labels), 1.0e-9
    end
  end

  property "tabu and annealing AZP give valid partitions that never worsen the start" do
    check all(
            {rows, cols, values} <- lattice_and_values(),
            seed <- integer(0..1000),
            method <- member_of([:tabu, :annealing]),
            max_runs: 30
          ) do
      g = lattice(rows, cols)
      x = [values]
      k = min(3, rows * cols)

      r =
        Region.azp(x, g,
          n_clusters: k,
          seed: seed,
          metric: :manhattan,
          method: method,
          init_temperature: 1.0
        )

      assert r.method == method
      assert length(Enum.uniq(r.labels)) == k
      assert Region.contiguous?(r.labels, g)
      assert r.objective <= Region.objective(x, r.initial_labels) + 1.0e-9
      assert_in_delta r.objective, Region.objective(x, r.labels), 1.0e-9
    end
  end

  property "max_p regions are contiguous and meet the threshold" do
    check all(
            {rows, cols, values} <- lattice_and_values(),
            threshold <- integer(1..4),
            seed <- integer(0..1000),
            max_runs: 30
          ) do
      g = lattice(rows, cols)
      n = rows * cols

      r =
        Region.max_p([values], g,
          threshold_variable: List.duplicate(1.0, n),
          threshold: threshold,
          seed: seed,
          max_iterations_construction: 10
        )

      assert Region.contiguous?(r.labels, g)
      assert Enum.all?(regions(r.labels), &(length(&1) >= threshold))
      assert length(Enum.uniq(r.labels)) == r.p
    end
  end

  property "relabel/1 is idempotent and preserves the partition" do
    check all(labels <- list_of(integer(0..6), min_length: 1, max_length: 30)) do
      r = Region.relabel(labels)
      assert Region.relabel(r) == r
      assert regions(r) == regions(labels)
    end
  end
end
