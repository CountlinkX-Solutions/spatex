defmodule Spatex.ClassifyPropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  @moduletag :capture_log

  alias Spatex.Classify

  defp vector, do: list_of(float(min: -1.0e6, max: 1.0e6), min_length: 12, max_length: 80)
  defp tied_vector, do: list_of(integer(0..8), min_length: 12, max_length: 80)

  defp methods do
    [
      &Classify.equal_interval(&1, k: 4),
      &Classify.quantiles(&1, k: 5),
      &Classify.box_plot/1,
      &Classify.std_mean/1,
      &Classify.percentiles/1,
      &Classify.maximum_breaks(&1, k: 4),
      &Classify.head_tail_breaks/1
    ]
  end

  defp assert_valid(y, fun) do
    c =
      ExUnit.CaptureLog.with_log(fn -> fun.(y) end)
      |> elem(0)

    n = length(y)
    assert Enum.sum(c.counts) == n
    assert length(c.yb) == n
    assert Enum.all?(c.yb, &(&1 in 0..(length(c.bins) - 1)))
    assert c.bins == Enum.sort(c.bins)
    assert c.counts == Enum.map(0..(length(c.bins) - 1), fn i -> Enum.count(c.yb, &(&1 == i)) end)
    assert c.adcm >= 0 and c.tss >= 0
    assert c.gadf <= 1.0 + 1.0e-12
  end

  property "continuous data: counts sum to n, yb in range, bins non-decreasing" do
    check all(y <- vector(), method <- member_of(methods())) do
      if Enum.min(y) != Enum.max(y), do: assert_valid(y, method)
    end
  end

  property "tied data" do
    check all(y <- tied_vector(), method <- member_of(methods())) do
      if Enum.min(y) != Enum.max(y), do: assert_valid(y, method)
    end
  end

  # Integer data keeps the float32 Fisher-Jenks arithmetic exact, so optimality can be checked
  # against exhaustive enumeration of the contiguous partitions of the sorted values.
  defp small_ints, do: list_of(integer(0..60), min_length: 6, max_length: 12)

  defp ss(v) do
    m = Enum.sum(v) / length(v)
    Enum.sum(for a <- v, do: (a - m) * (a - m))
  end

  defp brute_min(x, 1), do: ss(x)

  defp brute_min(x, k) do
    for i <- 1..(length(x) - k + 1) do
      {l, r} = Enum.split(x, i)
      ss(l) + brute_min(r, k - 1)
    end
    |> Enum.min()
  end

  property "fisher_jenks attains the exhaustive optimum and no classifier beats it" do
    check all(y <- small_ints(), k <- integer(2..4)) do
      if length(Enum.uniq(y)) >= k, do: assert_optimal(y, k)
    end
  end

  defp assert_optimal(y, k) do
    fj = Classify.fisher_jenks(y, k: k)
    assert length(fj.bins) == k
    assert_in_delta fj.tss, brute_min(Enum.sort(y), k), 1.0e-9

    others =
      [
        Classify.quantiles(y, k: k),
        Classify.jenks_caspall(y, k: k),
        Classify.natural_breaks(y, k: k),
        Classify.maximum_breaks(y, k: k)
      ]

    for o <- others, length(o.bins) == k, do: assert(fj.tss <= o.tss + 1.0e-9)
  end

  property "optimal classifiers keep gadf in [0, 1] and classify every observation" do
    check all(y <- vector(), k <- integer(2..6)) do
      if length(Enum.uniq(y)) >= k do
        forced =
          try do
            [Classify.jenks_caspall_forced(y, k: k)]
          rescue
            # Ties can leave fewer quantile classes than k; mapclassify fails the same way.
            ArgumentError -> []
          end

        for c <- [
              Classify.fisher_jenks(y, k: k),
              Classify.jenks_caspall(y, k: k),
              Classify.natural_breaks(y, k: k) | forced
            ] do
          assert Enum.sum(c.counts) == length(y)
          assert c.gadf >= -1.0e-12 and c.gadf <= 1.0 + 1.0e-12
          assert c.bins == Enum.sort(c.bins)
        end
      end
    end
  end

  property "natural_breaks and the sampled classifiers are pure functions of the seed" do
    check all(y <- vector(), seed <- integer(0..1000)) do
      if length(Enum.uniq(y)) >= 4, do: assert_seeded(y, seed)
    end
  end

  defp assert_seeded(y, seed) do
    assert Classify.natural_breaks(y, k: 3, seed: seed) ==
             Classify.natural_breaks(y, k: 3, seed: seed)

    assert Classify.jenks_caspall_sampled(y, k: 2, pct: 0.5, seed: seed) ==
             Classify.jenks_caspall_sampled(y, k: 2, pct: 0.5, seed: seed)
  end
end
