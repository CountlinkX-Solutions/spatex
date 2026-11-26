defmodule Spatex.ClassifyTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias Spatex.Classify

  doctest Spatex.Classify

  @y [1, 7, 4, 12, 4, 31]

  describe "equal_interval/2" do
    test "mapclassify example" do
      c = Classify.equal_interval(@y, k: 5)
      assert %Classify{method: :equal_interval, k: 5} = c
      assert c.bins == [7.0, 13.0, 19.0, 25.0, 31.0]
      assert c.yb == [0, 0, 0, 1, 0, 4]
      assert c.counts == [4, 1, 0, 0, 1]
      assert c.lowest == 1.0
    end

    test "constant input raises" do
      assert_raise ArgumentError, ~r/All values/, fn -> Classify.equal_interval([2, 2, 2]) end
    end

    test "validates options" do
      assert_raise ArgumentError, ~r/:k/, fn -> Classify.equal_interval(@y, k: 0) end
      assert_raise ArgumentError, ~r/:k/, fn -> Classify.equal_interval(@y, k: 2.5) end
      assert_raise ArgumentError, ~r/y/, fn -> Classify.equal_interval([]) end
      assert_raise ArgumentError, ~r/y/, fn -> Classify.equal_interval([1.0, :nan]) end
    end
  end

  describe "quantiles/2" do
    test "duplicates collapse with a warning" do
      log =
        capture_log(fn ->
          c = Classify.quantiles([1, 1, 1, 1, 2, 5], k: 4)
          send(self(), {:c, c})
        end)

      assert_received {:c, c}
      assert log =~ "Not enough unique values"
      assert c.k == 3
      assert length(c.bins) == 3
      assert length(c.warnings) == 1
      assert Enum.sum(c.counts) == 6
    end
  end

  describe "box_plot/2" do
    test "bins include fences; outlier ids" do
      c = Classify.box_plot([1, 2, 3, 4, 5, 6, 7, 100])
      assert c.k == 6
      assert c.info.high_outlier_ids == [7]
      assert c.info.low_outlier_ids == []
      assert c.info.hinge == 1.5
    end
  end

  describe "std_mean/2, percentiles/2, maximum_breaks/2, head_tail_breaks/1" do
    test "std_mean appends the maximum" do
      c = Classify.std_mean([1, 2, 3, 4, 50], multiples: [-1, 1])
      assert length(c.bins) == 3
      assert List.last(c.bins) == 50.0
    end

    test "percentiles defaults" do
      c = Classify.percentiles(Enum.to_list(1..100))
      assert c.bins |> length() == 6
      assert List.last(c.bins) == 100.0
    end

    test "maximum_breaks" do
      c = Classify.maximum_breaks([1, 2, 3, 10, 11, 30, 31], k: 3)
      assert c.bins == [6.5, 20.5, 31.0]
    end

    test "head_tail_breaks" do
      c = Classify.head_tail_breaks([1, 1, 1, 1, 2, 2, 10, 50])
      assert hd(c.bins) == 8.5
      assert c.k == length(c.bins)
    end
  end

  describe "user_defined/3" do
    test "appends the maximum when the last bin is smaller" do
      c = Classify.user_defined([1, 5, 20], [2, 10])
      assert c.bins == [2.0, 10.0, 20.0]
      assert c.yb == [0, 1, 2]
      assert c.k == 3
    end

    test "lowest feeds the legend only" do
      c = Classify.user_defined([1, 5, 20], [2, 10], lowest: 0)
      assert c.lowest == 0.0
      assert hd(Classify.legend(c)) == "[ 0.00,  2.00]"
    end
  end

  @clusters [1, 2, 3, 10, 11, 12, 30, 31]

  describe "fisher_jenks/2" do
    test "separates the obvious clusters" do
      c = Classify.fisher_jenks(@clusters, k: 3)
      assert %Classify{method: :fisher_jenks, k: 3} = c
      assert c.bins == [3.0, 12.0, 31.0]
      assert c.counts == [3, 3, 2]
    end

    test "needs at least k unique values" do
      assert_raise ArgumentError, ~r/Fewer unique values/, fn ->
        Classify.fisher_jenks([1, 1, 2], k: 3)
      end
    end

    test "validates k" do
      assert_raise ArgumentError, ~r/:k/, fn -> Classify.fisher_jenks(@clusters, k: 0) end
    end
  end

  describe "jenks_caspall_forced/2 with non-contiguous classes" do
    test "returns ascending bins where mapclassify returns unordered class maxima" do
      y = [
        0.0,
        0.0,
        1_000_000.0,
        1_000_000.0,
        24_291.9921875,
        250_000.0,
        -887_329.1015625,
        861_099.2431640625,
        875_061.03515625,
        321_594.23828125,
        0.0,
        -214_843.75
      ]

      c = Classify.jenks_caspall_forced(y, k: 6)
      assert c.bins == Enum.sort(c.bins)
      assert Enum.sum(c.counts) == length(y)
      assert c.gadf >= 0.0 and c.gadf <= 1.0
    end
  end

  describe "jenks_caspall/2 and jenks_caspall_forced/2" do
    test "converge to the clusters and report iterations" do
      c = Classify.jenks_caspall(@clusters, k: 3)
      assert c.bins == [3.0, 12.0, 31.0]
      assert c.info.iterations >= 1
      f = Classify.jenks_caspall_forced(@clusters, k: 3)
      assert length(f.bins) == 3
      assert is_integer(f.info.iterations)
    end

    test "forced rejects constant input" do
      assert_raise ArgumentError, ~r/All values/, fn ->
        Classify.jenks_caspall_forced([4, 4, 4], k: 2)
      end
    end
  end

  describe "natural_breaks/2" do
    test "is deterministic per seed and matches Fisher-Jenks on separated clusters" do
      a = Classify.natural_breaks(@clusters, k: 3, seed: 11)
      assert a.bins == [3.0, 12.0, 31.0]
      assert a == Classify.natural_breaks(@clusters, k: 3, seed: 11)
    end

    test "every distinct value becomes a class when there are fewer than k" do
      log =
        capture_log(fn -> send(self(), {:c, Classify.natural_breaks([1, 1, 2, 5], k: 5)}) end)

      assert_received {:c, c}
      assert c.bins == [1.0, 2.0, 5.0]
      assert c.k == 3
      assert log =~ "Not enough unique values"
    end

    test "validates :initial and :seed" do
      assert_raise ArgumentError, ~r/:initial/, fn ->
        Classify.natural_breaks(@clusters, initial: 0)
      end

      assert_raise ArgumentError, ~r/:seed/, fn ->
        Classify.natural_breaks(@clusters, seed: -1)
      end
    end
  end

  describe "sampled classifiers" do
    test "are deterministic per seed, keep k bins and classify all of y" do
      y = for i <- 1..500, do: rem(i * 37, 101) + i / 7
      a = Classify.fisher_jenks_sampled(y, k: 4, pct: 0.2, seed: 5)
      assert a == Classify.fisher_jenks_sampled(y, k: 4, pct: 0.2, seed: 5)
      assert length(a.bins) == 4
      assert Enum.sum(a.counts) == 500
      assert List.last(a.bins) == Enum.max(y) * 1.0
      j = Classify.jenks_caspall_sampled(y, k: 4, pct: 0.2, seed: 5)
      assert List.last(j.bins) == Enum.max(y) * 1.0
      assert Enum.sum(j.counts) == 500
    end

    test "validates :pct" do
      assert_raise ArgumentError, ~r/:pct/, fn ->
        Classify.fisher_jenks_sampled(@clusters, pct: 0)
      end
    end
  end

  describe "classify/3" do
    test "dispatches mapclassify scheme names" do
      assert Classify.classify(@clusters, :fisher_jenks, k: 3).bins == [3.0, 12.0, 31.0]
      assert Classify.classify(@clusters, "FisherJenks", k: 3).method == :fisher_jenks
      assert Classify.classify(@clusters, :quantiles, k: 2).method == :quantiles
      assert Classify.classify(@clusters, "equalinterval", k: 2).method == :equal_interval
      assert Classify.classify(@clusters, :user_defined, bins: [5, 20]).bins == [5.0, 20.0, 31.0]
      assert Classify.classify(@clusters, :head_tail_breaks).method == :head_tail_breaks
    end

    test "rejects unknown schemes" do
      assert_raise ArgumentError, ~r/unknown classification scheme/, fn ->
        Classify.classify(@clusters, :nope)
      end
    end
  end

  describe "find_bin/2" do
    test "clamps values outside the bins" do
      c = Classify.user_defined([1, 5, 20], [2, 10, 20])
      assert Classify.find_bin(c, [-5, 2, 3, 10, 99]) == [0, 0, 1, 1, 2]
      assert Classify.find_bin(c, 11) == 2
    end
  end

  describe "legend/2 and intervals/1" do
    test "match mapclassify get_legend_classes" do
      c = Classify.equal_interval(@y, k: 5)

      assert Classify.legend(c) ==
               [
                 "[ 1.00,  7.00]",
                 "( 7.00, 13.00]",
                 "(13.00, 19.00]",
                 "(19.00, 25.00]",
                 "(25.00, 31.00]"
               ]

      assert Classify.legend(c, decimals: 0) |> hd() == "[ 1,  7]"
      assert Classify.intervals(c) |> hd() == {1.0, 7.0}
    end

    test "lower open interval uses -inf" do
      c = Classify.box_plot([1.0, 1.0, 1.0, 1.0, 2.0, 5.0], hinge: 0.5)
      assert hd(Classify.legend(c)) == "(-inf, 0.62]"
      assert hd(Classify.intervals(c)) == {:neg_infinity, 0.625}
    end
  end

  describe "pretty_breaks and max_p" do
    test "pretty_breaks needs k >= 2 and distinct values" do
      assert_raise ArgumentError, ~r/k >= 2/, fn -> Classify.pretty_breaks([1, 2, 3], k: 1) end
      assert_raise ArgumentError, ~r/equal/, fn -> Classify.pretty_breaks([4, 4, 4]) end

      assert Classify.classify([1, 7, 4, 12, 4, 31], "Pretty_Breaks", k: 6).method ==
               :pretty_breaks
    end

    test "max_p is deterministic and validates its seeds" do
      y = Enum.map(1..40, &(&1 * 1.0 + rem(&1 * 7, 5)))
      a = Classify.max_p(y, k: 4)
      assert a == Classify.max_p(y, k: 4)
      assert length(a.bins) == 4 and Enum.sum(a.counts) == 40
      assert Classify.classify(y, :max_p, k: 4) == a

      assert_raise ArgumentError, ~r/seed1/, fn -> Classify.max_p(y, seed1: -1) end
      assert_raise ArgumentError, ~r/seed2/, fn -> Classify.max_p(y, seed2: 2 ** 32) end
      assert_raise ArgumentError, ~r/initial/, fn -> Classify.max_p(y, initial: 0) end
    end
  end
end
