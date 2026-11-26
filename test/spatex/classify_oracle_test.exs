defmodule Spatex.ClassifyOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle
  @moduletag :capture_log

  import ExUnit.CaptureLog

  alias Spatex.{Classify, Reference}

  @files ~w(columbus_crime columbus_hoval columbus_inc lognormal500 heavy_ties
            edge_small edge_dups edge_constant edge_two quantiles_drift well_separated
            max_p_crime max_p_lognormal)

  @by_k ~w(equal_interval quantiles maximum_breaks fisher_jenks jenks_caspall
           jenks_caspall_forced natural_breaks pretty_breaks)

  defp run(y, %{"method" => m, "params" => p}) do
    case m do
      "natural_breaks" -> Classify.natural_breaks(y, k: p["k"], seed: 0)
      m when m in @by_k -> apply(Classify, String.to_atom(m), [y, [k: p["k"]]])
      "box_plot" -> Classify.box_plot(y, hinge: p["hinge"])
      "std_mean" -> Classify.std_mean(y, multiples: p["multiples"], anchor: p["anchor"])
      "percentiles" -> Classify.percentiles(y, pct: p["pct"])
      "head_tail_breaks" -> Classify.head_tail_breaks(y)
      "user_defined" -> Classify.user_defined(y, p["bins"], lowest: p["lowest"])
      "max_p" -> Classify.max_p(y, k: p["k"], initial: p["initial"])
    end
  end

  for name <- @files do
    describe "fixture #{name}" do
      setup do
        {:ok, fx: Reference.load!("classify/#{unquote(name)}.json")}
      end

      test "records the mapclassify version", %{fx: fx} do
        assert fx["meta"]["mapclassify"] == "2.11.0"
      end

      test "every case matches mapclassify", %{fx: fx} do
        for case <- fx["cases"] do
          label = "#{case["method"]} #{inspect(case["params"])}"

          if case["error"] do
            assert_raise ArgumentError, fn ->
              capture_log(fn -> run(fx["y"], case) end)
              run(fx["y"], case)
            end
          else
            c = run(fx["y"], case)

            if case["method"] == "natural_breaks",
              do: assert_natural_breaks(c, case, label),
              else: assert_case(c, case, fx, label)
          end
        end
      end
    end
  end

  describe "natural breaks on separated clusters" do
    test "k = 3 on three clusters equals mapclassify" do
      fx = Reference.load!("classify/well_separated.json")

      for case <- fx["cases"], case["method"] == "natural_breaks", case["params"]["k"] == 3 do
        c = run(fx["y"], case)
        assert c.bins == case["bins"], "k=#{case["params"]["k"]}"
        assert c.yb == case["yb"]
      end
    end
  end

  describe "fixture fisher_jenks_stress" do
    test "float32 dynamic program matches mapclassify on every sample" do
      fx = Reference.load!("classify/fisher_jenks_stress.json")
      assert length(fx["samples"]) == 40

      for %{"y" => y, "k" => k, "bins" => bins} <- fx["samples"] do
        assert Classify.fisher_jenks(y, k: k).bins == bins
      end
    end
  end

  # NaturalBreaks: mapclassify's k-means starts from an unseeded random generator, so compare
  # statistically (see known_deviations.exs). Spatex must reach the same or a lower within-class
  # variance (optimal partitions need not be unique, so bins are compared only on
  # well separated clusters below).
  defp assert_natural_breaks(c, case, label) do
    assert c.tss <= case["tss"] * (1 + 1.0e-6) + 1.0e-9, "#{label}: tss #{c.tss} > #{case["tss"]}"
    assert c.k == length(case["bins"]), "#{label}: k"
    assert Enum.sum(c.counts) == length(c.yb)
  end

  defp assert_case(c, case, fx, label) do
    quantiles? = case["method"] == "quantiles"
    # Known deviation: Spatex reports the real number of quantile bins as k.
    expected_k = if quantiles?, do: length(case["bins"]), else: case["k"]
    assert c.k == expected_k, "#{label}: k"
    drift? = case["drift"] == true

    if drift? do
      # Known deviation: mapclassify leaves max(y) in class 0 (see known_deviations.exs).
      assert case["max_class"] == 0, "#{label}: PySAL behavior documented"
      assert List.last(c.bins) == Enum.max(fx["y"])

      assert Enum.at(c.yb, Enum.find_index(fx["y"], &(&1 == Enum.max(fx["y"])))) ==
               length(c.bins) - 1

      Reference.assert_close(Enum.drop(c.bins, -1), Enum.drop(case["bins"], -1), 1.0e-12)
    else
      Reference.assert_close(c.bins, case["bins"], 1.0e-12)
      assert c.yb == case["yb"], "#{label}: yb"
      assert c.counts == case["counts"], "#{label}: counts"
    end

    unless drift? do
      Reference.assert_close(c.tss, case["tss"], 1.0e-9)
      Reference.assert_close(c.adcm, case["adcm"], 1.0e-9)
      Reference.assert_close(c.gadf, case["gadf"], 1.0e-9)
      assert Classify.legend(c) == case["legend"], "#{label}: legend"

      assert Classify.find_bin(c, case["find_bin"]["x"]) == case["find_bin"]["bin"],
             "#{label}: find_bin"
    end

    # The Jenks-Caspall family emits a varying number of internal numpy/quantile warnings.
    unless case["method"] in ["jenks_caspall", "jenks_caspall_forced"],
      do: assert(length(c.warnings) == case["warnings"], "#{label}: warnings")

    if case["method"] in ["jenks_caspall", "jenks_caspall_forced"] do
      assert c.info.iterations == case["iterations"], "#{label}: iterations"
    end

    if case["method"] == "box_plot" do
      assert c.info.low_outlier_ids == case["low_outlier_ids"]
      assert c.info.high_outlier_ids == case["high_outlier_ids"]
      Reference.assert_close(c.info.iqr, case["iqr"], 1.0e-12)
    end
  end
end
