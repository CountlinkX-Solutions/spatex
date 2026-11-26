defmodule Spatex.ClassifyGadfTest do
  use ExUnit.Case, async: true

  @moduletag :capture_log

  alias Spatex.{Classify, Reference}

  @methods %{
    "Quantiles" => :quantiles,
    "FisherJenks" => :fisher_jenks,
    "MaximumBreaks" => :maximum_breaks
  }

  describe "gadf/2 against mapclassify.gadf" do
    @tag :oracle
    test "k, gadf and bins match for every case" do
      fx = Reference.load!("classify/gadf.json")

      for c <- fx["cases"] do
        y = fx["vectors"][c["vector"]]
        method = Map.fetch!(@methods, c["method"])
        label = "#{c["vector"]} #{c["method"]} pct=#{c["pct"]}"

        {k, cl, gadf} = Classify.gadf(y, method: method, maxk: c["maxk"], pct: c["pct"])

        assert k == c["k"], label
        assert_in_delta gadf, c["gadf"], 1.0e-9, label
        assert cl.gadf == gadf, label
        Reference.assert_close(cl.bins, c["bins"], 1.0e-12)
      end
    end
  end

  test "gadf/2 accepts scheme names and validates options" do
    y = [1, 2, 3, 10, 11, 12, 30, 31, 32, 60]
    assert {k, %Classify{method: :fisher_jenks}, g} = Classify.gadf(y, method: "FisherJenks")
    assert k >= 2 and g > 0.8

    assert_raise ArgumentError, ~r/:maxk/, fn -> Classify.gadf(y, maxk: 1) end
    assert_raise ArgumentError, ~r/:pct/, fn -> Classify.gadf(y, pct: 1.5) end
  end

  test "k_classifiers/2 keeps the smallest k, ties broken by the largest gadf" do
    y = [1, 2, 3, 10, 11, 12, 30, 31, 32, 60, 61, 62]
    r = Classify.k_classifiers(y)

    assert Map.keys(r.results) |> Enum.sort() ==
             Enum.sort([:quantiles, :fisher_jenks, :natural_breaks, :maximum_breaks])

    {method, k, cl, gadf} = r.best
    assert {^k, ^cl, ^gadf} = r.results[method]

    for {_m, {k1, _cl, g1}} <- r.results do
      assert k <= k1
      if k1 == k, do: assert(gadf >= g1)
    end
  end
end
