defmodule Spatex.APIConventionsTest do
  use ExUnit.Case, async: true

  alias Spatex.{Classify, Graph, Region}

  @attrs [[1.0, 1.1, 9.0, 9.2]]
  defp path, do: Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])

  test "Region entry points take opts \\\\ [] and reject unknown or non-boolean options" do
    assert_raise ArgumentError, ~r/n_clusters/, fn -> Region.ward_spatial(@attrs, path()) end

    assert_raise ArgumentError, ~r/unknown option\(s\) \[:n_cluster\]/, fn ->
      Region.skater(@attrs, path(), n_cluster: 2)
    end

    assert_raise ArgumentError, ~r/:symmetrize must be a boolean/, fn ->
      Region.azp(@attrs, path(), n_clusters: 2, symmetrize: "yes")
    end

    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Region.objective(@attrs, [0, 0, 1, 1], kinds: :center)
    end
  end

  test "Classify functions reject unknown or non-boolean options" do
    assert_raise ArgumentError, ~r/unknown option\(s\) \[:classes\]/, fn ->
      Classify.quantiles([1, 2, 3, 4, 5], classes: 3)
    end

    assert_raise ArgumentError, ~r/:anchor must be a boolean/, fn ->
      Classify.std_mean([1, 2, 3, 4, 5], anchor: 1)
    end

    assert %Classify{} = Classify.classify([1, 2, 3, 10, 11, 12], :fisher_jenks, k: 2)
  end

  test "GWR.fit/4 defaults its options and names the missing bandwidth" do
    assert_raise ArgumentError, ~r/bw is required/, fn ->
      Spatex.GWR.fit([{0, 0}, {1, 1}, {2, 0}], [1, 2, 3], [[1, 2, 2]])
    end
  end
end
