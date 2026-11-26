defmodule Spatex.RegionSa3Test do
  use ExUnit.Case, async: true

  import Spatex.RegionHelper

  alias Spatex.Graph
  alias Spatex.Region
  alias Spatex.Region.Sa3

  defp path(n) do
    {f, nb} = Enum.unzip(for i <- 0..(n - 2), pair <- [{i, i + 1}, {i + 1, i}], do: pair)
    Graph.from_arrays(f, nb, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(n - 1)))
  end

  @two_groups [[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]]

  test "two tight groups become two clusters with no noise" do
    r = Region.sa3(@two_groups, path(6), min_cluster_size: 3)
    assert %Sa3{n_clusters: 2, n_noise: 0, n_connected_components: 1} = r
    assert regions(r.labels) == [[0, 1, 2], [3, 4, 5]]
  end

  test "noise is -1 and defaults are spopt's (min_cluster_size 15, eom)" do
    r = Region.sa3(@two_groups, path(6))
    assert r.labels == List.duplicate(-1, 6)
    assert r.n_clusters == 0 and r.n_noise == 6
    assert r.components == []
  end

  test "leaf extraction keeps a root that cannot split as one cluster" do
    x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

    assert Region.sa3(x, path(5), min_cluster_size: 3, extraction: :eom).labels ==
             List.duplicate(-1, 5)

    assert Region.sa3(x, path(5), min_cluster_size: 3, extraction: :leaf).n_clusters == 1
  end

  test "disconnected graphs are clustered per component" do
    g = Graph.from_arrays([0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4])
    r = Region.sa3([[0.0, 0.1, 0.2, 5.0, 5.1, 5.2]], g, min_cluster_size: 2)
    assert r.n_connected_components == 2
    assert length(r.components) == 2
  end

  test "attribute tables are accepted in the usual shapes" do
    a = Region.sa3(@two_groups, path(6), min_cluster_size: 3)
    b = Region.sa3(hd(@two_groups), path(6), min_cluster_size: 3)

    c =
      Region.sa3(Nx.tensor(Enum.map(hd(@two_groups), &[&1]), type: :f64), path(6),
        min_cluster_size: 3
      )

    assert a.labels == b.labels and b.labels == c.labels
  end

  describe "validation" do
    test "min_cluster_size must be an integer >= 2" do
      for bad <- [1, 0, -3, 2.5, "3"] do
        assert_raise ArgumentError, ~r/min_cluster_size/, fn ->
          Region.sa3(@two_groups, path(6), min_cluster_size: bad)
        end
      end
    end

    test "extraction must be :eom or :leaf and unknown options raise" do
      assert_raise ArgumentError, ~r/extraction/, fn ->
        Region.sa3(@two_groups, path(6), extraction: :other)
      end

      assert_raise ArgumentError, ~r/unknown option/, fn ->
        Region.sa3(@two_groups, path(6), n_clusters: 2)
      end
    end

    test "the attribute table must match the graph" do
      assert_raise ArgumentError, fn -> Region.sa3([[1.0, 2.0]], path(6), min_cluster_size: 3) end
    end
  end
end
