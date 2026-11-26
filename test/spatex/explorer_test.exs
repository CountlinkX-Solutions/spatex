defmodule Spatex.ExplorerTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Spatex.ESDA.{GLocal, MoranLocal}

  defp path_graph, do: Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])

  defp moran_local(opts),
    do: Spatex.ESDA.moran_local([1, 5, 2, 8], path_graph(), opts)

  test "MoranLocal becomes a frame with one row per site" do
    m = moran_local(permutations: 99, seed: 7)
    df = Spatex.Explorer.to_dataframe(m)

    assert %DataFrame{} = df
    assert DataFrame.n_rows(df) == 4
    assert ~w(id is q z lag p_sim cluster) -- DataFrame.names(df) == []
    assert Series.to_list(df["id"]) == [0, 1, 2, 3]
    assert Series.to_list(df["q"]) == m.q
    assert Series.to_list(df["is"]) == m.is
    assert Series.to_list(df["cluster"]) == MoranLocal.cluster_labels(m)
    assert Series.to_list(df["p_sim"]) == m.p_sim
  end

  test "cluster labels honor :crit and permutations 0 drops p_sim columns" do
    m = moran_local(permutations: 99, seed: 7)
    df = Spatex.Explorer.to_dataframe(m, crit: 1.0)
    assert Series.to_list(df["cluster"]) == MoranLocal.cluster_labels(m, 1.0)

    df0 = Spatex.Explorer.to_dataframe(moran_local(permutations: 0))
    refute "p_sim" in DataFrame.names(df0)
    refute "cluster" in DataFrame.names(df0)
    refute "sim" in DataFrame.names(df0)
  end

  test "non-finite values and nil moments are handled" do
    m = %MoranLocal{
      n: 2,
      permutations: 0,
      is: [1.5, :nan],
      q: [1, 3],
      z: [0.5, :infinity],
      ids: [:a, :b]
    }

    df = Spatex.Explorer.to_dataframe(m)
    assert Series.to_list(df["is"]) == [1.5, :nan]
    assert Series.to_list(df["z"]) == [0.5, :infinity]
    assert Series.to_list(df["id"]) == ["a", "b"]
    refute "z_y" in DataFrame.names(df)
  end

  test "GLocal becomes a frame" do
    g = Spatex.ESDA.g_local([1, 5, 2, 8], path_graph(), permutations: 99, seed: 3)
    df = Spatex.Explorer.to_dataframe(g)
    assert DataFrame.n_rows(df) == 4
    assert Series.to_list(df["gs"]) == g.gs
    assert Series.to_list(df["p_sim"]) == g.p_sim
    assert %GLocal{} = g
  end

  test "Classify becomes a class column" do
    c = Spatex.Classify.quantiles([1, 2, 3, 4, 5, 6, 7, 8], k: 4)
    df = Spatex.Explorer.to_dataframe(c)
    assert DataFrame.names(df) == ["yb"]
    assert Series.to_list(df["yb"]) == c.yb
  end

  describe "region results" do
    @attrs [[1.0, 1.1, 9.0, 9.2]]

    test "every regionalization result becomes a labels column" do
      g = path_graph()

      results = [
        Spatex.Region.skater(@attrs, g, n_clusters: 2),
        Spatex.Region.ward_spatial(@attrs, g, n_clusters: 2),
        Spatex.Region.region_kmeans(@attrs, g, n_clusters: 2, seed: 1),
        Spatex.Region.azp(@attrs, g, n_clusters: 2, seed: 1)
      ]

      for r <- results do
        df = Spatex.Explorer.to_dataframe(r)
        assert DataFrame.names(df) == ["labels"], inspect(r.__struct__)
        assert Series.to_list(df["labels"]) == r.labels
      end
    end

    test "max_p labels keep nil for dropped areas" do
      r = %Spatex.Region.MaxP{labels: [0, nil, 1, 1], p: 2}
      df = Spatex.Explorer.to_dataframe(r)
      assert Series.to_list(df["labels"]) == [0, nil, 1, 1]
    end
  end

  test "unsupported inputs raise ArgumentError" do
    assert_raise ArgumentError, ~r/unsupported result/, fn ->
      Spatex.Explorer.to_dataframe(%{a: 1})
    end

    assert_raise ArgumentError, ~r/crit/, fn ->
      Spatex.Explorer.to_dataframe(moran_local(permutations: 9, seed: 1), crit: "x")
    end
  end
end
