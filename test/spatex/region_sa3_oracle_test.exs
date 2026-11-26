defmodule Spatex.RegionSa3OracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  import Spatex.RegionHelper

  alias Spatex.Reference
  alias Spatex.Region
  alias Spatex.Region.Sa3

  setup_all do
    %{fx: Reference.load!("region/sa3.json")}
  end

  defp columns(rows), do: rows |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

  test "labels equal spopt's SA3 for every dataset, min_cluster_size and extraction", %{fx: fx} do
    for ds <- fx["datasets"], c <- ds["cases"] do
      g = graph(ds["graph"])

      r =
        Region.sa3(columns(ds["x"]), g,
          min_cluster_size: c["min_cluster_size"],
          extraction: String.to_atom(c["extraction"])
        )

      assert %Sa3{} = r
      assert r.labels == c["labels"], "#{ds["name"]} #{c["min_cluster_size"]} #{c["extraction"]}"
      assert r.n_clusters == c["labels"] |> Enum.reject(&(&1 == -1)) |> Enum.uniq() |> length()
      assert r.n_noise == Enum.count(c["labels"], &(&1 == -1))
    end
  end

  test "the linkage of every clustered component matches scikit-learn to 1.0e-12", %{fx: fx} do
    for ds <- fx["datasets"], c <- ds["cases"] do
      g = graph(ds["graph"])

      r =
        Region.sa3(columns(ds["x"]), g,
          min_cluster_size: c["min_cluster_size"],
          extraction: String.to_atom(c["extraction"])
        )

      assert length(r.components) == length(c["linkage"])
      name = "#{ds["name"]} #{c["min_cluster_size"]}"

      for {mine, theirs} <- Enum.zip(r.components, c["linkage"]) do
        assert Enum.map(mine.linkage, fn {a, b, _d, _s} -> [a, b] end) == theirs["children"],
               name

        assert Enum.map(mine.linkage, fn {_a, _b, _d, s} -> s end) == theirs["sizes"], name
        Reference.assert_close(Enum.map(mine.linkage, &elem(&1, 2)), theirs["distances"], 1.0e-12)
        assert Enum.sum(Enum.map(mine.linkage, &elem(&1, 2))) > 0
      end
    end
  end

  test "the fixtures exercise the monotonicity correction", %{fx: fx} do
    corrected =
      for ds <- fx["datasets"], c <- ds["cases"], l <- c["linkage"], l["corrected"], do: l

    assert corrected != []
    assert Enum.any?(corrected, &(&1["raw_distances"] != &1["distances"]))
  end
end
