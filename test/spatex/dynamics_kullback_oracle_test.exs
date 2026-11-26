defmodule Spatex.DynamicsKullbackOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Dynamics
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9

  setup_all do
    {:ok, fx: Reference.load!("dynamics/kullback.json")}
  end

  defp norm(:nan), do: :nan
  defp norm(v), do: v

  test "giddy_compatible matches giddy.markov.kullback", %{fx: fx} do
    for c <- fx["cases"] do
      k = Dynamics.kullback(c["f"], giddy_compatible: true)
      l = c["label"]
      Reference.assert_close([k.statistic], [c["statistic"]], @tol)
      assert k.dof == c["dof"], l
      assert_p(k.p_value, c["p_value"], l)
    end
  end

  test "the default is Kullback's statistic with the row totals of each stratum", %{fx: fx} do
    for c <- fx["cases"] do
      k = Dynamics.kullback(c["f"])
      l = c["label"]
      Reference.assert_close([k.statistic], [c["statistic_intended"]], @tol)
      assert k.dof == c["dof"], l
      assert_p(k.p_value, c["p_value_intended"], l)
    end
  end

  test "both agree on giddy's docstring table, whose strata have equal margins", %{fx: fx} do
    c = Enum.find(fx["cases"], &(&1["label"] == "kullback 1962 table 9.2"))
    assert_in_delta Dynamics.kullback(c["f"]).statistic, 160.961, 5.0e-4
    assert c["statistic"] == c["statistic_intended"]
  end

  test "a spatial markov result gives the same test as giddy on Spatial_Markov.T", %{fx: fx} do
    for c <- fx["cases"], c["graph"] do
      g = fx["graphs"][c["graph"]]

      graph =
        g["focal"]
        |> Graph.from_arrays(g["neighbor"], g["weight"],
          ids: Enum.to_list(0..(length(g["ids"]) - 1))
        )
        |> Graph.transform(:r)

      opts = for {key, v} <- c["opts"], do: {String.to_existing_atom(key), v}
      sm = Dynamics.spatial_markov(c["y"], graph, [fill_empty_classes: true] ++ opts)
      k = Dynamics.kullback(sm, giddy_compatible: true)
      Reference.assert_close([k.statistic], [c["statistic"]], @tol)
      assert k.dof == c["dof"], c["label"]
    end
  end

  defp assert_p(actual, :nan, l), do: assert(norm(actual) == :nan, l)
  defp assert_p(actual, expected, l), do: assert_in_delta(actual, expected, 1.0e-9, l)
end
