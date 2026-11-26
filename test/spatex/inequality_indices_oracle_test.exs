defmodule Spatex.InequalityIndicesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Inequality
  alias Spatex.Reference

  @tol 1.0e-9

  setup_all do
    {:ok, fx: Reference.load!("dynamics/inequality_indices.json")}
  end

  test "atkinson matches inequality.Atkinson", %{fx: fx} do
    for c <- fx["atkinson"] do
      r = Inequality.atkinson(c["y"], c["epsilon"])
      Reference.assert_close([r.a, r.ede], [c["a"], c["ede"]], @tol)
    end
  end

  test "schutz matches inequality.Schutz", %{fx: fx} do
    for c <- fx["schutz"] do
      r = Inequality.schutz(c["y"])

      Reference.assert_close(
        [r.distance, r.intersection_point, r.coefficient],
        [c["distance"], c["intersection_point"], c["coefficient"]],
        @tol
      )
    end
  end

  test "wolfson and lorenz_curve match inequality.wolfson", %{fx: fx} do
    for c <- fx["wolfson"] do
      Reference.assert_close(Inequality.wolfson(c["y"]), c["wolfson"], @tol)
      {pop, share} = Inequality.lorenz_curve(c["y"])
      Reference.assert_close(pop, c["population"], @tol)
      Reference.assert_close(share, c["income_share"], @tol)
    end
  end
end
