defmodule Spatex.RegressionPanelREOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  test "ML_LagRE matches spreg with and without time effects" do
    fx = Reference.load!("regression/panel_re.json")
    g = fx["graph"]

    graph =
      Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
        ids: Enum.to_list(0..(fx["n"] - 1))
      )

    for c <- fx["cases"] do
      te = c["params"]["time_effects"]

      r =
        Regression.ml_lag_re(fx["y"], [fx["x1"], fx["x2"]], graph,
          time_effects: te,
          name_x: ["x1", "x2"]
        )

      label = "time_effects=#{te}"

      assert r.name_x == c["name_x"], label
      assert_in_delta r.phi, c["phi"], 1.0e-6, label
      assert_in_delta r.rho, c["rho"], 1.0e-6, label
      Reference.assert_close(r.betas, c["betas"], 1.0e-6)
      Reference.assert_close(List.flatten(r.vm), c["vm"], 1.0e-5)
      assert_in_delta r.logll, c["logll"], 1.0e-6, label
      assert_in_delta r.aic, c["aic"], 1.0e-6, label
      assert_in_delta r.sig2, c["sig2"], 1.0e-6, label
      Reference.assert_close(r.u, c["u"], 1.0e-6)
      {adi, aii, ati} = r.sp_multipliers.simple
      m = c["multipliers"]
      Reference.assert_close([adi, aii, ati], [m["adi"], m["aii"], m["ati"]], 1.0e-6)
    end
  end
end
