defmodule Spatex.RegressionPanelFEOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  test "ML_LagFE and ML_ErrorFE match spreg with and without time effects" do
    fx = Reference.load!("regression/panel_fe.json")
    g = fx["graph"]

    graph =
      Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
        ids: Enum.to_list(0..(fx["n"] - 1))
      )

    x = [fx["x1"], fx["x2"]]

    for c <- fx["cases"] do
      te = c["params"]["time_effects"]
      label = "#{c["model"]} time_effects=#{te}"
      opts = [time_effects: te, name_y: "y", name_x: ["x1", "x2"]]

      r =
        case c["model"] do
          "lag" -> Regression.ml_lag_fe(fx["y_lag"], x, graph, opts)
          "error" -> Regression.ml_error_fe(fx["y_err"], x, graph, opts)
        end

      assert r.name_x == c["name_x"], label
      Reference.assert_close(r.betas, c["betas"], 1.0e-6)
      Reference.assert_close(List.flatten(r.vm), c["vm"], 1.0e-6)
      assert_in_delta r.logll, c["logll"], 1.0e-6, label
      assert_in_delta r.aic, c["aic"], 1.0e-6, label
      assert_in_delta r.schwarz, c["schwarz"], 1.0e-6, label
      assert_in_delta r.sig2, c["sig2"], 1.0e-6, label
      Reference.assert_close(r.u, c["u"], 1.0e-6)
      Reference.assert_close(r.fixed_effects, c["mu_i"], 1.0e-6)

      if c["multipliers"] do
        {adi, aii, ati} = r.sp_multipliers.simple

        Reference.assert_close(
          [adi, aii, ati],
          [c["multipliers"]["adi"], c["multipliers"]["aii"], c["multipliers"]["ati"]],
          1.0e-6
        )
      end
    end
  end
end
