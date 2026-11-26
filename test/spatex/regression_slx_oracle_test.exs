defmodule Spatex.RegressionSlxOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  setup_all do
    fx = Reference.load!("regression/slx.json")
    g = fx["graph"]

    graph =
      Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
        ids: Enum.to_list(0..(g["n"] - 1))
      )

    {:ok, fx: fx, graph: graph}
  end

  defp cases(fx, model), do: Enum.filter(fx["cases"], &(&1["model"] == model))

  defp opts(c) do
    p = c["params"]
    slx_vars = if p["slx_vars"], do: p["slx_vars"], else: :all

    [slx_lags: p["slx_lags"], slx_vars: slx_vars, name_y: "CRIME", name_x: ["INC", "HOVAL"]] ++
      if(p["w_lags"], do: [w_lags: p["w_lags"]], else: [])
  end

  defp check(r, c, tol) do
    label = "#{c["model"]} #{inspect(c["params"])}"
    k = length(r.betas)
    # spreg's GM_Lag leaves W_y out of name_x.
    m = min(k, length(c["name_x"]))
    assert Enum.take(r.name_x, m) == Enum.take(c["name_x"], m), label
    Reference.assert_close(r.betas, Enum.take(c["betas"], k), tol)
    Reference.assert_close(List.flatten(r.vm), Enum.take(c["vm"], k * k), tol * 10)
  end

  test "OLS with SLX terms", %{fx: fx, graph: graph} do
    for c <- cases(fx, "ols") do
      r = Regression.ols(fx["y"], fx["x"], [graph: graph] ++ opts(c))
      check(r, c, 1.0e-9)
      assert r.lm_wx == nil
    end
  end

  test "ML spatial Durbin model and its impacts", %{fx: fx, graph: graph} do
    for c <- cases(fx, "ml_lag") do
      r = Regression.ml_lag(fx["y"], fx["x"], graph, opts(c))
      check(r, c, 1.0e-6)
      assert_in_delta r.rho, c["rho"], 1.0e-6
      effects = r.impacts[:simple]
      assert Enum.map(effects, & &1.name) == c["impacts"]["names"]
      Reference.assert_close(Enum.map(effects, & &1.total), c["impacts"]["total"], 1.0e-6)
      Reference.assert_close(Enum.map(effects, & &1.direct), c["impacts"]["direct"], 1.0e-6)
      Reference.assert_close(Enum.map(effects, & &1.indirect), c["impacts"]["indirect"], 1.0e-6)
    end
  end

  test "ML and GMM SLX-error models", %{fx: fx, graph: graph} do
    [c] = cases(fx, "ml_error")
    r = Regression.ml_error(fx["y"], fx["x"], graph, opts(c))
    check(r, c, 1.0e-6)
    assert_in_delta r.lam, c["lam"], 1.0e-6

    for est <- [:kp98, :het, :hom] do
      [c] = cases(fx, "gm_error_#{est}")
      r = Regression.gm_error(fx["y"], fx["x"], graph, [estimator: est] ++ opts(c))
      check(r, c, 5.0e-5)
      # spreg's GM_Error keeps lambda only as the last beta.
      assert_in_delta r.lam, c["lam"] || List.last(c["betas"]), 5.0e-5
    end
  end

  test "GM spatial Durbin model instruments with the higher-order lags", %{fx: fx, graph: graph} do
    for c <- cases(fx, "gm_lag") do
      r = Regression.gm_lag(fx["y"], fx["x"], graph, [spat_impacts: nil] ++ opts(c))
      check(r, c, 1.0e-9)
      assert_in_delta r.rho, List.last(c["betas"]), 1.0e-9
    end
  end
end
