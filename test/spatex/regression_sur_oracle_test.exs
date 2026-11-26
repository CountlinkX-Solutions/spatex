defmodule Spatex.RegressionSurOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  defp close(a, b, tol), do: Reference.assert_close(List.flatten(a), List.flatten(b), tol)
  defp triple({a, b, c}), do: [a, b * 1.0, c]

  test "SUR matches spreg: estimates, diagnostics and the iterated estimator" do
    fx = Reference.load!("regression/sur.json")
    g = fx["graph"]

    graph =
      Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
        ids: Enum.to_list(0..(g["n"] - 1))
      )

    for c <- fx["cases"] do
      ys = Enum.map(c["y_vars"], &fx["data"][&1])
      xs = Enum.map(c["x_vars"], fn vars -> Enum.map(vars, &fx["data"][&1]) end)
      o = c["opts"]

      r =
        Regression.sur(ys, xs,
          iter: o["iter"] || false,
          spat_diag: o["spat_diag"] || false,
          graph: graph,
          name_bigy: c["y_vars"],
          name_bigx: c["x_vars"]
        )

      label = c["label"]
      assert r.name_x == c["name_bigX"], label
      assert r.iterations == c["niter"], label
      close(r.b_ols, c["bOLS"], 1.0e-9)
      close(r.b_sur, c["bSUR"], 1.0e-9)
      close(r.varb, c["varb"], 1.0e-9)
      close(r.sig, c["sig"], 1.0e-9)
      close(r.sig_ols, c["sig_ols"], 1.0e-9)
      close(r.corr, c["corr"], 1.0e-9)
      assert_in_delta r.llik, c["llik"], 1.0e-8, label

      close(
        Enum.map(r.sur_inf, fn eq -> Enum.map(eq, &Tuple.to_list/1) end),
        c["sur_inf"],
        1.0e-8
      )

      close(triple(r.lr_test), c["lrtest"], 1.0e-8)
      close(triple(r.lm_test), c["lmtest"], 1.0e-8)
      close(r.residuals, c["bigE"], 1.0e-9)

      if c["surchow"],
        do: close(Enum.map(r.chow, &triple/1), c["surchow"], 1.0e-8),
        else: assert(r.chow == nil, label)

      if c["lmEtest"] do
        close(triple(r.lm_error), c["lmEtest"], 1.0e-8)
        close(triple(r.lm_lag), c["lmlagtest"], 1.0e-8)
      end
    end
  end
end
