defmodule Spatex.RegressionGMKKPOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  @fixture "regression/gm_kkp.json"

  defp graph_of(g, n),
    do: Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: Enum.to_list(0..(n - 1)))

  defp inputs(fx, "ncovr"), do: {fx["y"], [fx["rd"], fx["ps"]]}
  defp inputs(fx, _), do: {fx["y"], [fx["x1"], fx["x2"]]}

  defp name_x("ncovr"), do: ["RD", "PS"]
  defp name_x(_), do: ["x1", "x2"]

  defp opts(key, c),
    do: [
      time_effects: c["params"]["time_effects"],
      full_weights: c["params"]["full_weights"],
      name_x: name_x(key)
    ]

  defp fit(fx, key, c) do
    {y, x} = inputs(fx[key], key)
    Regression.gm_kkp(y, x, graph_of(fx[key]["graph"], fx[key]["n"]), opts(key, c))
  end

  defp at(fx, key, c, params) do
    {y, x} = inputs(fx[key], key)
    Regression.gm_kkp_at(y, x, graph_of(fx[key]["graph"], fx[key]["n"]), params, opts(key, c))
  end

  # spreg's two optimizer outcomes: the first-step (lambda, sigma_v), then (lambda, sigma_v, sigma_1).
  defp spreg_point(c) do
    [[l1, sv1], [l2, sv2, s12]] = Enum.map(c["calls"], & &1["spreg"])
    {l1, sv1, l2, sv2, s12}
  end

  defp assert_result(r, s, tol, label) do
    Reference.assert_close(r.betas, s["betas"], tol)
    Reference.assert_close(List.flatten(r.vm), s["vm"], tol)
    Reference.assert_close(r.std_err, s["std_err"], tol)
    assert_in_delta r.lam, s["lam"], tol * max(1.0, abs(s["lam"])), label
    assert_in_delta r.sig2_v, s["sig2_v"], tol * max(1.0, s["sig2_v"]), label
    assert_in_delta r.sig2_1, s["sig2_1"], tol * max(1.0, s["sig2_1"]), label
  end

  setup_all do
    {:ok, fx: Reference.load!(@fixture)}
  end

  for key <- ~w(ncovr columbus rook5x5) do
    describe "#{key}" do
      test "every quantity matches spreg at spreg's own optimizer results", %{fx: fx} do
        for c <- fx[unquote(key)]["cases"] do
          r = at(fx, unquote(key), c, spreg_point(c))
          s = c["spreg"]
          label = c["label"]

          assert r.name_x == c["name_x"] ++ [], label
          assert_result(r, s, 1.0e-9, label)
          Reference.assert_close(r.u, s["u"], 1.0e-9)
          Reference.assert_close(r.e_filtered, s["e_filtered"], 1.0e-9)

          Reference.assert_close(
            Enum.map(r.z_stat, &elem(&1, 0)),
            Enum.map(s["z_stat"], &hd/1),
            1.0e-8
          )

          Reference.assert_close(
            Enum.map(r.z_stat, &elem(&1, 1)),
            Enum.map(s["z_stat"], &List.last/1),
            1.0e-8
          )

          assert_in_delta r.pr2, s["pr2"], 1.0e-9, label
          assert_in_delta r.theta, 1 - :math.sqrt(s["sig2_v"]) / :math.sqrt(s["sig2_1"]), 1.0e-12
        end
      end

      test "the fit equals spreg run with its optimizer results polished to the exact minimum",
           %{fx: fx} do
        for c <- fx[unquote(key)]["cases"] do
          r = fit(fx, unquote(key), c)
          assert_result(r, c["exact"], 1.0e-6, c["label"])
        end
      end

      test "the fit stays within spreg's L-BFGS-B error of spreg's own result", %{fx: fx} do
        for c <- fx[unquote(key)]["cases"] do
          r = fit(fx, unquote(key), c)
          s = c["spreg"]
          # spreg's optimizer stops early: lambda is within 1e-5, the variances within 5e-4
          # relative (and the coefficients within 1e-3) on these cases.
          assert_in_delta r.lam, s["lam"], 1.0e-5, c["label"]
          assert_in_delta r.sig2_v, s["sig2_v"], 5.0e-4 * s["sig2_v"], c["label"]
          assert_in_delta r.sig2_1, s["sig2_1"], 5.0e-4 * s["sig2_1"], c["label"]
          Reference.assert_close(r.betas, s["betas"], 2.0e-3)
        end
      end
    end
  end
end
