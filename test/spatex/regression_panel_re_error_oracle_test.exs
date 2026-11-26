defmodule Spatex.RegressionPanelREErrorOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  @fixture "regression/ml_error_re.json"

  defp graph_of(g, n),
    do: Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: Enum.to_list(0..(n - 1)))

  defp inputs(fx, "ncovr"), do: {fx["y"], [fx["rd"], fx["ps"]]}
  defp inputs(fx, _), do: {fx["y"], [fx["x1"], fx["x2"]]}

  defp name_x("ncovr"), do: ["RD", "PS"]
  defp name_x(_), do: ["x1", "x2"]

  defp fit(fx, key, c, opts \\ []) do
    {y, x} = inputs(fx[key], key)

    Regression.ml_error_re(
      y,
      x,
      graph_of(fx[key]["graph"], fx[key]["n"]),
      [time_effects: c["params"]["time_effects"], name_x: name_x(key)] ++ opts
    )
  end

  defp at(fx, key, c, {lam, phi_opt}) do
    {y, x} = inputs(fx[key], key)

    Regression.ml_error_re_at(
      y,
      x,
      graph_of(fx[key]["graph"], fx[key]["n"]),
      {lam, phi_opt * phi_opt},
      time_effects: c["params"]["time_effects"],
      name_x: name_x(key)
    )
  end

  # The objective is `-(logll + nt/2 ln 2pi + nt/2)`.
  defp objective(r), do: -(r.logll + r.n * r.t / 2 * :math.log(2 * :math.pi()) + r.n * r.t / 2)

  setup_all do
    {:ok, fx: Reference.load!(@fixture)}
  end

  for key <- ~w(ncovr rook4x6 columbus) do
    describe "#{key} (simple spectrum)" do
      test "every quantity matches spreg at spreg's own (lam, phi)", %{fx: fx} do
        for c <- fx[unquote(key)]["cases"] do
          r = at(fx, unquote(key), c, {c["lam"], c["phi_opt"]})
          label = c["label"]

          assert r.name_x == c["name_x"], label
          Reference.assert_close(r.betas, c["betas"], 1.0e-9)
          Reference.assert_close(List.flatten(r.vm), c["vm"], 1.0e-8)
          Reference.assert_close(r.u, c["u"], 1.0e-9)
          assert_in_delta r.lam, c["lam"], 1.0e-12, label
          assert_in_delta r.phi, c["phi"], 1.0e-12, label
          assert_in_delta r.sig2, c["sig2"], 1.0e-9 * max(1.0, c["sig2"]), label
          assert_in_delta r.sig2_u, c["sig2_u"], 1.0e-9 * max(1.0, c["sig2_u"]), label
          assert_in_delta r.logll, c["logll"], 1.0e-9 * abs(c["logll"]), label
          assert_in_delta r.aic, c["aic"], 1.0e-9 * abs(c["aic"]), label
          assert_in_delta r.schwarz, c["schwarz"], 1.0e-9 * abs(c["schwarz"]), label
        end
      end

      test "the concentrated objective matches spreg at probe points", %{fx: fx} do
        for c <- fx[unquote(key)]["cases"], p <- c["probes"] do
          r = at(fx, unquote(key), c, {p["lam"], p["phi_opt"]})
          assert_in_delta objective(r), p["objective"], 1.0e-9 * abs(p["objective"]), c["label"]
        end
      end

      test "the optimum agrees with the exact optimum of spreg's objective", %{fx: fx} do
        for c <- fx[unquote(key)]["cases"] do
          r = fit(fx, unquote(key), c)
          e = c["exact"]
          assert_in_delta r.lam, e["lam"], 1.0e-6, c["label"]
          assert_in_delta :math.sqrt(r.phi), e["phi_opt"], 1.0e-5, c["label"]
          assert objective(r) <= e["objective"] + 1.0e-8 * abs(e["objective"]), c["label"]
          # spreg stops L-BFGS-B at ftol 1e-9: its lam / phi sit within about 1e-5 of the optimum.
          assert_in_delta r.lam, c["lam"], 1.0e-5, c["label"]
          assert_in_delta r.phi, c["phi"], 1.0e-4 * max(1.0, c["phi"]), c["label"]
        end
      end
    end
  end

  describe "rook5x5 (symmetric with repeated eigenvalues)" do
    test "everything but the likelihood matches spreg at spreg's point", %{fx: fx} do
      [c] = fx["rook5x5"]["cases"]
      r = at(fx, "rook5x5", c, {c["lam"], c["phi_opt"]})
      Reference.assert_close(r.betas, c["betas"], 1.0e-9)
      Reference.assert_close(List.flatten(r.vm), c["vm"], 1.0e-8)
      Reference.assert_close(r.u, c["u"], 1.0e-9)
      assert_in_delta r.sig2, c["sig2"], 1.0e-9 * max(1.0, c["sig2"])
      # spreg's la.eig basis is not orthogonal inside a degenerate eigenspace, so its objective
      # is not the likelihood; Spatex uses the orthonormal symmetric eigenvectors.
      refute_in_delta r.logll, c["logll"], 1.0e-3
    end

    test "the optimum is the exact optimum of the symmetric (eigh) objective", %{fx: fx} do
      [c] = fx["rook5x5"]["cases"]
      r = fit(fx, "rook5x5", c)
      e = c["exact"]
      assert e["objective_basis"] == "eigh"
      assert_in_delta r.lam, e["lam"], 1.0e-5
      assert_in_delta :math.sqrt(r.phi), e["phi_opt"], 1.0e-4
      assert_in_delta objective(r), e["objective"], 1.0e-7 * max(1.0, abs(e["objective"]))
      refute_in_delta r.lam, c["lam"], 1.0e-4
    end
  end
end
