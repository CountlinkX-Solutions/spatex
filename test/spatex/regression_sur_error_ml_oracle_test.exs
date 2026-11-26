defmodule Spatex.RegressionSurErrorMlOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  # spreg's L-BFGS-B stops up to ~1e-5 from the optimum of each search (measured: lambda within
  # 8e-6 of spreg's, 2e-5 of the epsilon = 1e-10 run); the coefficients and covariances that follow
  # differ by up to ~1e-4 (relative to the values). Both loops take the same number of searches.
  @lam_tol 5.0e-5
  @loose 5.0e-4

  defp close(a, b, tol), do: Reference.assert_close(List.flatten(a), List.flatten(b), tol)
  defp triple({a, b, c}), do: [a, b * 1.0, c]

  defp graph(fx, key) do
    g = fx["graphs"][key]
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: Enum.to_list(0..(g["n"] - 1)))
  end

  defp inputs(fx, c) do
    {Enum.map(c["y_vars"], &fx["data"][&1]),
     Enum.map(c["x_vars"], fn vars -> Enum.map(vars, &fx["data"][&1]) end)}
  end

  defp opts(c), do: [name_bigy: c["y_vars"], name_bigx: c["x_vars"], spat_diag: true, vm: true]

  defp check(r, c, ref, tol) do
    close(r.b_sur, ref["bSUR"], tol)
    close(r.varb, ref["varb"], tol)
    close(r.sig, ref["sig"], tol)
    close(r.corr, ref["corr"], tol)
    # residuals carry the coefficient error times the regressors (up to ~100 in the data)
    close(r.residuals, ref["bigE"], tol * if(tol > 1.0e-8, do: 10, else: 1))
    assert_in_delta r.llik_sur_err, ref["surerrllik"], max(tol, 1.0e-8) * 100
    close(Enum.map(r.sur_inf, fn eq -> Enum.map(eq, &Tuple.to_list/1) end), c["sur_inf"], tol)
  end

  test "every formula matches spreg at 1e-9 given spreg's lambdas" do
    fx = Reference.load!("regression/sur_error_ml.json")

    for c <- fx["cases"] do
      {ys, xs} = inputs(fx, c)
      r = Regression.sur_error_ml_at(ys, xs, graph(fx, c["weights"]), c["lam_path"], opts(c))
      assert r.name_x == c["name_bigX"], c["label"]
      assert r.k == Enum.map(c["bigK"], &trunc/1), c["label"]
      assert r.lam == c["lamsur"], c["label"]
      check(r, c, c, 1.0e-9)

      # the equation-by-equation start and the classic SUR do not depend on the search
      assert r.lam_ols == c["lamols"]
      assert_in_delta r.clik_err, c["clikerr"], 1.0e-8
      assert_in_delta r.llik_err, c["errllik"], 1.0e-8
      assert_in_delta r.llik, c["llik"], 1.0e-9
      close(r.b_sur0, c["bSUR0"], 1.0e-9)
      assert_in_delta r.clik_sur_err, c["cliksurerr"], 1.0e-8

      # tests
      close(triple(r.lr_test), c["lrtest"], 1.0e-8)
      close(triple(r.lr_lambda), c["likrlambda"], 1.0e-8)
      close(r.vm, c["vm"], 1.0e-9)
      assert length(r.vm) == c["vm_size"]
      close(Enum.map(r.lam_setp, &Tuple.to_list/1), c["lamsetp"] |> transpose(), 1.0e-9)
      close(triple(r.lam_test), c["lamtest"], 1.0e-8)
      close(triple(r.join_lam), c["joinlam"], 1.0e-8)

      if c["surchow"],
        do: close(Enum.map(r.chow, &triple/1), c["surchow"], 1.0e-8),
        else: assert(r.chow == nil, c["label"])
    end
  end

  # spreg stores (se, t, p) as three n_eq lists
  defp transpose(rows), do: Enum.zip_with(rows, & &1)

  test "the fitted model follows spreg's optimizer within its own tolerance" do
    fx = Reference.load!("regression/sur_error_ml.json")

    for c <- fx["cases"] do
      {ys, xs} = inputs(fx, c)
      r = Regression.sur_error_ml(ys, xs, graph(fx, c["weights"]), opts(c))
      # the per-equation Brent searches stop within their tolerance (1e-7), not at spreg's point
      close(r.lam_ols, c["lamols"], 2.0e-7)
      for {l, s} <- Enum.zip(r.lam, c["lamsur"]), do: assert_in_delta(l, s, @lam_tol)
      for {l, s} <- Enum.zip(r.lam, c["tight"]["lamsur"]), do: assert_in_delta(l, s, @lam_tol)
      check(r, c, c["tight"], @loose)
      assert r.iterations == c["searches"]
    end
  end
end
