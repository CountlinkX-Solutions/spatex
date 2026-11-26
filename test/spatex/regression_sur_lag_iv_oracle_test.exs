defmodule Spatex.RegressionSurLagIvOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  defp close(a, b, tol), do: Reference.assert_close(List.flatten(a), List.flatten(b), tol)
  defp triple({a, b, c}), do: [a, b * 1.0, c]

  defp graph(fx, key) do
    g = fx["graphs"][key]
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: Enum.to_list(0..(g["n"] - 1)))
  end

  defp cols(fx, vars), do: Enum.map(vars, &fx["data"][&1])

  defp run(fx, c) do
    ys = Enum.map(c["y_vars"], &fx["data"][&1])
    xs = Enum.map(c["x_vars"], &cols(fx, &1))

    # spreg names user variables end_<i>_<eq> and inst_<i>_<eq> (the last endogenous one is W y)
    user =
      if c["yend_vars"] do
        [
          yends: Enum.map(c["yend_vars"], &cols(fx, &1)),
          qs: Enum.map(c["q_vars"], &cols(fx, &1)),
          name_bigyend: Enum.map(c["name_bigyend"], &Enum.drop(&1, -1)),
          name_bigq:
            Enum.zip_with(c["q_vars"], c["name_bigq"], fn q, names ->
              Enum.take(names, length(q))
            end)
        ]
      else
        []
      end

    Regression.sur_lag_iv(
      ys,
      xs,
      graph(fx, c["weights"]),
      [
        w_lags: c["w_lags"],
        lag_q: c["lag_q"],
        spat_diag: true,
        name_bigy: c["y_vars"],
        name_bigx: c["x_vars"]
      ] ++ user
    )
  end

  test "SURlagIV matches spreg: names, 2SLS and 3SLS estimates, covariance, inference, tests" do
    fx = Reference.load!("regression/sur_lag_iv.json")

    for c <- fx["cases"] do
      label = c["label"]
      r = run(fx, c)

      assert r.name_y == c["name_bigy"], label
      assert r.name_x == c["name_bigX"], label
      assert r.name_yend == c["name_bigyend"], label
      assert r.name_q == c["name_bigq"], label
      assert r.k == Enum.map(c["bigK"], &trunc/1), label
      close(r.b_2sls, c["b2SLS"], 1.0e-9)
      close(r.residuals_2sls, c["tslsE"], 1.0e-9)
      close(r.b_3sls, c["b3SLS"], 1.0e-9)
      close(r.varb, c["varb"], 1.0e-9)
      close(r.sig, c["sig"], 1.0e-9)
      close(r.corr, c["corr"], 1.0e-9)
      # y - Z b multiplies the coefficient error (~1e-10) by W y (~40 on this data)
      close(r.residuals, c["bigE"], 1.0e-8)
      close(r.rho, Enum.map(c["b3SLS"], &List.last/1), 1.0e-9)

      close(
        Enum.map(r.tsls_inf, fn eq -> Enum.map(eq, &Tuple.to_list/1) end),
        c["tsls_inf"],
        1.0e-8
      )

      if c["surchow"],
        do: close(Enum.map(r.chow, &triple/1), c["surchow"], 1.0e-8),
        else: assert(r.chow == nil, label)

      close(triple(r.joinrho), c["joinrho"], 1.0e-8)
    end
  end
end
