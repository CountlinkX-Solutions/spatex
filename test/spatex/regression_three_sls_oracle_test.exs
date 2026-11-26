defmodule Spatex.RegressionThreeSlsOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Reference, Regression}

  defp close(a, b, tol), do: Reference.assert_close(List.flatten(a), List.flatten(b), tol)
  defp triple({a, b, c}), do: [a, b * 1.0, c]

  test "ThreeSLS matches spreg: 2SLS and 3SLS estimates, covariance, inference and Chow tests" do
    fx = Reference.load!("regression/three_sls.json")

    for c <- fx["cases"] do
      label = c["label"]
      data = fx["data"][if(String.starts_with?(label, "columbus"), do: "columbus", else: label)]
      cols = fn vars -> Enum.map(vars, &data[&1]) end

      r =
        Regression.three_sls(
          Enum.map(c["y_vars"], &data[&1]),
          Enum.map(c["x_vars"], cols),
          Enum.map(c["yend_vars"], cols),
          Enum.map(c["q_vars"], cols),
          name_bigy: c["y_vars"],
          name_bigx: c["x_vars"],
          name_bigyend: c["yend_vars"],
          name_bigq: c["q_vars"]
        )

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
      close(r.residuals, c["bigE"], 1.0e-9)

      close(
        Enum.map(r.tsls_inf, fn eq -> Enum.map(eq, &Tuple.to_list/1) end),
        c["tsls_inf"],
        1.0e-8
      )

      if c["surchow"],
        do: close(Enum.map(r.chow, &triple/1), c["surchow"], 1.0e-8),
        else: assert(r.chow == nil, label)
    end
  end
end
