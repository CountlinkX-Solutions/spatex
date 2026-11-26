defmodule Spatex.RegressionRegimesOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Reference, Regression}

  defp opts(o) do
    [name_y: "CRIME", name_x: ["INC", "HOVAL"]] ++
      Enum.flat_map(o, fn
        {"regime_err_sep", v} -> [regime_err_sep: v]
        {"constant_regi", v} -> [constant_regi: String.to_existing_atom(v)]
        {"cols2regi", v} -> [cols2regi: v]
        {"robust", "white"} -> [robust: :white]
      end)
  end

  test "betas, covariance and Chow tests match spreg for every mode" do
    fx = Reference.load!("regression/ols_regimes.json")

    for c <- fx["cases"] do
      regimes =
        if String.starts_with?(c["label"], "four"),
          do: fx["regimes"]["four"],
          else: fx["regimes"]["nsa"]

      r = Regression.ols_regimes(fx["y"], fx["x"], regimes, opts(c["opts"]))
      label = c["label"]

      assert r.name_x == c["name_x"], label
      assert {r.kr, r.kf, r.nr} == {c["kr"], c["kf"], c["nr"]}, label
      Reference.assert_close(r.betas, c["betas"], 1.0e-9)
      Reference.assert_close(List.flatten(r.vm), c["vm"], 1.0e-9)
      Reference.assert_close(Tuple.to_list(r.chow.joint), c["chow_joint"], 1.0e-8)

      Reference.assert_close(
        Enum.flat_map(r.chow.regi, &[elem(&1, 1), elem(&1, 2)]),
        List.flatten(c["chow_regi"]),
        1.0e-8
      )

      Reference.assert_close(r.u, c["u"], 1.0e-9)

      if c["multi"] do
        for {label_r, m} <- c["multi"] do
          fit = Enum.find_value(r.multi, fn {k, v} -> to_string(k) == label_r && v end)
          assert fit.name_x == m["name_x"], label
          Reference.assert_close(fit.betas, m["betas"], 1.0e-9)
          assert_in_delta fit.sig2, m["sig2"], 1.0e-9
          assert_in_delta fit.r2, m["r2"], 1.0e-12
        end
      end
    end
  end
end
