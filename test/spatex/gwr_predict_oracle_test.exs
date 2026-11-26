defmodule Spatex.GWRPredictOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.Prediction
  alias Spatex.Reference

  @tol 1.0e-8
  @fx Reference.load!("regression/gwr_predict.json")

  describe "predict (mgwr 2.2.1 GWR.predict)" do
    for {key, _} <- @fx["cases"] do
      @key key
      test "#{key}" do
        c = @fx["cases"][@key]
        ds = @fx["datasets"][c["dataset"]]
        o = c["options"]

        pick = fn list, idx -> Enum.map(idx, &Enum.at(list, &1)) end
        coords = pick.(ds["coords"], c["train"])
        y = pick.(ds["y"], c["train"])
        x = Enum.map(ds["x"], &pick.(&1, c["train"]))
        new_coords = pick.(ds["coords"], c["test"])
        x_new = Enum.map(ds["x"], &pick.(&1, c["test"]))

        opts =
          [
            bw: if(o["bw"] == :inf, do: :infinity, else: o["bw"]),
            kernel: String.to_atom(o["kernel"] || "bisquare"),
            fixed: o["fixed"] || false,
            spherical: o["spherical"] || false,
            constant: Map.get(o, "constant", true),
            sigma2_v1: o["sigma2_v1"],
            family: String.to_atom(o["family"])
          ]
          |> put_if(:offset, c["dataset"] == "poisson" && pick.(ds["offset"], c["train"]))
          |> put_if(:exog_scale, c["exog_scale"])
          |> put_if(:exog_resid, c["exog_resid"])

        r = GWR.predict(coords, y, x, new_coords, x_new, opts)

        assert %Prediction{m: m, n: n} = r
        assert m == length(c["test"]) and n == length(c["train"])
        assert r.family == String.to_atom(o["family"])

        for field <- ~w(params bse tvalues predictions)a do
          Reference.assert_close(Map.fetch!(r, field), c[Atom.to_string(field)], @tol)
        end

        if o["family"] == "gaussian",
          do: Reference.assert_close(r.local_r2, c["local_r2"], @tol),
          else: assert(r.local_r2 == nil)
      end
    end
  end

  defp put_if(opts, _key, nil), do: opts
  defp put_if(opts, _key, false), do: opts
  defp put_if(opts, key, value), do: Keyword.put(opts, key, value)
end
