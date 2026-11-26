defmodule Spatex.GWRHatOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.Reference

  @tol 1.0e-8
  @fixture "regression/gwr_hat.json"

  describe "hat_matrix: true (mgwr 2.2.1 GWR(hat_matrix=True).fit().S)" do
    for {key, _} <- Reference.load!(@fixture)["cases"] do
      @key key
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["cases"][@key]
        ds = fx["datasets"][c["dataset"]]
        o = c["options"]
        family = String.to_atom(o["family"])

        opts =
          [
            bw: if(o["bw"] == :inf, do: :infinity, else: o["bw"]),
            kernel: String.to_atom(o["kernel"] || "bisquare"),
            fixed: o["fixed"] || false,
            spherical: o["spherical"] || false,
            constant: Map.get(o, "constant", true),
            family: family,
            hat_matrix: true
          ]
          |> then(&if(family == :poisson, do: Keyword.put(&1, :offset, ds["offset"]), else: &1))

        r = GWR.fit(ds["coords"], ds["y"], ds["x"], opts)

        n = length(ds["y"])
        assert length(r.hat_matrix) == n and Enum.all?(r.hat_matrix, &(length(&1) == n))
        Reference.assert_close(r.hat_matrix, c["s"], @tol)
        Reference.assert_close(r.tr_s, c["tr_s"], @tol)

        # The same fit without the option leaves everything else untouched.
        plain = GWR.fit(ds["coords"], ds["y"], ds["x"], Keyword.delete(opts, :hat_matrix))
        assert plain.hat_matrix == nil
        assert %{plain | hat_matrix: nil} == %{r | hat_matrix: nil}
      end
    end
  end
end
