defmodule Spatex.GWRCollinearityOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.Collinearity
  alias Spatex.Reference

  @tol 1.0e-8
  @fx Reference.load!("regression/gwr_collinearity.json")

  describe "local_collinearity with one bandwidth (GWRResults.local_collinearity)" do
    for {key, _} <- @fx["gwr"] do
      @key key
      test "#{key}" do
        c = @fx["gwr"][@key]
        ds = @fx["datasets"][c["dataset"]]
        o = c["options"]

        r =
          GWR.local_collinearity(ds["coords"], ds["x"],
            bw: if(o["bw"] == :inf, do: :infinity, else: o["bw"]),
            kernel: String.to_atom(o["kernel"] || "bisquare"),
            fixed: o["fixed"] || false,
            spherical: o["spherical"] || false,
            constant: Map.get(o, "constant", true)
          )

        assert %Collinearity{n: n} = r
        assert n == length(ds["y"])
        assert r.bws == nil

        Reference.assert_close(r.corr, c["corr"], @tol)
        Reference.assert_close(r.vif, c["vif"], @tol)
        Reference.assert_close(r.condition_number, c["condition_number"], @tol)
        Reference.assert_close(r.vdp, c["vdp"], @tol)
      end
    end
  end

  describe "local_collinearity with one bandwidth per column (MGWRResults)" do
    for {key, _} <- @fx["mgwr"] do
      @key key
      test "#{key}" do
        c = @fx["mgwr"][@key]
        o = c["options"]

        coords =
          if String.starts_with?(@key, "georgia"),
            do: @fx["mgwr_georgia_coords"],
            else: @fx["mgwr_coords"]

        r =
          GWR.local_collinearity(coords, c["x"],
            bws: c["bws"],
            kernel: String.to_atom(o["kernel"] || "bisquare"),
            fixed: o["fixed"] || false
          )

        assert %Collinearity{corr: nil, vif: nil, corr_pairs: nil} = r
        assert r.bws == Enum.map(c["bws"], &if(o["fixed"], do: &1, else: trunc(&1)))
        Reference.assert_close(r.condition_number, c["condition_number"], @tol)
        Reference.assert_close(r.vdp, c["vdp"], @tol)
      end
    end
  end
end
