defmodule Spatex.GWRSearchOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.Bandwidth
  alias Spatex.Reference

  @fixture "regression/gwr_search.json"

  describe "search: :scipy (mgwr 2.2.1 Sel_BW.search(search_method='scipy'))" do
    for {key, _} <- Reference.load!(@fixture)["scipy"] do
      @key key
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["scipy"][@key]
        ds = fx["datasets"][c["dataset"]]
        r = GWR.select_bandwidth(ds["coords"], ds["y"], ds["x"], opts(c, ds))

        assert %Bandwidth{history: []} = r
        # The optimizer follows SciPy's iteration path, so adaptive searches agree to ~1e-9. A
        # fixed bandwidth sits where the criterion is flat to rounding (differences of ~1e-13
        # decide SciPy's parabolic-versus-golden branches), so it agrees to 1e-6 relative and its
        # evaluation count to a few calls.
        Reference.assert_close(r.bw, c["bw"], if(c["options"]["fixed"], do: 1.0e-6, else: 1.0e-9))
        Reference.assert_close(r.score, c["score"], 1.0e-9)

        if c["options"]["fixed"],
          do: assert(abs(r.nfev - c["nfev"]) <= 4),
          else: assert(r.nfev == c["nfev"])

        assert r.converged == c["success"]
      end
    end
  end

  describe "x_glob (mgwr 2.2.1 Sel_BW(X_glob=...): only the default search bounds)" do
    for {key, _} <- Reference.load!(@fixture)["x_glob_cases"] do
      @key key
      test "#{key}" do
        fx = Reference.load!(@fixture)
        c = fx["x_glob_cases"][@key]
        ds = fx["datasets"]["georgia"]
        x_glob = fx["x_glob"]
        o = c["options"]

        opts =
          [
            kernel: String.to_atom(o["kernel"] || "bisquare"),
            fixed: o["fixed"] || false,
            constant: Map.get(o, "constant", true),
            criterion: String.to_atom(String.downcase(o["criterion"] || "aicc"))
          ]
          |> put_if(:bw_min, o["bw_min"])
          |> put_if(:bw_max, o["bw_max"])
          |> put_if(:search, o["search_method"] && :scipy)
          |> put_if(:x_glob, x_glob(@key, x_glob))

        r = GWR.select_bandwidth(ds["coords"], ds["y"], ds["x"], opts)

        if o["fixed"] do
          Reference.assert_close(r.bw, c["bw"], 1.0e-6)
        else
          if o["search_method"] == "scipy",
            do: Reference.assert_close(r.bw, c["bw"], 1.0e-9),
            else: assert(r.bw == c["bw"])
        end

        Reference.assert_close(r.score, c["score"], 1.0e-9)
        assert length(r.history) == length(c["history"])

        for {{bw, score}, [ebw, escore]} <- Enum.zip(r.history, c["history"]) do
          if o["fixed"], do: Reference.assert_close(bw, ebw, 1.0e-6), else: assert(bw == ebw)
          Reference.assert_close(score, escore, 1.0e-9)
        end
      end
    end

    test "the extra columns move the first probes of the default bounds" do
      cases = Reference.load!(@fixture)["x_glob_cases"]
      none = cases["georgia_none"]["history"] |> hd() |> hd()
      one = cases["georgia_one_column"]["history"] |> hd() |> hd()
      assert none != one
    end
  end

  defp x_glob(key, columns) do
    cond do
      String.contains?(key, "none") -> nil
      String.contains?(key, "three") -> columns["three"]
      true -> columns["one"]
    end
  end

  defp opts(c, ds) do
    o = c["options"]

    [
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      spherical: o["spherical"] || false,
      constant: Map.get(o, "constant", true),
      criterion: String.to_atom(String.downcase(o["criterion"] || "aicc")),
      family: String.to_atom(o["family"]),
      search: :scipy
    ]
    |> put_if(:bw_min, o["bw_min"])
    |> put_if(:bw_max, o["bw_max"])
    |> put_if(:offset, o["family"] == "poisson" && ds["offset"])
  end

  defp put_if(opts, _key, nil), do: opts
  defp put_if(opts, _key, false), do: opts
  defp put_if(opts, key, value), do: Keyword.put(opts, key, value)
end
