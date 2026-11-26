defmodule Spatex.MGWROracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.MGWRResult
  alias Spatex.Reference

  @fx Reference.load!("regression/gwr_mgwr.json")
  @tol 1.0e-8

  for {key, _} <- @fx["cases"] do
    @key key
    test "MGWR #{key}" do
      c = @fx["cases"][@key]
      coords = coords_for(@key)
      r = GWR.mgwr(coords, c["y"], c["x"], mgwr_opts(c["options"]))

      assert %MGWRResult{} = r
      assert r.n == length(c["y"])
      assert r.k == length(c["x"]) + 1
      assert r.converged

      Reference.assert_close(r.bw_init, c["bw_init"], 1.0e-6)
      Reference.assert_close(r.bws, c["bws"], 1.0e-6)
      assert length(r.bws_history) == length(c["bws_history"])
      Reference.assert_close(r.bws_history, c["bws_history"], 1.0e-6)
      assert r.iterations == length(c["scores"])
      Reference.assert_close(r.scores, c["scores"], 1.0e-7)

      for field <- ~w(params predy resid)a do
        Reference.assert_close(Map.fetch!(r, field), c[Atom.to_string(field)], @tol)
      end

      for field <- ~w(bse tvalues)a do
        Reference.assert_close(Map.fetch!(r, field), c[Atom.to_string(field)], @tol)
      end

      Reference.assert_close(r.cct, c["cct"], @tol)
      Reference.assert_close(r.enp_j, c["enp_j"], @tol)

      for field <- ~w(tr_s sigma2 resid_ss llf aic aicc bic r2 adj_r2)a do
        Reference.assert_close(Map.fetch!(r, field), c[Atom.to_string(field)], @tol)
      end

      assert length(r.adj_alpha_j) == r.k
      Reference.assert_close(r.adj_alpha_j, c["adj_alpha_j"], @tol)
      Reference.assert_close(r.critical_tval, c["critical_tval"], @tol)
      Reference.assert_close(GWR.filter_tvals(r), c["filter_tvals"], @tol)
    end
  end

  defp coords_for("synthetic100" <> _), do: @fx["coords2"]
  defp coords_for("georgia" <> _), do: @fx["georgia_coords"]
  defp coords_for(_), do: @fx["coords"]

  defp mgwr_opts(o) do
    [
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      criterion: String.to_atom(String.downcase(o["criterion"] || "aicc")),
      rss_score: o["rss_score"] || false
    ]
    |> put_list(:multi_bw_min, o["multi_bw_min"])
    |> put_list(:multi_bw_max, o["multi_bw_max"])
  end

  defp put_list(opts, _key, nil), do: opts
  defp put_list(opts, key, [single]), do: Keyword.put(opts, key, single)
  defp put_list(opts, key, list), do: Keyword.put(opts, key, list)
end
