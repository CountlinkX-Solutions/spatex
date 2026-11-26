defmodule Spatex.GWROracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, Result}
  alias Spatex.Reference

  @tol 1.0e-8
  @score_tol 1.0e-9

  @fixtures ["regression/gwr_georgia.json", "regression/gwr_synthetic.json"]

  for fixture <- @fixtures do
    @fixture fixture
    describe "bandwidth selection (#{fixture})" do
      for {key, _} <- Reference.load!(fixture)["selection"] do
        @key key
        test "#{key}" do
          fx = Reference.load!(@fixture)
          sel = fx["selection"][@key]
          result = GWR.select_bandwidth(fx["coords"], fx["y"], fx["x"], select_opts(sel))

          assert %Bandwidth{} = result
          expected = sel["bw"]

          if fixed?(sel) do
            assert abs(result.bw - expected) <= 1.0e-6 * max(1.0, abs(expected)),
                   "bw #{result.bw} vs #{expected}"
          else
            assert result.bw == expected
          end

          Reference.assert_close(result.score, sel["score"], @score_tol)
          assert length(result.history) == length(sel["history"])

          for {{bw, score}, [ebw, escore]} <- Enum.zip(result.history, sel["history"]) do
            cond do
              ebw == :inf -> assert bw == :infinity
              fixed?(sel) -> Reference.assert_close(bw, ebw, 1.0e-6)
              true -> assert bw == ebw
            end

            Reference.assert_close(score, escore, @score_tol)
          end
        end
      end
    end

    describe "fit (#{fixture})" do
      for {key, _} <- Reference.load!(fixture)["fits"] do
        @key key
        test "#{key}" do
          fx = Reference.load!(@fixture)
          e = fx["fits"][@key]
          r = GWR.fit(fx["coords"], fx["y"], fx["x"], fit_opts(e["options"]))

          assert %Result{} = r
          assert r.n == length(fx["y"])
          assert r.k == length(hd(e["params"]))

          for field <- ~w(params bse tvalues)a do
            Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
          end

          for field <- ~w(local_r2 predy resid influ cooks_d std_res)a do
            Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
          end

          for field <-
                ~w(resid_ss tr_s tr_sts enp sigma2 llf aic aicc bic cv r2 adj_r2 df_resid)a do
            Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
          end

          Reference.assert_close(r.adj_alpha, e["adj_alpha"], @tol)
          Reference.assert_close(r.critical_tval, e["critical_tval"], 1.0e-8)
          Reference.assert_close(GWR.critical_tval(r, 0.05), e["critical_tval_05"], 1.0e-8)
          Reference.assert_close(GWR.filter_tvals(r), e["filter_tvals"], @tol)
          Reference.assert_close(GWR.filter_tvals(r, alpha: 0.05), e["filter_tvals_05"], @tol)
        end
      end
    end
  end

  defp fixed?(sel), do: sel["options"]["fixed"] == true

  defp select_opts(sel) do
    o = sel["options"]

    [
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      spherical: o["spherical"] || false,
      constant: Map.get(o, "constant", true),
      criterion: String.to_atom(String.downcase(o["criterion"] || "aicc"))
    ]
    |> put_if(:bw_min, o["bw_min"])
    |> put_if(:bw_max, if(o["bw_max"] != :inf, do: o["bw_max"]))
    |> put_if(:include_global, o["bw_max"] == :inf)
    |> put_if(:search, o["search_method"] && :interval)
    |> put_if(:interval, o["interval"])
  end

  defp fit_opts(o) do
    bw = if o["bw"] == :inf, do: :infinity, else: o["bw"]

    [
      bw: bw,
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      spherical: o["spherical"] || false,
      constant: Map.get(o, "constant", true),
      sigma2_v1: o["sigma2_v1"]
    ]
  end

  defp put_if(opts, _key, nil), do: opts
  defp put_if(opts, _key, false), do: opts
  defp put_if(opts, key, value), do: Keyword.put(opts, key, value)
end
