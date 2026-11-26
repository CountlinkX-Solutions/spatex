defmodule Spatex.GWRGLMOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.GWR
  alias Spatex.GWR.{Bandwidth, Result}
  alias Spatex.Reference

  @fixture "regression/gwr_glm.json"
  @tol 1.0e-8
  @score_tol 1.0e-9

  @fx Reference.load!(@fixture)

  describe "bandwidth selection (Poisson and Binomial)" do
    for {key, _} <- @fx["selection"] do
      @key key
      test "#{key}" do
        sel = @fx["selection"][@key]
        result = GWR.select_bandwidth(@fx["coords"], y_for(sel), @fx["x"], select_opts(sel))

        assert %Bandwidth{family: family} = result
        assert family == String.to_atom(sel["options"]["family"])

        if sel["options"]["fixed"] == true do
          assert abs(result.bw - sel["bw"]) <= 1.0e-6 * max(1.0, abs(sel["bw"]))
        else
          assert result.bw == sel["bw"]
        end

        Reference.assert_close(result.score, sel["score"], @score_tol)
        assert length(result.history) == length(sel["history"])

        for {{bw, score}, [ebw, escore]} <- Enum.zip(result.history, sel["history"]) do
          if sel["options"]["fixed"] == true,
            do: Reference.assert_close(bw, ebw, 1.0e-6),
            else: assert(bw == ebw)

          Reference.assert_close(score, escore, @score_tol)
        end
      end
    end
  end

  describe "fit (Poisson and Binomial)" do
    for {key, _} <- @fx["fits"] do
      @key key
      test "#{key}" do
        e = @fx["fits"][@key]
        r = GWR.fit(@fx["coords"], y_for(e), @fx["x"], fit_opts(e))

        assert %Result{family: family} = r
        assert family == String.to_atom(e["options"]["family"])
        assert r.k == 3
        assert r.scale == 1.0
        assert r.r2 == nil and r.adj_r2 == nil and r.local_r2 == nil
        assert length(r.iterations) == 80 and Enum.all?(r.iterations, &(&1 in 1..20))

        for field <- ~w(params bse tvalues predy resid influ cooks_d std_res)a do
          Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
        end

        for field <-
              ~w(resid_ss tr_s tr_sts enp sigma2 llf aic aicc bic cv d2 adj_d2 null_deviance
                 global_deviance df_resid)a do
          Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
        end

        Reference.assert_close(r.adj_alpha, e["adj_alpha"], @tol)
        Reference.assert_close(r.critical_tval, e["critical_tval"], @tol)
        Reference.assert_close(GWR.filter_tvals(r), e["filter_tvals"], @tol)

        if Enum.any?(e["deviance"], &(&1 == :nan)) do
          # mgwr evaluates 0 * ln(0) = NaN for a zero count; Spatex uses 0 ln 0 = 0.
          assert Enum.all?(r.deviance, &is_float/1)
          assert Enum.all?(r.p_dev, &is_float/1)
          assert Enum.all?(r.resid_deviance, &is_float/1)
        else
          for field <- ~w(deviance resid_deviance p_dev)a do
            Reference.assert_close(Map.fetch!(r, field), e[Atom.to_string(field)], @tol)
          end
        end
      end
    end

    test "the offset only changes a Poisson model" do
      e = @fx["fits"]["poisson_no_offset"]
      ones = List.duplicate(1.0, 80)
      a = GWR.fit(@fx["coords"], @fx["y_poisson"], @fx["x"], bw: 55, family: :poisson)

      b =
        GWR.fit(@fx["coords"], @fx["y_poisson"], @fx["x"], bw: 55, family: :poisson, offset: ones)

      assert a.params == b.params
      Reference.assert_close(a.params, e["params"], @tol)
    end
  end

  describe "validation" do
    test "family and offset are checked" do
      pts = @fx["coords"]
      x = @fx["x"]
      neg = List.replace_at(@fx["y_poisson"], 3, -1.0)

      assert_raise ArgumentError, ~r/family/, fn ->
        GWR.fit(pts, @fx["y_poisson"], x, bw: 55, family: :gamma)
      end

      assert_raise ArgumentError, ~r/non-negative/, fn ->
        GWR.fit(pts, neg, x, bw: 55, family: :poisson)
      end

      assert_raise ArgumentError, ~r/offset/, fn ->
        GWR.fit(pts, @fx["y_poisson"], x,
          bw: 55,
          family: :poisson,
          offset: List.duplicate(0.0, 80)
        )
      end

      assert_raise ArgumentError, ~r/offset/, fn ->
        GWR.fit(pts, @fx["y_poisson"], x, bw: 55, family: :poisson, offset: [1.0])
      end

      assert_raise ArgumentError, ~r/offset/, fn ->
        GWR.fit(pts, @fx["y_binomial"], x, bw: 55, family: :binomial, offset: @fx["offset"])
      end

      assert_raise ArgumentError, ~r/0 or 1/, fn ->
        GWR.fit(pts, @fx["y_poisson"], x, bw: 55, family: :binomial)
      end

      assert_raise ArgumentError, ~r/iwls_tol/, fn ->
        GWR.fit(pts, @fx["y_poisson"], x, bw: 55, family: :poisson, iwls_tol: -1.0)
      end
    end

    test "IWLS settings are honoured" do
      a = GWR.fit(@fx["coords"], @fx["y_binomial"], @fx["x"], bw: 60, family: :binomial)

      b =
        GWR.fit(@fx["coords"], @fx["y_binomial"], @fx["x"],
          bw: 60,
          family: :binomial,
          iwls_max_iter: 1
        )

      assert Enum.all?(b.iterations, &(&1 == 1))
      assert a.params != b.params
    end
  end

  defp y_for(entry), do: @fx[entry["y_key"]]

  defp select_opts(sel) do
    o = sel["options"]

    [
      family: String.to_atom(o["family"]),
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false,
      criterion: String.to_atom(String.downcase(o["criterion"] || "aicc"))
    ]
    |> put_offset(sel)
  end

  defp fit_opts(e) do
    o = e["options"]

    [
      bw: if(o["bw"] == :inf, do: :infinity, else: o["bw"]),
      family: String.to_atom(o["family"]),
      kernel: String.to_atom(o["kernel"] || "bisquare"),
      fixed: o["fixed"] || false
    ]
    |> put_offset(e)
  end

  defp put_offset(opts, %{"uses_offset" => true}), do: Keyword.put(opts, :offset, @fx["offset"])
  defp put_offset(opts, _), do: opts
end
