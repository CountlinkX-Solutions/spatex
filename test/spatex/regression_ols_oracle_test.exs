defmodule Spatex.RegressionOLSOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference
  alias Spatex.Regression
  alias Spatex.Regression.OLS

  @tol 1.0e-9
  @tight 1.0e-10
  @p_rel 1.0e-8

  defp graph(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp opts(c) do
    o = c["options"]

    [
      constant: c["constant"],
      name_x: c["name_x"],
      robust: o["robust"] && String.to_existing_atom(o["robust"]),
      sig2n_k: o["sig2n_k"],
      spat_diag: o["spat_diag"],
      moran: o["moran"],
      white_test: o["white_test"],
      vif: o["vif"]
    ]
    |> maybe(:graph, c["w"] && graph(c["w"]))
    |> maybe(:gwk, c["gwk"] && graph(c["gwk"]))
  end

  defp maybe(opts, _key, nil), do: opts
  defp maybe(opts, key, value), do: Keyword.put(opts, key, value)

  # p-values span many orders of magnitude, so compare them relatively.
  defp assert_p(actual, expected, label) do
    assert abs(actual - expected) <= @p_rel * expected + 1.0e-300,
           "#{label}: p #{inspect(actual)} vs #{inspect(expected)}"
  end

  defp assert_pair({v, p}, [ev, ep], label) do
    Reference.assert_close(v, ev, @tol)
    assert_p(p, ep, label)
  end

  defp assert_diag(map, expected, stat_key, label) do
    Reference.assert_close(Map.fetch!(map, stat_key), expected[Atom.to_string(stat_key)], @tol)
    assert map.df == round(expected["df"])
    assert_p(map.pvalue, expected["pvalue"], label)
  end

  defp check(%OLS{} = r, e, label) do
    assert r.n == e["n"]
    assert r.k == e["k"]
    Reference.assert_close(r.betas, e["betas"], @tight)
    Reference.assert_close(r.vm, e["vm"], @tight)
    Reference.assert_close(r.std_err, e["std_err"], @tight)
    Reference.assert_close(r.predy, e["predy"], @tol)
    Reference.assert_close(r.u, e["u"], @tol)
    Reference.assert_close(r.xtx, e["xtx"], @tol)
    Reference.assert_close(r.xtxi, e["xtxi"], @tol)

    for key <- ~w(utu sig2 sig2n sig2n_k mean_y std_y r2 ar2 logll aic schwarz sig2ml)a do
      Reference.assert_close(Map.fetch!(r, key), e[Atom.to_string(key)], @tol)
    end

    r.t_stat
    |> Enum.zip(e["t_stat"])
    |> Enum.with_index()
    |> Enum.each(fn {{got, exp}, i} -> assert_pair(got, exp, "#{label} t_stat[#{i}]") end)

    assert_pair(r.f_stat, e["f_stat"], "#{label} f_stat")
    Reference.assert_close(r.condition_number, e["condition_number"], @tol)
    assert_diag(r.jarque_bera, e["jarque_bera"], :jb, "jb")
    assert_diag(r.breusch_pagan, e["breusch_pagan"], :bp, "bp")
    assert_diag(r.koenker_bassett, e["koenker_bassett"], :kb, "kb")
    check_white(r.white, e["white"])
    check_vif(r.vif, e["vif"])
    check_spatial(r, e, label)
  end

  defp check_white(nil, nil), do: :ok

  defp check_white({:not_computed, :multicollinearity}, "Not computed due to multicollinearity."),
    do: :ok

  defp check_white(w, e), do: assert_diag(w, e, :wh, "white")

  defp check_vif(nil, nil), do: :ok

  defp check_vif(vif, expected) do
    assert length(vif) == length(expected)

    for {got, exp} <- Enum.zip(vif, expected) do
      case exp do
        nil -> assert got == nil
        [v, t] -> Reference.assert_close(Tuple.to_list(got), [v, t], @tol)
      end
    end
  end

  @pairs ~w(lm_error lm_lag rlm_error rlm_lag lm_sarma lm_wx lm_spdurbin rlm_wx rlm_durlag lm_slxerr)a

  defp check_spatial(r, e, label) do
    for key <- @pairs do
      case e[Atom.to_string(key)] do
        nil -> assert Map.fetch!(r, key) == nil, "#{label} #{key} should be nil"
        pair -> assert_pair(Map.fetch!(r, key), pair, "#{label} #{key}")
      end
    end

    case e["moran_res"] do
      nil ->
        assert r.moran_res == nil

      [i, z, p] ->
        {gi, gz, gp} = r.moran_res
        Reference.assert_close([gi, gz], [i, z], @tol)
        assert_p(gp, p, "#{label} moran p")
    end
  end

  for file <- ~w(ols_columbus ols_synthetic ols_lattice) do
    describe file do
      setup do
        %{fx: Reference.load!("regression/#{unquote(file)}.json")}
      end

      test "every case matches spreg 1.9.1", %{fx: fx} do
        for c <- fx["cases"], c["label"] != "crime~inc+x2/collinear" do
          result = Regression.ols(c["y"], c["x"], opts(c))
          check(result, c["expected"], c["label"])
        end
      end
    end
  end

  describe "special cases" do
    setup do
      %{
        cases:
          Map.new(Reference.load!("regression/ols_columbus.json")["cases"], &{&1["label"], &1})
      }
    end

    test "collinear design: White is not computed", %{cases: cases} do
      c = cases["crime~inc+x2/collinear"]
      r = Regression.ols(c["y"], c["x"], opts(c))
      assert c["expected"]["white"] == "Not computed due to multicollinearity."
      assert r.white == {:not_computed, :multicollinearity}
      assert r.condition_number > 30
    end

    test "extra constant column is dropped with spreg's warning", %{cases: cases} do
      c = cases["crime~inc+hoval+extra_constant"]
      r = Regression.ols(c["y"], c["x"], opts(c))
      assert [warning] = r.warnings
      assert warning =~ "C3" and warning =~ "removed for being constant"
      assert r.name_x == ["CONSTANT", "INC", "HOVAL"]
    end

    test "HAC vm is replicated unsymmetrized", %{cases: cases} do
      c = cases["hoval~inc+crime/hac"]
      r = Regression.ols(c["y"], c["x"], opts(c))
      assert r.robust == :hac
      [row0 | _] = r.vm
      [col0 | _] = Enum.zip_with(r.vm, & &1)
      refute_in_delta Enum.at(row0, 1), Enum.at(col0, 1), 1.0e-6
    end

    test "spreg doc probe values (crime ~ inc + hoval, rook-r)", %{cases: cases} do
      c = cases["crime~inc+hoval/rook_r/spat"]
      r = Regression.ols(c["y"], c["x"], opts(c))
      assert_in_delta hd(r.betas), 68.61896109502523, 1.0e-10
      assert_in_delta r.utu, 6014.892735784364, 1.0e-8
      {i, z, _p} = r.moran_res
      assert_in_delta i, 0.24986247924902968, 1.0e-10
      assert_in_delta z, 2.9367861608440484, 1.0e-9
    end
  end
end
