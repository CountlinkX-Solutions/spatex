defmodule Spatex.RegressionSurErrorGmOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.{Graph, Reference, Regression}

  @lam_tol 5.0e-6
  # derived quantities follow spreg's lambda, which is up to ~5e-6 from the exact minimizer
  @loose 5.0e-4

  defp close(a, b, tol), do: Reference.assert_close(List.flatten(a), List.flatten(b), tol)
  defp triple({a, b, c}), do: [a, b * 1.0, c]

  defp graph(fx, key) do
    g = fx["graphs"][key]
    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"], ids: Enum.to_list(0..(g["n"] - 1)))
  end

  defp inputs(fx, c) do
    {Enum.map(c["y_vars"], &fx["data"][&1]),
     Enum.map(c["x_vars"], fn vars -> Enum.map(vars, &fx["data"][&1]) end)}
  end

  defp opts(c), do: [name_bigy: c["y_vars"], name_bigx: c["x_vars"]]

  defp check(r, c, tol, label) do
    close(r.b_sur, c["bSUR"], tol)
    close(r.varb, c["varb"], tol)
    close(r.sig, c["sig"], tol)
    close(r.corr, c["corr"], tol)
    close(r.residuals, c["bigE"], tol)

    close(
      Enum.map(r.sur_inf, fn eq -> Enum.map(eq, &Tuple.to_list/1) end),
      c["sur_inf"],
      max(tol, 1.0e-8)
    )

    if c["surchow"],
      do: close(Enum.map(r.chow, &triple/1), c["surchow"], max(tol, 1.0e-8)),
      else: assert(r.chow == nil, label)
  end

  test "every formula matches spreg at 1e-9 given spreg's lambdas" do
    fx = Reference.load!("regression/sur_error_gm.json")

    for c <- fx["cases"] do
      {ys, xs} = inputs(fx, c)
      r = Regression.sur_error_gm_at(ys, xs, graph(fx, c["weights"]), c["lamsur"], opts(c))
      assert r.name_x == c["name_bigX"], c["label"]
      assert r.k == Enum.map(c["bigK"], &trunc/1), c["label"]
      assert r.lam == c["lamsur"], c["label"]
      check(r, c, 1.0e-9, c["label"])
    end
  end

  test "the estimated lambdas are the exact minimizers, within 5e-6 of spreg's L-BFGS-B" do
    fx = Reference.load!("regression/sur_error_gm.json")
    limited = []

    limited =
      for c <- fx["cases"], reduce: limited do
        acc ->
          {ys, xs} = inputs(fx, c)
          r = Regression.sur_error_gm(ys, xs, graph(fx, c["weights"]), opts(c))
          assert length(r.lam) == length(c["optim"])

          for {lam, o} <- Enum.zip(r.lam, c["optim"]) do
            assert_in_delta lam, o["lam_exact"], @lam_tol
            assert o["f_exact"] <= o["f"] * (1 + 1.0e-6) + 1.0e-12, c["label"]
          end

          if Enum.all?(c["optim"], &(abs(&1["lam"] - &1["lam_exact"]) <= @lam_tol)) do
            for {lam, s} <- Enum.zip(r.lam, c["lamsur"]), do: assert_in_delta(lam, s, @lam_tol)
            check(r, c, @loose, c["label"])
            acc
          else
            [c["label"] | acc]
          end
      end

    # the flat objective of binary weights is where spreg's optimizer stops far from the minimum
    assert Enum.sort(limited) == ["three_equal_rook_binary", "two_rook_binary"]
  end
end
