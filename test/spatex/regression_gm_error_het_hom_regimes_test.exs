defmodule Spatex.RegressionGMErrorHetHomRegimesTest do
  use ExUnit.Case, async: true

  alias Spatex.{Graph, Reference, Regression}

  # Two disconnected rings of ten observations: the regime subgraphs are the rings themselves.
  @n 20
  @y for i <- 0..19, do: 3.0 + :math.sin(i * 1.3) * 2.0 + rem(i * 7, 5) * 0.4
  @x1 for i <- 0..19, do: 2.0 + :math.cos(i * 0.9) * 1.5 + rem(i * 3, 4) * 0.5
  @x2 for i <- 0..19, do: 1.0 + :math.sin(i * 2.1 + 1.0) + rem(i, 3) * 0.3
  @yend for i <- 0..19, do: 2.5 + :math.cos(i * 1.7) * 1.2 + Enum.at(@x1, i) * 0.4
  @q for i <- 0..19, do: 1.0 + :math.sin(i * 0.7 + 2.0) * 1.4 + Enum.at(@x2, i) * 0.3
  @regimes List.duplicate(:a, 10) ++ List.duplicate(:b, 10)

  defp ring(offset) do
    for i <- 0..9, j <- [rem(i + 1, 10), rem(i + 9, 10)], do: {offset + i, offset + j}
  end

  defp graph(extra \\ []) do
    {f, t} = Enum.unzip(ring(0) ++ ring(10) ++ extra)

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(@n - 1)))
    |> Graph.transform(:r)
  end

  defp ring_graph do
    {f, t} = Enum.unzip(ring(0))

    f
    |> Graph.from_arrays(t, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..9))
    |> Graph.transform(:r)
  end

  defp err(opts),
    do: Regression.gm_error_regimes(@y, [@x1, @x2], @regimes, graph(), opts)

  defp combo(opts),
    do: Regression.gm_combo_regimes(@y, [@x1, @x2], @regimes, graph(), opts)

  defp endog(opts), do: err([yend: [@yend], q: [@q]] ++ opts)

  defp half(list, :a), do: Enum.slice(list, 0, 10)
  defp half(list, :b), do: Enum.slice(list, 10, 10)

  test "the pooled Het and Hom fits report lambda in the covariance, as spreg" do
    for est <- [:het, :hom] do
      r = err(estimator: est)
      assert r.mode == :pooled and r.estimator == est
      assert r.name_x == ~w(a_CONSTANT a_var_1 a_var_2 b_CONSTANT b_var_1 b_var_2 lambda)
      assert length(r.betas) == 7 and length(r.vm) == 7 and length(r.std_err) == 7
      assert length(r.z_stat) == 7
      assert {r.kr, r.kf, r.kryd, r.nr} == {3, 1, 0, 2}
      assert r.lam == List.last(r.betas) and r.lambdas |> List.last() |> is_float()
      assert r.iteration == 1 and r.iter_stop =~ "Maximum number of iterations"
      assert (est == :het and r.sig2 == nil) or (est == :hom and is_float(r.sig2))
      assert length(r.chow.regi) == 3

      assert Enum.zip_with(@y, r.predy, &(&1 - &2))
             |> Enum.zip(r.u)
             |> Enum.all?(fn {a, b} -> abs(a - b) < 1.0e-12 end)
    end
  end

  test "regime_err_sep fits one Het or Hom model per regime on its own subgraph" do
    for {est, extra} <- [het: [step1c: true, max_iter: 3], hom: [a1: :hom, max_iter: 3]] do
      r = err([estimator: est, regime_err_sep: true] ++ extra)
      assert r.mode == :separate and r.lam == nil and r.sig2 == nil
      # spreg counts lambda in kr, and the Chow tests cover it as one more varying coefficient
      assert {r.kr, r.kf, r.kryd} == {4, 0, 0}
      assert length(r.chow.regi) == 4
      assert Enum.map(r.chow.regi, &elem(&1, 0)) == ~w(CONSTANT var_1 var_2 lambda)

      for label <- [:a, :b] do
        sub =
          Regression.gm_error(
            half(@y, label),
            [half(@x1, label), half(@x2, label)],
            ring_graph(),
            [estimator: est] ++ extra
          )

        Reference.assert_close(r.multi[label].betas, sub.betas, 1.0e-12)
        assert r.lams[label] == sub.lam
        assert r.lambdas[label] == sub.lambdas
      end

      assert length(r.vm) == 8 and length(r.std_err) == 8
      assert r.betas == r.multi.a.betas ++ r.multi.b.betas
    end
  end

  test "the combo with regime_lag_sep fits one Het or Hom combo per regime" do
    for est <- [:het, :hom] do
      r = combo(estimator: est, regime_lag_sep: true)
      assert r.mode == :separate and r.rho == nil and map_size(r.rhos) == 2
      assert {r.kr, r.kf} == {5, 0}

      sub =
        Regression.gm_combo(half(@y, :a), [half(@x1, :a), half(@x2, :a)], ring_graph(),
          estimator: est
        )

      assert r.rhos[:a] == sub.rho
      Reference.assert_close(r.multi[:a].betas, sub.betas, 1.0e-12)
    end
  end

  test "the pooled combo carries rho and the reduced-form prediction" do
    for est <- [:het, :hom] do
      r = combo(estimator: est)
      assert r.rho == Enum.at(r.betas, -2) and r.lam == List.last(r.betas)
      assert length(r.predy_e) == @n and is_float(r.pr2_e)
      assert List.last(r.name_yend) == "_Global_W_dep_var"
    end
  end

  test "Hom defaults to A1 = :het, as spreg's regimes classes do" do
    default = err(estimator: :hom)
    assert default.a1 == :het
    assert default.lam == err(estimator: :hom, a1: :het).lam
    assert default.lam != err(estimator: :hom, a1: :hom_sc).lam
    assert err(estimator: :het).a1 == nil and err(estimator: :het, step1c: true).step1c
  end

  test "iterations follow spreg: the regimes models filter with the first-step lambda" do
    # spreg's GM_Error_Het_Regimes uses the previous lambda (like the base model); the other five
    # classes keep the first-step lambda in every iteration, so a second pass only moves lambda
    for est <- [:het, :hom] do
      one = err(estimator: est, max_iter: 1)
      two = err(estimator: est, max_iter: 2, epsilon: 0.0)
      assert two.iteration == 2 and length(two.lambdas) == length(one.lambdas) + 1
      assert two.lam != one.lam
    end

    one = endog(estimator: :hom, max_iter: 1)
    two = endog(estimator: :hom, max_iter: 2, epsilon: 0.0)
    Reference.assert_close(one.betas |> Enum.drop(-1), two.betas |> Enum.drop(-1), 1.0e-12)
    assert one.lam != two.lam
  end

  test "estimator-specific options are validated" do
    assert_raise ArgumentError, ~r/step1c.*:het/, fn -> err(estimator: :hom, step1c: true) end
    assert_raise ArgumentError, ~r/a1.*:hom/, fn -> err(estimator: :het, a1: :hom) end
    assert_raise ArgumentError, ~r/only apply/, fn -> err(max_iter: 2) end
    assert_raise ArgumentError, ~r/only apply/, fn -> err(a1: :hom) end
    assert_raise ArgumentError, ~r/max_iter/, fn -> err(estimator: :het, max_iter: 0) end
    assert_raise ArgumentError, ~r/epsilon/, fn -> err(estimator: :het, epsilon: -1.0) end
    assert_raise ArgumentError, ~r/step1c/, fn -> err(estimator: :het, step1c: 1) end
    assert_raise ArgumentError, ~r/a1/, fn -> err(estimator: :hom, a1: :other) end
    assert_raise ArgumentError, ~r/:estimator/, fn -> err(estimator: :gls) end
    assert err(estimator: :kp98).mode == :pooled
  end

  test "the options spreg's Het and Hom regimes classes take beyond the port are rejected" do
    for opt <- [:slx_lags, :slx_vars, :hard_bound, :inv_method, :cores] do
      assert_raise ArgumentError, ~r/unknown option/, fn -> err([{opt, 1}, estimator: :het]) end
    end
  end

  test "constant_regi, cols2regi and spreg_compatible apply as in the KP98 models" do
    r = endog(estimator: :hom, constant_regi: :one)
    assert "_Global_CONSTANT" in r.name_x
    legacy = endog(estimator: :hom, constant_regi: :one, spreg_compatible: true)
    assert r.betas == legacy.betas and r.vm == legacy.vm
    assert r.chow.joint != legacy.chow.joint

    assert_raise ArgumentError, ~r/All coefficients must vary/, fn ->
      err(estimator: :het, regime_err_sep: true, cols2regi: [true, false])
    end
  end

  test "the internal lambda hook replaces the optimizer, in call order" do
    fit = err(estimator: :het, step1c: true, max_iter: 2, epsilon: 0.0)
    assert length(fit.lambdas) == 4

    hooked =
      Regression.gm_error_regimes_at(@y, [@x1, @x2], @regimes, graph(), [0.1, 0.2, 0.3, 0.4],
        estimator: :het,
        step1c: true,
        max_iter: 2,
        epsilon: 0.0
      )

    assert hooked.lambdas == [0.1, 0.2, 0.3, 0.4] and hooked.lam == 0.4

    assert_raise ArgumentError, ~r/lambdas/, fn ->
      Regression.gm_error_regimes_at(@y, [@x1, @x2], @regimes, graph(), [0.1], estimator: :kp98)
    end
  end
end
