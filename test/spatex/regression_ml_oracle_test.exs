defmodule Spatex.RegressionMLOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Graph
  alias Spatex.Reference
  alias Spatex.Regression
  alias Spatex.Regression.{MLError, MLLag}

  # rho/lambda come from a bounded Brent search with xatol = 1e-7 on an objective that is flat at
  # the optimum: a one-ulp difference in a function value (BLAS/LAPACK summation order, which no
  # port can reproduce bit for bit) can flip a late comparison and move the result by up to ~xatol.
  # Most cases follow spreg's path exactly (rho to ~1e-11, same evaluation count); the rest end
  # within 1e-7. Everything else is then compared at a tolerance that follows the rho gap, and the
  # at-parameter tests below verify every formula at 1e-9 independently of the optimizer.
  @rho_tol 1.0e-7
  @exact_gap 1.0e-9
  @tol_exact 1.0e-8
  @tol_gap 2.0e-6
  @p_rel_exact 1.0e-7
  @p_rel_gap 1.0e-3

  defp graph(%{"n" => n, "focal" => f, "neighbor" => nb, "weight" => w}),
    do: Graph.from_arrays(f, nb, w, ids: Enum.to_list(0..(n - 1)))

  defp impacts_opt(c) do
    if map_size(c["expected"]["sp_multipliers"] || %{}) == 3, do: :all, else: :simple
  end

  defp fit(c) do
    g = graph(c["w"])

    base = [
      method: String.to_existing_atom(c["method"]),
      epsilon: c["epsilon"],
      name_x: c["name_x"]
    ]

    case c["model"] do
      "lag" -> Regression.ml_lag(c["y"], c["x"], g, base ++ [spat_impacts: impacts_opt(c)])
      "error" -> Regression.ml_error(c["y"], c["x"], g, base)
    end
  end

  # Tiny p-values (z of 30 or more) move by orders of magnitude with the last digits of z, which is
  # compared directly; for those only the logarithm is checked.
  defp assert_p(actual, expected, _p_rel, label) when expected < 1.0e-30 do
    assert abs(:math.log(actual) - :math.log(expected)) <= 1.0e-6 * abs(:math.log(expected)),
           "#{label}: p #{inspect(actual)} vs #{inspect(expected)}"
  end

  defp assert_p(actual, expected, p_rel, label) do
    assert abs(actual - expected) <= p_rel * expected + 1.0e-300,
           "#{label}: p #{inspect(actual)} vs #{inspect(expected)}"
  end

  defp assert_z(z_stat, expected, tol, p_rel, label) do
    z_stat
    |> Enum.zip(expected)
    |> Enum.with_index()
    |> Enum.each(fn {{{z, p}, [ez, ep]}, i} ->
      Reference.assert_close(z, ez, tol)
      assert_p(p, ep, p_rel, "#{label} z_stat[#{i}]")
    end)
  end

  # Derived quantities move with rho at a sensitivity of up to ~1e3.
  defp scaled(gap, floor), do: min(@tol_gap, max(floor, 1.0e3 * gap))

  defp rho_of(%MLLag{rho: rho}), do: rho
  defp rho_of(%MLError{lam: lam}), do: lam

  # Compares a fitted model with spreg's attributes. `:fit` mode follows the rho gap (see above);
  # `:at` mode (parameter fed in) demands 1e-9 everywhere.
  defp check(r, e, mode, label) do
    key = if match?(%MLLag{}, r), do: "rho", else: "lam"
    gap = abs(rho_of(r) - e[key])

    {rho_tol, tol, p_rel} =
      case mode do
        :at -> {1.0e-12, 1.0e-9, @p_rel_exact}
        :fit when gap < @exact_gap -> {@rho_tol, scaled(gap, @tol_exact), @p_rel_exact}
        :fit -> {@rho_tol, @tol_gap, @p_rel_gap}
      end

    check(r, e, rho_tol, tol, p_rel, label)
  end

  # Everything both models report.
  @common ~w(logll aic schwarz sig2 pr2 mean_y std_y)a

  defp check_common(r, e, rho_key, tol, rho_tol, p_rel, label) do
    Reference.assert_close(Map.fetch!(r, rho_key), e[Atom.to_string(rho_key)], rho_tol)
    Reference.assert_close(r.betas, e["betas"], tol)
    Reference.assert_close(r.vm, e["vm"], tol)
    Reference.assert_close(r.vm1, e["vm1"], tol)
    Reference.assert_close(r.std_err, e["std_err"], tol)
    assert_z(r.z_stat, e["z_stat"], tol, p_rel, label)
    Reference.assert_close(r.u, e["u"], tol)
    Reference.assert_close(r.predy, e["predy"], tol)
    assert r.n == e["n"]
    assert r.k == e["k"]
    assert r.name_x == e["name_x"]

    for key <- @common do
      Reference.assert_close(Map.fetch!(r, key), e[Atom.to_string(key)], tol)
    end

    # The likelihood is flat at the optimum, so it agrees far better than rho does.
    Reference.assert_close(r.logll, e["logll"], 1.0e-9)
  end

  defp check(%MLLag{} = r, e, rho_tol, tol, p_rel, label) do
    check_common(r, e, :rho, tol, rho_tol, p_rel, label)
    Reference.assert_close(r.utu, e["utu"], tol)
    Reference.assert_close(r.predy_e, e["predy_e"], tol)
    Reference.assert_close(r.e_pred, e["e_pred"], tol)
    Reference.assert_close(r.pr2_e, e["pr2_e"], tol)
    check_multipliers(r, e["sp_multipliers"], tol)
  end

  defp check(%MLError{} = r, e, rho_tol, tol, p_rel, label) do
    check_common(r, e, :lam, tol, rho_tol, p_rel, label)
    Reference.assert_close(r.e_filtered, e["e_filtered"], tol)
  end

  defp check_multipliers(r, expected, tol) do
    for {name, [adi, aii, ati]} <- expected do
      method = String.to_existing_atom(name)
      assert {a, i, t} = r.sp_multipliers[method]
      Reference.assert_close([a, i, t], [adi, aii, ati], tol)

      # Effects are the slopes times the multipliers.
      slopes = r.betas |> Enum.take(r.k - 1) |> tl()

      for {effect, b} <- Enum.zip(r.impacts[method], slopes) do
        Reference.assert_close(
          [effect.direct, effect.indirect, effect.total],
          [adi * b, aii * b, ati * b],
          tol
        )
      end
    end

    assert map_size(r.sp_multipliers) == map_size(expected)
  end

  defp cases(file), do: Reference.load!("regression/#{file}.json")["cases"]

  defp by_label(file), do: Map.new(cases(file), &{&1["label"], &1})

  for file <- ~w(ml_columbus ml_synthetic) do
    describe "#{file}: full and lu" do
      test "match spreg (lu is evaluated like full, so only its optimum is compared)" do
        for c <- cases(unquote(file)), c["method"] in ["full", "lu"] do
          check(fit(c), c["expected"], :fit, c["label"])
        end
      end
    end

    describe "#{file}: ord" do
      test "asymmetric patterns (complex eigenvalues) match spreg's ord" do
        for c <- cases(unquote(file)), c["method"] == "ord", not c["pysal_symmetrized"] do
          check(fit(c), c["expected"], :fit, c["label"])
        end
      end

      test "symmetric row-standardized patterns: spreg's rho, and spreg's full standard errors" do
        # spreg's ord variance substitutes the symmetrized matrix for W (documented deviation):
        # the correct reference for everything but rho is the same model with method full.
        all = by_label(unquote(file))

        for c <- Map.values(all), c["method"] == "ord", c["pysal_symmetrized"] do
          r = fit(c)
          key = if c["model"] == "lag", do: "rho", else: "lam"
          assert_in_delta rho_of(r), c["expected"][key], @rho_tol, c["label"]
          twin = all[String.replace(c["label"], "/ord", "/full")]
          check(r, twin["expected"], :fit, c["label"])
        end
      end
    end
  end

  describe "at-parameter formulas (spreg's own rho/lambda fed back in)" do
    test "all back-out quantities match at 1e-9" do
      for file <- ~w(ml_columbus ml_synthetic ml_ord_deviation),
          c <- cases(file),
          c["method"] == "full" do
        e = c["expected"]
        g = graph(c["w"])
        opts = [name_x: c["name_x"], epsilon: c["epsilon"]]

        case c["model"] do
          "lag" ->
            opts = opts ++ [spat_impacts: impacts_opt(c)]
            check(Regression.ml_lag_at(c["y"], c["x"], g, e["rho"], opts), e, :at, c["label"])

          "error" ->
            check(Regression.ml_error_at(c["y"], c["x"], g, e["lam"], opts), e, :at, c["label"])
        end
      end
    end
  end

  describe "optimizer" do
    test "the literal Brent port follows spreg's path in most full-method cases" do
      full =
        for file <- ~w(ml_columbus ml_synthetic ml_ord_deviation),
            c <- cases(file),
            c["method"] == "full" do
          e = c["expected"]
          r = fit(c)
          key = if c["model"] == "lag", do: "rho", else: "lam"
          {abs(rho_of(r) - e[key]) < 1.0e-9 and r.iterations == e["nfev"], c["label"]}
        end

      matched = Enum.count(full, &elem(&1, 0))
      # Observed 13 of 17 (all lag fits); the others end within xatol (see the module comment).
      assert matched >= 0.6 * length(full),
             "path reproduced in #{matched} of #{length(full)}: #{inspect(Enum.reject(full, &elem(&1, 0)))}"
    end

    test "named cases reproduce rho to 1e-10 with spreg's evaluation count" do
      all = by_label("ml_columbus")

      for label <- [
            "hoval~inc+crime/queen_r/lag/full",
            "crime~inc+hoval/rook_binary/lag/full",
            "crime~inc+hoval/rook_binary/error/full",
            "crime~inc+hoval/knn4_r/error/full"
          ] do
        c = all[label]
        r = fit(c)
        key = if c["model"] == "lag", do: "rho", else: "lam"
        assert_in_delta rho_of(r), c["expected"][key], 1.0e-10, label
        assert r.iterations == c["expected"]["nfev"], label
        assert r.converged
      end
    end
  end

  describe "ord deviation (symmetric inverse-distance weights)" do
    # PySAL: `asymmetry(intrinsic=False) == []` selects `symmetrize(w)`, the eigenvalues of the
    # row-standardized *binary* matrix, which are not those of W (bug 1), and the symmetrized
    # matrix then replaces W in the variance (bug 2). Spatex uses the eigenvalues of W.
    # known_deviations.exs: ml_ord_deviation.json
    setup do
      %{all: by_label("ml_ord_deviation")}
    end

    test "PySAL's ord disagrees with PySAL's full; Spatex's ord agrees with full", %{all: all} do
      for model <- ~w(lag error) do
        key = if model == "lag", do: "rho", else: "lam"
        full = all["crime~inc+hoval/idw_r/#{model}/full"]
        ord = all["crime~inc+hoval/idw_r/#{model}/ord"]
        assert ord["pysal_symmetrized"]

        # PySAL's ord parameter is off in the third digit (and its likelihood in the second) ...
        assert abs(ord["expected"][key] - full["expected"][key]) > 1.0e-3
        assert abs(ord["expected"]["logll"] - full["expected"]["logll"]) > 1.0e-3

        # ... Spatex's ord matches PySAL's full.
        r = fit(ord)
        assert_in_delta rho_of(r), full["expected"][key], @rho_tol
        check(r, full["expected"], :fit, "idw/#{model}/ord")
      end
    end

    test "PySAL's ord standard errors differ even where its rho is right (bug 2)" do
      c = by_label("ml_columbus")
      ord = c["hoval~inc+crime/queen_r/lag/ord"]
      full = c["hoval~inc+crime/queen_r/lag/full"]
      assert ord["pysal_symmetrized"]

      assert abs(List.last(ord["expected"]["std_err"]) - List.last(full["expected"]["std_err"])) >
               1.0e-3

      r = fit(ord)
      assert_in_delta List.last(r.std_err), List.last(full["expected"]["std_err"]), 1.0e-6
    end

    test "full on the same weights matches spreg", %{all: all} do
      for {_, c} <- all, c["method"] == "full" do
        check(fit(c), c["expected"], :fit, c["label"])
      end
    end
  end

  describe "spreg doc probe values (crime ~ inc + hoval, queen-r)" do
    setup do
      %{all: by_label("ml_columbus")}
    end

    test "lag", %{all: all} do
      r = fit(all["crime~inc+hoval/queen_r/lag/full"])
      assert_in_delta r.rho, 0.4233254173534828, 1.0e-9
      assert_in_delta r.logll, -182.67397201013392, 1.0e-8
      assert_in_delta r.pr2_e, 0.580646624896824, 1.0e-8
      assert {1.0, aii, ati} = r.sp_multipliers.simple
      assert_in_delta aii, 0.7340802422931958, 1.0e-9
      assert_in_delta ati, 1.7340802422931958, 1.0e-9
      assert_in_delta r.sig2, 96.8571813851897, 1.0e-6
    end

    test "error", %{all: all} do
      r = fit(all["crime~inc+hoval/queen_r/error/full"])
      assert_in_delta r.lam, 0.5467530269102509, @rho_tol
      assert_in_delta r.logll, -183.7494280620572, 1.0e-9
      # k excludes lambda: aic = 2 * 3 - 2 * logll.
      assert r.k == 3
      assert_in_delta r.aic, 373.4988561241144, 1.0e-8
      assert_in_delta r.sig2, 97.6742325440519, 1.0e-5
    end
  end
end
