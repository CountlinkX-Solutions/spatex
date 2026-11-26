defmodule Spatex.DynamicsOracleTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Dynamics
  alias Spatex.Graph
  alias Spatex.Reference

  @tol 1.0e-9

  # Fixtures decode non-finite floats as :inf / :neg_inf; results use :infinity / :neg_infinity.
  defp norm(:inf), do: :infinity
  defp norm(:neg_inf), do: :neg_infinity
  defp norm(list) when is_list(list), do: Enum.map(list, &norm/1)
  defp norm(other), do: other

  defp close(actual, expected, label) do
    Reference.assert_close(actual, norm(expected), @tol)
  rescue
    e in ExUnit.AssertionError ->
      reraise %{e | message: label <> " " <> e.message}, __STACKTRACE__
  end

  # Known deviation: for a reducible chain whose system is numerically singular but not rejected,
  # giddy returns meaningless first passage times (negative or beyond 1e12, e.g. -5.3e15 in
  # place of infinity). Spatex returns :infinity there; the expected value is replaced by it.
  defp scrub_garbage(expected, actual) when is_list(expected) and is_list(actual),
    do: Enum.zip_with(expected, actual, &scrub_garbage/2)

  defp scrub_garbage(e, a) when is_number(e) and (e < 0 or e > 1.0e12) do
    assert a == :infinity, "giddy garbage #{e} must be :infinity in Spatex, got #{inspect(a)}"
    a
  end

  defp scrub_garbage(e, _a), do: e

  defp graph(fx, name) do
    g = fx["graphs"][name]

    Graph.from_arrays(g["focal"], g["neighbor"], g["weight"],
      ids: Enum.to_list(0..(length(g["ids"]) - 1))
    )
  end

  defp row_graph(fx, name), do: fx |> graph(name) |> Graph.transform(:r)

  # Binomial-style band for two independent P-permutation estimates of the same p.
  defp assert_p_close(actual, expected, perms, label) do
    sd = :math.sqrt(2 * max(expected * (1 - expected), 1.0e-4) / perms) + 1 / perms

    assert abs(actual - expected) <= 5 * sd,
           "#{label}: #{actual} vs #{expected} (5 sd = #{5 * sd})"
  end

  defp sorted_rows(ss) when is_list(ss) do
    if is_list(hd(ss)),
      do: Enum.sort_by(ss, fn row -> Enum.find_index(row, &(&1 > 1.0e-12)) end),
      else: ss
  end

  import ExUnit.Assertions

  describe "markov" do
    test "matches giddy.Markov on every case" do
      fx = Reference.load!("dynamics/markov.json")

      for c <- fx["cases"] do
        out = c["out"]
        fill = c["fill_empty_classes"]

        m =
          if c["y"],
            do: Dynamics.markov(c["y"], k: c["k"], fill_empty_classes: fill),
            else: Dynamics.markov(c["ids"], fill_empty_classes: fill)

        l = c["label"]
        assert m.classes == out["classes"], l
        assert m.k == out["k"], l
        close(m.transitions, out["transitions"], l)
        close(m.p, out["p"], l)
        close(sorted_rows(m.steady_state), sorted_rows(out["steady_state"]), l)
        close(m.sojourn_time, out["sojourn_time"], l)
        assert Enum.sort(m.communication_classes) == out["cclasses"], l
        assert Enum.sort(m.recurrent_classes) == out["rclasses"], l
        assert length(m.transient_classes) == out["num_tclasses"], l
        assert m.absorbing_states == out["astates"], l

        if out["mfpt_error"] do
          # Known deviation: giddy raises LinAlgError on a recurrent cycle; Spatex returns the
          # mean first passage times that are certain and infinity elsewhere.
          assert length(m.mfpt) == m.k
        else
          close(m.mfpt, scrub_garbage(norm(out["mfpt"]), m.mfpt), l)
        end
      end
    end

    test "steady_state/2, mfpt/2 and sojourn_time/2 on hand matrices" do
      fx = Reference.load!("dynamics/ergodic.json")

      for c <- fx["cases"] do
        close(
          sorted_rows(Dynamics.steady_state(c["p"])),
          sorted_rows(c["steady_state"]),
          c["label"]
        )

        close(Dynamics.mfpt(c["p"]), c["mfpt"], c["label"])
      end
    end
  end

  describe "spatial_markov" do
    test "matches giddy.Spatial_Markov" do
      fx = Reference.load!("dynamics/spatial_markov.json")

      for c <- fx["cases"] do
        o = c["out"]
        l = c["label"]
        opts = c["opts"]

        perms = c["permutations"] || 0

        sm =
          Dynamics.spatial_markov(
            c["y"],
            row_graph(fx, c["graph"]),
            [fill_empty_classes: c["fill_empty_classes"], permutations: perms, seed: 11] ++
              for({key, v} <- opts, do: {String.to_existing_atom(key), v})
          )

        assert {sm.k, sm.m} == {o["k"], o["m"]}, l
        close(sm.cutoffs, o["cutoffs"], l)
        close(sm.lag_cutoffs, o["lag_cutoffs"], l)
        assert sm.class_ids == o["class_ids"], l
        # Known deviation: after a permutation test giddy's `lclass_ids` is left holding the
        # lag classes of the last permuted replicate (`_calc` overwrites the attribute).
        if perms == 0, do: assert(sm.lclass_ids == o["lclass_ids"], l)
        close(sm.transitions, o["transitions"], l)
        close(sm.p, o["p"], l)
        close(sm.t, o["t"], l)
        close(sm.p_cond, o["p_cond"], l)
        close(sm.s, o["s"], l)
        close(sm.s_cond, o["s_cond"], l)
        close(sm.f, scrub_garbage(o["f"], sm.f), l)
        close(sm.f_cond, scrub_garbage(o["f_cond"], sm.f_cond), l)

        close(
          [sm.q, sm.q_p_value, sm.lr, sm.lr_p_value],
          [o["q"], o["q_p_value"], o["lr"], o["lr_p_value"]],
          l
        )

        assert sm.dof_hom == o["dof_hom"], l
        close(Enum.map(sm.shtest, &Tuple.to_list/1), o["shtest"], l)
        close(Enum.map(sm.chi2, &Tuple.to_list/1), o["chi2"], l)
        close([sm.x2, sm.x2_pvalue], [o["x2"], o["x2_pvalue"]], l)
        assert sm.x2_dof == o["x2_dof"], l

        if perms > 0 do
          assert_p_close(sm.x2_rpvalue, c["x2_rpvalue"], perms, l)
          mean = Enum.sum(sm.x2_realizations) / perms
          se = c["x2_realizations_std"] / :math.sqrt(perms)
          assert abs(mean - c["x2_realizations_mean"]) <= 6 * se * :math.sqrt(2), l
        end
      end
    end
  end

  describe "lisa_markov" do
    test "matches giddy.LISA_Markov" do
      fx = Reference.load!("dynamics/lisa_markov.json")

      for c <- fx["cases"] do
        o = c["out"]
        l = c["label"]
        perms = c["permutations"]

        lm =
          Dynamics.lisa_markov(c["y"], row_graph(fx, c["graph"]), permutations: perms, seed: 9)

        assert lm.q == o["q"], l
        close(lm.transitions, o["transitions"], l)
        close(lm.p, o["p"], l)
        close(sorted_rows(lm.steady_state), sorted_rows(o["steady_state"]), l)
        close(lm.mfpt, o["mfpt"], l)
        close(lm.sojourn_time, o["sojourn_time"], l)
        assert lm.move_types == o["move_types"], l

        # Known deviation: giddy's own chi_2 builds its marginal chains from the (t, n) array
        # (transitions between neighboring locations); Spatex uses the (n, t) orientation.
        {stat, p, dof} = lm.chi_2
        close([stat, p, dof], o["chi_2_intended"], l)
        close(lm.expected_t, o["expected_t_intended"], l)

        if perms > 0 do
          pv = List.flatten(lm.p_values)
          ref = List.flatten(o["p_values"])
          worst = Enum.zip(pv, ref) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
          assert worst <= 0.04, "#{l}: max |dp| = #{worst}"

          mine = List.flatten(lm.significant_moves)
          theirs = List.flatten(o["significant_moves"])
          agree = Enum.zip(mine, theirs) |> Enum.count(fn {a, b} -> a == b end)
          assert agree / length(mine) >= 0.97, "#{l}: #{agree}/#{length(mine)} moves agree"
        end
      end
    end
  end

  describe "rank mobility" do
    test "kendall_tau matches giddy.rank.Tau" do
      for c <- Reference.load!("dynamics/tau.json")["cases"] do
        t = Dynamics.kendall_tau(c["x"], c["y"])

        close(
          [t.tau, t.p, t.concordant, t.discordant, t.extra_x, t.extra_y],
          [c["tau"], c["tau_p"], c["concordant"], c["discordant"], c["extra_x"], c["extra_y"]],
          c["label"]
        )
      end
    end

    test "spatial_tau matches giddy.rank.SpatialTau" do
      fx = Reference.load!("dynamics/spatial_tau.json")

      for c <- fx["cases"] do
        l = c["label"]
        perms = c["permutations"]

        r =
          Dynamics.spatial_tau(c["x"], c["y"], graph(fx, c["graph"]),
            permutations: perms,
            seed: 4
          )

        close(
          [r.tau, r.tau_p, r.concordant, r.discordant, r.extra_x, r.extra_y],
          [c["tau"], c["tau_p"], c["concordant"], c["discordant"], c["extra_x"], c["extra_y"]],
          l
        )

        close(
          [r.tau_spatial, r.concordant_spatial, r.discordant_spatial],
          [c["tau_spatial"], c["concordant_spatial"], c["discordant_spatial"]],
          l
        )

        assert r.pairs_spatial == c["pairs_spatial"], l
        assert_p_close(r.tau_spatial_psim, c["tau_spatial_psim"], perms, l)
        mean = Enum.sum(r.taus) / perms
        assert abs(mean - c["taus_mean"]) <= 6 * c["taus_std"] / :math.sqrt(perms) * 1.5, l
      end
    end

    test "theta matches giddy.rank.Theta" do
      for c <- Reference.load!("dynamics/theta.json")["cases"] do
        l = c["label"]
        perms = c["permutations"]
        r = Dynamics.theta(c["y"], c["regime"], permutations: perms, seed: 6)
        close(r.theta, c["theta"], l)
        close(r.total, c["total"], l)
        close(r.max_total, c["max_total"], l)
        close(r.ranks, c["ranks"], l)

        for {mine, ref} <- [
              {r.pvalue_left, c["pvalue_left"]},
              {r.pvalue_right, c["pvalue_right"]}
            ],
            {a, b} <- Enum.zip(mine, ref),
            do: assert_p_close(a, b, perms, l)
      end
    end
  end
end
