defmodule Spatex.PointPatsQuadratTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.PointPats.Quadrat
  alias Spatex.Reference

  @tol 1.0e-9
  @fx Reference.load!("pointpats/quadrat.json")
  @cases @fx["cases"]

  defp pts(list), do: Enum.map(list, fn [x, y] -> {x, y} end)

  defp opts(%{"kwargs" => kw}) do
    Enum.flat_map(kw, fn
      {"shape", s} -> [shape: String.to_atom(s)]
      {"nx", v} -> [nx: v]
      {"ny", v} -> [ny: v]
      {"rectangle_width", v} -> [rectangle_width: v]
      {"rectangle_height", v} -> [rectangle_height: v]
      {"lh", v} -> [lh: v]
    end)
  end

  describe "oracle: rectangle and hexagon grids" do
    for {c, i} <- Enum.with_index(@cases) do
      test "case #{i}: #{c["points_name"]} #{inspect(c["kwargs"])}" do
        c = Enum.at(@cases, unquote(i))
        q = PointPats.quadrat_test(pts(c["points"]), opts(c))
        assert %Quadrat{} = q
        assert q.cell_ids == c["cell_ids"]
        assert q.counts == c["counts"]
        assert q.df == c["df"]
        assert q.num_cells == length(c["counts"])
        Reference.assert_close(q.chi2, c["chi2"], @tol)
        Reference.assert_close(q.chi2_pvalue, c["chi2_pvalue"], @tol)
        Reference.assert_close(q.chi2_contrib, c["chi2_contrib"], @tol)
        Reference.assert_close(Tuple.to_list(q.mbb), c["mbb"], @tol)
        assert q.count_column == c["count_column"]

        if c["kwargs"]["shape"] == "hexagon" do
          assert q.count_row_even == c["count_row_even"]
          assert q.count_row_odd == c["count_row_odd"]
        else
          assert q.count_row == c["count_row"]
          Reference.assert_close(q.rectangle_width, c["rectangle_width"], @tol)
          Reference.assert_close(q.rectangle_height, c["rectangle_height"], @tol)
        end
      end
    end

    test "probed R30 values" do
      c = Enum.find(@cases, &(&1["points_name"] == "r30" and &1["kwargs"]["nx"] == 3))
      q = PointPats.quadrat_test(pts(c["points"]))
      assert q.counts == [4, 3, 3, 4, 3, 3, 3, 6, 1]
      assert_in_delta q.chi2, 4.2, 1.0e-12
      assert q.df == 8
    end
  end

  describe "known deviation: HexagonM.num" do
    test "spatex counts the real cells where pointpats' formula can disagree" do
      c =
        Enum.find(@cases, fn c ->
          c["points_name"] == "r30" and c["kwargs"]["lh"] == 35
        end)

      # pointpats reports num = 7 although the grid has 8 cells
      assert c["pysal_num"] == 7 and length(c["counts"]) == 8
      q = PointPats.quadrat_test(pts(c["points"]), shape: :hexagon, lh: 35)
      assert q.num_cells == 8
      refute Map.has_key?(q, :num)
    end
  end

  describe "realizations" do
    test "Monte Carlo p-value is statistically close to pointpats" do
      for r <- @fx["realizations"] do
        o =
          Enum.flat_map(r["kwargs"], fn
            {"shape", s} -> [shape: String.to_atom(s)]
            {"nx", v} -> [nx: v]
            {"ny", v} -> [ny: v]
            {"lh", v} -> [lh: v]
          end)

        q = PointPats.quadrat_test(pts(r["points"]), [permutations: 499, seed: 7] ++ o)
        Reference.assert_close(q.chi2, r["chi2"], @tol)
        assert length(q.chi2_realizations) == 499
        mean = Enum.sum(q.chi2_realizations) / 499
        # the simulated statistic follows the same null distribution
        se = r["r_sd"] / :math.sqrt(499) * :math.sqrt(2)
        assert abs(mean - r["r_mean"]) <= 6 * se
        assert abs(q.chi2_r_pvalue - r["r_pvalue"]) <= 0.15
        assert q.chi2_r_pvalue >= 1 / 500 and q.chi2_r_pvalue <= 1.0
      end
    end

    test "is deterministic per seed and absent without realizations" do
      p = pts(Enum.at(@cases, 0)["points"])
      a = PointPats.quadrat_test(p, permutations: 50, seed: 1)
      assert a == PointPats.quadrat_test(p, permutations: 50, seed: 1)

      assert a.chi2_realizations !=
               PointPats.quadrat_test(p, permutations: 50, seed: 2).chi2_realizations

      q = PointPats.quadrat_test(p)
      assert q.chi2_realizations == nil and q.chi2_r_pvalue == nil
    end
  end

  describe "validation" do
    test "rejects degenerate extents and bad options" do
      assert_raise ArgumentError, ~r/extent/, fn ->
        PointPats.quadrat_test([{1, 0}, {1, 3}, {1, 5}])
      end

      p = pts(Enum.at(@cases, 0)["points"])
      assert_raise ArgumentError, ~r/shape/, fn -> PointPats.quadrat_test(p, shape: :triangle) end
      assert_raise ArgumentError, ~r/nx/, fn -> PointPats.quadrat_test(p, nx: 0) end

      assert_raise ArgumentError, ~r/lh/, fn ->
        PointPats.quadrat_test(p, shape: :hexagon, lh: 0)
      end
    end
  end
end
