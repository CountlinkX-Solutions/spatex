defmodule Spatex.GWRPredictTest do
  use ExUnit.Case, async: true

  alias Spatex.GWR
  alias Spatex.GWR.{Collinearity, Prediction}

  # 7 x 7 jittered lattice with a slope that grows to the east.
  defp sample do
    :rand.seed(:exsss, {11, 22, 33})

    pts =
      for r <- 0..6, c <- 0..6 do
        {c + :rand.uniform() * 0.4, r + :rand.uniform() * 0.4}
      end

    x1 = for _ <- pts, do: :rand.uniform() * 4 - 2
    x2 = for {_, v} <- Enum.with_index(pts), do: :rand.uniform() + rem(v, 5) * 0.3

    y =
      for {{px, py}, a, b} <- Enum.zip([pts, x1, x2]),
          do: 1 + 0.3 * py + (0.5 + 0.2 * px) * a + 0.4 * b

    {pts, y, [x1, x2]}
  end

  describe "predict/6" do
    test "predicting at the calibration points reproduces the fit" do
      {pts, y, x} = sample()
      fit = GWR.fit(pts, y, x, bw: 20)
      p = GWR.predict(pts, y, x, pts, x, bw: 20)

      assert %Prediction{m: 49, n: 49, k: 3, family: :gaussian, bw: 20} = p
      assert_close(p.params, fit.params)
      assert_close(p.bse, fit.bse)
      assert_close(p.tvalues, fit.tvalues)
      assert_close(p.predictions, fit.predy)
      assert_close(p.local_r2, fit.local_r2)
      assert_close(p.scale, fit.scale)
      assert p.warnings == []
    end

    test "predicts at new points given as tuples, lists or an Nx tensor" do
      {pts, y, x} = sample()
      new = [{1.5, 2.5}, {3.0, 3.0}]
      x_new = [[0.5, -1.0], [1.0, 2.0]]
      a = GWR.predict(pts, y, x, new, x_new, bw: 20)

      b =
        GWR.predict(pts, y, x, [[1.5, 2.5], [3.0, 3.0]], Nx.tensor([[0.5, 1.0], [-1.0, 2.0]]),
          bw: 20
        )

      assert a.m == 2 and length(a.params) == 2 and length(hd(a.params)) == 3
      assert_close(a.params, b.params)
    end

    test "exog_scale and exog_resid are used instead of the calibration fit" do
      {pts, y, x} = sample()
      base = GWR.predict(pts, y, x, [{1.5, 2.5}], [[0.5], [1.0]], bw: 20)
      zero = List.duplicate(0.0, 49)

      p =
        GWR.predict(pts, y, x, [{1.5, 2.5}], [[0.5], [1.0]],
          bw: 20,
          exog_scale: 4.0 * base.scale,
          exog_resid: zero
        )

      assert_close(p.params, base.params)
      assert_close(p.bse, Enum.map(base.bse, fn row -> Enum.map(row, &(2 * &1)) end))
      assert_close(p.local_r2, [1.0])
    end

    test "Poisson and binomial predictions are linear predictors, without local R2" do
      {pts, _y, x} = sample()
      counts = for i <- 0..48, do: 1 + rem(i * 7, 5)
      p = GWR.predict(pts, counts, x, [{2.0, 2.0}], [[0.1], [0.2]], bw: 30, family: :poisson)
      assert %Prediction{family: :poisson, local_r2: nil, scale: 1.0} = p
      [row] = p.params
      [pred] = p.predictions
      assert_close(pred, Enum.at(row, 0) + 0.1 * Enum.at(row, 1) + 0.2 * Enum.at(row, 2))

      bin = for i <- 0..48, do: rem(i * 3, 2)
      b = GWR.predict(pts, bin, x, [{2.0, 2.0}], [[0.1], [0.2]], bw: 30, family: :binomial)
      assert b.family == :binomial
    end

    test "constant columns dropped from x are dropped from x_new as well" do
      {pts, y, [x1, x2]} = sample()
      ones = List.duplicate(3.0, 49)
      a = GWR.predict(pts, y, [x1, ones, x2], [{1.5, 2.5}], [[0.5], [3.0], [1.0]], bw: 20)
      b = GWR.predict(pts, y, [x1, x2], [{1.5, 2.5}], [[0.5], [1.0]], bw: 20)
      assert_close(a.params, b.params)
      assert length(a.warnings) == 1
    end

    test "more prediction points than observations are fine" do
      {pts, y, x} = sample()
      new = for i <- 0..59, do: {i * 0.1, 3.0}
      x_new = [List.duplicate(0.5, 60), List.duplicate(1.0, 60)]
      assert GWR.predict(pts, y, x, new, x_new, bw: 20).m == 60
    end

    test "validates its arguments" do
      {pts, y, x} = sample()
      new = [{1.5, 2.5}]
      x_new = [[0.5], [1.0]]
      call = fn opts -> GWR.predict(pts, y, x, new, x_new, opts) end

      assert_raise ArgumentError, ~r/bw is required/, fn -> call.([]) end
      assert_raise ArgumentError, ~r/unknown option.*:foo/, fn -> call.(bw: 20, foo: 1) end

      assert_raise ArgumentError, ~r/both be given or both omitted/, fn ->
        call.(bw: 20, exog_scale: 1.0)
      end

      assert_raise ArgumentError, ~r/both be given or both omitted/, fn ->
        call.(bw: 20, exog_resid: List.duplicate(0.0, 49))
      end

      assert_raise ArgumentError, ~r/exog_resid/, fn ->
        call.(bw: 20, exog_scale: 1.0, exog_resid: [0.0, 1.0])
      end

      assert_raise ArgumentError, ~r/exog_scale must be a positive number/, fn ->
        call.(bw: 20, exog_scale: -1.0, exog_resid: List.duplicate(0.0, 49))
      end

      assert_raise ArgumentError, ~r/x_new has 1 columns, expected 2/, fn ->
        GWR.predict(pts, y, x, new, [[0.5]], bw: 20)
      end

      assert_raise ArgumentError, ~r/x_new column 1/, fn ->
        GWR.predict(pts, y, x, new, [[0.5, 0.7], [1.0, 2.0]], bw: 20)
      end

      assert_raise ArgumentError, ~r/bw must be between 2 and n/, fn -> call.(bw: 1) end
      assert_raise ArgumentError, ~r/offset is only used/, fn -> call.(bw: 20, offset: y) end
      assert_raise ArgumentError, ~r/exceeds max_n/, fn -> call.(bw: 20, max_n: 10) end
    end

    test "a prediction point without neighbors names the point" do
      {pts, y, x} = sample()

      assert_raise ArgumentError, ~r/prediction point 1/, fn ->
        GWR.predict(pts, y, x, [{1.0, 1.0}, {500.0, 500.0}], [[0.5, 0.5], [1.0, 1.0]],
          bw: 0.8,
          fixed: true,
          exog_scale: 1.0,
          exog_resid: List.duplicate(0.0, 49)
        )
      end
    end
  end

  describe "local_collinearity/3" do
    defp wide do
      {pts, _y, [x1, x2]} = sample()
      {pts, [x1, x2]}
    end

    test "returns the four indicators with consistent shapes" do
      {pts, x} = wide()
      r = GWR.local_collinearity(pts, x, bw: 25)

      assert %Collinearity{n: 49, k: 3, constant: true, bw: 25, bws: nil} = r
      assert r.corr_pairs == [{1, 2}]
      assert length(r.corr) == 49 and Enum.all?(r.corr, &(length(&1) == 1))
      assert length(r.vif) == 49 and Enum.all?(r.vif, &(length(&1) == 2))
      assert length(r.condition_number) == 49
      assert length(r.vdp) == 49 and Enum.all?(r.vdp, &(length(&1) == 3))
      assert Enum.all?(r.condition_number, &(&1 >= 1.0))
    end

    test "the multiscale form with equal bandwidths equals the single-bandwidth one" do
      {pts, x} = wide()
      a = GWR.local_collinearity(pts, x, bw: 25)
      b = GWR.local_collinearity(pts, x, bws: [25, 25, 25])
      assert_close(a.condition_number, b.condition_number)
      assert_close(a.vdp, b.vdp)
      assert b.bw == nil and b.bws == [25, 25, 25] and b.corr == nil and b.vif == nil
    end

    test "an infinite bandwidth is accepted" do
      {pts, x} = wide()
      r = GWR.local_collinearity(pts, x, bw: :infinity)
      [first | rest] = r.condition_number
      assert Enum.all?(rest, &(&1 == first))
      assert r.bw == :infinity

      assert %Collinearity{bws: [:infinity, 30, 30]} =
               GWR.local_collinearity(pts, x, bws: [:infinity, 30, 30])
    end

    test "validates its arguments" do
      {pts, [x1, _] = x} = wide()

      assert_raise ArgumentError, ~r/bw .* or bws .* is required/, fn ->
        GWR.local_collinearity(pts, x, [])
      end

      assert_raise ArgumentError, ~r/either bw or bws/, fn ->
        GWR.local_collinearity(pts, x, bw: 25, bws: [25, 25, 25])
      end

      assert_raise ArgumentError, ~r/one bandwidth per design column \(3,/, fn ->
        GWR.local_collinearity(pts, x, bws: [25, 25])
      end

      assert_raise ArgumentError, ~r/at least two explanatory variables/, fn ->
        GWR.local_collinearity(pts, [x1], bw: 25)
      end

      assert_raise ArgumentError, ~r/unknown option.*:family/, fn ->
        GWR.local_collinearity(pts, x, bw: 25, family: :poisson)
      end

      assert_raise ArgumentError, ~r/bw must be between 2 and n/, fn ->
        GWR.local_collinearity(pts, x, bw: 1)
      end

      assert_raise ArgumentError, ~r/exceeds max_n/, fn ->
        GWR.local_collinearity(pts, x, bw: 25, max_n: 10)
      end
    end
  end

  defp assert_close(a, b), do: Spatex.Reference.assert_close(a, b, 1.0e-8)
end
