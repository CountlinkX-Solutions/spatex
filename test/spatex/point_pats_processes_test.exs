defmodule Spatex.PointPatsProcessesTest do
  use ExUnit.Case, async: true

  alias Spatex.PointPats
  alias Spatex.Reference

  @fx Reference.load!("pointpats/processes.json")

  # pointpats draws from unseeded numpy generators, so the committed fixture holds summaries over
  # many pointpats realizations (test/reference/generate_pointpats_processes.py) and these tests
  # compare Spatex's own seeded summaries with them inside Monte Carlo bands.
  @moduletag :statistical

  defp inside?(points, {x0, y0, x1, y1}),
    do: Enum.all?(points, fn {x, y} -> x >= x0 and x <= x1 and y >= y0 and y <= y1 end)

  defp nn_mean(points) do
    tree = Enum.with_index(points)

    ds =
      for {{x, y}, i} <- tree do
        for({{a, b}, j} <- tree, j != i, do: (a - x) * (a - x) + (b - y) * (b - y))
        |> Enum.min()
        |> :math.sqrt()
      end

    Enum.sum(ds) / length(ds)
  end

  defp mean(list), do: Enum.sum(list) / length(list)

  defp sd(list) do
    m = mean(list)
    :math.sqrt(Enum.sum(Enum.map(list, &((&1 - m) * (&1 - m)))) / (length(list) - 1))
  end

  defp moments(patterns) do
    pts = List.flatten(patterns) |> Enum.map(&Tuple.to_list/1) |> Enum.map(&List.to_tuple/1)
    n = length(pts)
    mx = Enum.sum(for {x, _} <- pts, do: x) / n
    my = Enum.sum(for {_, y} <- pts, do: y) / n
    vx = Enum.sum(for {x, _} <- pts, do: (x - mx) * (x - mx)) / n
    vy = Enum.sum(for {_, y} <- pts, do: (y - my) * (y - my)) / n
    cxy = Enum.sum(for {x, y} <- pts, do: (x - mx) * (y - my)) / n
    {mx, my, vx, vy, cxy}
  end

  defp cov_opt(nil), do: []
  defp cov_opt(c) when is_number(c), do: [cov: c]
  defp cov_opt(rows), do: [cov: rows]

  defp center_opt(nil), do: []
  defp center_opt([x, y]), do: [center: {x, y}]

  describe "normal/2" do
    for {c, i} <- Enum.with_index(@fx["normal"]) do
      test "moments match pointpats (#{c["name"]})" do
        c = Enum.at(@fx["normal"], unquote(i))
        [x0, y0, x1, y1] = c["window"]
        window = {x0, y0, x1, y1}

        opts =
          [size: {c["n"], c["realizations"]}, seed: 3] ++
            cov_opt(c["cov"]) ++ center_opt(c["center"])

        patterns = PointPats.normal(window, opts)

        assert length(patterns) == c["realizations"]
        assert Enum.all?(patterns, &(length(&1) == c["n"] and inside?(&1, window)))

        {mx, my, vx, vy, cxy} = moments(patterns)
        [fmx, fmy] = c["mean"]
        [fvx, fvy] = c["var"]
        assert_in_delta mx, fmx, 0.05
        assert_in_delta my, fmy, 0.05
        assert_in_delta vx, fvx, 0.03 * fvx
        assert_in_delta vy, fvy, 0.03 * fvy
        assert_in_delta cxy, c["cov_xy"], 0.05
      end
    end

    test "is deterministic per seed and returns a single pattern or a list" do
      a = PointPats.normal({0, 0, 10, 5}, size: 50, seed: 4)
      assert length(a) == 50 and Enum.all?(a, &match?({_, _}, &1))
      assert a == PointPats.normal({0, 0, 10, 5}, size: 50, seed: 4)
      assert a != PointPats.normal({0, 0, 10, 5}, size: 50, seed: 5)
      assert [p, q] = PointPats.normal({0, 0, 10, 5}, size: {20, 2}, seed: 4)
      assert length(p) == 20 and p != q
      assert [{_, _}] = PointPats.normal({0, 0, 10, 5}, size: 1, seed: 4)
      assert length(PointPats.normal({0, 0, 10, 5}, seed: 4)) == 100
    end

    test "stays inside a convex window" do
      hull = [{0, 0}, {10, 0}, {5, 8}]
      pts = PointPats.normal({:convex, hull}, size: 300, seed: 2)
      assert length(pts) == 300

      assert Enum.all?(pts, fn {x, y} ->
               y >= -1.0e-9 and y <= 8 * min(x, 10 - x) / 5 + 1.0e-9
             end)
    end

    test "validates its options" do
      w = {0, 0, 10, 5}

      assert_raise ArgumentError, ~r/unknown option :n_seeds/, fn ->
        PointPats.normal(w, n_seeds: 2)
      end

      assert_raise ArgumentError, ~r/size/, fn -> PointPats.normal(w, size: 0) end
      assert_raise ArgumentError, ~r/cov/, fn -> PointPats.normal(w, cov: "x") end

      assert_raise ArgumentError, ~r/cov/, fn ->
        PointPats.normal(w, cov: [[1, 0], [0, 1], [0, 0]])
      end

      assert_raise ArgumentError, ~r/cov/, fn -> PointPats.normal(w, cov: -1) end

      assert_raise ArgumentError, ~r/positive definite/, fn ->
        PointPats.normal(w, cov: [[1, 2], [2, 1]])
      end

      assert_raise ArgumentError, ~r/center/, fn -> PointPats.normal(w, center: {1, :a}) end

      assert_raise ArgumentError, ~r/inside the window/, fn ->
        PointPats.normal(w, center: {1.0e6, 1.0e6}, cov: 1)
      end
    end
  end

  describe "cluster_normal/2" do
    for {c, i} <- Enum.with_index(@fx["cluster_normal"]) do
      test "clustering matches pointpats (n_seeds #{c["n_seeds"]}, cov #{inspect(c["cov"])})" do
        c = Enum.at(@fx["cluster_normal"], unquote(i))
        [x0, y0, x1, y1] = c["window"]
        window = {x0, y0, x1, y1}
        opts = [size: {c["n"], 60}, n_seeds: c["n_seeds"], seed: 8] ++ cov_opt(c["cov"])
        patterns = PointPats.cluster_normal(window, opts)

        assert length(patterns) == 60
        assert Enum.all?(patterns, &(length(&1) == c["n"] and inside?(&1, window)))

        means = Enum.map(patterns, &nn_mean/1)
        %{"mean" => fm, "sd" => fsd, "n" => fn_} = c["nn"]
        # the difference of two sample means: 6 standard errors
        se = :math.sqrt(fsd * fsd / fn_ + sd(means) * sd(means) / 60)
        assert_in_delta mean(means), fm, 6 * se

        if c["cov"] do
          assert mean(means) < 0.6 * c["csr_nn"]["mean"]
        end
      end
    end

    test "is deterministic per seed; clusters start at their centre" do
      w = {0, 0, 100, 100}
      a = PointPats.cluster_normal(w, size: 31, n_seeds: 3, cov: 4, seed: 1)
      assert a == PointPats.cluster_normal(w, size: 31, n_seeds: 3, cov: 4, seed: 1)
      assert length(a) == 31
      assert [p, q] = PointPats.cluster_normal(w, size: {20, 2}, cov: [[4, 1], [1, 4]], seed: 1)
      assert length(p) == 20 and p != q
    end

    test "validates its options" do
      w = {0, 0, 10, 10}

      assert_raise ArgumentError, ~r/too many clusters/, fn ->
        PointPats.cluster_normal(w, size: 3, n_seeds: 5, seed: 1)
      end

      assert_raise ArgumentError, ~r/cov is required/, fn ->
        PointPats.cluster_normal(w, size: 10, n_seeds: 1, seed: 1)
      end

      assert_raise ArgumentError, ~r/unknown option :cluster_radius/, fn ->
        PointPats.cluster_normal(w, cluster_radius: 1)
      end

      assert_raise ArgumentError, ~r/n_seeds/, fn -> PointPats.cluster_normal(w, n_seeds: 0) end
    end
  end

  defp close_pairs(points, r) do
    for {{x, y}, i} <- Enum.with_index(points),
        {{a, b}, j} <- Enum.with_index(points),
        j > i,
        (a - x) * (a - x) + (b - y) * (b - y) <= r * r,
        reduce: 0,
        do: (acc -> acc + 1)
  end

  describe "strauss/2" do
    for {c, i} <- Enum.with_index(@fx["strauss"]) do
      test "inhibition matches pointpats (gamma #{c["gamma"]})" do
        c = Enum.at(@fx["strauss"], unquote(i))
        [x0, y0, x1, y1] = c["window"]
        window = {x0, y0, x1, y1}

        patterns =
          PointPats.strauss(window,
            size: {c["n"], 40},
            gamma: c["gamma"],
            r: c["r"],
            n_iter: c["n_iter"],
            seed: 6
          )

        assert length(patterns) == 40
        assert Enum.all?(patterns, &(length(&1) == c["n"] and inside?(&1, window)))

        pairs = Enum.map(patterns, &close_pairs(&1, c["r"]))
        %{"mean" => fm, "sd" => fsd, "n" => fn_} = c["pairs"]
        se = :math.sqrt(fsd * fsd / fn_ + sd(pairs) * sd(pairs) / 40)
        # six standard errors of the difference, never tighter than one pair
        assert_in_delta mean(pairs), fm, max(6 * se, 1.0)

        nns = Enum.map(patterns, &nn_mean/1)
        %{"mean" => nm, "sd" => nsd} = c["nn"]
        nse = :math.sqrt(nsd * nsd / fn_ + sd(nns) * sd(nns) / 40)
        assert_in_delta mean(nns), nm, 6 * nse
      end
    end

    test "close pairs fall as gamma falls, and gamma = 1 is binomial" do
      w = {0, 0, 1, 1}

      mean_pairs = fn gamma ->
        pats = PointPats.strauss(w, size: {50, 15}, gamma: gamma, r: 0.08, n_iter: 3000, seed: 2)
        mean(Enum.map(pats, &close_pairs(&1, 0.08)))
      end

      [g1, g_mid, g_low] = Enum.map([1.0, 0.2, 0.01], mean_pairs)
      assert g_low < g_mid and g_mid < g1
      # E[pairs] of 50 uniform points is C(50, 2) * (pi r^2 - 8/3 r^3 + r^4 / 2) = 23
      assert g1 > 17 and g1 < 29
    end

    test "is deterministic per seed and returns a single pattern or a list" do
      w = {0, 0, 1, 1}
      a = PointPats.strauss(w, size: 20, n_iter: 300, seed: 4)
      assert length(a) == 20 and inside?(a, w)
      assert a == PointPats.strauss(w, size: 20, n_iter: 300, seed: 4)
      assert a != PointPats.strauss(w, size: 20, n_iter: 300, seed: 5)
      assert [p, q] = PointPats.strauss(w, size: {20, 2}, n_iter: 300, seed: 4)
      assert length(p) == 20 and p != q
      assert [_, _] = PointPats.strauss(w, intensity: 20, size: 2, n_iter: 300, seed: 1)
      assert length(PointPats.strauss(w, n_iter: 100, seed: 1)) == 100
    end

    test "works in a convex window" do
      hull = [{0, 0}, {10, 0}, {5, 8}]
      pts = PointPats.strauss({:convex, hull}, size: 30, r: 0.5, gamma: 0.1, n_iter: 500, seed: 3)
      assert length(pts) == 30

      assert Enum.all?(pts, fn {x, y} -> y >= -1.0e-9 and y <= 8 * min(x, 10 - x) / 5 + 1.0e-9 end)
    end

    test "validates its options" do
      w = {0, 0, 1, 1}
      assert_raise ArgumentError, ~r/gamma/, fn -> PointPats.strauss(w, gamma: -1) end
      assert_raise ArgumentError, ~r/gamma/, fn -> PointPats.strauss(w, gamma: "a") end
      assert_raise ArgumentError, ~r/\br\b/, fn -> PointPats.strauss(w, r: 0) end
      assert_raise ArgumentError, ~r/n_iter/, fn -> PointPats.strauss(w, n_iter: -1) end
      assert_raise ArgumentError, ~r/max_iter/, fn -> PointPats.strauss(w, max_iter: 0) end
      assert_raise ArgumentError, ~r/unknown option :cov/, fn -> PointPats.strauss(w, cov: 1) end

      assert_raise ArgumentError, ~r/too large|gamma/, fn ->
        PointPats.strauss(w, size: 50, gamma: 0.0, r: 0.5, n_iter: 2000, seed: 1)
      end
    end
  end
end
