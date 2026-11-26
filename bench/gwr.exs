# Gaussian GWR bandwidth selection (adaptive bisquare, AICc golden section), n = 1000.
Code.require_file("support.exs", __DIR__)

n = 1_000
pts = Bench.points(n, 11)
x1 = Bench.normal(n, 12)
x2 = Bench.normal(n, 13)
e = Bench.normal(n, 14)

y =
  [pts, x1, x2, e]
  |> Enum.zip()
  |> Enum.map(fn {{px, py}, a, b, u} -> 1.0 + (1.0 + px) * a + (1.0 - py) * b + 0.3 * u end)

Bench.run("gwr select_bandwidth (n=1000, k=3)", fn ->
  Spatex.GWR.select_bandwidth(pts, y, [x1, x2],
    kernel: :bisquare,
    fixed: false,
    criterion: :aicc
  )
end)
