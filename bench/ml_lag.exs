# Maximum-likelihood spatial lag, Ord log-determinant, on a 50x50 rook lattice (n = 2500).
Code.require_file("support.exs", __DIR__)

g = Bench.lattice_graph(50)
n = 2_500
x1 = Bench.normal(n, 8)
x2 = Bench.normal(n, 9)
e = Bench.normal(n, 10)

y =
  [x1, x2, e]
  |> Enum.zip()
  |> Enum.map(fn {a, b, u} -> 1.0 + 0.5 * a - 0.3 * b + u end)

Bench.run("ml_lag ord (n=2500)", fn ->
  Spatex.Regression.ml_lag(y, [x1, x2], g, method: :ord)
end)
