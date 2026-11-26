# OLS with 3 regressors, default non-spatial diagnostics, LM spatial diagnostics and the Moran of
# the residuals, on the 100x100 rook lattice (n = 10000).
Code.require_file("support.exs", __DIR__)

g = Bench.lattice_graph(100)
x1 = Bench.normal(10_000, 4)
x2 = Bench.normal(10_000, 5)
x3 = Bench.normal(10_000, 6)
e = Bench.normal(10_000, 7)

y =
  [x1, x2, x3, e]
  |> Enum.zip()
  |> Enum.map(fn {a, b, c, u} -> 1.0 + 0.5 * a - 0.3 * b + 0.2 * c + u end)

Bench.run("ols + spatial diagnostics (n=10000, k=4)", fn ->
  Spatex.Regression.ols(y, [x1, x2, x3], graph: g, spat_diag: true, moran: true)
end)
