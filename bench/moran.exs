# Global Moran's I with 999 permutations on the 100x100 rook lattice (n = 10000, row-standardized).
Code.require_file("support.exs", __DIR__)

g = Bench.lattice_graph(100)
y = Bench.normal(10_000, 2)

Bench.run("moran (n=10000, 999 perms)", fn ->
  Spatex.ESDA.moran(y, g, permutations: 999, seed: 0)
end)
