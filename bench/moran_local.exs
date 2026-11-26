# Local Moran (LISA) with 999 permutations on the 100x100 rook lattice (n = 10000).
Code.require_file("support.exs", __DIR__)

g = Bench.lattice_graph(100)
y = Bench.normal(10_000, 2)

# The simulation matrix kept as nested lists (keep_simulations: true).
Bench.run("moran_local (n=10000, 999 perms)", fn ->
  Spatex.ESDA.moran_local(y, g, permutations: 999, seed: 0, keep_simulations: true)
end)

# Without the n x 999 simulation matrix (keep_simulations: false), to separate the inference cost
# from the cost of materializing the simulations.
Bench.run("moran_local (n=10000, 999 perms, no sims kept)", fn ->
  Spatex.ESDA.moran_local(y, g, permutations: 999, seed: 0, keep_simulations: false)
end)

# The matrix kept as an Nx.Tensor built straight from the native binary (no list decoding).
Bench.run("moran_local (n=10000, 999 perms, keep_simulations: :tensor)", fn ->
  Spatex.ESDA.moran_local(y, g, permutations: 999, seed: 0, keep_simulations: :tensor)
end)
