# Queen/rook contiguity from WKB polygons: columbus (49) and a 100x100 synthetic lattice.
Code.require_file("support.exs", __DIR__)

{columbus, _} = Bench.columbus()
lattice = Bench.lattice_wkb(100, 100)

Bench.run("build_contiguity columbus (n=49, rook)", fn ->
  Spatex.Graph.build_contiguity(columbus, rook: true)
end)

Bench.run("build_contiguity lattice (n=10000, rook)", fn ->
  Spatex.Graph.build_contiguity(lattice, rook: true)
end)
