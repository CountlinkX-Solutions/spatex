# k-nearest-neighbor graph (k = 8) of seeded uniform random points.
Code.require_file("support.exs", __DIR__)

for n <- [10_000, 100_000] do
  pts = Bench.points(n, 1)
  Bench.run("build_knn (n=#{n}, k=8)", fn -> Spatex.Graph.build_knn(pts, 8) end)
end
