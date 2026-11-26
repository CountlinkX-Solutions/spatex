# Exact Fisher-Jenks (k = 5) of 5000 seeded lognormal values.
Code.require_file("support.exs", __DIR__)

y = for v <- Bench.normal(5_000, 3), do: :math.exp(v)

Bench.run("fisher_jenks (n=5000, k=5)", fn -> Spatex.Classify.fisher_jenks(y, k: 5) end)
