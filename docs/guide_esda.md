# ESDA guide

A short path through exploratory spatial data analysis with Spatex: pick a notion of "neighbour",
measure global and local autocorrelation, and report only what survives multiple testing. The
worked example is [`notebooks/tour.livemd`](https://github.com/CountlinkX-Solutions/spatex/blob/main/notebooks/tour.livemd).

## Choosing weights

Every statistic depends on the `Spatex.Graph` you give it, so the graph is a modelling decision,
not a detail.

| Data | Start with | Notes |
|---|---|---|
| Polygons (tracts, counties) | `Spatex.Graph.build_contiguity/2`, queen (`rook: false`) | Queen counts shared vertices, rook only shared edges. Queen is the usual default; rook suits regular grids. |
| Points, or polygons of very uneven size | `build_knn/3` | Every site gets exactly `k` neighbours, so there are no islands. Not symmetric by construction. |
| Points where distance matters | `build_distance_band/3` or `build_kernel/2` | A band can leave islands in sparse areas; pick a threshold that connects most sites. Distances are Euclidean: project lon/lat data first. |
| Group membership (regions, regimes) | `build_block_contiguity/2` | Sites in the same group are neighbours. |

Check the graph before using it: `Spatex.Graph.isolates/1` (sites without neighbours),
`cardinalities/1` and `symmetric?/1`. Islands cannot be assessed locally (their pseudo p-value is
the smallest possible, an esda quirk Spatex keeps for parity), so think about whether to drop them
or connect them.

## Transforms

`Spatex.Graph.transform/2` rescales the weights: `:b` binary, `:r` row-standardized (each row sums
to one, so the lag `Wy` is the neighbours' average), `:d` double-standardized, `:v` variance
stabilizing. Moran's I and the other ESDA functions default to `transform: :r`, as PySAL does.
Row-standardization stops sites with many neighbours dominating, but it makes the weights
asymmetric. The regression diagnostics in `Spatex.Regression.ols/3` never transform implicitly:
pass `Spatex.Graph.transform(graph, :r)` yourself. `g_local/3` is the exception on the ESDA side:
give it the graph as built and let it apply `:transform`.

## Permutation inference, seeds and determinism

Analytic p-values (`p_norm`, `p_rand`) assume a distribution for the statistic. Pseudo p-values
(`p_sim`) instead reshuffle the data over the sites and count how often a statistic at least as
extreme appears: `(larger + 1) / (permutations + 1)`. So:

* The smallest possible `p_sim` is `1 / (permutations + 1)`. With 999 permutations nothing can be
  below 0.001; use 9999 when you need finer p-values or have many tests to correct.
* Local statistics use conditional randomization: a site's own value stays fixed and its neighbours'
  values are drawn from the other sites.
* Pass `seed:` (a non-negative integer) for a reproducible result. The same seed gives bit-for-bit
  identical simulations whatever the number of threads. Without `seed:` a random one is drawn and
  stored in the result's `seed` field, so any run can be repeated afterwards.
* The generator is not numpy's, so Spatex and PySAL agree on permutation p-values statistically, not
  draw by draw. Statistics and analytic values agree to about 1e-9
  (see [PySAL parity](parity.md) and `notebooks/pysal_comparison.livemd`).

## Multiple testing and FDR

A local statistic gives one test per site. At the 5% level, 49 sites produce about two false
positives even when there is no pattern at all. `Spatex.ESDA.fdr/2` returns the false-discovery-rate
cut-off for a list of p-values: keep the sites whose `p_sim` is below it.

```elixir
gi = Spatex.ESDA.g_local(crime, graph, star: true, permutations: 9999, seed: 1)
cutoff = Spatex.ESDA.fdr(gi.p_sim, 0.05)
hot_or_cold = for {z, p} <- Enum.zip(gi.zs, gi.p_sim), p < cutoff, do: z > 0
```

The result is a threshold, not a vector of adjusted p-values. When no site qualifies it falls back to
the conservative Bonferroni bound `alpha / n`. Apply it to LISA's `p_sim` the same way. Cutting on a
raw 0.05 is acceptable for exploration, but say so.

## Interpreting LISA

`Spatex.ESDA.moran_local/3` places each site in a quadrant of the Moran scatterplot, comparing its
standardized value `z` with the lag of its neighbours:

| Quadrant | Label | Meaning |
|---|---|---|
| 1 | High-High | high value surrounded by high values: a cluster |
| 2 | Low-High | low value among high neighbours: a spatial outlier |
| 3 | Low-Low | low value among low neighbours: a cluster |
| 4 | High-Low | high value among low neighbours: a spatial outlier |

`Spatex.ESDA.MoranLocal.cluster_labels/2` returns these labels where `p_sim` is below the level you
give and `"Insignificant"` elsewhere. Remember:

* Clusters of the same label next to each other are one region, not independent findings, because
  neighbouring sites share neighbours and so their statistics are correlated.
* Quadrants only describe the sign relative to the mean. A High-High site is above average, not
  necessarily large in absolute terms.
* Check the global Moran's I first. If it is near zero, expect few local clusters beyond chance.
* G and G* find hot and cold spots (concentrations of high or low values) but do not identify
  outliers; use both when you want both views.
* Confirm structure with `Spatex.Regression.ols/3` and its Lagrange multiplier tests when you model
  the variable: residual autocorrelation means the model is missing a spatial term.
