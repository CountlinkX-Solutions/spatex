defmodule Spatex.PointPats.DistanceTest do
  @moduledoc """
  Result of `Spatex.PointPats.g_test/2` and its siblings: an observed distance function with a
  seeded Monte Carlo test against complete spatial randomness (`pointpats.distance_statistics`
  `*_test`).

    * `function` - `:f`, `:g`, `:j`, `:k` or `:l`.
    * `support` - the distances at which the statistic is evaluated.
    * `statistic` - the observed values at `support`.
    * `pvalue` - per-support pseudo p-value `min(p, 1 - p)` with
      `p = (count + 1) / (permutations + 1)`, `count` being the number of simulated values
      at least as large as the observed one, as in pointpats, so it is at most
      0.5 and can be `0.0` (for example at distance 0, where every statistic is 0). `:nan` when
      `permutations` is 0.
    * `simulations` - `permutations` curves (lists aligned with `support`) when
      `keep_simulations: true`, a `{permutations, length(support)}` `Nx.Tensor` of type
      `{:f, 64}` with `keep_simulations: :tensor`, else `nil`.
    * `envelope_low`, `envelope_high` - per-support minimum and maximum over the simulations
      (a Spatex convenience, not part of pointpats); `:nan` without simulations.

  Non-finite floats are the atoms `:nan`, `:infinity` and `:neg_infinity`.
  """

  defstruct [
    :function,
    :support,
    :statistic,
    :pvalue,
    :simulations,
    :permutations,
    :seed,
    :envelope_low,
    :envelope_high
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          function: :f | :g | :j | :k | :l,
          support: [float()],
          statistic: [value()],
          pvalue: [value()],
          simulations: [[value()]] | Nx.Tensor.t() | nil,
          permutations: non_neg_integer(),
          seed: non_neg_integer(),
          envelope_low: [value()],
          envelope_high: [value()]
        }
end
