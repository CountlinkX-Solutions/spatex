defmodule Spatex.Region.RandomRegion do
  @moduledoc """
  Result of `Spatex.Region.random_region/2`: one random partition of the areas.

    * `labels` - region of each area, `0..n_regions-1` in construction order; `nil` when the
      request was infeasible.
    * `regions` - the areas (0-based positions) of each region, in the order spopt lists them
      (construction order; area order inside a region is not meaningful). `[]` when infeasible.
    * `n_regions` - number of regions (`0` when infeasible).
    * `feasible` - `false` when no contiguous solution was found in `maxiter` attempts (spopt's
      `feasible = False`, with empty `regions`).
    * `attempts` - attempts used (always `1` without a graph).
    * `seed` - the seed used.
  """

  defstruct labels: nil, regions: [], n_regions: 0, feasible: true, attempts: 1, seed: 0

  @type t :: %__MODULE__{
          labels: [non_neg_integer()] | nil,
          regions: [[non_neg_integer()]],
          n_regions: non_neg_integer(),
          feasible: boolean(),
          attempts: pos_integer(),
          seed: non_neg_integer()
        }
end
