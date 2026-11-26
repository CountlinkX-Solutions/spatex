defmodule Spatex.Region.MaxP do
  @moduledoc """
  Result of `Spatex.Region.max_p/3`.

    * `labels` - region of each area, `0..p-1` (spopt numbers them `1..p`); `nil` for areas
      removed with `policy: :drop`.
    * `p` - number of regions.
    * `objective` - within-region pairwise cityblock dissimilarity of the returned partition.
    * `n_candidates` - construction results that reached the maximum `p` and were annealed.
    * `seed` - the seed used.
  """

  defstruct labels: [], p: 0, objective: 0.0, n_candidates: 0, seed: 0

  @type t :: %__MODULE__{
          labels: [non_neg_integer() | nil],
          p: pos_integer(),
          objective: float(),
          n_candidates: pos_integer(),
          seed: non_neg_integer()
        }
end
