defmodule Spatex.Region.RegionKMeans do
  @moduledoc """
  Result of `Spatex.Region.region_kmeans/3`.

    * `labels` - region of each area (the region of `seeds[i]` is `i`).
    * `centroids` - the final region means, one `[x1, ..., xp]` row per region.
    * `iters` - iterations of the improvement loop (spopt starts counting at 1).
    * `seeds` - the seed areas used.
    * `warnings` - set when the improvement loop hit `max_iterations`.
  """

  defstruct labels: [], centroids: [], iters: 0, seeds: [], warnings: []

  @type t :: %__MODULE__{
          labels: [non_neg_integer()],
          centroids: [[float()]],
          iters: pos_integer(),
          seeds: [non_neg_integer()],
          warnings: [String.t()]
        }
end
