defmodule Spatex.Region.Sa3 do
  @moduledoc """
  Result of `Spatex.Region.sa3/3`.

    * `labels` - cluster of each area, in graph order: `0`-based and numbered across connected
      components, or `-1` for noise (areas in components no larger than `min_cluster_size`, and
      areas the extraction leaves out). spopt's `labels_` uses the same numbering and noise value.
    * `n_clusters` - number of clusters (noise is not a cluster).
    * `n_noise` - number of noise areas.
    * `n_connected_components` - connected components of the graph. Components are clustered
      separately; one with at most `min_cluster_size` areas is all noise.
    * `components` - one `%{nodes: [non_neg_integer()], linkage: [{left, right, distance, size}]}`
      for every component that was clustered, in component order (by lowest area). `nodes` are
      the areas in ascending order and `linkage` is the scipy-style linkage matrix SA3 extracts
      from, after spopt's monotonicity correction: leaves are `0..length(nodes) - 1` (positions in
      `nodes`), merge `i` is node `length(nodes) + i`, `distance` is scikit-learn's Ward distance
      (`sqrt(2 * inertia)`) and `size` the number of areas merged.
  """

  defstruct labels: [], n_clusters: 0, n_noise: 0, n_connected_components: 1, components: []

  @type linkage_row ::
          {non_neg_integer(), non_neg_integer(), float(), pos_integer()}

  @type component :: %{nodes: [non_neg_integer()], linkage: [linkage_row()]}

  @type t :: %__MODULE__{
          labels: [integer()],
          n_clusters: non_neg_integer(),
          n_noise: non_neg_integer(),
          n_connected_components: pos_integer(),
          components: [component()]
        }
end
