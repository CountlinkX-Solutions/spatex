defmodule Spatex.Region.Ward do
  @moduledoc """
  Result of `Spatex.Region.ward_spatial/3`.

    * `labels` - cluster of each area, numbered like scikit-learn does.
    * `children` - the merge tree: merge `i` joins nodes `{a, b}` (`a < b`) into node `n + i`.
      With large `n_clusters` scikit-learn stops the tree early and only `n - n_clusters`
      merges are present.
    * `distances` - `sqrt(2 * inertia)` of every merge. They are not monotone, because only
      neighboring clusters can merge.
    * `n_connected_components` - components of the input graph. Above one, the graph was
      completed with links between the most similar areas of each pair of components (in
      attribute space, as scikit-learn does).
    * `n_leaves` - number of areas.
    * `warnings` - messages equivalent to scikit-learn's warnings.
  """

  defstruct labels: [],
            children: [],
            distances: [],
            n_connected_components: 1,
            n_leaves: 0,
            warnings: []

  @type t :: %__MODULE__{
          labels: [non_neg_integer()],
          children: [{non_neg_integer(), non_neg_integer()}],
          distances: [float()],
          n_connected_components: pos_integer(),
          n_leaves: pos_integer(),
          warnings: [String.t()]
        }
end
