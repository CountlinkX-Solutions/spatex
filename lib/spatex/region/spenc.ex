defmodule Spatex.Region.Spenc do
  @moduledoc """
  Result of `Spatex.Region.spenc/3`.

    * `labels` - cluster of each area, in graph order, `0`-based. Spectral clustering does not
      guarantee that clusters are contiguous on the graph.
    * `n_clusters` - number of distinct clusters in `labels`.
    * `assign_labels` - `:discretize`, `:kmeans` or `:hierarchical`, the strategy used (spopt
      switches to `:hierarchical` itself when `n_clusters` is `:infinity`).
    * `seed` - the seed used for the random stages (a drawn one is recorded here); `nil` for
      `:hierarchical`, which is deterministic.
    * `start_row` - for `:discretize`, the embedding row the random rotation was initialised from
      (spopt draws it from `random_state`; `sklearn`'s `discretize` is a function of it alone).
      `nil` otherwise.
    * `eigenvalues` - the `n_clusters` smallest eigenvalues of the normalized Laplacian, ascending
      (`[]` for `:hierarchical`).
    * `embedding` - the `n x n_clusters` spectral embedding spopt calls `embedding_`: eigenvectors
      divided by the square root of the affinity degrees, each with its largest-magnitude entry
      positive. Rows of lists, or an `Nx.Tensor` with `keep_embedding: :tensor`; `nil` unless
      kept and for `:hierarchical`.
    * `affinity_matrix` - the dense `n x n` affinity (`affinity_matrix_`) when `keep_affinity` is
      set, else `nil`.
    * `warnings` - for example a disconnected graph, which spopt also warns about.
  """

  defstruct labels: [],
            n_clusters: 0,
            assign_labels: :discretize,
            seed: nil,
            start_row: nil,
            eigenvalues: [],
            embedding: nil,
            affinity_matrix: nil,
            warnings: []

  @type t :: %__MODULE__{
          labels: [non_neg_integer()],
          n_clusters: non_neg_integer(),
          assign_labels: :discretize | :kmeans | :hierarchical,
          seed: non_neg_integer() | nil,
          start_row: non_neg_integer() | nil,
          eigenvalues: [float()],
          embedding: [[float()]] | Nx.Tensor.t() | nil,
          affinity_matrix: [[float()]] | Nx.Tensor.t() | nil,
          warnings: [String.t()]
        }
end
