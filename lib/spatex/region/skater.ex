defmodule Spatex.Region.Skater do
  @moduledoc """
  Result of `Spatex.Region.skater/3`.

    * `labels` - region of each area, numbered by the lowest area they contain.
    * `n_regions` - number of regions found (fewer than requested when `floor` blocks every cut;
      see `warnings`).
    * `msf_edges` - the pruned spanning forest as `{i, j}` pairs, `i < j`, in row-major order.
    * `score` - sum over regions of the distances of their areas to the region mean.
    * `trace` - with `trace: true`, one `%{labels, in_node, out_node, score}` map per pruning
      step, starting with the initial forest (`in_node`/`out_node` `nil`, `score` `:infinity`).
    * `warnings` - messages equivalent to spopt's `OptimizeWarning`s.
  """

  defstruct labels: [], n_regions: 0, msf_edges: [], score: 0.0, trace: [], warnings: []

  @type t :: %__MODULE__{
          labels: [non_neg_integer()],
          n_regions: pos_integer(),
          msf_edges: [{non_neg_integer(), non_neg_integer()}],
          score: float(),
          trace: [map()],
          warnings: [String.t()]
        }
end
