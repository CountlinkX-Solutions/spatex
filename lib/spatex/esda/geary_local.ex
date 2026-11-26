defmodule Spatex.ESDA.GearyLocal do
  @moduledoc """
  Result of `Spatex.ESDA.geary_local/3`: local Geary (`esda.Geary_Local`, Anselin 1995).

  Every per-site field is a list aligned with the graph's node order (`ids`). Field names are
  PySAL's attributes in snake_case (`localG` is `local_g`, `rlocalG` is `sim`).

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{n, permutations}` instead of nested lists.

    * `local_g` - `sum_j w_ij (z_i - z_j)^2` over the row-standardized graph and the
      standardized values (ddof 0). Small values mean similarity with the neighbors.
    * `p_sim` - pseudo p-value of the conditional randomization (see the `:alternative`
      option); `sim` - the simulated statistics, one list of `permutations` values per site.
      Both are `nil` with `permutations: 0`; `sim` is also `nil` with
      `keep_simulations: false`.
    * `labs` - only with `labels: true`: `1` outlier, `2` cluster, `3` other, `4` not
      significant at `sig`, `nil` where no rule applies (a tie with the mean).

  Islands keep their place in the output with `local_g` 0 (esda drops them from `localG` and
  misaligns it with `p_sim`) and, as in esda, the smallest possible `p_sim`,
  `1 / (permutations + 1)`.
  """

  @enforce_keys [:n, :permutations]
  defstruct [
    :local_g,
    :p_sim,
    :sim,
    :labs,
    :n,
    :permutations,
    :seed,
    :ids,
    sig: 0.05,
    alternative: :directed
  ]

  @type t :: %__MODULE__{
          local_g: [float()],
          p_sim: [float()] | nil,
          sim: [[float()]] | Nx.Tensor.t() | nil,
          labs: [1..4 | nil] | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()] | nil,
          sig: float(),
          alternative: atom()
        }
end
