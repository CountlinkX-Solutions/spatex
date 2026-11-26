defmodule Spatex.ESDA.JoinCountsLocal do
  @moduledoc """
  Result of `Spatex.ESDA.join_counts_local/3`: univariate local join counts
  (`esda.Join_Counts_Local`, Anselin and Li 2019).

  Every per-site field is a list aligned with the graph's node order (`ids`).

    * `ljc` - number of neighbors with `y = 1` of every site with `y = 1` (0 elsewhere,
      islands included), as floats.
    * `p_sim` - pseudo p-value of the conditional randomization, `:nan` where `ljc` is 0 (as
      in esda); `sim` - the simulated counts, one list of `permutations` values per site
      (esda's `rjoins`). Both are `nil` with `permutations: 0`; `sim` is also `nil` with
      `keep_simulations: false`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{n, permutations}` instead of nested lists.
  """

  @enforce_keys [:n, :permutations]
  defstruct [:ljc, :p_sim, :sim, :n, :permutations, :seed, :ids, alternative: :directed]

  @type t :: %__MODULE__{
          ljc: [float()],
          p_sim: [float() | :nan] | nil,
          sim: [[float()]] | Nx.Tensor.t() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()] | nil,
          alternative: atom()
        }
end
