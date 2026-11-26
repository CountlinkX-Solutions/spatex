defmodule Spatex.ESDA.GearyLocalMV do
  @moduledoc """
  Result of `Spatex.ESDA.geary_local_mv/3`: multivariate local Geary
  (`esda.Geary_Local_MV`, Anselin 2019).

  Every per-site field is a list aligned with the graph's node order (`ids`).

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{n, permutations}` instead of nested lists.

    * `local_g` - `sum_v sum_j w_ij (z_vi - z_vj)^2 / k` over the `k` standardised variables
      (ddof 0) and the row-standardised graph; 0 for islands.
    * `p_sim` - folded pseudo p-value of the conditional randomization (esda's only test);
      `sim` - the simulated statistics, one list of `permutations` values per site. Both are
      `nil` with `permutations: 0`; `sim` is also `nil` with `keep_simulations: false`.
  """

  @enforce_keys [:n, :k, :permutations]
  defstruct [:local_g, :p_sim, :sim, :n, :k, :permutations, :seed, :ids]

  @type t :: %__MODULE__{
          local_g: [float()],
          p_sim: [float()] | nil,
          sim: [[float()]] | Nx.Tensor.t() | nil,
          n: non_neg_integer(),
          k: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()]
        }
end
