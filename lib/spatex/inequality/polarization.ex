defmodule Spatex.Inequality.Polarization do
  @moduledoc """
  Result of `Spatex.Inequality.polarization/3` (`inequality.polarization.S`).

    * `statistic` - the spatial polarization index `S = 1 - (c - k) / max(m - k, 1)`, 1 when every
      group forms one connected block of the graph. `c` is the number of connected components of
      the subgraph joining neighbors of the same group, `m` the observations that have a group and
      `k = max(n_g_components, n_a_components)`.
    * `n` - number of observations; `n_a_components` - number of attribute groups (`k` or
      `length(bins) - 1`, nominal even when quantile edges collapse); `n_g_components` - connected
      components of the graph (islands included); `n_i_components` - components of the same-group
      subgraph.
    * `i_labels`, `a_labels`, `g_labels` - per observation, the same-group component, the
      attribute group and the graph component, each numbered from 0 in order of first appearance.
      Observations that have no group (the lowest bin edge, or constant data) have `nil` in
      `i_labels` and `a_labels`.
    * `p_value` - pseudo p-value, (the number of simulated indices at least `statistic`, plus one) over
      `permutations + 1`; `sim` - the
      simulated indices (a list, or a `{permutations}` `Nx.Tensor` of type `{:f, 64}` with
      `keep_simulations: :tensor`); both `nil` when `permutations` is 0 (`sim` also with
      `keep_simulations: false`). `seed` is the seed used (`nil` without permutations).
  """
  @enforce_keys [:statistic]
  defstruct [
    :statistic,
    :n,
    :n_a_components,
    :n_g_components,
    :n_i_components,
    :i_labels,
    :a_labels,
    :g_labels,
    :p_value,
    :sim,
    :seed,
    permutations: 0
  ]

  @type t :: %__MODULE__{
          statistic: float(),
          n: non_neg_integer(),
          n_a_components: pos_integer(),
          n_g_components: pos_integer(),
          n_i_components: non_neg_integer(),
          i_labels: [non_neg_integer() | nil],
          a_labels: [non_neg_integer() | nil],
          g_labels: [non_neg_integer()],
          p_value: float() | nil,
          sim: [float()] | Nx.Tensor.t() | nil,
          seed: non_neg_integer() | nil,
          permutations: non_neg_integer()
        }
end
