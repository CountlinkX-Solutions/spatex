defmodule Spatex.Dynamics.Markov do
  @moduledoc """
  Result of `Spatex.Dynamics.markov/2` (`giddy.markov.Markov`), also returned by
  `Spatex.Dynamics.full_rank_markov/2` and `geo_rank_markov/2`.

  Matrices are lists of rows. Mean first passage times and sojourn times can be `:infinity`.

    * `classes` - the class labels in row/column order, `k` their number.
    * `transitions` - counts of moves between consecutive periods; `p` - row-normalized
      probabilities (an all-zero row stays zero unless `fill_empty_classes: true` made it
      absorbing).
    * `steady_state` - the stationary distribution (a list of `k` floats) of an irreducible
      chain; otherwise one distribution per recurrent class (a list of such lists).
    * `mfpt` - mean first passage times (`mfpt[i][j]`: expected steps from `i` to first reach
      `j`; the diagonal is the recurrence time); `sojourn_time` - `1 / (1 - p_ii)`.
      Like giddy's lazily computed properties, these three always treat empty rows as absorbing.
    * `communication_classes`, `recurrent_classes`, `transient_classes` - state indices
      (zero-based), `absorbing_states` - states with `p_ii == 1`.
    * `ranks` - the `n x t` rank states of `full_rank_markov/2` / `geo_rank_markov/2`; `nil`
      for `markov/2`.
  """

  @enforce_keys [:classes, :k]
  defstruct [
    :classes,
    :k,
    :transitions,
    :p,
    :steady_state,
    :mfpt,
    :sojourn_time,
    :communication_classes,
    :recurrent_classes,
    :transient_classes,
    :absorbing_states,
    :ranks,
    fill_empty_classes: false
  ]

  @type value :: float() | :infinity
  @type matrix :: [[float()]]

  @type t :: %__MODULE__{
          classes: [term()],
          k: pos_integer(),
          transitions: matrix(),
          p: matrix(),
          steady_state: [float()] | [[float()]],
          mfpt: [[value()]],
          sojourn_time: [value()],
          communication_classes: [[non_neg_integer()]],
          recurrent_classes: [[non_neg_integer()]],
          transient_classes: [[non_neg_integer()]],
          absorbing_states: [non_neg_integer()],
          ranks: [[pos_integer()]] | nil,
          fill_empty_classes: boolean()
        }
end
