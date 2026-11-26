defmodule Spatex.Dynamics.LisaMarkov do
  @moduledoc """
  Result of `Spatex.Dynamics.lisa_markov/3` (`giddy.markov.LISA_Markov`).

  A Markov chain on the local Moran quadrants (`classes` are `1` High-High, `2` Low-High,
  `3` Low-Low, `4` High-Low) with the `Spatex.Dynamics.Markov` fields `transitions`, `p`,
  `steady_state`, `mfpt`, `sojourn_time` and the class structure, plus:

    * `q` - `n x t` quadrants.
    * `move_types` - `n x (t - 1)` transition type `4 (q_s - 1) + q_(s+1)` in `1..16`.
    * `p_values`, `significant_moves` - `n x t` simulated p-values (`p_z_sim`) and the `n x
      (t - 1)` move types refined by the significance of both end points
      (`16 * idx + move_type`, `idx` 0 when both are significant, 1 origin only, 2 destination
      only, 3 neither), for `permutations > 0`; `nil` otherwise.
    * `chi_2`, `expected_t` - test of independence between the own and the lag moves
      (`{chi2, p_value, dof}`) and the expected transition counts. Unlike giddy, whose marginal
      chains run across neighboring locations because it hands `Markov` a `(t, n)` array,
      Spatex builds them across periods (see `docs/parity.md`).
  """

  @enforce_keys [:q, :classes]
  defstruct [
    :q,
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
    :move_types,
    :significant_moves,
    :p_values,
    :chi_2,
    :expected_t,
    :seed,
    significance_level: 0.05,
    permutations: 0,
    fill_empty_classes: false
  ]

  @type t :: %__MODULE__{
          q: [[1..4]],
          classes: [1..4],
          k: 4,
          transitions: [[float()]],
          p: [[float()]],
          steady_state: [float()] | [[float()]],
          mfpt: [[float() | :infinity]],
          sojourn_time: [float() | :infinity],
          communication_classes: [[non_neg_integer()]],
          recurrent_classes: [[non_neg_integer()]],
          transient_classes: [[non_neg_integer()]],
          absorbing_states: [non_neg_integer()],
          move_types: [[1..16]],
          significant_moves: [[1..64]] | nil,
          p_values: [[float() | :nan]] | nil,
          chi_2: {float(), float() | :nan, integer()},
          expected_t: [[float()]],
          seed: non_neg_integer() | nil,
          significance_level: float(),
          permutations: non_neg_integer(),
          fill_empty_classes: boolean()
        }
end
