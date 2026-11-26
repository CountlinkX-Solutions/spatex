defmodule Spatex.Dynamics.SpatialMarkov do
  @moduledoc """
  Result of `Spatex.Dynamics.spatial_markov/3` (`giddy.markov.Spatial_Markov`).

  Field names are giddy's attributes in snake_case; the upper-case conditional attributes carry
  a `_cond` suffix: `T` is `t`, `P` is `p_cond`, `S` is `s_cond`, `F` is `f_cond`.

    * `k`, `m` - number of classes of the variable and of its spatial lag; `cutoffs` /
      `lag_cutoffs` - class upper bounds without the final infinite one (`nil` when classified
      period by period or `discrete: true`); `class_ids` / `lclass_ids` - `n x t` class indices.
    * `transitions`, `p` - the classic Markov counts and probabilities; `t` - the `m` conditional
      `k x k` count matrices, `p_cond` their row-normalized versions.
    * `s`, `f` - steady state and mean first passage times of `p`; `s_cond`, `f_cond` - the
      same per lag class. They are `nil` for a matrix with an empty row unless
      `fill_empty_classes: true` (giddy raises there).
    * `q`, `q_p_value`, `lr`, `lr_p_value`, `dof_hom` - the Bickenbach-Bode homogeneity tests of
      `t`; `shtest` - per lag class `{chi2, p_value, dof}` for the difference between the
      conditional and overall steady states; `chi2` - per lag class `{chi2, p_value, dof}` of
      the conditional counts against the overall probabilities; `x2`, `x2_pvalue`, `x2_dof` -
      their sum.
    * `x2_rpvalue`, `x2_realizations` - permutation pseudo p-value and replicates of `x2`
      (`nil` with `permutations: 0`).
  """

  @enforce_keys [:k, :m]
  defstruct [
    :k,
    :m,
    :classes,
    :cutoffs,
    :lag_cutoffs,
    :class_ids,
    :lclass_ids,
    :transitions,
    :p,
    :t,
    :p_cond,
    :s,
    :s_cond,
    :f,
    :f_cond,
    :q,
    :q_p_value,
    :lr,
    :lr_p_value,
    :dof_hom,
    :shtest,
    :chi2,
    :x2,
    :x2_pvalue,
    :x2_dof,
    :x2_rpvalue,
    :x2_realizations,
    :seed,
    discrete: false,
    permutations: 0,
    fill_empty_classes: false
  ]

  @type test :: {float() | :nan, float() | :nan, integer()}

  @type t :: %__MODULE__{
          k: pos_integer(),
          m: pos_integer(),
          classes: [term()],
          cutoffs: [float()] | nil,
          lag_cutoffs: [float()] | nil,
          class_ids: [[non_neg_integer()]],
          lclass_ids: [[non_neg_integer()]],
          transitions: [[float()]],
          p: [[float()]],
          t: [[[float()]]],
          p_cond: [[[float()]]],
          s: [float()] | [[float()]] | nil,
          s_cond: [[float()] | [[float()]] | nil],
          f: [[float() | :infinity]] | nil,
          f_cond: [[[float() | :infinity]] | nil],
          q: float() | :nan,
          q_p_value: float() | :nan,
          lr: float() | :nan,
          lr_p_value: float() | :nan,
          dof_hom: integer(),
          shtest: [test() | nil],
          chi2: [test()],
          x2: float(),
          x2_pvalue: float() | :nan,
          x2_dof: non_neg_integer(),
          x2_rpvalue: float() | nil,
          x2_realizations: [float()] | nil,
          seed: non_neg_integer() | nil,
          discrete: boolean(),
          permutations: non_neg_integer(),
          fill_empty_classes: boolean()
        }
end
