defmodule Spatex.Inequality.TheilD do
  @moduledoc """
  Result of `Spatex.Inequality.theil_d/3` (`inequality.TheilD` / `TheilDSim`).

  `t` is Theil's T, `bg` its between-group part, `wg = t - bg` the within-group part. For a
  vector input they are floats, for an `n x t` matrix lists with one value per column.
  `bg_pvalue` is the permutation pseudo p-value of `bg` (the share of replicates with `bg*` at least `bg`, counting the observation: `(1 + count) / (P + 1)`) and is
  `nil` with `permutations: 0`.
  """
  defstruct [:t, :bg, :wg, :bg_pvalue, :seed, permutations: 0]

  @type t :: %__MODULE__{
          t: float() | [float()],
          bg: float() | [float()],
          wg: float() | [float()],
          bg_pvalue: float() | [float()] | nil,
          seed: non_neg_integer() | nil,
          permutations: non_neg_integer()
        }
end
