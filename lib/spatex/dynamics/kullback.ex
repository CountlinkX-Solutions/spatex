defmodule Spatex.Dynamics.Kullback do
  @moduledoc """
  Result of `Spatex.Dynamics.kullback/2` (`giddy.markov.kullback`), Kullback's information test of
  Markov homogeneity across strata.

    * `statistic` - the conditional homogeneity statistic (giddy's `"Conditional homogeneity"`).
    * `dof` - degrees of freedom, `r (s - 1) (r - 1)` for `s` strata and `r` classes.
    * `p_value` - `1 - chi2.cdf(statistic, dof)`; `:nan` when `dof` is 0.
  """
  @enforce_keys [:statistic, :dof, :p_value]
  defstruct [:statistic, :dof, :p_value]

  @type t :: %__MODULE__{
          statistic: float() | :nan | :infinity | :neg_infinity,
          dof: non_neg_integer(),
          p_value: float() | :nan
        }
end
