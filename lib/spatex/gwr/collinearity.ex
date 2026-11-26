defmodule Spatex.GWR.Collinearity do
  @moduledoc """
  Result of `Spatex.GWR.local_collinearity/3`: local indicators of multicollinearity within
  the geographically weighted design matrix, after `GWRResults.local_collinearity` and
  `MGWRResults.local_collinearity` of mgwr 2.2.1.

  Matrices are lists of `n` rows (one per location). Non-finite numbers are the atoms `:nan`,
  `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - locations and design columns (the intercept included).
    * `constant`, `kernel`, `fixed`, `spherical` - the weighting that was used.
    * `bw` - the single bandwidth (a GWR), `bws` - one bandwidth per design column (an MGWR);
      exactly one of them is set. Adaptive bandwidths are integer neighbor counts.
    * `corr` - local correlation coefficients between the explanatory variables (the intercept
      excluded): one column per pair of `corr_pairs`, which lists the pairs as `{i, j}` with
      `i < j` in lexicographic order, `i` and `j` being 0-based design columns. `nil` for `bws`.
    * `vif` - local variance inflation factors, one column per explanatory variable (the
      intercept excluded, so `k - 1` columns with an intercept). `nil` for `bws`.
    * `condition_number` - the local condition number of the weighted design (the largest over
      the smallest singular value of the column-scaled weighted design matrix): one value per
      location. Belsley, Kuh and Welsch suggest values above 30 indicate harmful collinearity.
    * `vdp` - the local variance-decomposition proportions of the smallest singular value, `k`
      columns, in design column order (the intercept first when the model has one).
    * `warnings` - messages such as a truncated adaptive bandwidth.
  """

  defstruct [
    :n,
    :k,
    :constant,
    :kernel,
    :fixed,
    :spherical,
    :bw,
    :bws,
    :corr,
    :corr_pairs,
    :vif,
    :condition_number,
    :vdp,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type bandwidth :: pos_integer() | float() | :infinity

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          constant: boolean(),
          kernel: :bisquare | :gaussian | :exponential,
          fixed: boolean(),
          spherical: boolean(),
          bw: bandwidth() | nil,
          bws: [bandwidth()] | nil,
          corr: [[value()]] | nil,
          corr_pairs: [{non_neg_integer(), non_neg_integer()}] | nil,
          vif: [[value()]] | nil,
          condition_number: [value()],
          vdp: [[value()]],
          warnings: [String.t()]
        }
end
