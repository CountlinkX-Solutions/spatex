defmodule Spatex.ESDA.Losh do
  @moduledoc """
  Result of `Spatex.ESDA.losh/3`: local spatial heteroscedasticity (`esda.LOSH`, Ord and Getis
  2012).

  Every per-site field is a list aligned with the graph's node order (`ids`). Field names are
  PySAL's attributes in snake_case (`Hi` is `hi`, `yresid` is `yresid`, `VarHi` is `var_hi`).

    * `hi` - local heteroscedasticity; `ylag` - spatial mean of the neighbors; `yresid` -
      `|y - ylag|^a`; `var_hi` - variance of `hi`.
    * `pval` - chi-square p-values, only with `inference: :chi2` and `a == 2`, else `nil`.

  Islands: esda turns every value `NaN` as soon as one site has no neighbors. Spatex computes
  the statistic over the sites that have neighbors (equal to esda on the island-free graph)
  and reports `:nan` for the islands themselves. Non-finite floats are the atoms `:nan`,
  `:infinity` and `:neg_infinity`.
  """

  @enforce_keys [:n, :a]
  defstruct [:hi, :ylag, :yresid, :var_hi, :pval, :n, :a, :ids, inference: nil]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          hi: [value()],
          ylag: [value()],
          yresid: [value()],
          var_hi: [value()],
          pval: [value()] | nil,
          n: pos_integer(),
          a: float(),
          ids: [term()] | nil,
          inference: :chi2 | nil
        }
end
