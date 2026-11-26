defmodule Spatex.GWR.Bandwidth do
  @moduledoc """
  Result of `Spatex.GWR.select_bandwidth/4` (`mgwr.sel_bw.Sel_BW`).

    * `bw` - the selected bandwidth: an integer neighbor count for adaptive kernels, a distance
      (rounded to two decimals, like mgwr) for fixed ones, or `:infinity` when the global model
      won the `include_global` check.
    * `score` - the criterion value of the (unrounded) selected bandwidth.
    * `history` - every evaluation as `{bw, score}`, in mgwr's order (duplicates included); empty
      for `search: :scipy`, like mgwr's `sel_hist`.
    * `nfev`, `converged` - the evaluation count and success of SciPy's optimizer
      (`search: :scipy` only, `nil` otherwise). With that search `bw` is SciPy's float, neither
      truncated (adaptive) nor rounded (fixed).
    * `criterion`, `family`, `kernel`, `fixed`, `spherical`, `constant` - the settings of the search.
  """

  defstruct [
    :bw,
    :score,
    :history,
    :criterion,
    :family,
    :kernel,
    :fixed,
    :spherical,
    :constant,
    :nfev,
    :converged
  ]

  @type t :: %__MODULE__{
          bw: pos_integer() | float() | :infinity,
          nfev: pos_integer() | nil,
          converged: boolean() | nil,
          score: float(),
          history: [{pos_integer() | float() | :infinity, float()}],
          criterion: :aicc | :aic | :bic | :cv,
          family: :gaussian | :poisson | :binomial,
          kernel: :bisquare | :gaussian | :exponential,
          fixed: boolean(),
          spherical: boolean(),
          constant: boolean()
        }
end
