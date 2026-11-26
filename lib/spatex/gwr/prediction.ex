defmodule Spatex.GWR.Prediction do
  @moduledoc """
  Result of `Spatex.GWR.predict/6`: local coefficients and predictions at new locations, after
  `GWR.predict` and the `GWRResults` it returns in mgwr 2.2.1.

  Matrices (`params`, `bse`, `tvalues`) are lists of `m` rows with `k` entries, the intercept
  first when the model has one; per-point vectors have `m` entries. Non-finite numbers are the
  atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `m` - the number of prediction points; `n`, `k` - calibration observations and design
      columns (the intercept included).
    * `family`, `bw`, `kernel`, `fixed`, `spherical`, `constant` - the model that was calibrated.
      An adaptive `bw` is the integer number of nearest calibration points.
    * `params` - the local coefficients at each prediction point.
    * `bse`, `tvalues` - their standard errors and t-values. The standard errors scale the local
      covariance by `scale` (the calibration residual variance, or `exog_scale`).
    * `predictions` - `sum_j x_new_ij * params_ij`. For the Gaussian family this is the predicted
      response; like mgwr, for `:poisson` and `:binomial` it is the *linear predictor* (log mean
      or log-odds), not the mean.
    * `local_r2` - the geographically weighted R-squared at each prediction point from the
      calibration residuals (Gaussian only; `nil` for the other families, where mgwr raises).
    * `scale` - the scale used for the covariances: the calibration `sigma2` for the Gaussian
      family, `1.0` for the others, or the `exog_scale` that was passed.
    * `warnings` - messages such as a truncated adaptive bandwidth.
  """

  defstruct [
    :m,
    :n,
    :k,
    :family,
    :bw,
    :kernel,
    :fixed,
    :spherical,
    :constant,
    :params,
    :bse,
    :tvalues,
    :predictions,
    :local_r2,
    :scale,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          m: pos_integer(),
          n: pos_integer(),
          k: pos_integer(),
          family: :gaussian | :poisson | :binomial,
          bw: pos_integer() | float() | :infinity,
          kernel: :bisquare | :gaussian | :exponential,
          fixed: boolean(),
          spherical: boolean(),
          constant: boolean(),
          params: [[value()]],
          bse: [[value()]],
          tvalues: [[value()]],
          predictions: [value()],
          local_r2: [value()] | nil,
          scale: value(),
          warnings: [String.t()]
        }
end
