defmodule Spatex.GWR.Result do
  @moduledoc """
  Result of `Spatex.GWR.fit/4`, with the attribute names of `mgwr.gwr.GWRResults` (mgwr 2.2.1)
  in snake_case, for the Gaussian, Poisson and binomial families.

  Per-location matrices (`params`, `bse`, `tvalues`) are lists of `n` rows with `k` entries,
  the intercept first when the model has one. Per-location vectors have `n` entries. Non-finite
  numbers are the atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and design columns (the intercept included).
    * `family` - `:gaussian`, `:poisson` or `:binomial`.
    * `bw`, `kernel`, `fixed`, `spherical`, `constant` - the model that was fitted. An adaptive
      `bw` is the integer number of nearest neighbors (the focal point included).
    * `params`, `bse`, `tvalues` - local coefficients, standard errors and t-values.
    * `local_r2` (Gaussian only), `predy` (the fitted means), `resid` (response residuals),
      `influ` (diagonal of the hat matrix), `std_res`, `cooks_d`.
    * `resid_ss`, `tr_s` (`tr(S)`), `tr_sts` (`tr(S'S)`), `enp` (effective number of parameters),
      `sigma2`, `df_resid` (`n - 2 tr(S) + tr(S'S)`).
    * `scale` - the residual variance for the Gaussian family, `1.0` otherwise.
    * `llf`, `aic`, `aicc`, `bic`, `cv` - log-likelihood and information criteria. The Gaussian
      ones follow mgwr's `get_AIC`/`get_AICc`/`get_BIC`; for Poisson and binomial they are
      deviance based: `aic = deviance + 2 tr(S)`, `aicc = aic + 2 tr(S) (tr(S) + 1) / (n - tr(S) - 1)`,
      `bic = deviance + tr(S) ln(n)`.
    * `r2`, `adj_r2` - global (deviance based) R-squared and its adjusted form (Gaussian only).
    * `d2`, `adj_d2` - the percent deviance explained `1 - global_deviance / null_deviance` and
      its adjusted form, for every family; `null_deviance`, `global_deviance` are its parts.
    * `deviance`, `resid_deviance`, `p_dev` - local deviance of the weighted-mean model, local
      residual deviance and the local percent deviance `1 - resid_deviance / deviance` (GLM
      families; `nil` for Gaussian). `iterations` - IWLS iterations per location (GLM only).
    * `adj_alpha` - multiple-testing corrected alphas for the 0.1, 0.05 and 0.001 levels
      (da Silva and Fotheringham, 2016); `critical_tval` - the t critical value for the
      corrected 0.05 level (`n - 1` degrees of freedom, as in mgwr).
    * `hat_matrix` - the `n x n` hat matrix `S` (`S y` are `predy`, `diag(S)` is `influ`) as a list
      of rows, or an `Nx.Tensor` with `hat_matrix: :tensor`; `nil` unless `fit/4` was asked for it.
    * `warnings` - messages such as a truncated adaptive bandwidth.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
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
    :local_r2,
    :predy,
    :resid,
    :influ,
    :std_res,
    :cooks_d,
    :resid_ss,
    :tr_s,
    :tr_sts,
    :enp,
    :sigma2,
    :df_resid,
    :llf,
    :aic,
    :aicc,
    :bic,
    :cv,
    :r2,
    :adj_r2,
    :scale,
    :d2,
    :adj_d2,
    :null_deviance,
    :global_deviance,
    :deviance,
    :resid_deviance,
    :p_dev,
    :iterations,
    :adj_alpha,
    :critical_tval,
    :hat_matrix,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
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
          local_r2: [value()] | nil,
          predy: [value()],
          resid: [value()],
          influ: [value()],
          std_res: [value()],
          cooks_d: [value()],
          resid_ss: value(),
          tr_s: value(),
          tr_sts: value(),
          enp: value(),
          sigma2: value(),
          df_resid: value(),
          llf: value(),
          aic: value(),
          aicc: value(),
          bic: value(),
          cv: value(),
          r2: value() | nil,
          adj_r2: value() | nil,
          scale: value(),
          d2: value(),
          adj_d2: value(),
          null_deviance: value(),
          global_deviance: value(),
          deviance: [value()] | nil,
          resid_deviance: [value()] | nil,
          p_dev: [value()] | nil,
          iterations: [pos_integer()] | nil,
          adj_alpha: [value()],
          critical_tval: value(),
          hat_matrix: [[value()]] | Nx.Tensor.t() | nil,
          warnings: [String.t()]
        }
end
