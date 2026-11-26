defmodule Spatex.Regression.MLLag do
  @moduledoc """
  Result of `Spatex.Regression.ml_lag/4`: the maximum-likelihood spatial lag model
  `y = rho W y + X beta + e`, with the attribute names of `spreg.ML_Lag` (spreg 1.9.1) in
  snake_case.

  Matrices (`vm`, `vm1`) are lists of rows. `betas`, `std_err` and `z_stat` list the regressors
  first and `rho` last, matching `name_x` (whose last entry is `"W_<name_y>"`). Non-finite numbers
  are the atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and coefficients; like spreg, `k` counts `rho`.
    * `betas`, `rho`, `vm` (covariance of `betas`), `vm1` (covariance of `betas`, `rho` and
      `sigma^2`), `std_err`, `z_stat` (`[{z, p}]`).
    * `logll`, `aic`, `schwarz`, `sig2` (`u'u / n`), `utu`.
    * `u` residuals, `predy` (`X beta + rho W y`), `predy_e` (the reduced-form prediction
      `(I - rho W)^-1 X beta` by power expansion, truncated at `epsilon`), `e_pred` (`y - predy_e`).
    * `pr2` (squared correlation of `y` and `predy`), `pr2_e` (same with `predy_e`).
    * `mean_y`, `std_y` (sample standard deviation).
    * `sp_multipliers` - `%{simple: {adi, aii, ati}, full: ..., power: ...}` for the requested
      `spat_impacts`; `impacts` - the same keys mapping to a list of
      `%{name, direct, indirect, total}` per non-constant regressor; `power_terms` - powers used by
      the power approximation. All `nil` with `spat_impacts: nil`.
    * `method`, `epsilon`, `iterations` (objective evaluations), `converged`, `name_x`, `name_y`,
      `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :betas,
    :rho,
    :vm,
    :vm1,
    :std_err,
    :z_stat,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :utu,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :pr2,
    :pr2_e,
    :mean_y,
    :std_y,
    :sp_multipliers,
    :impacts,
    :power_terms,
    :method,
    :epsilon,
    :iterations,
    :converged,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type pair :: {value(), value()}
  @type multipliers :: {value(), value(), value()}

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          betas: [value()],
          rho: value(),
          vm: [[value()]],
          vm1: [[value()]],
          std_err: [value()],
          z_stat: [pair()],
          logll: value(),
          aic: value(),
          schwarz: value(),
          sig2: value(),
          utu: value(),
          u: [value()],
          predy: [value()],
          predy_e: [value()],
          e_pred: [value()],
          pr2: value(),
          pr2_e: value() | nil,
          mean_y: value(),
          std_y: value(),
          sp_multipliers: %{optional(atom()) => multipliers()} | nil,
          impacts: %{optional(atom()) => [map()]} | nil,
          power_terms: pos_integer() | nil,
          method: :full | :ord | :lu,
          epsilon: float(),
          iterations: pos_integer(),
          converged: boolean(),
          name_x: [String.t()],
          name_y: String.t(),
          warnings: [String.t()]
        }
end
