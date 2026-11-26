defmodule Spatex.Regression.GMCombo do
  @moduledoc """
  Result of `Spatex.Regression.gm_combo/4`: the spatial lag model with spatially autocorrelated
  errors, `y = rho W y + X beta + u`, `u = lambda W u + e` (`GM_Combo`, `GM_Combo_Het`,
  `GM_Combo_Hom`; spreg 1.9.1) in snake_case.

  Matrices are lists of rows. `betas` lists the constant, the exogenous and ordinary endogenous
  regressors, `rho` and `lam` last, matching `name_x`. Non-finite numbers are the atoms `:nan`,
  `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and regressors (`rho` included, `lam` not).
    * `estimator` - `:kp98`, `:het` or `:hom`.
    * `betas`, `rho`, `lam`, `vm`, `std_err`, `z_stat` (`[{z, p}]`): with `:kp98` the covariance is
      `k x k` (no `lam` entry), with `:het` and `:hom` `(k + 1) x (k + 1)`.
    * `sig2` - `nil` for `:het`.
    * `u` (`y - Z beta`), `predy`, `predy_e` (reduced-form prediction, zeros when `|rho| >= 1`),
      `e_pred`, `e_filtered`, `pr2`, `pr2_e` (`nil` when `|rho| >= 1`).
    * `iteration`, `iter_stop`, `step1c`, `max_iter`, `epsilon`, `a1`, `lambdas` as in
      `Spatex.Regression.GMError`.
    * `w_lags`, `lag_q`, `name_x`, `name_y`, `name_yend`, `name_q`, `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :estimator,
    :betas,
    :rho,
    :lam,
    :vm,
    :std_err,
    :z_stat,
    :sig2,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :e_filtered,
    :pr2,
    :pr2_e,
    :iteration,
    :iter_stop,
    :step1c,
    :max_iter,
    :epsilon,
    :a1,
    :lambdas,
    :w_lags,
    :lag_q,
    :name_x,
    :name_y,
    :name_yend,
    :name_q,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          estimator: :kp98 | :het | :hom,
          betas: [value()],
          rho: value(),
          lam: value(),
          vm: [[value()]],
          std_err: [value()],
          z_stat: [{value(), value()}],
          sig2: value() | nil,
          u: [value()],
          predy: [value()],
          predy_e: [value()],
          e_pred: [value()],
          e_filtered: [value()],
          pr2: value(),
          pr2_e: value() | nil,
          iteration: pos_integer() | nil,
          iter_stop: String.t() | nil,
          step1c: boolean() | nil,
          max_iter: pos_integer() | nil,
          epsilon: float() | nil,
          a1: :hom_sc | :hom | :het | nil,
          lambdas: [value()],
          w_lags: pos_integer(),
          lag_q: boolean(),
          name_x: [String.t()],
          name_y: String.t(),
          name_yend: [String.t()],
          name_q: [String.t()],
          warnings: [String.t()]
        }
end
