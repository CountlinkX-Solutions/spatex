defmodule Spatex.Regression.GMError do
  @moduledoc """
  Result of `Spatex.Regression.gm_error/4`: a GMM spatial error model
  `y = X beta + u`, `u = lambda W u + e` for any of the three estimators (`GM_Error`,
  `GM_Error_Het`, `GM_Error_Hom` and their `Endog` variants with `yend` / `q`), with the
  attribute names of spreg 1.9.1 in snake_case.

  Matrices are lists of rows. `betas` lists the regressors and `lam` last, matching `name_x`
  (whose last entry is `"lambda"`). Non-finite numbers are the atoms `:nan`, `:infinity` and
  `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and regressors (`lam` not counted).
    * `estimator` - `:kp98`, `:het` or `:hom`.
    * `betas`, `lam`, `vm`, `std_err`, `z_stat` (`[{z, p}]`, normal). With `:kp98` the covariance
      is `k x k` and `lam` has no standard error (`std_err` and `z_stat` have `k` entries); with
      `:het` and `:hom` it is `(k + 1) x (k + 1)` including `lam`.
    * `sig2` - `nil` for `:het` (spreg reports none).
    * `u` (`y - X beta`), `predy`, `e_filtered` (`u - lam W u`), `pr2`.
    * `iteration`, `iter_stop` (spreg's message), `step1c`, `max_iter`, `epsilon`, `a1`
      (`:hom` only) - `nil` for `:kp98`, which has no iterations.
    * `lambdas` - every lambda the moment optimizer returned, in call order (first step, then each
      iteration).
    * `name_x` (coefficient names, `"lambda"` last), `name_y`, `name_yend`, `name_q`, `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :estimator,
    :betas,
    :lam,
    :vm,
    :std_err,
    :z_stat,
    :sig2,
    :u,
    :predy,
    :e_filtered,
    :pr2,
    :iteration,
    :iter_stop,
    :step1c,
    :max_iter,
    :epsilon,
    :a1,
    :lambdas,
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
          lam: value(),
          vm: [[value()]],
          std_err: [value()],
          z_stat: [{value(), value()}],
          sig2: value() | nil,
          u: [value()],
          predy: [value()],
          e_filtered: [value()],
          pr2: value(),
          iteration: pos_integer() | nil,
          iter_stop: String.t() | nil,
          step1c: boolean() | nil,
          max_iter: pos_integer() | nil,
          epsilon: float() | nil,
          a1: :hom_sc | :hom | :het | nil,
          lambdas: [value()],
          name_x: [String.t()],
          name_y: String.t(),
          name_yend: [String.t()],
          name_q: [String.t()],
          warnings: [String.t()]
        }
end
