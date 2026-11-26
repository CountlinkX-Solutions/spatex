defmodule Spatex.Regression.GMLag do
  @moduledoc """
  Result of `Spatex.Regression.gm_lag/4`: the spatial two-stage least squares model
  `y = rho W y + X beta + e` (`spreg.GM_Lag`, spreg 1.9.1) in snake_case.

  Matrices are lists of rows. `betas`, `std_err` and `z_stat` list the constant, the exogenous
  and ordinary endogenous regressors, and `rho` last, matching `name_x`. Non-finite numbers are
  the atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and coefficients (`rho` included).
    * `betas`, `rho`, `vm` (`sig2 * varb`, or the White / HAC covariance), `varb`
      (`(Z'H (H'H)^-1 H'Z)^-1`), `std_err`, `z_stat` (`[{z, p}]`, normal).
    * `sig2` (`u'u / n`, or `u'u / (n - k)` with `sig2n_k: true`), `u` (structural residuals,
      `y - Z betas`), `predy` (`Z betas`), `predy_e` (reduced-form prediction
      `(I - rho W)^-1 X beta`, zeros when `|rho| >= 1`), `e_pred` (`y - predy_e`).
    * `pr2`, `pr2_e` - squared correlations of `y` with `predy` and `predy_e` (`pr2_e` is `nil`
      when `|rho| >= 1`).
    * `ak_test` - Anselin-Kelejian test `{statistic, p}` for residual spatial dependence (`nil`
      with `spat_diag: false` or `robust: :hac`).
    * `sp_multipliers` - `%{simple: {adi, aii, ati}, full: ..., power: ...}` for the requested
      `spat_impacts`; `impacts` - the same keys mapping to a list of
      `%{name, direct, indirect, total}` per exogenous non-constant regressor; `power_terms`.
    * `robust`, `w_lags`, `lag_q`, `name_x`, `name_y`, `name_yend` (ordinary endogenous
      regressors), `name_q` (user instruments), `warnings`.

  The instrument matrix `h` and regressor matrix `z` spreg stores are not returned.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :betas,
    :rho,
    :vm,
    :varb,
    :std_err,
    :z_stat,
    :sig2,
    :u,
    :predy,
    :predy_e,
    :e_pred,
    :pr2,
    :pr2_e,
    :ak_test,
    :sp_multipliers,
    :impacts,
    :power_terms,
    :robust,
    :w_lags,
    :lag_q,
    :name_x,
    :name_y,
    :name_yend,
    :name_q,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type multipliers :: {value(), value(), value()}

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          betas: [value()],
          rho: value(),
          vm: [[value()]],
          varb: [[value()]],
          std_err: [value()],
          z_stat: [{value(), value()}],
          sig2: value(),
          u: [value()],
          predy: [value()],
          predy_e: [value()],
          e_pred: [value()],
          pr2: value(),
          pr2_e: value() | nil,
          ak_test: {value(), value()} | nil,
          sp_multipliers: %{optional(atom()) => multipliers()} | nil,
          impacts: %{optional(atom()) => [map()]} | nil,
          power_terms: pos_integer() | nil,
          robust: nil | :white | :hac,
          w_lags: pos_integer(),
          lag_q: boolean(),
          name_x: [String.t()],
          name_y: String.t(),
          name_yend: [String.t()],
          name_q: [String.t()],
          warnings: [String.t()]
        }
end
