defmodule Spatex.Regression.OLS do
  @moduledoc """
  Result of `Spatex.Regression.ols/3`: coefficients, inference and diagnostics, with the
  attribute names of `spreg.OLS` (spreg 1.9.1) in snake_case.

  Matrices (`vm`, `xtx`, `xtxi`) are lists of rows. Tests that spreg reports as `(statistic,
  p-value)` are `{statistic, p}` tuples; those it reports as dictionaries are maps with spreg's
  keys. Fields for diagnostics that were not requested are `nil`. Non-finite numbers are the
  atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and columns of the final design (the constant included).
    * `betas`, `vm`, `std_err`, `t_stat` (`[{t, p}]`), `predy`, `u`, `utu`.
    * `sig2` (the estimator the options chose), `sig2n` (`utu / n`), `sig2n_k` (`utu / (n - k)`),
      `sig2ml` (`sig2n`), `mean_y`, `std_y`, `r2`, `ar2`, `xtx`, `xtxi`.
    * `f_stat` (`{f, p}`), `logll`, `aic`, `schwarz`, `condition_number` (spreg's `mulColli`),
      `jarque_bera` (`%{df, jb, pvalue}`), `breusch_pagan` (`%{df, bp, pvalue}`),
      `koenker_bassett` (`%{df, kb, pvalue}`) - with `nonspat_diag: true` (the default).
    * `white` - `%{df, wh, pvalue}`, `{:not_computed, :multicollinearity}` or `nil`.
    * `vif` - per column `{vif, tolerance}`, `nil` for the constant (zero variance).
    * `lm_error`, `lm_lag`, `rlm_error`, `rlm_lag`, `lm_sarma` and the Koley-Bera tests `lm_wx`,
      `lm_spdurbin`, `rlm_wx`, `rlm_durlag`, `lm_slxerr` - `{statistic, p}` with `spat_diag: true`.
    * `moran_res` - `{i, z, p}`, Moran's I of the residuals, with `moran: true`.
    * `robust` (`nil | :white | :hac`), `name_x`, `name_y`, `warnings`.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :betas,
    :vm,
    :std_err,
    :t_stat,
    :predy,
    :u,
    :utu,
    :sig2,
    :sig2n,
    :sig2n_k,
    :sig2ml,
    :mean_y,
    :std_y,
    :r2,
    :ar2,
    :xtx,
    :xtxi,
    :f_stat,
    :logll,
    :aic,
    :schwarz,
    :condition_number,
    :jarque_bera,
    :breusch_pagan,
    :koenker_bassett,
    :white,
    :vif,
    :lm_error,
    :lm_lag,
    :rlm_error,
    :rlm_lag,
    :lm_sarma,
    :lm_wx,
    :lm_spdurbin,
    :rlm_wx,
    :rlm_durlag,
    :lm_slxerr,
    :moran_res,
    :robust,
    :name_x,
    :name_y,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type pair :: {value(), value()}

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          betas: [value()],
          vm: [[value()]],
          std_err: [value()],
          t_stat: [pair()],
          predy: [value()],
          u: [value()],
          utu: value(),
          sig2: value(),
          sig2n: value(),
          sig2n_k: value(),
          sig2ml: value() | nil,
          mean_y: value(),
          std_y: value(),
          r2: value(),
          ar2: value(),
          xtx: [[value()]],
          xtxi: [[value()]],
          f_stat: pair() | nil,
          logll: value() | nil,
          aic: value() | nil,
          schwarz: value() | nil,
          condition_number: value() | nil,
          jarque_bera: map() | nil,
          breusch_pagan: map() | nil,
          koenker_bassett: map() | nil,
          white: map() | {:not_computed, :multicollinearity} | nil,
          vif: [{value(), value()} | nil] | nil,
          lm_error: pair() | nil,
          lm_lag: pair() | nil,
          rlm_error: pair() | nil,
          rlm_lag: pair() | nil,
          lm_sarma: pair() | nil,
          lm_wx: pair() | nil,
          lm_spdurbin: pair() | nil,
          rlm_wx: pair() | nil,
          rlm_durlag: pair() | nil,
          lm_slxerr: pair() | nil,
          moran_res: {value(), value(), value()} | nil,
          robust: nil | :white | :hac,
          name_x: [String.t()],
          name_y: String.t(),
          warnings: [String.t()]
        }
end
