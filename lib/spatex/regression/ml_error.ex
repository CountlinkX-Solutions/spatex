defmodule Spatex.Regression.MLError do
  @moduledoc """
  Result of `Spatex.Regression.ml_error/4`: the maximum-likelihood spatial error model
  `y = X beta + u`, `u = lambda W u + e`, with the attribute names of `spreg.ML_Error`
  (spreg 1.9.1) in snake_case.

  Matrices (`vm`, `vm1`) are lists of rows. `betas`, `std_err` and `z_stat` list the regressors
  first and `lam` last, matching `name_x` (whose last entry is `"lambda"`). Non-finite numbers are
  the atoms `:nan`, `:infinity` and `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and regressors; unlike `Spatex.Regression.MLLag`, `k` does not count
      `lam` (so `aic` and `schwarz` use `k`, exactly as spreg does).
    * `betas`, `lam`, `vm` (block diagonal: `beta` and `lam` are uncorrelated), `vm1` (the 2x2
      covariance of `lam` and `sigma^2`), `std_err`, `z_stat` (`[{z, p}]`).
    * `logll`, `aic`, `schwarz`, `sig2` (`e_filtered'e_filtered / n`).
    * `u` (`y - X beta`, not filtered), `predy` (`X beta`), `e_filtered` (`u - lam W u`).
    * `pr2` (squared correlation of `y` and `predy`), `mean_y`, `std_y`.
    * `method`, `epsilon`, `iterations` (objective evaluations), `converged`, `name_x`, `name_y`,
      `warnings`.
  """

  defstruct [
    :n,
    :k,
    :betas,
    :lam,
    :vm,
    :vm1,
    :std_err,
    :z_stat,
    :logll,
    :aic,
    :schwarz,
    :sig2,
    :u,
    :predy,
    :e_filtered,
    :pr2,
    :mean_y,
    :std_y,
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

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          betas: [value()],
          lam: value(),
          vm: [[value()]],
          vm1: [[value()]],
          std_err: [value()],
          z_stat: [pair()],
          logll: value(),
          aic: value(),
          schwarz: value(),
          sig2: value(),
          u: [value()],
          predy: [value()],
          e_filtered: [value()],
          pr2: value(),
          mean_y: value(),
          std_y: value(),
          method: :full | :ord | :lu,
          epsilon: float(),
          iterations: pos_integer(),
          converged: boolean(),
          name_x: [String.t()],
          name_y: String.t(),
          warnings: [String.t()]
        }
end
