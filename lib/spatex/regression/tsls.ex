defmodule Spatex.Regression.TSLS do
  @moduledoc """
  Result of `Spatex.Regression.tsls/5`: two-stage least squares, with the attribute names of
  `spreg.TSLS` (spreg 1.9.1) in snake_case.

  Matrices are lists of rows. Non-finite numbers are the atoms `:nan`, `:infinity` and
  `:neg_infinity`.

  ## Fields

    * `n`, `k` - observations and coefficients (constant, exogenous and endogenous columns).
    * `betas`, `vm` (`sig2 * varb`, or the White / HAC covariance), `varb`
      (`(Z'H (H'H)^-1 H'Z)^-1`), `std_err`, `z_stat` (`[{z, p}]`, normal).
    * `sig2` (`u'u / n`, or `u'u / (n - k)` with `sig2n_k: true`), `utu`, `u`, `predy`, `pr2`.
    * `robust` (`nil | :white | :hac`), `name_x` (coefficient names), `name_y`, `name_yend`,
      `name_q`, `warnings`.
  """

  defstruct [
    :n,
    :k,
    :betas,
    :vm,
    :varb,
    :std_err,
    :z_stat,
    :sig2,
    :utu,
    :u,
    :predy,
    :pr2,
    :robust,
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
          betas: [value()],
          vm: [[value()]],
          varb: [[value()]],
          std_err: [value()],
          z_stat: [{value(), value()}],
          sig2: value(),
          utu: value(),
          u: [value()],
          predy: [value()],
          pr2: value(),
          robust: nil | :white | :hac,
          name_x: [String.t()],
          name_y: String.t(),
          name_yend: [String.t()],
          name_q: [String.t()],
          warnings: [String.t()]
        }
end
