defmodule Spatex.GWR.MGWRResult do
  @moduledoc """
  Result of `Spatex.GWR.mgwr/4`, with the attribute names of `mgwr.gwr.MGWRResults` (mgwr 2.2.1)
  in snake_case.

  Matrices (`params`, `bse`, `tvalues`, `cct`) are lists of `n` rows with `k` entries, the
  intercept first when the model has one; `bws_history` has one row of `k` bandwidths per
  backfitting iteration. Non-finite numbers are the atoms `:nan`, `:infinity` and
  `:neg_infinity`.

  ## Fields

    * `n`, `k`, `kernel`, `fixed`, `spherical`, `constant` - the model that was fitted.
    * `bws` - one bandwidth per column (neighbor counts when adaptive, distances when `fixed`).
    * `bw_init`, `bws_history`, `scores`, `iterations`, `converged` - the backfitting trace:
      the starting bandwidth, the bandwidths and convergence score of every iteration, and
      whether the score fell below `tol_multi` before `max_iter_multi`.
    * `sel_hist` - the evaluations `{bw, score}` of every bandwidth search of the backfitting
      (one list per searched covariate and iteration, in order; mgwr's `Sel_BW.sel_hist` of a
      multi-bandwidth selector, which `Spatex.GWR.bandwidth_intervals/2` reads).
    * `params`, `predy` (`sum_j x_ij * params_ij`), `resid`, `resid_ss`, `r2`.
    * Inference (`nil` when fitted with `inference: false`): `bse`, `tvalues`, `cct`,
      `enp_j` (effective number of parameters per covariate), `tr_s` (their sum), `sigma2`
      (`resid_ss / (n - tr_s)`), `llf`, `aic`, `aicc`, `bic`, `adj_r2`, `adj_alpha_j`
      (`k` rows of the 0.1, 0.05 and 0.001 levels divided by `enp_j`) and `critical_tval` (one
      t critical value per covariate for the corrected 0.05 level, `n - 1` degrees of freedom).
    * `influ`, `std_res`, `cooks_d`, `cv` - always `nil`: mgwr cannot compute them for MGWR.
    * `warnings` - messages such as a backfitting that did not converge.
  """

  # credo:disable-for-next-line Credo.Check.Warning.StructFieldAmount
  defstruct [
    :n,
    :k,
    :kernel,
    :fixed,
    :spherical,
    :constant,
    :bws,
    :bw_init,
    :bws_history,
    :sel_hist,
    :scores,
    :iterations,
    :converged,
    :params,
    :predy,
    :resid,
    :resid_ss,
    :r2,
    :bse,
    :tvalues,
    :cct,
    :enp_j,
    :tr_s,
    :sigma2,
    :llf,
    :aic,
    :aicc,
    :bic,
    :adj_r2,
    :adj_alpha_j,
    :critical_tval,
    :influ,
    :std_res,
    :cooks_d,
    :cv,
    warnings: []
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type bw :: pos_integer() | float() | :infinity

  @type t :: %__MODULE__{
          n: pos_integer(),
          k: pos_integer(),
          kernel: :bisquare | :gaussian | :exponential,
          fixed: boolean(),
          spherical: boolean(),
          constant: boolean(),
          bws: [bw()],
          bw_init: bw(),
          bws_history: [[bw()]],
          sel_hist: [[{bw(), value()}]],
          scores: [value()],
          iterations: pos_integer(),
          converged: boolean(),
          params: [[value()]],
          predy: [value()],
          resid: [value()],
          resid_ss: value(),
          r2: value(),
          bse: [[value()]] | nil,
          tvalues: [[value()]] | nil,
          cct: [[value()]] | nil,
          enp_j: [value()] | nil,
          tr_s: value() | nil,
          sigma2: value() | nil,
          llf: value() | nil,
          aic: value() | nil,
          aicc: value() | nil,
          bic: value() | nil,
          adj_r2: value() | nil,
          adj_alpha_j: [[value()]] | nil,
          critical_tval: [value()] | nil,
          influ: nil,
          std_res: nil,
          cooks_d: nil,
          cv: nil,
          warnings: [String.t()]
        }
end
